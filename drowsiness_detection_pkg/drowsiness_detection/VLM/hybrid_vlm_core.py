#!/usr/bin/env python3
"""
Hybrid Drowsiness Detection - Core VLM Logic (No ROS, No Camera)
Extracted from webcam script for use in ROS2 subscriber node.

Classes:
- EventTracker: tracks events from start to end with temporal continuity
- VLMRequestHandler: manages async Qwen2.5-VL calls with retry logic
- detect_ambiguity_flags(): detects occlusion/suspicious conditions
- encode_frame_b64(): encodes frames for VLM
- compute_perceptual_hash(): perceptual hashing for deduplication
"""

import os
import cv2
import numpy as np
import threading
import base64
import logging
from collections import deque
from datetime import datetime
from PIL import Image
import time
import json
import requests
import imagehash
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type

logger = logging.getLogger(__name__)

# ----------------------------
# Ollama HTTP Config
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"

# ----------------------------
# SYSTEM PROMPT (strict JSON schema)
# ----------------------------
SYSTEM_PROMPT = """
SYSTEM: You are Qwen2.5-VL (qwen2.5vl:3b) for automated driver behavior evaluation.

⚠️ CRITICAL: The camera is an INFRARED (IR) camera, NOT a regular RGB camera.
- Images will appear grayscale/monochrome - this is NORMAL, not a lighting issue.
- IR cameras detect thermal radiation, so facial features may appear differently than in visible light.
- Do NOT report "poor lighting" or "dark environment" - grayscale is the expected appearance.
- Focus on detecting facial landmarks, head pose, eye state, and behaviors despite the IR format.

You will receive 8 consecutive frames from a driver's face/upper torso.
Produce JSON-only output following exactly this schema:


{
  "clip_id": "<string>",
  "drowsy": {"label": "yes"|"no"|"uncertain","confidence": 0.0-1.0 | null},
  "behaviors": {
      "yawn": {"detected": false,"confidence":0.0},
      "cover_mouth": {"detected": false,"confidence":0.0},
      "eat": {"detected": false,"confidence":0.0},
      "drink": {"detected": false,"confidence":0.0},
      "sneeze": {"detected": false,"confidence":0.0},
      "cry": {"detected": false,"confidence":0.0},
      "micro_sleep": {"detected": false,"confidence":0.0},
      "talking": {"detected": false,"confidence":0.0},
      "hands_on_face": {"detected": false,"confidence":0.0},
      "glasses": {"detected": false,"confidence":0.0},
      "sunglasses": {"detected": false,"confidence":0.0},
      "mask": {"detected": false,"confidence":0.0},
      "other_occlusion": {"detected": false,"confidence":0.0}
  },
  "eye_visibility": {"left_eye":"open"|"closed"|"occluded"|"uncertain","right_eye":"open"|"closed"|"occluded"|"uncertain"},
  "mouth_visibility": "open"|"closed"|"occluded"|"uncertain",
  "head_posture": {"pitch":"neutral","yaw":"center","roll":"neutral"},
  "occlusion": {"face_occluded": true|false, "reason":"mask"|"hand"|"sun_glare"|"other"|null},
  "lighting": {"level":"infrared_normal","issue":null},
  "evidence_frames": [0,1,2,3,4,5,6,7],
  "notes": ""
}


- Always fill all fields. Use "uncertain" when visibility is unclear.
- Use numeric confidence scores 0.0-1.0, or null if completely uncertain.
- Do not output per-frame arrays, PERCLOS, blink rates, or markdown. Return only the JSON object.
- Remember: grayscale appearance is NORMAL for IR cameras - it does NOT indicate poor lighting.
"""

# ----------------------------
# USER PROMPT TEMPLATE
# ----------------------------
USER_PROMPT_TEMPLATE = """
You will analyze the following 8 consecutive frames from a driver's cabin using Qwen2.5VL:3b

⚠️ REMINDER: These images are from an INFRARED camera - grayscale/monochrome appearance is NORMAL.

Treat all 8 frames together as a single 2-second scene. Do NOT evaluate frames independently or produce per-frame outputs.
Aggregate all observed behaviors over the scene: yawning, covering mouth, eating, drinking, sneezing, crying, micro-sleep, eyes closed, looking away, talking, hands on face, wearing glasses/sunglasses, mask occlusion, or any other gestures.
Also report mouth visibility as "open", "closed", or "occluded".
Report face occlusion in "occlusion" field with reason (mask, hand, sun_glare, other, or null).
Provide a short textual summary of the scene in the "notes" field.


Clip id: {clip_id}
Frames: images[0..7] in order (0 = earliest, 7 = latest)


Answer only JSON that strictly conforms to the SYSTEM_PROMPT schema.
Always fill all fields, even if uncertain. Do NOT include per-frame outputs, metrics like PERCLOS or blink rates, or markdown.
"""

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    # Sampling configuration
    "sampling_hz": 4.0,                 # 4 frames per second
    "buffer_duration": 2.0,             # 2 seconds
    # buffer_size is DERIVED: int(sampling_hz * buffer_duration) = 8 frames
    
    # Classical thresholds (from literature)
    "ear_low_threshold": 0.26,          # Soukupová & Čech 2016
    "ear_extreme_threshold": 0.10,      # Very low (potential occlusion)
    "mar_yawn_threshold": 0.50,         # Abtahi 2014
    
    # Blink filtering logic
    "blink_duration_max": 0.3,          # Normal blink: 100-300ms
    "drowsiness_duration_min": 2.0,     # Drowsiness: 2+ seconds
    
    # Brightness thresholds REMOVED for IR camera
    # IR cameras don't have meaningful "brightness" - CLAHE normalization handles histogram
    # These were: brightness_very_dark, brightness_dark, brightness_normal_min, etc.
    
    # Edge density (for detecting occlusion/sunglasses/hands)
    "edge_density_threshold": 0.15,
    
    # Pre-event buffer
    "pre_buffer_size": 4,               # Frames to keep before event starts
    
    # Frame encoding
    "frame_width": 480,
    "frame_height": 360,
    "jpeg_quality": 70,
    
    # VLM triggers
    "vlm_trigger_on_ambiguity": True,
    
    # VLM drowsiness triggers (CRITICAL SAFETY FIX)
    # Trigger VLM not just on ambiguity, but also on clear drowsiness indicators
    "vlm_trigger_drowsy_ear": 0.20,     # Trigger if EAR < 0.20 (very drowsy)
    "vlm_trigger_drowsy_duration": 2.0, # For at least 2 seconds
    "vlm_trigger_yawn_mar": 0.5,       # Trigger if MAR > 0.65 (yawning)
    "vlm_trigger_yawn_duration": 1.0,   # For at least 1 second
    
    # Perceptual hashing for deduplication
    "perceptual_hash_size": 8,          # pHash dimension (8x8 = 64-bit hash)
    "dedup_hash_threshold": 5,          # Hamming distance threshold
    
    # Temporal overlap for evolving behaviors
    "temporal_overlap_ratio": 0.5,      # 50% overlap between submissions
    
    # Event duration limits
    "min_heartbeat_interval": 5.0,      # Min 4s between heartbeat submissions (Increased to avoid congestion)
    "max_event_duration": 60.0,         # Force submit after 60s
    
    # HTTP & threading
    "http_timeout_per_frame": 20,       # Dynamic timeout: 20s * num_frames
    "http_retry_attempts": 3,           # Number of retry attempts
    "http_retry_backoff": 2.0,          # Exponential backoff multiplier
    "max_concurrent_vlm": 2,            # Max parallel VLM requests
    
    # Logging
    "log_level": "INFO",                # DEBUG, INFO, WARNING, ERROR
    
    # Per-stream state
    "per_stream_state": True,           # Enable stream isolation
    
    # Output
    "output_dir": "./vlm_triggers",
}

# Derive buffer size from sampling configuration
CONFIG["buffer_size"] = int(CONFIG["sampling_hz"] * CONFIG["buffer_duration"])

# Setup logging
logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"]),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Helper: Perceptual Hashing
# ----------------------------
def compute_perceptual_hash(frame_gray):
    """
    Compute perceptual hash that captures scene semantics.
    Uses pHash algorithm which is robust to minor changes.
    Designed for grayscale IR camera input.
    
    Args:
        frame_gray: OpenCV grayscale image (IR camera)
    
    Returns:
        ImageHash object (can compute Hamming distance with -)
    """
    try:
        # Input is already grayscale from IR camera
        pil_img = Image.fromarray(frame_gray)
            
        return imagehash.phash(pil_img, hash_size=CONFIG["perceptual_hash_size"])
    except Exception as e:
        logger.error(f"Perceptual hash computation failed: {e}")
        return None

def compute_temporal_signature(frames):
    """
    Create signature from multiple frames for better deduplication.
    Samples every other frame to reduce computation while maintaining temporal coverage.
    
    Args:
        frames: List of grayscale frames (IR camera)
    
    Returns:
        List of ImageHash objects
    """
    if not frames:
        return []
    
    # Sample every other frame (reduces computation by 50%)
    sampled_frames = frames[::2]
    hashes = []
    
    for frame in sampled_frames:
        h = compute_perceptual_hash(frame)
        if h is not None:
            hashes.append(h)
    
    return hashes

# ----------------------------
# Helper: Ambiguity Detection
# ----------------------------
def detect_ambiguity_flags(frame_gray, face_bbox, ear, mar, face_conf):
    """
    Detect situations requiring VLM validation.
    
    IMPORTANT: These flags indicate "Classical metrics may be unreliable - defer to VLM".
    They are TRIGGERS for secondary validation, NOT final assessments of drowsiness.
    
    Args:
        frame_gray: OpenCV grayscale image (IR camera input)
        face_bbox: Tuple (x, y, w, h) or None
        ear: Eye Aspect Ratio or None
        mar: Mouth Aspect Ratio or None
        face_conf: Face detection confidence or None
    
    Returns:
        List of flags: ["no_face", "extreme_low_ear", "conflicting_eye_mouth", "busy_edges"]
    """
    flags = []

    # Check face detection
    # RELAXED LOGIC: Accept if we have a bbox OR high confidence (e.g. from EAR proxy)
    # Fail only if:
    # 1. Low confidence (explicitly unreliable)
    # 2. OR No bbox AND no confidence info (missing everything)
    low_conf = face_conf is not None and face_conf < 0.7
    missing_all_info = face_bbox is None and face_conf is None
    
    if low_conf or missing_all_info:
        flags.append("no_face")
        logger.debug("Ambiguity: No reliable face detection")
        return flags  # Can't do reliable face-specific checks without a face

    # Extreme EAR values
    if ear is not None and ear < CONFIG["ear_extreme_threshold"]:
        flags.append("extreme_low_ear")
        logger.debug(f"Ambiguity: Extreme low EAR = {ear:.3f}")

    # Conflicting signals (eyes closed + mouth open = yawning or occlusion?)
    if ear is not None and mar is not None:
        eyes_closed = ear < CONFIG["ear_low_threshold"]
        mouth_open = mar > CONFIG["mar_yawn_threshold"]
        if eyes_closed and mouth_open:
            flags.append("conflicting_eye_mouth")
            logger.debug(f"Ambiguity: Conflicting EAR={ear:.3f} MAR={mar:.3f}")

  

    # Edge density - USE FACE ROI ONLY (already grayscale)
    if face_bbox:
        x, y, w, h = face_bbox
        # Ensure bbox is within frame bounds
        frame_h, frame_w = frame_gray.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        
        if w > 0 and h > 0:
            face_roi = frame_gray[y:y+h, x:x+w]
            # Already grayscale - no conversion needed
            edges = cv2.Canny(face_roi, 30, 100)
            edge_density = np.mean(edges > 0)
            
            if edge_density > CONFIG["edge_density_threshold"]:
                flags.append("busy_edges")
                logger.debug(f"Ambiguity: Busy edges in face ROI ({edge_density:.3f})")

    return flags

def should_trigger_vlm(has_ambiguity, ear, mar, event_duration):
    """
    Determine if VLM should be triggered.
    
    CRITICAL SAFETY FIX: Triggers on BOTH ambiguity AND clear drowsiness.
    Prevents missing high-confidence drowsiness that lacks ambiguity flags.
    
    Args:
        has_ambiguity: Boolean - are ambiguity flags present?
        ear: Current EAR value or None
        mar: Current MAR value or None
        event_duration: How long the event has been active (seconds)
    
    Returns:
        Tuple: (should_trigger: bool, reason: str)
    """
    # Trigger 1: Ambiguity (existing behavior)
    if has_ambiguity:
        return True, "ambiguity"
    
    # Trigger 2: Clear drowsiness - very low EAR sustained
    if ear is not None and ear < CONFIG["vlm_trigger_drowsy_ear"]:
        if event_duration >= CONFIG["vlm_trigger_drowsy_duration"]:
            return True, f"drowsy_ear_{ear:.3f}"
    
    # Trigger 3: Clear yawning - high MAR sustained
    if mar is not None and mar > CONFIG["vlm_trigger_yawn_mar"]:
        if event_duration >= CONFIG["vlm_trigger_yawn_duration"]:
            return True, f"yawning_mar_{mar:.3f}"
    
    return False, None

def encode_frame_b64(frame, quality=70):
    """
    Encode frame to base64 JPEG.
    CRITICAL: Converts input to Grayscale before encoding to match IR camera nature
    and ensure VLM sees pure monochrome data.
    """
    try:
        # Convert to grayscale if not already (Senior Engineer Best Practice: Data Normalization)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        # Encode (imencode handles single channel correctly)
        success, buf = cv2.imencode('.jpg', frame_gray, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if success:
            return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:
        logger.error(f"Frame encoding failed: {e}")
    
    return None

# ----------------------------
# Event Tracker
# ----------------------------
class EventTracker:
    """
    Tracks drowsiness/occlusion events with temporal continuity and per-stream isolation.
    
    Key improvements:
    - Maintains event history during heartbeats (no clearing)
    - Tracks metrics timeline throughout event
    - Timestamp alignment for all frames
    - Temporal overlap for evolving behaviors
    - Per-stream state isolation
    """

    def __init__(self, stream_id="default"):
        self.stream_id = stream_id
        self.buffer_size = CONFIG["buffer_size"]
        
        # Event state
        self.event_active = False
        self.current_event_id = None
        self.event_counter = 0  # Counter to prevent ID collisions
        
        # Buffers - Larger to support overlap and continuity
        max_buffer = self.buffer_size * 3  # Allow for overlap + accumulation
        self.event_frames = deque(maxlen=max_buffer)
        self.event_metrics = deque(maxlen=max_buffer)
        self.event_timestamps = deque(maxlen=max_buffer)
        
        # Pre-buffer for context before event starts
        self.pre_buffer_frames = deque(maxlen=CONFIG["pre_buffer_size"])
        self.pre_buffer_metrics = deque(maxlen=CONFIG["pre_buffer_size"])
        self.pre_buffer_timestamps = deque(maxlen=CONFIG["pre_buffer_size"])
        
        # Timing
        self.event_start_time = None
        self.last_submission_time = None
        self.last_heartbeat_time = None
        
        # State tracking
        self.last_flags = None
        self.accumulated_flags = set()  # Track all unique flags during event
        
        logger.info(f"EventTracker initialized for stream: {stream_id}")

    def update(self, frame, timestamp, has_flags, current_flags, metrics):
        """
        Update event state with new frame, metrics, and timestamp.
        
        Args:
            frame: BGR image
            timestamp: Unix timestamp or frame timestamp
            has_flags: Boolean - are there ambiguity flags?
            current_flags: List of current flag strings
            metrics: Dict of metrics (ear, mar, brightness, etc.)
        
        Returns:
            Tuple: (frames, timestamps, metrics_list, should_submit, event_id)
        """
        should_submit = False
        frames_to_submit = None
        timestamps_to_submit = None
        metrics_to_submit = None
        event_id_to_submit = None
        
        # CRITICAL FIX: Use frame timestamp consistently, not time.time()
        # This ensures temporal reasoning aligns with actual frame capture time
        current_time = timestamp

        # Always maintain pre-buffer while not in an event
        if not self.event_active:
            self.pre_buffer_frames.append(frame.copy())
            self.pre_buffer_timestamps.append(timestamp)
            self.pre_buffer_metrics.append(metrics.copy())

        # Event Start: Flags just triggered
        if has_flags and not self.event_active:
            self.event_active = True
            self.event_start_time = current_time  # Frame timestamp, not time.time()
            self.last_submission_time = current_time
            self.last_heartbeat_time = current_time
            
            # Generate unique event ID with microseconds and counter
            self.event_counter += 1
            timestamp_us = int(current_time * 1000000)  # Microseconds
            self.current_event_id = f"{self.stream_id}_{timestamp_us}_{self.event_counter}"
            
            # Clear and initialize with pre-buffer (instant replay)
            self.event_frames.clear()
            self.event_metrics.clear()
            self.event_timestamps.clear()
            
            self.event_frames.extend(self.pre_buffer_frames)
            self.event_metrics.extend(self.pre_buffer_metrics)
            self.event_timestamps.extend(self.pre_buffer_timestamps)
            
            # Add current frame
            self.event_frames.append(frame.copy())
            self.event_metrics.append(metrics.copy())
            self.event_timestamps.append(timestamp)
            
            self.last_flags = current_flags
            # Accumulate all unique flags seen during event
            self.accumulated_flags.update(current_flags)
            
            logger.info(f"[{self.stream_id}] Event {self.current_event_id} STARTED "
                       f"(pre-buffered {len(self.pre_buffer_frames)} frames)")
            logger.debug(f"[{self.stream_id}] Flags: {', '.join(current_flags)}")
            
            return None, None, None, False, None

        # Event Active: Continue recording
        elif has_flags and self.event_active:
            # Add frame to ongoing event
            self.event_frames.append(frame.copy())
            self.event_metrics.append(metrics.copy())
            self.event_timestamps.append(timestamp)
            self.last_flags = current_flags
            # Accumulate all unique flags
            self.accumulated_flags.update(current_flags)
            
            event_duration = current_time - self.event_start_time
            time_since_heartbeat = current_time - self.last_heartbeat_time
            
            # Check if we should send a heartbeat submission
            should_heartbeat = (
                time_since_heartbeat >= CONFIG["min_heartbeat_interval"] or
                event_duration >= CONFIG["max_event_duration"]
            )
            
            if should_heartbeat and len(self.event_frames) >= self.buffer_size:
                # Heartbeat submission with overlap preservation
                overlap_size = int(self.buffer_size * CONFIG["temporal_overlap_ratio"])
                
                # Submit most recent buffer_size frames
                frames_to_submit = list(self.event_frames)[-self.buffer_size:]
                timestamps_to_submit = list(self.event_timestamps)[-self.buffer_size:]
                metrics_to_submit = list(self.event_metrics)[-self.buffer_size:]
                
                # Unique heartbeat ID with microseconds
                timestamp_us = int(current_time * 1000000)
                event_id_to_submit = f"{self.current_event_id}_hb{timestamp_us}"
                
                # CRITICAL: Don't clear - maintain temporal continuity
                # Keep only overlap_size + buffer_size for memory management
                keep_size = overlap_size + self.buffer_size
                if len(self.event_frames) > keep_size * 2:
                    # Trim oldest frames to prevent unbounded growth
                    trim_count = len(self.event_frames) - keep_size
                    for _ in range(trim_count):
                        self.event_frames.popleft()
                        self.event_metrics.popleft()
                        self.event_timestamps.popleft()
                
                self.last_heartbeat_time = current_time
                self.last_submission_time = current_time
                
                logger.info(f"[{self.stream_id}] Event HEARTBEAT "
                           f"(duration={event_duration:.1f}s, submitted {len(frames_to_submit)} frames)")
                logger.debug(f"[{self.stream_id}] Flags: {', '.join(current_flags)}")
                
                return frames_to_submit, timestamps_to_submit, metrics_to_submit, True, event_id_to_submit
            
            else:
                # Still accumulating
                logger.debug(f"[{self.stream_id}] Recording frame {len(self.event_frames)}/{self.buffer_size}")
                return None, None, None, False, None

        # Event End: Flags cleared
        elif not has_flags and self.event_active:
            event_duration = current_time - self.event_start_time
            
            # CRITICAL FIX: Pad with last captured frame, not current frame
            # Don't fabricate post-event content into the event
            if len(self.event_frames) < self.buffer_size:
                # Get last frame from event history
                last_event_frame = self.event_frames[-1] if self.event_frames else frame
                last_event_timestamp = self.event_timestamps[-1] if self.event_timestamps else timestamp
                last_event_metrics = self.event_metrics[-1] if self.event_metrics else metrics
                
                padding_needed = self.buffer_size - len(self.event_frames)
                logger.warning(
                    f"[{self.stream_id}] Event ended short - padding with {padding_needed} frames "
                    f"(repeated last event frame)"
                )
                
                for _ in range(padding_needed):
                    self.event_frames.append(last_event_frame.copy())
                    # Same timestamp indicates this is a padded duplicate
                    self.event_timestamps.append(last_event_timestamp)
                    self.event_metrics.append(last_event_metrics.copy())
            
            # Submit final event
            frames_to_submit = list(self.event_frames)[-self.buffer_size:]
            timestamps_to_submit = list(self.event_timestamps)[-self.buffer_size:]
            metrics_to_submit = list(self.event_metrics)[-self.buffer_size:]
            event_id_to_submit = f"{self.current_event_id}_end"
            
            logger.info(f"[{self.stream_id}] Event {self.current_event_id} ENDED "
                       f"(duration={event_duration:.1f}s, submitted {len(frames_to_submit)} frames)")
            logger.debug(f"[{self.stream_id}] Final flags: {', '.join(self.last_flags)}")
            
            # Save accumulated flags BEFORE clearing (caller needs to access them)
            flags_to_return = set(self.accumulated_flags)  # Make a copy
            
            # Reset event state
            self.event_active = False
            self.current_event_id = None
            self.last_flags = None
            self.accumulated_flags = set()  # Reset for next event
            
            # Store the flags so caller can access them via self.last_event_flags
            self.last_event_flags = flags_to_return
            
            return frames_to_submit, timestamps_to_submit, metrics_to_submit, True, event_id_to_submit

        # No flags, no event
        return None, None, None, False, None

# ----------------------------
# VLM Request Handler
# ----------------------------
# VLM Request Handler
# ----------------------------
class VLMRequestHandler:
    """
    Thread-safe VLM request handler with comprehensive improvements.
    
    Features:
    - Per-stream state isolation
    - Perceptual hash deduplication
    - Lenient JSON parsing with recovery
    - HTTP retry with exponential backoff
    - Structured logging
    - Dynamic timeouts
    - Model versioning
    """

    def __init__(self, max_workers=2, output_dir="./vlm_triggers"):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VLM")
        self.semaphore = threading.BoundedSemaphore(max_workers)  # Use public API
        self.output_dir = output_dir
        
        # Per-stream state for deduplication
        self.stream_states = {}  # stream_id -> {last_hashes, last_flags, last_time}
        self.state_lock = threading.Lock()
        
        # Metrics
        self.pending_requests = 0
        self.completed_requests = 0
        self.dropped_requests = 0
        self.failed_requests = 0
        self.metrics_lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.VLMHandler")
        
        # Model versioning for provenance
        self.model_version = self._get_model_version()
        
        self.logger.info(f"VLMRequestHandler initialized (workers={max_workers})")

    def _get_model_version(self):
        """Get current Ollama model version for provenance tracking"""
        try:
            resp = requests.get(
                "http://localhost:11434/api/show",
                json={"name": MODEL_NAME},
                timeout=5
            )
            if resp.ok:
                info = resp.json()
                return {
                    "model": MODEL_NAME,
                    "modified_at": info.get("modified_at"),
                    "size": info.get("size"),
                }
        except Exception as e:
            self.logger.warning(f"Could not fetch model version: {e}")
        
        return {"model": MODEL_NAME, "version": "unknown"}

    def should_submit(self, stream_id, frames, clip_id, flags):
        """
        Advanced deduplication using perceptual hashing.
        
        NOTE: This only checks if submission should occur.
        State is NOT updated here - only after successful acquisition.
        
        Args:
            stream_id: Stream identifier
            frames: List of frames
            clip_id: Clip identifier (used to detect heartbeat events)
            flags: Current flags
        
        Returns:
            Tuple: (should_submit: bool, reason: str, hashes: list)
        """
        with self.state_lock:
            # Initialize stream state if needed
            if stream_id not in self.stream_states:
                self.stream_states[stream_id] = {
                    "last_hashes": [],
                    "last_flags": [],
                    "last_time": 0
                }
            
            # 0. SKIP DEDUPLICATION FOR HEARTBEATS
            # Heartbeats are essential for tracking "progress" of static events (e.g. eyes closed)
            # If we drop them as "visual duplicates", we lose the timeline.
            if "_hb" in clip_id:
                # Still compute hash (for consistent state tracking) but FORCE submission
                # current_hashes = compute_temporal_signature(frames)
                # We return True regardless of similarity
                # BUT we should probably compute hashes anyway to keep state somewhat fresh?
                # For safety, let's just return True and claim "heartbeat_forced"
                # The state update in submit_request will handle the rest.
                current_hashes = compute_temporal_signature(frames)
                return True, "heartbeat_forced", current_hashes
            
            state = self.stream_states[stream_id]
            
            # 1. State Change Check (flags changed)
            current_flags_sorted = sorted(flags) if flags else []
            last_flags_sorted = sorted(state["last_flags"]) if state["last_flags"] else []
            
            if current_flags_sorted != last_flags_sorted:
                self.logger.info(f"[{stream_id}] State changed: {state['last_flags']} → {flags}")
                # Compute hashes but DON'T update state yet
                current_hashes = compute_temporal_signature(frames)
                return True, "state_change", current_hashes
            
            # 2. ACTUALLY USE Perceptual Hash Check (visual similarity)
            current_hashes = compute_temporal_signature(frames)
            
            if not current_hashes:
                self.logger.warning(f"[{stream_id}] Failed to compute perceptual hash")
                return True, "hash_computation_failed", []
            
            if state["last_hashes"] and current_hashes:
                # CRITICAL FIX: Validate lengths match before zip
                if len(current_hashes) != len(state["last_hashes"]):
                    self.logger.warning(
                        f"[{stream_id}] Hash count mismatch: {len(current_hashes)} vs "
                        f"{len(state['last_hashes'])} - treating as novel"
                    )
                    return True, "hash_length_mismatch", current_hashes
                
                # Compare hamming distances
                try:
                    distances = [
                        hash1 - hash2  # Hamming distance via __sub__
                        for hash1, hash2 in zip(current_hashes, state["last_hashes"])
                    ]
                    avg_distance = np.mean(distances) if distances else float('inf')
                    
                    if avg_distance < CONFIG["dedup_hash_threshold"]:
                        self.logger.debug(f"[{stream_id}] Visual duplicate (distance={avg_distance:.1f})")
                        return False, "visual_duplicate", current_hashes
                    else:
                        self.logger.debug(f"[{stream_id}] Novel scene (distance={avg_distance:.1f})")
                except Exception as e:
                    self.logger.warning(f"[{stream_id}] Hash comparison failed: {e}")
                    return True, "hash_comparison_error", current_hashes
            
            # Novel scene - return hashes but don't update state yet
            return True, "novel_scene", current_hashes

    def submit_request(self, stream_id, frames, timestamps, metrics, clip_id, flags):
        """
        Submit VLM request to thread pool (non-blocking).
        
        CRITICAL FIX: State only updated AFTER successful semaphore acquisition.
        If request is dropped, deduplication state remains unchanged.
        
        Args:
            stream_id: Stream identifier
            frames: List of BGR frames
            timestamps: List of frame timestamps
            metrics: List of metric dicts
            clip_id: Clip identifier
            flags: Ambiguity flags
        
        Returns:
            Future object or None if dropped/skipped
        """
        # Deduplication check - does NOT mutate state
        should_submit, reason, current_hashes = self.should_submit(stream_id, frames, clip_id, flags)
        
        if not should_submit:
            self.logger.debug(f"[{stream_id}] Skipping {clip_id}: {reason}")
            return None
        
        with self.metrics_lock:
            self.pending_requests += 1
        
        # Try to acquire semaphore (non-blocking)
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            self.logger.warning(
                f"[{stream_id}] System busy (max {CONFIG['max_concurrent_vlm']} concurrent) "
                f"- DROPPING {clip_id} - State NOT updated"
            )
            with self.metrics_lock:
                self.pending_requests -= 1
                self.dropped_requests += 1
            # CRITICAL: State was NOT updated, so next submission will retry
            return None
        
        # SUCCESS - NOW update deduplication state
        with self.state_lock:
            state = self.stream_states[stream_id]
            state["last_hashes"] = current_hashes
            state["last_flags"] = list(flags)
            state["last_time"] = time.time()
        
        # Submit to thread pool
        try:
            future = self.executor.submit(
                self._vlm_request_blocking,
                stream_id, frames, timestamps, metrics, clip_id, flags
            )
            future.add_done_callback(lambda f: self._on_request_complete())
            return future
        except Exception as e:
            self.logger.error(f"[{stream_id}] Failed to submit request: {e}")
            with self.metrics_lock:
                self.pending_requests -= 1
            self.semaphore.release()
            return None

    def _on_request_complete(self):
        """Called when a VLM request completes"""
        with self.metrics_lock:
            self.pending_requests -= 1
            self.completed_requests += 1
        self.semaphore.release()

    def _http_request_with_retry(self, payload, timeout):
        """
        HTTP request with exponential backoff retry.
        
        CRITICAL FIX: Only retries TRANSIENT failures (timeout, connection, 5xx).
        Does NOT retry deterministic failures (4xx client errors - bad schema/prompt).
        """
        @retry(
            retry=retry_if_exception_type((
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            )),
            stop=stop_after_attempt(CONFIG["http_retry_attempts"]),
            wait=wait_exponential(multiplier=CONFIG["http_retry_backoff"], min=4, max=60),
            reraise=True
        )
        def _do_request():
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            
            # CRITICAL: Don't retry 4xx errors (client errors - bad prompt/schema)
            # These are deterministic failures that won't fix themselves
            if 400 <= resp.status_code < 500:
                self.logger.error(
                    f"Client error {resp.status_code} - NOT retrying (likely schema/prompt issue)"
                )
                resp.raise_for_status()  # Raises without triggering retry
            
            # Will retry 5xx errors (server errors - potentially transient)
            resp.raise_for_status()
            return resp
        
        return _do_request()

    def _vlm_request_blocking(self, stream_id, frames, timestamps, metrics, clip_id, flags):
        """
        Process VLM request in background thread.
        Enhanced with retry logic, lenient parsing, and full provenance.
        """
        start_time = time.time()
        
        # Dynamic timeout based on frame count
        timeout = len(frames) * CONFIG["http_timeout_per_frame"]
        
        # Encode frames
        imgs_b64 = [encode_frame_b64(f, CONFIG["jpeg_quality"]) for f in frames]
        if not all(imgs_b64):
            self.logger.error(f"[{stream_id}] Frame encoding failed for {clip_id}")
            return {"error": "frame_encoding_failed"}
        
        # Build payload
        payload = {
            "model": MODEL_NAME,
            "system": SYSTEM_PROMPT,
            "prompt": USER_PROMPT_TEMPLATE.format(clip_id=clip_id),
            "images": imgs_b64,
            "stream": False,
        }
        
        try:
            self.logger.info(
                f"[{stream_id}] Analyzing {clip_id} "
                f"({len(frames)} frames, timeout={timeout}s, flags={','.join(flags)})"
            )
            
            # HTTP request with retry
            resp = self._http_request_with_retry(payload, timeout)
            elapsed = time.time() - start_time
            
            resp.raise_for_status()
            vlm_json = resp.json()
            
            if "response" in vlm_json:
                response_text = vlm_json["response"]
                
                # Lenient JSON parsing
                vlm_analysis = self._parse_vlm_response(response_text, clip_id)
                
                if "error" not in vlm_analysis or vlm_analysis.get("partial", False):
                    drowsy_label = vlm_analysis.get("drowsy", {}).get("label", "unknown")
                    drowsy_conf = vlm_analysis.get("drowsy", {}).get("confidence", 0)
                    
                    self.logger.info(
                        f"[{stream_id}] VLM complete ({elapsed:.1f}s): "
                        f"drowsy={drowsy_label} (conf={drowsy_conf})"
                    )
                    
                    # Save results
                    self._save_clip(stream_id, clip_id, frames, timestamps, metrics,
                                   vlm_analysis, flags, elapsed)
                    return vlm_analysis
                else:
                    self.logger.warning(f"[{stream_id}] Parse error: {vlm_analysis.get('error')}")
                    self._save_clip(stream_id, clip_id, frames, timestamps, metrics,
                                   vlm_analysis, flags, elapsed)
                    return vlm_analysis
            else:
                return vlm_json
        
        except RetryError as e:
            self.logger.error(f"[{stream_id}] All retries exhausted for {clip_id}: {e}")
            with self.metrics_lock:
                self.failed_requests += 1
            return {"error": "retry_exhausted", "details": str(e)}
        
        except requests.Timeout:
            self.logger.error(f"[{stream_id}] Timeout after {timeout}s for {clip_id}")
            with self.metrics_lock:
                self.failed_requests += 1
            return {"error": "timeout", "timeout_duration": timeout}
        
        except Exception as e:
            self.logger.error(f"[{stream_id}] HTTP error for {clip_id}: {e}", exc_info=True)
            with self.metrics_lock:
                self.failed_requests += 1
            return {"error": str(e)}

    def _parse_vlm_response(self, response_text, clip_id):
        """
        Lenient JSON parsing with partial recovery.
        Attempts multiple strategies to extract valid JSON.
        """
        # Strategy 1: Strict parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.debug(f"Strict JSON parse failed: {e}")
        
        # Strategy 2: Extract from markdown code blocks
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                self.logger.info("Recovered JSON from markdown block")
                return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find first { to last }
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(response_text[start:end+1])
                self.logger.info("Recovered JSON from bracket extraction")
                return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Partial recovery - return error with raw text
        self.logger.warning(f"All JSON recovery strategies failed for {clip_id}")
        return {
            "error": "json_parse_failed",
            "raw_response": response_text[:500],  # Truncate to avoid huge logs
            "clip_id": clip_id,
            "partial": True
        }

    def _save_clip(self, stream_id, clip_id, frames, timestamps, metrics,
                   vlm_analysis, flags, inference_time):
        """Save frames and analysis with enhanced metadata and provenance"""
        out_dir = os.path.join(self.output_dir, "events", stream_id, clip_id)
        
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create directory {out_dir}: {e}")
            return
        
        # Save frames with timestamps in filename
        frames_saved = 0
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            try:
                frame_path = os.path.join(out_dir, f"frame_{i:02d}_{ts:.3f}.jpg")
                if cv2.imwrite(frame_path, frame):
                    frames_saved += 1
            except Exception as e:
                self.logger.error(f"Error saving frame {i}: {e}")
        
        # Save enhanced metadata with full provenance
        try:
            analysis = {
                "clip_id": clip_id,
                "stream_id": stream_id,
                "timestamp": datetime.now().isoformat(),
                "frame_timestamps": timestamps,
                "frames_count": frames_saved,
                "flags_detected": flags,
                "metrics_timeline": metrics,  # Full timeline, not just start
                "vlm_analysis": vlm_analysis,
                "inference_time_seconds": inference_time,
                "model_info": self.model_version,
                "config_snapshot": {
                    "sampling_hz": CONFIG["sampling_hz"],
                    "buffer_duration": CONFIG["buffer_duration"],
                    "buffer_size": CONFIG["buffer_size"],
                    "model": MODEL_NAME,
                    "ollama_url": OLLAMA_URL,
                }
            }
            
            json_path = os.path.join(out_dir, "analysis.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(analysis, fh, indent=2, ensure_ascii=False)
            
            self.logger.info(f"[{stream_id}] Saved {frames_saved}/{len(frames)} frames → {out_dir}")
        
        except Exception as e:
            self.logger.error(f"[{stream_id}] Save error: {e}", exc_info=True)

    def shutdown(self):
        """Gracefully shut down the handler."""
        self.logger.info("Shutting down VLM thread pool...")
        self.logger.info(
            f"Stats - Pending: {self.pending_requests}, "
            f"Completed: {self.completed_requests}, "
            f"Dropped: {self.dropped_requests}, "
            f"Failed: {self.failed_requests}"
        )
        self.executor.shutdown(wait=True)
        self.logger.info("VLM thread pool shutdown complete")