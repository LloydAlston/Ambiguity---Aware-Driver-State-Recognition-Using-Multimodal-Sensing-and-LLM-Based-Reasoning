#!/usr/bin/env python3
"""
Hybrid Driver Drowsiness Detection - CAPTURE FULL OCCLUSION EVENT
=====================================================================
KEY: Capture complete occlusion event from start to end
- Flag triggered: Start recording frames
- Flag persists: Keep recording frames
- Flag ends: Send full 8-frame sequence showing HOW occlusion occurred

This captures the EVENT PROGRESSION, not just snapshots.

With 2 VLM workers, you can process multiple events
simultaneously without blocking the camera.

OPTIMIZATIONS:
1. Increased HTTP timeout to 300s (5 mins)
2. Reduced image resolution (480x360) and JPEG quality (70)
3. Event-based submission: Capture full occlusion lifecycle
4. 8-frame temporal context @ 4Hz
5. Multi-threaded VLM processing (2 concurrent workers)

LOGIC:
- Normal blink: 100-200ms (eyes closed) → IGNORE
- Drowsiness eye closure: 2+ seconds → START CAPTURING EVENT
- Yawning + ambiguity: START CAPTURING EVENT
- Flags end → SUBMIT entire 8-frame clip showing progression
- VLM sees: "what happened in this 2-second scene?"

ARCHITECTURE:
1. Main thread: Camera capture (30 FPS)
2. Worker thread 1: Frame processing (classical metrics @ 4 Hz)
3. Event tracker: Captures frames when flag starts until it ends
4. Worker thread pool (2 threads): VLM requests (async, parallel)
"""

import os
import cv2
import numpy as np
import threading
import base64
from queue import Queue, Full, Empty
from collections import deque
from datetime import datetime
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor

# Mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Utils
from utils import calculate_avg_ear, mouth_aspect_ratio


# ----------------------------
# SYSTEM PROMPT (strict JSON schema with occlusion & notes)
# ----------------------------
SYSTEM_PROMPT = """
SYSTEM: You are Qwen2.5-VL (qwen2.5vl:3b-q8_0) for automated driver behavior evaluation.
You will receive 8 consecutive frames from a driver's face/upper torso.
Produce JSON-only output following exactly this schema:

{
  "clip_id": "<string>",
  "drowsy": {"label": "yes"|"no","confidence": 0.0},
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
  "eye_visibility": {"left_eye":"open","right_eye":"open"},
  "mouth_visibility": "open"|"closed"|"occluded",
  "head_posture": {"pitch":"neutral","yaw":"center","roll":"neutral"},
  "occlusion": {"face_occluded": true|false, "reason":"mask"|"hand"|"sun_glare"|"other"|null},
  "lighting": {"level":"normal","issue":null},
  "evidence_frames": [0,1,2,3,4,5,6,7],
  "notes": ""
}

- Always fill all fields, even if uncertain.
- Use numeric confidence scores 0.0-1.0.
- Do not output per-frame arrays, PERCLOS, blink rates, or markdown. Return only the JSON object.
"""

# ----------------------------
# USER PROMPT TEMPLATE (aggregated 2-second scene with occlusion and notes)
# ----------------------------
USER_PROMPT_TEMPLATE = """
You will analyze the following 8 consecutive frames from a driver's cabin using Qwen2.5VL:3b-q8_0.
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
# Ollama HTTP
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b-q8_0"

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    # Classical thresholds (from literature)
    "ear_low_threshold": 0.26,          # Soukupová & Čech 2016
    "ear_extreme_threshold": 0.10,      # Very low (potential occlusion)
    "mar_yawn_threshold": 0.60,         # Abtahi 2014
    
    # Blink filtering logic
    "blink_duration_max": 0.3,          # Normal blink: 100-300ms
    "drowsiness_duration_min": 2.0,     # Drowsiness: 2+ seconds
    
    # Brightness heuristics (empirical)
    "brightness_very_dark": 45,
    "brightness_dark": 80,
    "brightness_normal_min": 60,
    "brightness_normal_max": 180,
    "brightness_bright": 200,
    "brightness_overexposed": 230,
    
    # Edge density (for detecting occlusion/sunglasses)
    "edge_density_threshold": 0.18,
    
    # Sampling & buffering
    "sampling_hz": 4.0,                 # 4 frames per second
    "buffer_duration": 2.0,             # 2 seconds = 8 frames
    
    # Camera
    "frame_width": 480,
    "frame_height": 360,
    "jpeg_quality": 70,
    
    # VLM triggers
    "vlm_trigger_on_ambiguity": True,
    
    # HTTP & threading
    "http_timeout": 300,
    "max_concurrent_vlm": 2,
    
    # Output
    "output_dir": "./vlm_triggers",
}

# ----------------------------
# Helper: Occlusion & Ambiguity Detection
# ----------------------------
def detect_ambiguity_flags(frame_bgr, face_detected, ear, mar, brightness, face_conf):
    """
    Detect situations where classical metrics may be unreliable.
    """
    flags = []
    
    if not face_detected or face_conf < 0.7:
        flags.append("no_face")
    
    if ear is not None and ear < CONFIG["ear_extreme_threshold"]:
        flags.append("extreme_low_ear")
    
    if ear is not None and mar is not None:
        eyes_closed = ear < CONFIG["ear_low_threshold"]
        mouth_open = mar > CONFIG["mar_yawn_threshold"]
        if eyes_closed and mouth_open:
            flags.append("conflicting_eye_mouth")
    
    if brightness is not None:
        if brightness < CONFIG["brightness_very_dark"]:
            flags.append("very_dark")
        elif brightness > CONFIG["brightness_overexposed"]:
            flags.append("overexposed")
    
    if face_detected:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.mean(edges > 0)
        if edge_density > CONFIG["edge_density_threshold"]:
            flags.append("busy_edges")
    
    return flags


def encode_frame_b64(frame, quality=70):
    """Encode frame to base64 JPEG"""
    success, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if success:
        return base64.b64encode(buf.tobytes()).decode('ascii')
    return None


# ----------------------------
# Event Tracker: Captures full occlusion lifecycle
# ----------------------------
class OcclusionEventTracker:
    """Tracks occlusion events from start to end, captures 8-frame progression"""
    
    def __init__(self):
        self.event_active = False
        self.event_frames = deque(maxlen=8)  # Max 8 frames
        self.event_start_time = None
        self.last_flags = None
        self.event_metadata = {}
    
    def update(self, frame, has_flags, current_flags, metrics):
        """
        Update event state with new frame.
        Returns tuple: (frames_to_submit, should_submit)
        """
        should_submit = False
        frames_to_submit = None
        
        if has_flags and not self.event_active:
            # ✅ FLAG JUST TRIGGERED - START CAPTURING EVENT
            self.event_active = True
            self.event_start_time = time.time()
            self.event_frames.clear()
            self.event_frames.append(frame.copy())
            self.last_flags = current_flags
            self.event_metadata = metrics.copy()
            
            print(f"\n📍 EVENT STARTED: Capturing frames from now until occlusion ends...")
            return None, False
        
        elif has_flags and self.event_active:
            # ✅ FLAG STILL ACTIVE - KEEP RECORDING
            self.event_frames.append(frame.copy())
            self.last_flags = current_flags
            print(f"   📍 Recording frame {len(self.event_frames)}/8 of occlusion event...")
            return None, False
        
        elif not has_flags and self.event_active:
            # ✅ FLAG ENDED - SUBMIT FULL EVENT SEQUENCE
            self.event_active = False
            
            # If buffer has < 8 frames, pad with current frame to get 8
            while len(self.event_frames) < 8:
                self.event_frames.append(frame.copy())
            
            frames_to_submit = list(self.event_frames)
            should_submit = True
            
            event_duration = time.time() - self.event_start_time
            print(f"\n✅ EVENT ENDED: Submitting {len(frames_to_submit)} frames showing full progression")
            print(f"   Duration: {event_duration:.1f}s | Flags detected: {','.join(self.last_flags)}")
            
            return frames_to_submit, should_submit
        
        return None, False


# ----------------------------
# VLM Request Handler
# ----------------------------
class VLMRequestHandler:
    """Handles all VLM requests in a separate thread pool"""
    
    def __init__(self, max_workers=2, output_dir="./vlm_triggers"):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VLM")
        self.semaphore = threading.Semaphore(max_workers)
        self.output_dir = output_dir
        self.pending_requests = 0
        self.completed_requests = 0
        self.lock = threading.Lock()
    
    def submit_request(self, frames, clip_id, flags, metrics):
        """
        INSTANTLY submit VLM request to thread pool and return.
        """
        with self.lock:
            self.pending_requests += 1
        
        try:
            self.semaphore.acquire(blocking=False)
            future = self.executor.submit(
                self._vlm_request_blocking,
                frames, clip_id, flags, metrics
            )
            future.add_done_callback(lambda f: self._on_request_complete())
            return future
        except Exception as e:
            print(f"[VLM] ❌ Failed to submit request: {e}")
            with self.lock:
                self.pending_requests -= 1
            self.semaphore.release()
            return None
    
    def _on_request_complete(self):
        """Called when a VLM request completes"""
        with self.lock:
            self.pending_requests -= 1
            self.completed_requests += 1
        self.semaphore.release()
    
    def _vlm_request_blocking(self, frames, clip_id, flags, metrics):
        """
        Process VLM request in background thread.
        Takes 1-3 minutes but doesn't block the camera.
        """
        imgs_b64 = [encode_frame_b64(f, CONFIG["jpeg_quality"]) for f in frames]
        
        if not all(imgs_b64):
            print(f"[VLM] ❌ Frame encoding failed for {clip_id}")
            return {"error": "frame encoding failed"}
        
        payload = {
            "model": MODEL_NAME,
            "system_prompt": SYSTEM_PROMPT,
            "prompt": USER_PROMPT_TEMPLATE.format(clip_id=clip_id),
            "images": imgs_b64,
            "stream": False
        }
        
        try:
            print(f"[VLM] 🔄 Analyzing occlusion event for {clip_id} ({len(frames)} frames)...")
            print(f"     Flags: {','.join(flags)}")
            start = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=CONFIG["http_timeout"])
            elapsed = time.time() - start
            
            print(f"[VLM] ✅ Response received ({elapsed:.1f}s)")
            resp.raise_for_status()
            
            vlm_json = resp.json()
            if "response" in vlm_json:
                response_text = vlm_json["response"]
                try:
                    vlm_analysis = json.loads(response_text)
                    print(f"[VLM] ✅ Parsed VLM analysis")
                    print(f"     Drowsy: {vlm_analysis.get('drowsy', {}).get('label', 'unknown')} "
                          f"(confidence: {vlm_analysis.get('drowsy', {}).get('confidence', 0):.2f})")
                    print(f"     Notes: {vlm_analysis.get('notes', 'N/A')[:100]}")
                    self._save_clip(clip_id, frames, vlm_analysis, flags)
                    return vlm_analysis
                except json.JSONDecodeError as je:
                    print(f"[VLM] ❌ Response not valid JSON: {je}")
                    self._save_clip(clip_id, frames, {"error": "invalid json"}, flags)
                    return {"error": "invalid json response"}
            else:
                self._save_clip(clip_id, frames, vlm_json, flags)
                return vlm_json
                    
        except requests.Timeout:
            print(f"[VLM] ❌ TIMEOUT after {CONFIG['http_timeout']}s")
            return {"error": "timeout"}
        except Exception as e:
            print(f"[VLM] ❌ HTTP error: {e}")
            return {"error": str(e)}
    
    def _save_clip(self, clip_id, frames, vlm_analysis, flags):
        """Save frames and analysis to disk"""
        out_dir = os.path.join(self.output_dir, "events", clip_id)
        
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"[SAVE] ❌ Failed to create directory: {e}")
            return
        
        # Save frames
        frames_saved = 0
        for i, f in enumerate(frames):
            try:
                frame_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
                if cv2.imwrite(frame_path, f):
                    frames_saved += 1
            except Exception as e:
                print(f"[SAVE] ❌ Error saving frame {i}: {e}")
        
        # Save analysis with metadata
        try:
            analysis = {
                "clip_id": clip_id,
                "timestamp": datetime.now().isoformat(),
                "frames_count": frames_saved,
                "flags_detected": flags,
                "vlm_analysis": vlm_analysis
            }
            
            json_path = os.path.join(out_dir, "analysis.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(analysis, fh, indent=2, ensure_ascii=False)
            
            print(f"✅ SAVED: {frames_saved}/{len(frames)} frames showing event progression")
            print(f"   Directory: {out_dir}")
        except Exception as e:
            print(f"❌ SAVE error: {e}")
    
    def shutdown(self):
        """Gracefully shutdown VLM thread pool"""
        print("\n⏳ Shutting down VLM thread pool...")
        print(f"   Pending requests: {self.pending_requests}")
        print(f"   Completed requests: {self.completed_requests}")
        self.executor.shutdown(wait=True)
        print("✅ VLM thread pool shutdown complete")


# ----------------------------
# Camera + Mediapipe + Hybrid VLM Node
# ----------------------------
class HybridDrowsinessDetector:
    def __init__(self, camera_id=0, driver_id="driver1", model_path="face_landmarker.task", output_dir="./vlm_triggers"):
        self.camera_id = camera_id
        self.driver_id = driver_id
        self.cfg = CONFIG
        self.cfg["output_dir"] = output_dir
        self.running = True
        self.frame_count = 0
        self.submitted_events = []
        self.event_count = 0

        # Sampling parameters
        self.last_sample_time = 0.0
        self.sample_period = 1.0 / self.cfg["sampling_hz"]

        # Frame queue for async processing
        self.frame_queue = Queue(maxsize=8)

        # ✅ EVENT TRACKER - captures full occlusion lifecycle
        self.event_tracker = OcclusionEventTracker()

        # VLM handler
        self.vlm_handler = VLMRequestHandler(
            max_workers=self.cfg["max_concurrent_vlm"],
            output_dir=self.cfg["output_dir"]
        )

        # Mediapipe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.mp_face_detection = mp.solutions.face_detection
        self.face_det = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        # Worker thread for frame processing
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Track eye closure
        self._closure_start = None
        
        # Print config
        abs_output = os.path.abspath(self.cfg["output_dir"])
        print(f"✅ Initialized HybridDrowsinessDetector (driver_id={driver_id})")
        print(f"✅ EVENT TRACKING: Captures full occlusion progression (start → end)")
        print(f"✅ Submission: Full 8-frame sequence showing HOW event occurred")
        print(f"✅ OPTIMIZED: 480x360 resolution, JPEG Q70, Timeout 300s")
        print(f"✅ VLM workers: {self.cfg['max_concurrent_vlm']} parallel")
        print(f"✅ Output directory: {abs_output}")
        print()


    def worker_loop(self):
        """Process frames from queue - extracts metrics only"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                self._process_frame(frame)
            except Empty:
                continue
            except Exception as e:
                print(f"[Worker] Error: {e}")


    def _process_frame(self, frame):
        """
        Process single frame - extract metrics and track occlusion events.
        """
        self.frame_count += 1
        now = time.time()

        # Face detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.face_det.process(rgb)
        face_detected = bool(dets.detections)
        face_conf = float(dets.detections[0].score[0]) if face_detected else 0.0

        # Metrics
        ear, mar, brightness = None, None, None
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with self.lock:
                lm_res = self.landmarker.detect(mp_img)
            if lm_res.face_landmarks:
                landmarks = np.array([[p.x, p.y] for p in lm_res.face_landmarks[0]])
                ear = calculate_avg_ear(landmarks)
                mar = mouth_aspect_ratio(landmarks)
        except Exception:
            pass

        # Brightness
        if face_detected:
            bbox = dets.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = max(0, int(bbox.xmin*w))
            y = max(0, int(bbox.ymin*h))
            bw = max(1, int(bbox.width*w))
            bh = max(1, int(bbox.height*h))
            roi = frame[y:y+bh, x:x+bw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))

        # Classical metrics
        eyes_closed = ear is not None and ear < self.cfg["ear_low_threshold"]
        
        if eyes_closed:
            if self._closure_start is None:
                self._closure_start = now
            closure_duration = now - self._closure_start
            prolonged_closure = closure_duration >= self.cfg["drowsiness_duration_min"]
        else:
            self._closure_start = None
            prolonged_closure = False
        
        yawning = mar is not None and mar > self.cfg["mar_yawn_threshold"]

        # Detect flags
        ambiguity_flags = detect_ambiguity_flags(frame, face_detected, ear, mar, brightness, face_conf)
        
        # Determine if event is active (has flags)
        has_flags = bool(prolonged_closure or (yawning and ambiguity_flags))

        # Collect metrics
        metrics = {
            "ear": ear,
            "mar": mar,
            "brightness": brightness,
            "face_conf": face_conf
        }

        # ✅ UPDATE EVENT TRACKER
        frames_to_submit, should_submit = self.event_tracker.update(
            frame, has_flags, ambiguity_flags if has_flags else [], metrics
        )

        # If event ended, submit the full 8-frame progression
        if should_submit and frames_to_submit:
            clip_id = f"occlusion_event_{datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}"
            
            print(f"\n🔔 SUBMITTING OCCLUSION EVENT: {clip_id}")
            print(f"   Frames: {len(frames_to_submit)} showing event progression")
            print(f"   Flags: {','.join(self.event_tracker.last_flags)}")
            
            self.vlm_handler.submit_request(
                frames_to_submit, 
                clip_id, 
                self.event_tracker.last_flags,
                metrics
            )
            
            self.submitted_events.append(clip_id)
            self.event_count += 1

        # Display overlay
        disp = frame.copy()
        status = f"F:{self.frame_count} | Events: {self.event_count}"
        
        signals = []
        if self.event_tracker.event_active:
            signals.append(f"📍 EVENT RECORDING ({len(self.event_tracker.event_frames)}/8)")
        if eyes_closed:
            signals.append("EYES_CLOSED")
        if yawning:
            signals.append("YAWN")
        if ambiguity_flags:
            signals.append(f"FLAGS[{','.join(ambiguity_flags[:2])}]")
        
        signal_text = " | ".join(signals) if signals else "NORMAL"
        cv2.putText(disp, f"{status} | {signal_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if ear is not None:
            cv2.putText(disp, f"EAR:{ear:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        if mar is not None:
            cv2.putText(disp, f"MAR:{mar:.3f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        
        # VLM status
        pending = self.vlm_handler.pending_requests
        completed = self.vlm_handler.completed_requests
        vlm_status = f"VLM: {completed} done, {pending} pending"
        vlm_color = (0, 255, 0) if pending == 0 else (0, 165, 255)
        cv2.putText(disp, vlm_status, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vlm_color, 1)
        
        # Mode info
        cv2.putText(disp, f"⚡ EVENT CAPTURE MODE: Records full occlusion progression", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 100), 1)
        
        cv2.imshow("Hybrid Drowsiness Detector (q to quit)", disp)
        cv2.waitKey(1)


    def push_frame(self, frame):
        """Add frame to queue"""
        try:
            self.frame_queue.put_nowait(frame)
        except Full:
            _ = self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(frame)


    def run_camera(self):
        """Main camera loop"""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["frame_height"])
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Could not open camera")
            return

        print("🎥 Camera started (press 'q' to quit)\n")
        print("⚡ EVENT CAPTURE MODE:")
        print("   • Flag triggers: START capturing")
        print("   • Flag persists: CONTINUE capturing")
        print("   • Flag ends: SUBMIT full 8-frame progression")
        print("   • VLM analyzes: Complete 2-second scene\n")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                self.push_frame(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup(cap)


    def cleanup(self, cap=None):
        """Graceful shutdown"""
        self.running = False
        
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        
        self.vlm_handler.shutdown()
        
        try:
            self.worker_thread.join(timeout=2.0)
        except Exception:
            pass
        
        print(f"\n✅ Cleanup complete")
        print(f"   Total frames processed: {self.frame_count}")
        print(f"   Occlusion events submitted: {self.event_count}")
        print(f"   Event clips: {len(self.submitted_events)}")
        print(f"   Output saved to: {os.path.abspath(self.cfg['output_dir'])}")
        print()


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    MODEL_PATH = r"C:\Users\nikhi\drowsiness_detection_ros2\face_landmarker.task"
    OUTPUT_DIR = r"C:\Users\nikhi\drowsiness_detection_ros2\vlm_triggers"
    
    detector = HybridDrowsinessDetector(
        camera_id=0, 
        driver_id="test_driver", 
        model_path=MODEL_PATH, 
        output_dir=OUTPUT_DIR
    )
    try:
        detector.run_camera()
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()