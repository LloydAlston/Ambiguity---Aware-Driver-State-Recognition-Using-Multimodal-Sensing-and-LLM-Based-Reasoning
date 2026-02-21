#!/usr/bin/env python3
"""
ROS2 Node: Hybrid VLM Subscriber
Subscribes to:
  - /camera/image_raw (from PySpin IR camera node)
  - /ear_mar (EAR/MAR metrics from IR camera node)

Publishes:
  - NOTHING (runs Hybrid VLM analysis on received frames)

When an occlusion event is detected:
  1. OcclusionEventTracker captures 8 frames
  2. VLMRequestHandler submits to Qwen2.5-VL (async, non-blocking)
  3. Results saved to disk with event metadata
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from drowsiness_detection_msg.msg import EarMarValue
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time
import os
import json
from collections import deque

# Import core VLM logic (no ROS, no camera)
from drowsiness_detection.VLM.hybrid_vlm_core import (
    CONFIG,
    detect_ambiguity_flags,
    EventTracker,  # Renamed from OcclusionEventTracker
    VLMRequestHandler,
    should_trigger_vlm,  # NEW: For drowsiness triggers
)


class HybridVlmNode(Node):
    """
    Subscribes to IR camera topics and runs Hybrid drowsiness + VLM occlusion logic.
    
    Class workflow:
    1. Subscribe to /camera/image_raw (IR camera - grayscale)
    2. Apply CLAHE enhancement for IR normalization
    3. Monitor EAR/MAR for drowsiness + ambiguity flags
    4. Aggregate evidence + send to VLM on event boundaries
    5. Publish VLM result for downstream LLM
    """

    def __init__(self):
        super().__init__("hybrid_vlm_node")

        # Parameters
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value

        self.declare_parameter("output_dir", os.path.expanduser("~/DROWSINESS_DETECTION/vlm_triggers"))
        output_dir = self.get_parameter("output_dir").value
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Optimization Parameters
        self.declare_parameter("vlm_heartbeat_interval", 1.0)
        self.declare_parameter("vlm_buffer_size", 8)
        self.declare_parameter("vlm_pre_buffer_size", 4)
        self.declare_parameter("vlm_trigger_persistence", 3) # Debounce: must be true for 3 frames
        
        heartbeat = self.get_parameter("vlm_heartbeat_interval").value
        buff_size = self.get_parameter("vlm_buffer_size").value
        pre_buff_size = self.get_parameter("vlm_pre_buffer_size").value
        self.persistence_threshold = self.get_parameter("vlm_trigger_persistence").value
        self.flag_persistence_counter = 0

        # Bridge for image conversion
        self.bridge = CvBridge()

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.metrics_sub = self.create_subscription(
            EarMarValue,
            "/ear_mar",
            self.metrics_callback,
            10,
        )

        # Publisher for VLM results (Critical for LLM integration - RELIABLE QoS)
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.vlm_pub = self.create_publisher(
            String,
            "/vlm_occlusion_results",
            qos_reliable
        )

        # Publisher for System Health (Monitoring)
        self.health_pub = self.create_publisher(
            String,
            "/vlm_system_health",
            10
        )
        
        # Latest EAR/MAR from /ear_mar topic
        self.latest_ear = None
        self.latest_mar = None
        self.latest_metrics_stamp = None

        # Eye closure tracking
        self._closure_start = None

        # Counters
        self.frame_count = 0
        self.event_count = 0
        self.submitted_events = []

        # Core VLM components
        CONFIG["output_dir"] = output_dir
        CONFIG["min_heartbeat_interval"] = heartbeat
        CONFIG["buffer_size"] = buff_size
        CONFIG["pre_buffer_size"] = pre_buff_size
        
        self.event_tracker = EventTracker(stream_id=self.driver_id)
        self.vlm_handler = VLMRequestHandler(
            max_workers=CONFIG["max_concurrent_vlm"],
            output_dir=CONFIG["output_dir"],
        )

        # Lock for shared state
        self.lock = threading.Lock()

        self.get_logger().info(
            f"[HybridVLM] Node initialized (driver_id={self.driver_id})"
        )
        self.get_logger().info(
            f"[HybridVLM] Subscribing to /camera/image_raw and /ear_mar"
        )
        self.get_logger().info(f"[HybridVLM] Output dir: {output_dir}")
        self.get_logger().info(f"[HybridVLM] Config: Heartbeat={heartbeat}s, Buffer={buff_size}, PreBuffer={pre_buff_size}")
        self.get_logger().info(
            f"[HybridVLM] Triggers: Drowsiness + Yawning + ANY Ambiguity Flags"
        )
        
        # Rate Limiting (Downsampling)
        # Ensure VLM sees ~2s context (8 frames @ 4Hz) regardless of camera FPS
        self.target_hz = CONFIG.get("sampling_hz", 4.0)
        self.sampling_interval = 1.0 / self.target_hz
        self.last_process_time = 0.0
        
        # State Initialization
        self._flag_latch = False
        self._last_flags_state = False
        self._closure_start = None
        self._yawn_start = None
        self._latched_ambiguity_flags = set()
        self.get_logger().info(f"[HybridVLM] Downsampling input to {self.target_hz} Hz")

    # ---------------------- Metrics callback ----------------------
    def metrics_callback(self, msg: EarMarValue):
        """Receive EAR/MAR metrics from camera node"""
        with self.lock:
            self.latest_ear = float(msg.ear_value)
            self.latest_mar = float(msg.mar_value)
            self.latest_metrics_stamp = msg.header.stamp

    # ---------------------- Image callback -----------------------
    def image_callback(self, msg: Image):
        """Receive IR image from FLIR camera and run Hybrid analysis"""
        # Dual-Rate Logic: Monitor at FULL Speed (60Hz), Record at Sample Rate (4Hz)

        try:
            # FLIR IR camera: Convert directly to grayscale (skip BGR entirely)
            frame_gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            # Fallback: If mono8 fails, convert from whatever format
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if len(cv_image.shape) == 3:
                    frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = cv_image
            except Exception as e2:
                self.get_logger().error(f"[HybridVLM] CV bridge error: {e2}")
                return
        
        # Extract frame timestamp for temporal consistency
        frame_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Get latest metrics (may be slightly delayed)
        with self.lock:
            ear = self.latest_ear
            mar = self.latest_mar

        # Process frame with hybrid logic (grayscale input)
        self.process_frame(frame_gray, ear, mar, frame_timestamp)

    # ---------------------- Core Hybrid Logic ----------------------
    def process_frame(self, frame_gray: np.ndarray, ear: float, mar: float, frame_timestamp: float):
        """
        Main hybrid processing pipeline (IR Grayscale input):
        1. Apply CLAHE enhancement
        2. Detect ambiguity flags
        3. Check prolonged closure + yawning
        4. Update event tracker
        5. Submit VLM request if event ends
        """
        self.frame_count += 1

        # IR Camera Preprocessing: Apply CLAHE for histogram normalization
        # CLAHE enhances local contrast and normalizes brightness distribution
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_enhanced = clahe.apply(frame_gray)
        
        # Calculate brightness on CLAHE-enhanced image
        brightness = float(np.mean(frame_enhanced))

        # Treat "ear > 0" as proxy for face detected (IR camera produces landmarks)
        face_detected = ear is not None and ear > 0.0
        face_conf = 1.0 if face_detected else 0.0
        
        # IR camera doesn't provide bounding box - use None
        face_bbox = None

        # Classical metrics
        now = self.get_clock().now().nanoseconds * 1e-9

        eyes_closed = ear is not None and ear < CONFIG["ear_low_threshold"]
        if eyes_closed:
            if self._closure_start is None:
                self._closure_start = now
            closure_duration = now - self._closure_start
            prolonged_closure = closure_duration >= CONFIG["drowsiness_duration_min"]
        else:
            self._closure_start = None
            prolonged_closure = False

        yawning = mar is not None and mar > CONFIG["mar_yawn_threshold"]
        if yawning:
            if self._yawn_start is None:
                self._yawn_start = now
            yawn_duration = now - self._yawn_start
            prolonged_yawn = yawn_duration >= CONFIG["vlm_trigger_yawn_duration"]
        else:
            self._yawn_start = None
            prolonged_yawn = False

        # Detect ambiguity (hands, occlusion, no_face, lighting, etc.)
        # Use CLAHE-enhanced frame for better edge/occlusion detection
        ambiguity_flags = detect_ambiguity_flags(
            frame_enhanced,  # CLAHE-enhanced grayscale for IR camera
            face_bbox,  # None for IR camera
            ear,
            mar,
            face_conf,
        )

        # TRIGGER: Has flags? (any of: prolonged closure, yawning+ambiguity, or ambiguity alone)
        # CRITICAL FIX: Trigger on Ambiguity OR (Yawn > 1s) OR (Closure > 2s)
        # This decouples yawning from requiring ambiguity.
        
        has_ambiguity = len(ambiguity_flags) > 0
        
        # We already computed prolonged_closure (based on duration) and prolonged_yawn
        # prolonged_closure = (ear < 0.26 and dur > 2.0s) -> Wait, we want vlm_trigger_drowsy_ear (0.20)
        # Let's check the strict VLM triggers for "Clear Drowsiness"
        
        is_deep_drowsy = False
        if eyes_closed and ear < CONFIG["vlm_trigger_drowsy_ear"]: # 0.20
             # Re-check duration for strict threshold
             if self._closure_start and (now - self._closure_start) >= CONFIG["vlm_trigger_drowsy_duration"]:
                 is_deep_drowsy = True

        raw_flags_detected = bool(
            has_ambiguity or
            is_deep_drowsy or
            prolonged_yawn
        )

        # Adaptive Debouncing: Critical flags trigger faster
        CRITICAL_FLAGS = ["no_face", "conflicting_eye_mouth"]
        is_critical = any(f in CRITICAL_FLAGS for f in ambiguity_flags)
        
        # If critical, threshold is 1 (instant). Else use configured persistence (e.g. 3)
        current_threshold = 1 if is_critical else self.persistence_threshold

        # Debounce Logic
        if raw_flags_detected:
            self.flag_persistence_counter += 1
        else:
            self.flag_persistence_counter = 0
            
        has_flags_confirmed = self.flag_persistence_counter >= current_threshold

        # --- DUAL RATE LATCHING ---
        # Latch the flag if it appeared AT ANY POINT since last sample
        if not hasattr(self, "_flag_latch"): self._flag_latch = False
        if not hasattr(self, "_last_flags_state"): self._last_flags_state = False
        
        if has_flags_confirmed:
            self._flag_latch = True
            # CRITICAL FIX: Accumulate flags so we don't send "Blank Flags" if they flicker
            self._latched_ambiguity_flags.update(ambiguity_flags)
            
        # INTERRUPT LOGIC: Immediate Capture on Rising Edge
        # If this is a NEW event start, capture THIS EXACT FRAME immediately
        # to ensure we don't miss the "Smoking Gun" due to 4Hz sampling.
        is_rising_edge = has_flags_confirmed and not self._last_flags_state
        self._last_flags_state = has_flags_confirmed
        
        # Check if it's time to SAMPLE (Record to Buffer)
        now_ts = self.get_clock().now().nanoseconds * 1e-9
        time_to_sample = (now_ts - self.last_process_time) >= self.sampling_interval
        
        if not time_to_sample and not is_rising_edge:
            # NOT sample time yet, and no urgent trigger: just monitor.
            return 
            
        # RATE LIMIT REACHED (or FORCED by Interrupt): Update Tracker
        # If forced by interrupt, we reset the timer so next sample is 0.25s later
        self.last_process_time = now_ts
        
        # Use the latched flag + current metrics
        # (Pass current metrics, they are close enough)
        metrics = {
            "ear": ear,
            "mar": mar,
            "brightness": brightness,
            "face_conf": face_conf,
        }
        
        # Build comprehensive flag list - include ALL trigger reasons
        # This ensures VLM knows WHY it was triggered (not just ambiguity)
        eff_flags = []
        if self._flag_latch:
            # Add ambiguity flags from the ACCUMULATED set (prevents blank flags)
            eff_flags.extend(list(self._latched_ambiguity_flags))
            
            # Add drowsiness triggers (context)
            if prolonged_closure:
                eff_flags.append("prolonged_eye_closure")
            if prolonged_yawn: # Use prolonged logic
                eff_flags.append("yawning")
            if yawning:
                eff_flags.append("yawning")

        # Update event tracker with NEW signature
        # Use CLAHE-enhanced frame for better VLM analysis
        frames, timestamps, metrics_list, should_submit, event_id = self.event_tracker.update(
            frame_enhanced,  # CLAHE-enhanced grayscale for VLM
            frame_timestamp,  # NEW: Frame timestamp for temporal consistency
            self._flag_latch,
            eff_flags,
            metrics,
        )
        
        # Reset Latch and Accumulator
        self._flag_latch = False
        self._latched_ambiguity_flags.clear()

        # If event ended, submit to VLM
        if should_submit and frames:
            self.get_logger().info(
                f"[HybridVLM] 🔔 Submitting event: {event_id}"
            )
            # Use accumulated_flags to show ALL flags seen during event
            all_flags = list(self.event_tracker.accumulated_flags) if hasattr(self.event_tracker, 'accumulated_flags') else (self.event_tracker.last_flags or [])
            self.get_logger().info(
                f"[HybridVLM]    Frames: {len(frames)} "
                f"| Flags: {','.join(all_flags)}"
            )

            # NEW signature: requires stream_id, timestamps, event_id
            future = self.vlm_handler.submit_request(
                stream_id=self.driver_id,
                frames=frames,
                timestamps=timestamps,  # NEW
                metrics=metrics_list,  # NEW: Now a list
                clip_id=event_id,  # NEW: Use event_id from tracker
                flags=all_flags,  # Use accumulated flags
            )
            
            # Register callback to publish result when ready
            if future:
                future.add_done_callback(self._on_vlm_result_ready)

            self.submitted_events.append(event_id)
            self.event_count += 1

            # Log VLM status
            pending = self.vlm_handler.pending_requests
            completed = self.vlm_handler.completed_requests
            self.get_logger().info(
                f"[HybridVLM]    VLM status: {completed} completed, {pending} pending"
            )

        # Publish System Health (Every frame or on event? Every 30 frames to save bandwidth)
        if self.frame_count % 30 == 0:
            health_msg = String()
            stats = {
                "active_event": self.event_tracker.event_active,
                "queue_size": self.vlm_handler.pending_requests,
                "dropped_count": self.vlm_handler.dropped_requests,
                "total_events": self.event_count,
                "completed_count": self.vlm_handler.completed_requests
            }
            health_msg.data = json.dumps(stats)
            self.health_pub.publish(health_msg)

    def _on_vlm_result_ready(self, future):
        """Callback when VLM analysis is done"""
        try:
            result = future.result() # This will block if not ready, but callback implies ready
            if result and "error" not in result:
                msg = String()
                msg.data = json.dumps(result)
                self.vlm_pub.publish(msg)
                self.get_logger().info(f"[HybridVLM] 🚀 Published VLM result for LLM Node")
            else:
                self.get_logger().warn(f"[HybridVLM] VLM returned error or empty: {result}")
        except Exception as e:
            self.get_logger().error(f"[HybridVLM] Error publishing VLM result: {e}")

    # ---------------------- Cleanup -----------------------
    def destroy_node(self):
        """Graceful shutdown: wait for pending VLM requests, then exit"""
        self.get_logger().info("[HybridVLM] Shutting down HybridVlmNode...")
        self.get_logger().info(
            f"[HybridVLM] Total frames: {self.frame_count} | Events: {self.event_count}"
        )

        if hasattr(self, "vlm_handler"):
            self.vlm_handler.shutdown()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HybridVlmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[HybridVLM] Keyboard interrupt — shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()