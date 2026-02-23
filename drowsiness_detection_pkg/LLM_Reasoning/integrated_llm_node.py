#!/usr/bin/env python3

"""
ROS2 Node: Integrated Safety-Critical LLM Reasoning (v2.0 - Merged & Corrected)

Subscribes to:
  - /camera_predictions (Camera ML: 3-CLASS - Alert, Drowsy, Very Drowsy) - every 60s
  - /carla_predictions (CARLA ML: 3-CLASS - Alert, Drowsy, Very Drowsy) - every 60s
  - /vlm_occlusion_results (VLM JSON: async, when occlusion flags detected)

LLM Trigger:
  - Timer-based: Every 60 seconds (triggered by Camera + CARLA ML window)
  - VLM provides CONTEXT ONLY (not a trigger)
  - When 1-minute window completes:
    1. Gather camera + carla data (3-class predictions)
    2. Include latest VLM context (if available)
    3. Send to Ollama for safety-critical reasoning
    4. Publish final decision

FEATURES (v2.0):
  - Simplified VLM context model (Qwen2.5VL schema)
  - Enhanced LLM prompt with probabilistic VLM interpretation
  - Strict timing constraints (15s freshness, 2s grace, 75s watchdog)
  - Deterministic fallback with rule-based reasoning
  - Thread-safe data handling
  - Dual-mode persistence (JSON + CSV)

ERROR FIXES:
  - Fixed missing Thread import
  - Fixed variable naming bugs in sync timeout
  - Fixed duplicate function calls
  - Improved JSON parsing robustness
  - Added error handling and validation
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, String

import json
import time
import re
import os
from threading import Lock, Thread  # FIXED: Added Thread to imports
from datetime import datetime
import requests


class IntegratedSafetyCriticalLLMNode(Node):
    """
    Integrated LLM reasoning: Camera ML + CARLA ML (3-CLASS, every 1 min) + VLM context (async).
    
    v2.0: Merged features from both implementations with error corrections.
    """

    def __init__(self):
        super().__init__("integrated_llm_node")

        # === Parameters ===
        self.declare_parameter("driver_id", "maria")
        self.declare_parameter("model_name", "llama3.1")
        self.declare_parameter("vlm_confidence_threshold", 0.7)

        # Load params
        self.driver_id = self.get_parameter("driver_id").value
        self.model_name = self.get_parameter("model_name").value
        self.vlm_confidence_threshold = self.get_parameter("vlm_confidence_threshold").value

        self.declare_parameter("ollama_endpoint", "http://localhost:11434")
        self.ollama_endpoint = self.get_parameter("ollama_endpoint").value

        self.declare_parameter("ollama_model", "llama3.1:8b")
        self.ollama_model = self.get_parameter("ollama_model").value

        self.declare_parameter("confidence_threshold", 0.6)
        self.confidence_threshold = self.get_parameter("confidence_threshold").value

        # Few-shot learning parameters
        self.declare_parameter("use_few_shot", False)
        self.use_few_shot = self.get_parameter("use_few_shot").value

        self.declare_parameter("num_shots", 3)
        self.num_shots = self.get_parameter("num_shots").value

        self.declare_parameter("log_dir", os.path.expanduser("~/DROWSINESS_DETECTION/LLM_Output"))
        self.log_dir = self.get_parameter("log_dir").value

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "alert_history.csv")
        self._init_csv_header()

        # === Class names for both models ===
        self.class_names = {0: "Alert", 1: "Drowsy", 2: "Very Drowsy"}

        # === Subscribers ===
        # Use RELIABLE QoS (10) to ensure we don't miss the 1-minute updates
        self.camera_sub = self.create_subscription(
            Float64MultiArray,
            "/camera_predictions",
            self.camera_callback,
            10
        )

        self.carla_sub = self.create_subscription(
            Float64MultiArray,
            "/carla_predictions",
            self.carla_callback,
            10
        )

        # VLM provides CONTEXT ONLY (async, when occlusion flags detected)
        # Use RELIABLE to ensure safety warnings are received
        self.vlm_sub = self.create_subscription(
            String,
            "/vlm_occlusion_results",
            self.vlm_callback,
            10
        )

        # === Publisher ===
        self.alert_pub = self.create_publisher(
            String,
            "/drowsiness_alert",
            10
        )

        # === Data storage (thread-safe) ===
        self.data_lock = Lock()
        self.camera_data = None
        self.carla_data = None
        
        # Track which predictions have been used to prevent re-processing
        self.last_used_camera_timestamp = 0.0
        self.last_used_carla_timestamp = 0.0
        
        # VLM context (simplified: single event with TTL)
        self.vlm_context = None
        
        # Watchdog: Track last decision to prevent deadlocks
        self.last_decision_time = time.time()
        self.watchdog_timeout = 85.0  # 60s window + 25s grace

        # === Initialize Ollama ===
        self._init_ollama()

        # === Watchdog Timer (Safety Net) ===
        # Checks every 10s if the system is stalled (waiting for missing sensor)
        self.create_timer(10.0, self._watchdog_check)

        # === Partial Trigger Timer (Sync Manager) ===
        self.sync_timer = None

        self.get_logger().info(
            f"Integrated Safety-Critical LLM Node started (3-CLASS - v2.0)\n"
            f"   Driver ID: {self.driver_id}\n"
            f"   Ollama: {self.ollama_endpoint}/{self.ollama_model}\n"
            f"   Confidence Threshold: {self.confidence_threshold:.1%}\n"
            f"   Few-shot Learning: {'ENABLED' if self.use_few_shot else 'DISABLED'} ({self.num_shots} examples)\n"
            f"   Classes: 0=Alert, 1=Drowsy, 2=Very Drowsy"
        )
        self.get_logger().info(
            f"   Data Sources:\n"
            f"     - /camera_predictions (Camera ML - 3-CLASS, every 60s)\n"
            f"     - /carla_predictions (CARLA ML - 3-CLASS, every 60s)\n"
            f"     - /vlm_occlusion_results (VLM - async, context only)"
        )
        self.get_logger().info(
            f"   LLM Trigger: TIMER-BASED (every 60 seconds)\n"
            f"   VLM Role: Provides reliability context, does NOT trigger LLM"
        )

    def _init_csv_header(self):
        """Initialize CSV file with header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'w') as f:
                    f.write(
                        "timestamp,driver_id,final_state,intervention_action,llm_confidence,"
                        "camera_class,camera_conf,carla_class,carla_conf,vlm_conf,vlm_occlusion,reasoning\n"
                    )
                self.get_logger().info(f"CSV header initialized: {self.csv_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize CSV header: {e}")

    def _init_ollama(self):
        """Initialize Ollama connection."""
        try:
            response = requests.get(f"{self.ollama_endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models if isinstance(m, dict)]
                self.get_logger().info(f"Ollama connected. Models: {model_names}")

                if any(self.ollama_model in m for m in model_names):
                    self.get_logger().info(f"✓ {self.ollama_model} available")
                else:
                    self.get_logger().warn(f"⚠ {self.ollama_model} not found in available models")
            else:
                self.get_logger().error(f"Ollama API error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            self.get_logger().error(
                f"✗ Cannot connect to Ollama at {self.ollama_endpoint}\n"
                f"   Start Ollama with: ollama serve"
            )
        except Exception as e:
            self.get_logger().error(f"Ollama init error: {e}")

    def camera_callback(self, msg: Float64MultiArray):
        """Receive camera ML predictions (3-CLASS, every 60s from camera_ml_node)."""
        if len(msg.data) < 8:
            self.get_logger().warn("Camera message has insufficient data")
            return

        try:
            with self.data_lock:
                self.camera_data = {
                    'perclos': float(msg.data[0]),
                    'blink_rate': float(msg.data[1]),
                    'blink_duration_mean': float(msg.data[2]),
                    'prediction': int(msg.data[3]),
                    'alert_prob': float(msg.data[4]),
                    'drowsy_prob': float(msg.data[5]),
                    'very_drowsy_prob': float(msg.data[6]),
                    'confidence': float(msg.data[7]),
                    'class_name': self.class_names.get(int(msg.data[3]), "Unknown"),
                    'timestamp': time.time()
                }

            self.get_logger().info(
                f"Camera ML received: {self.camera_data['class_name']} "
                f"(conf: {self.camera_data['confidence']:.1%})"
            )

            # Check if we have both data sources ready
            self._check_sync_and_trigger()

        except (ValueError, IndexError) as e:
            self.get_logger().error(f"Error parsing camera data: {e}")

    def carla_callback(self, msg: Float64MultiArray):
        """Receive CARLA ML predictions (3-CLASS, every 60s from carla_ml_node)."""
        if len(msg.data) < 8:
            self.get_logger().warn("CARLA message has insufficient data")
            return

        try:
            with self.data_lock:
                self.carla_data = {
                    'entropy': float(msg.data[0]),
                    'steering_rate': float(msg.data[1]),
                    'sdlp': float(msg.data[2]),
                    'prediction': int(msg.data[3]),
                    'alert_prob': float(msg.data[4]),
                    'drowsy_prob': float(msg.data[5]),
                    'very_drowsy_prob': float(msg.data[6]),
                    'confidence': float(msg.data[7]),
                    'class_name': self.class_names.get(int(msg.data[3]), "Unknown"),
                    'timestamp': time.time()
                }

            self.get_logger().info(
                f"CARLA ML received: {self.carla_data['class_name']} "
                f"(conf: {self.carla_data['confidence']:.1%})"
            )

            # Check if we have both data sources ready
            self._check_sync_and_trigger()

        except (ValueError, IndexError) as e:
            self.get_logger().error(f"Error parsing CARLA data: {e}")

    def vlm_callback(self, msg: String):
        """
        Receive VLM occlusion analysis results (async, when flags detected).
        VLM provides CONTEXT ONLY - does NOT trigger LLM reasoning.
        
        Supports Qwen2.5VL schema with structured output.
        """
        try:
            vlm_json = json.loads(msg.data)

            with self.data_lock:
                # Add timestamp for expiry logic
                vlm_json['_receive_time'] = time.time()
                self.vlm_context = vlm_json

            # Extract and log key VLM data
            occlusion = vlm_json.get('occlusion', {})
            if isinstance(occlusion, dict):
                occ_reason = occlusion.get('reason', 'none') if occlusion.get('face_occluded') else 'none'
            else:
                occ_reason = str(occlusion)

            behaviors = vlm_json.get('behaviors', {})
            detected_behaviors = []
            if isinstance(behaviors, dict):
                for b_name, b_data in behaviors.items():
                    if isinstance(b_data, dict) and b_data.get('detected'):
                        detected_behaviors.append(b_name)

            self.get_logger().info(
                f"VLM Context received (ASYNC): "
                f"occlusion={occ_reason}, "
                f"behaviors={detected_behaviors if detected_behaviors else 'none'}, "
                f"confidence={vlm_json.get('confidence', 0.0):.2f}"
            )

        except json.JSONDecodeError as e:
            self.get_logger().error(f"VLM JSON parse error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in VLM callback: {e}")

    def _check_sync_and_trigger(self):
        """
        Check if we have sufficient data to trigger combined reasoning.
        Implements PARTIAL TRIGGER logic to prevent deadlocks.
        """
        with self.data_lock:
            has_cam = self.camera_data is not None
            has_carla = self.carla_data is not None

            # Check freshness (up to 70s because publisher is every 60s)
            now = time.time()
            cam_valid = has_cam and (now - self.camera_data['timestamp'] < 70.0)
            carla_valid = has_carla and (now - self.carla_data['timestamp'] < 70.0)

            # 1. FULL TRIGGER: Both ready, fresh, AND not already used
            cam_is_new = has_cam and self.camera_data['timestamp'] > self.last_used_camera_timestamp
            carla_is_new = has_carla and self.carla_data['timestamp'] > self.last_used_carla_timestamp

            if cam_valid and carla_valid and cam_is_new and carla_is_new:
                # Cancel any pending partial timer
                if self.sync_timer:
                    self.sync_timer.cancel()
                    self.sync_timer = None

                self.get_logger().info("SYNC: Full sensor data received. Triggering LLM.")

                # Mark as used but keep data for potential fallback
                c_data = self.camera_data.copy()
                k_data = self.carla_data.copy()
                self.last_used_camera_timestamp = c_data['timestamp']
                self.last_used_carla_timestamp = k_data['timestamp']

                # Get VLM context if available
                v_ctx = self.vlm_context.copy() if self.vlm_context else None
                self.last_decision_time = time.time()

                self._invoke_integrated_reasoning(c_data, k_data, v_ctx)
                return

            # 2. PARTIAL TRIGGER START: One new sensor ready, wait for other
            one_new_sensor = (cam_valid and cam_is_new) or (carla_valid and carla_is_new)
            if one_new_sensor and not self.sync_timer:
                self.get_logger().info("SYNC: One sensor ready. Starting 25.0s partial wait timer.")
                self.sync_timer = self.create_timer(25.0, self._on_sync_timeout)

            # Check VLM context expiry (clear if no events in last 60s)
            if self.vlm_context and '_receive_time' in self.vlm_context:
                vlm_age = time.time() - self.vlm_context['_receive_time']
                if vlm_age > 60.0:
                    self.get_logger().info(f"VLM Context expired (age: {vlm_age:.1f}s) - Clearing context.")
                    self.vlm_context = None

    def _on_sync_timeout(self):
        """FIXED: Called if 2nd sensor fails to arrive within 25s grace period."""
        self.get_logger().warn("SYNC TIMEOUT: Proceeding with PARTIAL sensor data.")
        if self.sync_timer:
            self.sync_timer.cancel()
            self.sync_timer = None

        with self.data_lock:
            # FIXED: Use descriptive variable names (not camera_copy/carla_copy/vlm_copy)
            c_data = self.camera_data if self.camera_data else self._get_dummy_sensor_data("camera")
            k_data = self.carla_data if self.carla_data else self._get_dummy_sensor_data("carla")
            v_ctx = self.vlm_context.copy() if self.vlm_context else None

            # Consume (reset for next window)
            self.camera_data = None
            self.carla_data = None

            # Reset Watchdog
            self.last_decision_time = time.time()

        # FIXED: Call once with proper variable names (not duplicate)
        self._invoke_integrated_reasoning(c_data, k_data, v_ctx)

    def _watchdog_check(self):
        """
        Safety Timeout: If no full decision made in >85s, force decision with partial data.
        Prevents system hang if one sensor dies.
        """
        with self.data_lock:
            elapsed = time.time() - self.last_decision_time
            if elapsed > self.watchdog_timeout:
                # Only trigger if we have AT LEAST one pending sensor or VLM context
                has_data = (self.camera_data is not None or
                           self.carla_data is not None or
                           self.vlm_context is not None)

                if has_data:
                    self.get_logger().warn(
                        f"WATCHDOG TRIGGERED: System stalled for {elapsed:.1f}s. Forcing decision."
                    )

                    # Partial data snapshot
                    cam_data = self.camera_data.copy() if self.camera_data else None
                    carla_data = self.carla_data.copy() if self.carla_data else None
                    vlm_ctx = self.vlm_context.copy() if self.vlm_context else None

                    # Clear buffers
                    self.camera_data = None
                    self.carla_data = None
                    self.last_decision_time = time.time()

                    # Use Deterministic Fallback (skip LLM for robustness)
                    fallback_result = self._deterministic_fallback(cam_data, carla_data, vlm_ctx)
                    fallback_result["reasoning"] = "[WATCHDOG] Forced decision due to sensor timeout. " + fallback_result["reasoning"]

                    # Fill with dummy data if missing for publishing
                    dummy_cam = cam_data if cam_data else self._get_dummy_sensor_data("camera")
                    dummy_carla = carla_data if carla_data else self._get_dummy_sensor_data("carla")

                    self._publish_alert(fallback_result, dummy_cam, dummy_carla, vlm_ctx)

    def _get_dummy_sensor_data(self, name):
        """Return empty sensor structure for logging when data is unavailable."""
        return {
            'class_name': 'Unknown',
            'prediction': -1,
            'confidence': 0.0,
            'alert_prob': 0.0,
            'drowsy_prob': 0.0,
            'very_drowsy_prob': 0.0,
            'perclos': 0.0,
            'blink_rate': 0.0,
            'blink_duration_mean': 0.0,
            'entropy': 0.0,
            'steering_rate': 0.0,
            'sdlp': 0.0
        }

    def _invoke_integrated_reasoning(self, camera_data, carla_data, vlm_context):
        """Invoke Ollama with integrated sensor data (3-CLASS) in a background thread."""
        # Log before starting thread
        self.get_logger().info(
            f"\n60-SECOND WINDOW: INTEGRATED LLM REASONING (3-CLASS)\n"
            f"Camera ML:  {camera_data['class_name']} ({camera_data['prediction']}) - "
            f"conf: {camera_data['confidence']:.1%}\n"
            f"CARLA ML:   {carla_data['class_name']} ({carla_data['prediction']}) - "
            f"conf: {carla_data['confidence']:.1%}\n"
            f"VLM Context: {'Available (from occlusion event)' if vlm_context else 'Not yet (no events)'}"
        )

        # Start background thread for blocking inference
        # FIXED: Now correctly imports Thread
        inference_thread = Thread(
            target=self._run_inference_thread,
            args=(camera_data, carla_data, vlm_context),
            daemon=True
        )
        inference_thread.start()

    def _run_inference_thread(self, camera_data, carla_data, vlm_context):
        """Background thread to build prompt, query Ollama, and publish result."""
        try:
            prompt = self._build_integrated_prompt(camera_data, carla_data, vlm_context)

            # This blocking call now happens in a thread, safe for ROS2
            response = self._query_ollama_blocking(prompt)

            if response:
                # Publishing is thread-safe in rclpy
                self._publish_alert(response, camera_data, carla_data, vlm_context)
            else:
                self.get_logger().warn("LLM returned no response - SWITCHING TO DETERMINISTIC FALLBACK")

                # Rule-Based Fallback logic
                fallback_alert = self._deterministic_fallback(camera_data, carla_data, vlm_context)
                fallback_alert["reasoning"] = "[FALLBACK] LLM failed. " + fallback_alert["reasoning"]

                self._publish_alert(fallback_alert, camera_data, carla_data, vlm_context)

        except Exception as e:
            self.get_logger().error(f"Error in inference thread: {e}")

    def _build_integrated_prompt(self, camera_data, carla_data, vlm_context) -> str:
        """Build integrated prompt with 3-CLASS sensor data and enhanced VLM interpretation."""

        system_prompt = """You are a safety-critical driver monitoring assistant.
You act as a decision judge that resolves conflicts between multiple sensors using a 3-level drowsiness classification system.

CLASSIFICATION LEVELS:
- 0 (Alert): Driver is fully alert and attentive.
- 1 (Drowsy): Driver shows signs of reduced attention or early-stage drowsiness.
- 2 (Very Drowsy): Driver shows strong evidence of severe drowsiness or micro-sleep risk.

RELIABILITY RULES (follow strictly):
- If eye-tracking metrics report low confidence (≤60%), classify eye data as UNRELIABLE.
- If steering/vehicle metrics report low confidence (≤60%), classify steering data as UNRELIABLE.

VLM INTERPRETATION GUIDE (Probabilistic - Enhanced):
- VLM Confidence (0.0 - 1.0) represents reliability of visual context.
- Confidence > 0.9: HIGH reliability. VLM context overrides standard Camera ML.
- Confidence 0.6 - 0.9: MODERATE. Weigh VLM evidence against Camera evidence.
- Confidence < 0.6: LOW. Treat VLM context as UNTRUSTWORTHY. Prioritize standard sensors.

EVENT RULES:

1. BLOCKING EVENTS (Invalidate Camera): ["sunglasses", "mask", "hands_covering_eyes"]
   → IF VLM Confidence > 0.7: Classify eye metrics as UNRELIABLE. Trust Steering.

2. SUPPORTING EVENTS (Validate Drowsiness): ["yawning", "rubbing_eyes", "nodding"]
   → IF VLM Confidence > 0.6: Use as strong evidence for Class 1/2.

3. NEUTRAL EVENTS: ["drinking", "talking"]
   → Proceed with normal analysis.

DECISION RULES (3-class output):
- "alert": Both reliable sensors indicate class 0, OR sensors conflict but no reliable evidence suggests drowsiness.
- "drowsy": At least one reliable sensor indicates class 1, OR mixed evidence leans toward mild drowsiness.
- "very_drowsy": Both reliable sensors indicate class 2, OR strong evidence from at least one reliable modality supports class 2.
- "unknown": Data is unreliable, insufficient, or strongly contradictory.

INTERVENTION RULES:
- "no_alert": Final classification = Alert (0) with agreement across reliable sensors.
- "soft_alert": Final classification = Drowsy (1), OR final_state = "unknown".
- "strong_alert": Final classification = Very Drowsy (2) from at least one reliable sensor.
- "takeover_request": Very Drowsy (2) supported by multiple reliable sensors.
"""

        # Build VLM context section
        vlm_section = ""
        if vlm_context:
            try:
                # Extract fields from Qwen Schema
                occ = vlm_context.get('occlusion', {})
                if isinstance(occ, dict):
                    occ_reason = occ.get('reason', 'none') if occ.get('face_occluded') else 'none'
                else:
                    occ_reason = str(occ)

                # Extract behaviors
                behaviors = vlm_context.get('behaviors', {})
                detected_behaviors = []
                if isinstance(behaviors, dict):
                    for b_name, b_data in behaviors.items():
                        if isinstance(b_data, dict) and b_data.get('detected'):
                            detected_behaviors.append(b_name)

                # Extract visibility states
                eye_vis = vlm_context.get('eye_visibility', vlm_context.get('eyes_visible', 'unknown'))
                mouth_vis = vlm_context.get('mouth_visibility', vlm_context.get('mouth_visible', 'unknown'))
                notes = vlm_context.get('notes', 'N/A')
                vlm_conf = float(vlm_context.get('confidence', 0.8))

                vlm_section = f"""
VLM CONTEXT (from recent occlusion event - async analysis):
- Occlusion Type: {occ_reason}
- Behaviors Detected: {', '.join(detected_behaviors) if detected_behaviors else 'None'}
- Lighting Condition: {vlm_context.get('lighting', {}).get('level', 'normal') if isinstance(vlm_context.get('lighting'), dict) else 'normal'}
- Eye Visibility: {eye_vis}
- Mouth Visibility: {mouth_vis}
- VLM Notes: {notes}
- VLM Confidence: {vlm_conf:.2f} (Weigh this against sensor data)
"""
            except Exception as e:
                self.get_logger().warn(f"Error extracting VLM context: {e}")
                vlm_section = "\nVLM CONTEXT: Error parsing VLM data"

        else:
            vlm_section = "\nVLM CONTEXT: No occlusion events detected yet (normal conditions assumed)"

        user_prompt = f"""Analyze this integrated 1-minute driver monitoring window (3-CLASS):

CAMERA/EYE METRICS (Facial Analysis - from camera_ml_node):
- PERCLOS (% eye closure): {camera_data['perclos']:.1f}%
- Blink Rate: {camera_data['blink_rate']:.1f} blinks/min
- Avg Blink Duration: {camera_data['blink_duration_mean']:.4f}s
- Camera Model Prediction: {camera_data['class_name']} (class {camera_data['prediction']})
- Probabilities: Alert={camera_data['alert_prob']:.1%}, Drowsy={camera_data['drowsy_prob']:.1%}, VeryDrowsy={camera_data['very_drowsy_prob']:.1%}
- Camera Model Confidence: {camera_data['confidence']:.1%}
- Camera Reliability: {"RELIABLE" if camera_data['confidence'] > 0.6 else "UNRELIABLE"}

VEHICLE/STEERING METRICS (Vehicle Control - from carla_ml_node):
- Steering Entropy (randomness): {carla_data['entropy']:.4f}
- Steering Rate (corrections/min): {carla_data['steering_rate']:.1f}
- Lane Position Deviation (SDLP): {carla_data['sdlp']:.4f}
- Steering Model Prediction: {carla_data['class_name']} (class {carla_data['prediction']})
- Probabilities: Alert={carla_data['alert_prob']:.1%}, Drowsy={carla_data['drowsy_prob']:.1%}, VeryDrowsy={carla_data['very_drowsy_prob']:.1%}
- Steering Model Confidence: {carla_data['confidence']:.1%}
- Steering Reliability: {"RELIABLE" if carla_data['confidence'] > 0.6 else "UNRELIABLE"}

{vlm_section}

DECISION TASK:
Output ONLY a valid JSON object with these exact fields:

{{
  "final_state": "alert|drowsy|very_drowsy|unknown",
  "intervention_action": "no_alert|soft_alert|strong_alert|takeover_request",
  "llm_confidence": 0.X (0.0 to 1.0),
  "reasoning": "Explain your decision by explicitly citing the reliable metrics."
}}

Consider:
1. Which sensors are reliable (confidence > 0.6)?
2. What are the class predictions from reliable sensors?
3. Does VLM indicate conditions that affect eye metric reliability?
4. Do reliable sensors agree or conflict?
5. What is the most conservative safe decision?

RESPOND WITH JSON ONLY:"""

        return system_prompt + "\n\n" + user_prompt

    def _deterministic_fallback(self, cam_data, carla_data, vlm_data) -> dict:
        """
        Rule-based safety logic if LLM fails or Watchdog triggers.
        Returns: {{final_state, intervention_action, llm_confidence, reasoning}}
        """
        reasoning = []
        cam_score = 0.0
        carla_score = 0.0
        valid_sensors = 0

        # 1. Check VLM Blocking
        camera_blocked = False
        if vlm_data:
            occ_type = str(vlm_data.get('occlusion', '')).lower()
            if isinstance(vlm_data.get('occlusion'), dict):
                occ_type = str(vlm_data.get('occlusion', {}).get('reason', '')).lower()

            blockers = ["sunglasses", "mask", "hands", "dark"]
            if any(b in occ_type for b in blockers):
                camera_blocked = True
                reasoning.append(f"VLM detects blocking: {occ_type}. Camera UNRELIABLE.")

        # 2. Assess Camera
        if cam_data and not camera_blocked:
            pred = cam_data.get('prediction', 0)
            conf = cam_data.get('confidence', 0.0)

            if conf > 0.6:
                if pred == 1:
                    cam_score = 0.5
                elif pred == 2:
                    cam_score = 1.0
                else:
                    cam_score = 0.0

                valid_sensors += 1
                reasoning.append(f"Camera: {cam_data.get('class_name')} ({conf:.1%}).")
            else:
                reasoning.append(f"Camera: Low confidence ({conf:.1%}).")
        elif not cam_data:
            reasoning.append("Camera: No data available.")

        # 3. Assess CARLA/Steering
        if carla_data:
            pred = carla_data.get('prediction', 0)
            conf = carla_data.get('confidence', 0.0)

            if conf > 0.4:  # Steering often has lower confidence
                if pred == 1:
                    carla_score = 0.5
                elif pred == 2:
                    carla_score = 1.0
                else:
                    carla_score = 0.0

                valid_sensors += 1
                reasoning.append(f"Steering: {carla_data.get('class_name')} ({conf:.1%}).")
            else:
                reasoning.append(f"Steering: Low confidence ({conf:.1%}).")
        else:
            reasoning.append("Steering: No data available.")

        # 4. Integrate scores
        final_state = "unknown"
        action = "soft_alert"  # Default fail-safe
        confidence = 0.0

        if valid_sensors > 0:
            avg_score = (cam_score + carla_score) / valid_sensors

            if avg_score >= 0.8:
                final_state = "very_drowsy"
                action = "strong_alert"
                confidence = 0.85
            elif avg_score >= 0.4:
                final_state = "drowsy"
                action = "soft_alert"
                confidence = 0.70
            else:
                final_state = "alert"
                action = "no_alert"
                confidence = 0.75
        else:
            reasoning.append("No reliable sensors available. Using fail-safe mode.")
            confidence = 0.5

        return {
            "final_state": final_state,
            "intervention_action": action,
            "llm_confidence": confidence,
            "reasoning": " ".join(reasoning)
        }

    def _query_ollama_blocking(self, prompt: str) -> dict:
        """Query Ollama with integrated prompt."""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 2048,
                "num_ctx": 8192
            }

            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                timeout=90
            )

            if response.status_code == 200:
                result = response.json()
                text_output = result.get("response", "")
                self.get_logger().debug(f"Raw response (first 300 chars): {text_output[:300]}")
                return self._parse_json_response(text_output)
            else:
                self.get_logger().error(f"Ollama API error: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            self.get_logger().error("Ollama request timed out (90s)")
            return None
        except Exception as e:
            self.get_logger().error(f"Ollama query error: {e}")
            return None

    def _parse_json_response(self, text_output: str) -> dict:
        """FIXED: Extract and parse JSON from Ollama response with robust regex."""
        try:
            # Clean markdown wrappers
            text_output = text_output.strip()
            text_output = text_output.replace("```json", "").replace("```", "").strip()

            # Try direct JSON parse first
            try:
                parsed = json.loads(text_output)
                required = ['final_state', 'intervention_action', 'llm_confidence', 'reasoning']
                if all(field in parsed for field in required):
                    return parsed
            except json.JSONDecodeError:
                pass  # Fall through to regex extraction

            # FIXED: More permissive regex that handles nested braces
            # Match from first { to last } containing required fields
            match_start = text_output.find('{')
            match_end = text_output.rfind('}')

            if match_start >= 0 and match_end > match_start:
                json_str = text_output[match_start:match_end + 1]

                try:
                    parsed = json.loads(json_str)
                    required = ['final_state', 'intervention_action', 'llm_confidence', 'reasoning']
                    if all(field in parsed for field in required):
                        return parsed
                except json.JSONDecodeError as e:
                    self.get_logger().warn(f"JSON parse error in extracted string: {e}")

            self.get_logger().warn(f"No valid JSON found in response (first 200 chars): {text_output[:200]}")
            return None

        except Exception as e:
            self.get_logger().error(f"JSON parsing error: {e}")
            return None

    def _publish_alert(self, alert_data: dict, camera_data: dict, carla_data: dict, vlm_data: dict = None):
        """Publish integrated safety-critical alert (every 60s)."""
        try:
            alert_json = {
                'driver_id': self.driver_id,
                'final_state': alert_data.get('final_state', 'unknown'),
                'intervention_action': alert_data.get('intervention_action', 'soft_alert'),
                'llm_confidence': float(alert_data.get('llm_confidence', 0.5)),
                'reasoning': str(alert_data.get('reasoning', 'Evaluation required')),
                'timestamp': time.time(),
                'llm_model': self.ollama_model,
                'window_duration_sec': 60,
                'sensors': {
                    'camera': {
                        'prediction': camera_data.get('class_name', 'Unknown'),
                        'confidence': float(camera_data.get('confidence', 0.0))
                    },
                    'carla': {
                        'prediction': carla_data.get('class_name', 'Unknown'),
                        'confidence': float(carla_data.get('confidence', 0.0))
                    },
                    'vlm': {
                        'occlusion': str(vlm_data.get('occlusion', 'none')) if vlm_data else 'none',
                        'confidence': float(vlm_data.get('confidence', 0.0)) if vlm_data else 0.0
                    }
                }
            }

            msg = String()
            msg.data = json.dumps(alert_json, indent=2)
            self.alert_pub.publish(msg)

            self.get_logger().info(
                f"\n60-SECOND ALERT PUBLISHED:\n"
                f"State: {alert_json['final_state']}\n"
                f"Action: {alert_json['intervention_action']}\n"
                f"LLM Confidence: {alert_json['llm_confidence']:.1%}\n"
                f"Reasoning: {alert_json['reasoning']}"
            )

            # Save to disk
            self._save_alert_to_disk(alert_json)

        except Exception as e:
            self.get_logger().error(f"Failed to publish alert: {e}")

    def _save_alert_to_disk(self, alert_json):
        """Save alert JSON to disk for persistence and append to CSV."""
        try:
            # 1. Save individual JSON
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alert_{timestamp_str}.json"
            filepath = os.path.join(self.log_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(alert_json, f, indent=2)

            self.get_logger().info(f"Saved alert JSON to: {filepath}")

            # 2. Append to CSV summary
            with open(self.csv_path, 'a') as f:
                cam_pred = alert_json['sensors']['camera']['prediction']
                cam_conf = alert_json['sensors']['camera']['confidence']
                carla_pred = alert_json['sensors']['carla']['prediction']
                carla_conf = alert_json['sensors']['carla']['confidence']

                # Get VLM data (safely)
                vlm = alert_json.get('sensors', {}).get('vlm', {})
                vlm_conf = vlm.get('confidence', 0.0)
                vlm_occ = vlm.get('occlusion', 'none')

                # Escape reasoning for CSV
                reasoning = alert_json['reasoning'].replace(",", ";").replace("\n", " ")

                line = (
                    f"{timestamp_str},{self.driver_id},{alert_json['final_state']},"
                    f"{alert_json['intervention_action']},{alert_json['llm_confidence']:.2f},"
                    f"{cam_pred},{cam_conf:.2f},{carla_pred},{carla_conf:.2f},"
                    f"{vlm_conf:.2f},{vlm_occ},{reasoning}\n"
                )
                f.write(line)

            self.get_logger().info(f"Appended to CSV: {self.csv_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to save alert to disk: {e}")


def main(args=None):
    """Main entry point for ROS2 node."""
    try:
        rclpy.init(args=args)
        node = IntegratedSafetyCriticalLLMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        rclpy.get_logger().info("Stopping Integrated Safety-Critical LLM Node...")
    except Exception as e:
        print(f"Error starting node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()