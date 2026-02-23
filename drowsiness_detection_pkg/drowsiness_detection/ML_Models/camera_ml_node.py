#!/usr/bin/env python3
"""
ROS2 Node: Camera ML Model Inference
- Subscribes to /ear_mar (60 FPS EAR/MAR values)
- Buffers for 60 seconds
- Computes minute-level metrics
- Runs Camera ML model (3 features: PERCLOS, BlinkRate, blink_duration_mean)
- Publishes to /camera_predictions (3-CLASS: 0=Alert, 1=Drowsy, 2=Very Drowsy)
"""


import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from drowsiness_detection_msg.msg import EarMarValue
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque
import joblib
from threading import Lock
import threading
import os
from datetime import datetime
import json



# Import utility functions from your utils.py
from drowsiness_detection.camera.utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
)




class CameraMLNode(Node):
    """Camera ML Model - Facial Metrics Inference (3-CLASS)."""



    def __init__(self):
        super().__init__("camera_ml_node")



        # === Parameters ===
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value
        
        # Log directory for persisting predictions
        self.log_dir = os.path.join(os.getenv('HOME'), 'camera_ml_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.declare_parameter("window_duration", 60)  # 60 seconds
        self.window_duration = self.get_parameter("window_duration").value
        
        self.declare_parameter("fps", 30)  # Camera runs at ~30-45 FPS
        self.fps = self.get_parameter("fps").value



        # === EAR/MAR thresholds ===
        self.ear_threshold = 0.26
        self.mar_threshold = 0.5
        self.min_consec_frames = 2
        
        # === Class names ===
        self.class_names = {0: "Alert", 1: "Drowsy", 2: "Very Drowsy"}
        
        # === Subscriber: Camera metrics ===
        self.declare_parameter("model_dir", os.path.expanduser("~/Team_vision/drowsiness_detection_ros2/models"))
        self.model_dir = self.get_parameter("model_dir").value

        self.subscription = self.create_subscription(
            EarMarValue,
            "/ear_mar",
            self.ear_mar_callback,
            10  # Use default RELIABLE QoS to match publisher
        )
        self.get_logger().info("✅ Subscribed to /ear_mar topic (RELIABLE QoS)")

        # === Publisher: Camera ML predictions ===
        self.predictions_pub = self.create_publisher(
            Float64MultiArray,
            "/camera_predictions",
            10
        )

        # === Data buffers (thread-safe) ===
        self.buffer_lock = Lock()
        self.max_buffer_size = self.fps * self.window_duration
        self.ear_buffer = deque(maxlen=self.max_buffer_size)
        self.mar_buffer = deque(maxlen=self.max_buffer_size)

        # === Last data timestamp (for monitoring) ===
        self.last_data_time = None
        
        # === ML models ===
        self._load_ml_models()
        
        # === Monitoring timer ===
        # Check if we're receiving data regularly
        self.monitor_timer = self.create_timer(5.0, self.check_data_flow)
        self.data_timeout_threshold = 10.0  # seconds

        # === Timer removed ===
        # Inference is triggered in callback when buffer is full (Fixed Segment)
        
        self.get_logger().info(
            f"Camera ML Node started (3-CLASS)\n"
            f"   Driver ID: {self.driver_id}\n"
            f"   Window: {self.window_duration}s @ {self.fps} FPS\n"
            f"   Model Dir: {self.model_dir}\n"
            f"   Classes: 0=Alert, 1=Drowsy, 2=Very Drowsy"
        )

    def _load_ml_models(self):
        """Load pre-trained ML model for camera."""
        try:
            model_path = os.path.join(self.model_dir, 'model_camera_rf.pkl')
            scaler_path = os.path.join(self.model_dir, 'model_camera_rf_scaler.pkl')

            # Detailed file checks
            if not os.path.exists(model_path):
                self.get_logger().error(f"❌ Model file not found: {model_path}")
                self.camera_model = None
                return
            
            if not os.path.exists(scaler_path):
                self.get_logger().error(f"❌ Scaler file not found: {scaler_path}")
                self.camera_model = None
                return
            
            # Check file sizes
            model_size = os.path.getsize(model_path)
            scaler_size = os.path.getsize(scaler_path)
            self.get_logger().info(f"Loading model ({model_size} bytes) and scaler ({scaler_size} bytes)...")

            self.camera_model = joblib.load(model_path)
            self.camera_scaler = joblib.load(scaler_path)
            self.get_logger().info(f"✅ Camera model loaded successfully from {model_path}")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load camera model: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.camera_model = None



    def check_data_flow(self):
        """Monitor if data is being received regularly."""
        if self.last_data_time is None:
            self.get_logger().warn(
                "⚠️  No data received yet from /ear_mar topic. "
                "Is camera_mediapipe_node running?"
            )
            return
        
        time_since_data = (self.get_clock().now().nanoseconds / 1e9) - self.last_data_time
        
        if time_since_data > self.data_timeout_threshold:
            self.get_logger().warn(
                f"⚠️  No data received for {time_since_data:.1f}s. "
                f"Last data at {self.last_data_time:.1f}. "
                "Check if camera_mediapipe_node is still running."
            )
        else:
            buffer_len = len(self.ear_buffer)
            if buffer_len > 0:
                progress_pct = (buffer_len / self.max_buffer_size) * 100
                self.get_logger().info(
                    f"📊 Data flowing OK. Buffer: {buffer_len}/{self.max_buffer_size} "
                    f"({progress_pct:.1f}%) - Need 100% for inference"
                )
    
    def ear_mar_callback(self, msg: EarMarValue):
        """Receive EAR/MAR values from camera node."""
        try:
            # Update last data timestamp
            self.last_data_time = self.get_clock().now().nanoseconds / 1e9
            
            # DEBUG: Log first callback
            if len(self.ear_buffer) == 0:
                self.get_logger().info("🎯 First /ear_mar message received! Callback is working.")
            
            ear = float(msg.ear_value)
            mar = float(msg.mar_value)
            
            # Validate data quality
            if np.isnan(ear) or np.isnan(mar):
                self.get_logger().warn(f"⚠️  Received NaN values (EAR={ear}, MAR={mar}), skipping")
                return
            
            if ear < 0 or ear > 1 or mar < 0 or mar > 2:
                self.get_logger().warn(
                    f"⚠️  Received out-of-range values (EAR={ear:.3f}, MAR={mar:.3f}), "
                    "but adding to buffer anyway"
                )

            with self.buffer_lock:
                self.ear_buffer.append(ear)
                self.mar_buffer.append(mar)
                
                # Progress logging every 5 seconds worth of data (more frequent)
                buffer_len = len(self.ear_buffer)
                if buffer_len > 0 and buffer_len % (self.fps * 5) == 0:
                    progress_pct = (buffer_len / self.max_buffer_size) * 100
                    self.get_logger().info(
                        f"📈 Buffer progress: {buffer_len}/{self.max_buffer_size} "
                        f"({progress_pct:.0f}%) - Inference triggers at 100%"
                    )
                
                # Check if buffer is full (Fixed Segment)
                if len(self.ear_buffer) >= self.max_buffer_size:
                    self.get_logger().info("🚀 Buffer full! Starting ML inference...")
                    self.run_ml_inference()
        
        except Exception as e:
            self.get_logger().error(f"❌ Error in ear_mar_callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())



    def run_ml_inference(self):
        """Run ML inference on buffered data.
        
        NOTE: This method is called while holding self.buffer_lock from ear_mar_callback().
        Do NOT add 'with self.buffer_lock:' here - it will cause deadlock!
        """
        try:
            # Lock is already held by caller (ear_mar_callback)
            if len(self.ear_buffer) < self.max_buffer_size:
                self.get_logger().warn(
                    f"⚠️  Inference called but buffer not full: "
                    f"{len(self.ear_buffer)}/{self.max_buffer_size}"
                )
                return
            
            self.get_logger().info("🔬 Running ML inference on 60-second data window...")

            # === CAMERA METRICS ===
            camera_metrics = self._compute_camera_metrics()

            self.get_logger().info(
                f"\nCAMERA METRICS (60s window):\n"
                f"  PERCLOS:              {camera_metrics['perclos']:>6.1f}%\n"
                f"  BlinkRate:            {camera_metrics['blink_rate']:>6.1f} blinks/min\n"
                f"  Blink Duration Mean:  {camera_metrics['blink_duration_mean']:>6.4f}s"
            )

            # === Run ML model ===
            camera_result = self._run_camera_model(
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['blink_duration_mean']
            )

            # === Publish results ===
            self._publish_predictions(camera_result, camera_metrics)

            # === FIXED SEGMENT: CLEAR BUFFERS ===
            self.ear_buffer.clear()
            self.mar_buffer.clear()
            self.get_logger().info("✅ Inference complete. Buffers cleared. Collecting next 60s window...")
        
        except Exception as e:
            self.get_logger().error(f"❌ Error during ML inference: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Clear buffers even on error to avoid stuck state
            self.ear_buffer.clear()
            self.mar_buffer.clear()



    def _compute_camera_metrics(self) -> dict:
        """Compute camera-based metrics."""
        ear_array = np.array(list(self.ear_buffer))
        mar_array = np.array(list(self.mar_buffer))


        perclos = calculate_perclos(
            ear_values=ear_array,
            ear_threshold=self.ear_threshold,
            min_consec_frames=self.min_consec_frames
        )


        blink_rate = calculate_blink_frequency(
            ear_values=ear_array,
            ear_threshold=self.ear_threshold,
            fps=self.fps,
            min_consec_frames=self.min_consec_frames
        )


        blink_duration_mean = self._calculate_blink_duration_mean(ear_array)


        return {
            'perclos': perclos,
            'blink_rate': blink_rate,
            'blink_duration_mean': blink_duration_mean
        }



    def _calculate_blink_duration_mean(self, ear_values: np.ndarray) -> float:
        """Calculate mean blink duration from EAR values."""
        if len(ear_values) == 0:
            return 0.0
        
        below_threshold = ear_values < self.ear_threshold
        blink_starts = np.where(np.diff(below_threshold.astype(int)) == 1)[0]
        blink_ends = np.where(np.diff(below_threshold.astype(int)) == -1)[0]
        
        if len(blink_starts) == 0 or len(blink_ends) == 0:
            return 0.0
        
        blink_durations_frames = []
        for start, end in zip(blink_starts, blink_ends):
            if end > start:
                blink_durations_frames.append(end - start)
        
        if len(blink_durations_frames) == 0:
            return 0.0
        
        mean_duration_frames = np.mean(blink_durations_frames)
        mean_duration_seconds = mean_duration_frames / self.fps
        
        return max(0.0, mean_duration_seconds)



    def _run_camera_model(self, perclos, blink_rate, blink_duration_mean):
        """Run Camera Model (Random Forest on facial metrics) - 3-CLASS."""
        if self.camera_model is None:
            self.get_logger().error(
                "❌ Camera model not loaded - cannot run inference! "
                "Check model loading errors above."
            )
            return None


        try:
            features = np.array([[perclos, blink_rate, blink_duration_mean]])
            features_scaled = self.camera_scaler.transform(features)
            
            # Get prediction and probabilities for 3 classes
            prediction = self.camera_model.predict(features_scaled)[0]
            proba = self.camera_model.predict_proba(features_scaled)[0]
            
            # proba = [P(Alert), P(Drowsy), P(Very Drowsy)]
            result = {
                'prediction': int(prediction),
                'class_name': self.class_names[prediction],
                'alert_prob': float(proba[0]),
                'drowsy_prob': float(proba[1]),
                'very_drowsy_prob': float(proba[2]),
                'confidence': float(max(proba))
            }
            
            self.get_logger().info(
                f"\nCAMERA MODEL PREDICTION (3-CLASS):\n"
                f"  Predicted Class:     {result['class_name']} ({result['prediction']})\n"
                f"  Alert Prob:          {result['alert_prob']:.1%}\n"
                f"  Drowsy Prob:         {result['drowsy_prob']:.1%}\n"
                f"  Very Drowsy Prob:    {result['very_drowsy_prob']:.1%}\n"
                f"  Model Confidence:    {result['confidence']:.1%}"
            )
            
            return result
        except Exception as e:
            self.get_logger().error(f"Camera model error: {e}")
            return None



    def _publish_predictions(self, camera_result, camera_metrics):
        """
        Publish camera ML predictions to /camera_predictions topic.
        
        Message format (Float64MultiArray):
          data[0] = PERCLOS (%)
          data[1] = BlinkRate (blinks/min)
          data[2] = Blink Duration Mean (seconds)
          data[3] = Predicted Class (0=Alert, 1=Drowsy, 2=Very Drowsy)
          data[4] = Alert Probability (0-1)
          data[5] = Drowsy Probability (0-1)
          data[6] = Very Drowsy Probability (0-1)
          data[7] = Model Confidence (0-1)
        """
        msg = Float64MultiArray()
        
        if camera_result:
            msg.data.extend([
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['blink_duration_mean'],
                float(camera_result['prediction']),
                camera_result['alert_prob'],
                camera_result['drowsy_prob'],
                camera_result['very_drowsy_prob'],
                camera_result['confidence']
            ])
            
            self.get_logger().info(
                f"\nPUBLISHED TO /camera_predictions:\n"
                f"  PERCLOS:              {camera_metrics['perclos']:.1f}%\n"
                f"  BlinkRate:            {camera_metrics['blink_rate']:.1f} blinks/min\n"
                f"  Blink Duration Mean:  {camera_metrics['blink_duration_mean']:.4f}s\n"
                f"  Prediction:           {camera_result['class_name']} ({camera_result['prediction']})\n"
                f"  Alert Prob:           {camera_result['alert_prob']:.1%}\n"
                f"  Drowsy Prob:          {camera_result['drowsy_prob']:.1%}\n"
                f"  Very Drowsy Prob:     {camera_result['very_drowsy_prob']:.1%}\n"
                f"  Confidence:           {camera_result['confidence']:.1%}"
            )
            # Save prediction to disk for later analysis
            self._save_prediction_to_disk(camera_result, camera_metrics)
        else:
            msg.data.extend([
                camera_metrics['perclos'],
                camera_metrics['blink_rate'],
                camera_metrics['blink_duration_mean'],
                1.0,  # Default to Drowsy (middle class)
                0.33, 0.33, 0.33, 0.33
            ])
            self.get_logger().warn(
                "Model failed - published metrics with default probabilities"
            )
        
        self.predictions_pub.publish(msg)

    def _save_prediction_to_disk(self, result, metrics):
        """Persist Camera ML prediction and raw metrics as a JSON file.
        Runs in a separate thread to avoid blocking ROS callbacks.
        """
        def save_task():
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"camera_pred_{timestamp}.json"
                filepath = os.path.join(self.log_dir, filename)
                data = {
                    'timestamp': timestamp,
                    'metrics': metrics,
                    'prediction': result
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                self.get_logger().info(f"Saved Camera ML prediction to: {filepath}")
            except Exception as e:
                self.get_logger().error(f"Failed to save camera prediction: {e}")

        # Fire and forget thread
        try:
            threading.Thread(target=save_task, daemon=True).start()
        except Exception as e:
            self.get_logger().error(f"Failed to start save thread: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = CameraMLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping Camera ML Node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == "__main__":
    main()