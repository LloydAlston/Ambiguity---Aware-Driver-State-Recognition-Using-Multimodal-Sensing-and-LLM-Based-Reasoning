#!/usr/bin/env python3
"""
ROS2 Node: CARLA ML Model Inference
- Subscribes to /carla_metrics (1Hz steering metrics)
- Aggregates 60 seconds of metrics
- Runs ML model inference
- Publishes predictions to /carla_predictions
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque
import joblib
from threading import Lock
import os


class CarlaMLNode(Node):
    def __init__(self):
        super().__init__("carla_ml_node")
        
        # Metrics come at 1Hz (60 per minute)
        self.metrics_rate = 1.0  # Hz
        
        # Window for aggregation: 60 seconds
        self.declare_parameter("aggregation_window", 60)
        self.aggregation_window = self.get_parameter("aggregation_window").value
        
        # Buffer size: 60 seconds × 1 message/second = 60 messages
        self.buffer_size = int(self.aggregation_window * self.metrics_rate)  # = 60
        
        # Thread safety
        self.buffer_lock = Lock()
        
        # Model directory parameter
        self.declare_parameter("model_dir", os.path.expanduser("~/Team_vision/drowsiness_detection_ros2/models"))
        self.model_dir = self.get_parameter("model_dir").value
        
        # Buffers
        self.entropy_buffer = deque(maxlen=self.buffer_size)
        self.steering_rate_buffer = deque(maxlen=self.buffer_size)
        self.sdlp_buffer = deque(maxlen=self.buffer_size)
        
        # Load ML model
        self._load_ml_model()
        
        # Subscriber
        self.metrics_sub = self.create_subscription(
            Vector3Stamped,
            "/carla_metrics",
            self.metrics_callback,
            10
        )
        
        # Publisher
        self.predictions_pub = self.create_publisher(
            Float64MultiArray,
            "/carla_predictions",
            10
        )
        
        # Run model every 62 seconds (allow buffer to fill slightly more than 60s)
        self.create_timer(self.aggregation_window + 2, self.run_aggregated_inference)
        
        self.get_logger().info(
            f"CARLA ML Node initialized\n"
            f"   Aggregation window: {self.aggregation_window}s\n"
            f"   Buffer size: {self.buffer_size} messages\n"
            f"   Running model every {self.aggregation_window} seconds"
        )
    
    def _load_ml_model(self):
        """Load pre-trained ML model."""
        try:
            model_path = os.path.join(self.model_dir, 'model_carla_steering_rf.pkl')
            scaler_path = os.path.join(self.model_dir, 'model_carla_steering_rf_scaler.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                self.get_logger().error(f"CARLA model files not found in {self.model_dir}")
                self.carla_model = None
                self.carla_scaler = None
                self.model_loaded = False
                return
            
            self.carla_model = joblib.load(model_path)
            self.carla_scaler = joblib.load(scaler_path)
            self.get_logger().info(f"✅ CARLA ML model loaded from {model_path}")
            self.model_loaded = True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load CARLA model: {e}")
            self.carla_model = None
            self.carla_scaler = None
            self.model_loaded = False
    
    def metrics_callback(self, msg):
        """Store metrics (1Hz)."""
        with self.buffer_lock:
            self.entropy_buffer.append(msg.vector.x)
            self.steering_rate_buffer.append(msg.vector.y)
            self.sdlp_buffer.append(msg.vector.z)
            
            # Debug: show buffer fill rate
            if len(self.entropy_buffer) % 10 == 0:  # Every 10 messages
                self.get_logger().info(
                    f"Buffer: {len(self.entropy_buffer)}/{self.buffer_size} messages",
                    throttle_duration_sec=5.0
                )
    
    def run_aggregated_inference(self):
        """Aggregate 60 seconds of metrics and run ML model."""
        with self.buffer_lock:
            # Tolerate minor drop (58/60 is fine)
            if len(self.entropy_buffer) < (self.buffer_size - 2):
                self.get_logger().warn(
                    f"Not enough data. Have {len(self.entropy_buffer)} messages, need ~{self.buffer_size}"
                )
                return
            
            # === AGGREGATION STRATEGY ===
            # Option 1: Mean of 60 one-second metrics
            aggregated_entropy = np.mean(list(self.entropy_buffer))
            aggregated_steering_rate = np.mean(list(self.steering_rate_buffer))
            aggregated_sdlp = np.mean(list(self.sdlp_buffer))
            
            # Option 2: Maximum (worst-case) from the minute
            # aggregated_entropy = np.max(list(self.entropy_buffer))
            # aggregated_steering_rate = np.max(list(self.steering_rate_buffer))
            # aggregated_sdlp = np.max(list(self.sdlp_buffer))
            
            # Option 3: Weighted average (recent more important)
            # weights = np.linspace(0.5, 1.5, len(self.entropy_buffer))
            # aggregated_entropy = np.average(list(self.entropy_buffer), weights=weights)
            
            self.get_logger().info(
                f"\n=== 60-SECOND AGGREGATION ===\n"
                f"Window: {self.aggregation_window}s\n"
                f"Samples: {len(self.entropy_buffer)} messages\n"
                f"Aggregated Entropy: {aggregated_entropy:.4f}\n"
                f"Aggregated Steering Rate: {aggregated_steering_rate:.1f} changes/min\n"
                f"Aggregated SDLP: {aggregated_sdlp:.4f}"
            )
            
            # Run ML model with aggregated features
            result = self._run_ml_model(
                aggregated_entropy,
                aggregated_steering_rate,
                aggregated_sdlp
            )
            
            self._publish_prediction(result, aggregated_entropy, aggregated_steering_rate, aggregated_sdlp)
    
    def _run_ml_model(self, entropy, steering_rate, sdlp):
        """Run ML model inference."""
        if not self.model_loaded:
            self.get_logger().warn("Model not loaded - skipping inference")
            return None
        
        try:
            # Create feature vector
            features = np.array([[entropy, steering_rate, sdlp]])
            features_scaled = self.carla_scaler.transform(features)
            
            # Get prediction and probabilities
            prediction = self.carla_model.predict(features_scaled)[0]
            proba = self.carla_model.predict_proba(features_scaled)[0]
            
            result = {
                'prediction': int(prediction),
                'probabilities': proba.tolist(),
                'confidence': float(max(proba))
            }
            
            self.get_logger().info(
                f"ML Prediction: Class {result['prediction']} "
                f"(confidence: {result['confidence']:.1%})"
            )
            
            return result
        except Exception as e:
            self.get_logger().error(f"ML model error: {e}")
            return None
    
    def _publish_prediction(self, result, entropy, steering_rate, sdlp):
        """Publish prediction to /carla_predictions."""
        msg = Float64MultiArray()
        
        if result:
            # Format: [entropy, steering_rate, sdlp, prediction, prob_0, prob_1, prob_2, ...]
            msg.data = [
                float(entropy),
                float(steering_rate),
                float(sdlp),
                float(result['prediction'])
            ]
            msg.data.extend(result['probabilities'])
            msg.data.append(result['confidence'])
        else:
            # Fallback if model failed
            msg.data = [
                float(entropy),
                float(steering_rate),
                float(sdlp),
                1.0,  # Default prediction
                0.33, 0.33, 0.33,  # Equal probabilities
                0.33  # Low confidence
            ]
        
        self.predictions_pub.publish(msg)
        self.get_logger().info(
            f"Published prediction to /carla_predictions",
            throttle_duration_sec=5.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = CarlaMLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping CARLA ML Node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()