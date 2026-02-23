#!/usr/bin/env python3
"""
ROS2 Node: CARLA Steering Metrics Generator
- Subscribes to /carla/hero/vehicle_control_cmd (60Hz steering data)
- Subscribes to /carla/lane_offset (60Hz lane position)
- Aggregates 1-second windows
- Computes: Entropy, Steering Rate, SDLP
- Publishes to /carla_metrics (1Hz)
"""

import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleControl
from geometry_msgs.msg import Vector3Stamped
from drowsiness_detection_msg.msg import LanePosition
import numpy as np
from collections import deque
from threading import Lock

# Import standardized metric calculation function
from drowsiness_detection.camera.utils import vehicle_feature_extraction


class CarlaSteeringMetricsNode(Node):
    def __init__(self):
        super().__init__("carla_steering_metrics_node")
        
        # Parameters
        self.declare_parameter("window_duration", 1)  # 1-second window
        self.window_duration = self.get_parameter("window_duration").value
        
        self.declare_parameter("sampling_rate", 60)  # 60Hz from manual control
        self.sampling_rate = self.get_parameter("sampling_rate").value
        
        # Calculate buffer size: 1 second × 60Hz = 60 samples
        self.window_size = int(self.window_duration * self.sampling_rate)  # = 60
        
        # Thread safety
        self.buffer_lock = Lock()
        
        # Buffers for 1-second window (60 samples @ 60Hz)
        self.steering_buffer = deque(maxlen=self.window_size)
        self.lane_buffer = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
        
        # Subscribe to both steering and lane position
        self.steering_sub = self.create_subscription(
            CarlaEgoVehicleControl,  
            "/carla/hero/vehicle_control_cmd",
            self.steering_callback,
            10
        )
        
        self.lane_sub = self.create_subscription(
            LanePosition,
            "/carla/lane_offset",
            self.lane_callback,
            10
        )
        
        # Publisher for metrics (1Hz)
        self.metrics_pub = self.create_publisher(
            Vector3Stamped,
            "/carla_metrics",
            10
        )
        
        # Timer for 1-second processing
        self.create_timer(1.0, self.process_window)
        
        self.get_logger().info(
            f"✅ CARLA Steering Metrics Node started\n"
            f"   Window: {self.window_duration}s\n"
            f"   Sampling rate: {self.sampling_rate}Hz\n"
            f"   Window size: {self.window_size} samples\n"
            f"   Publishing: 1 message per second"
        )
    
    def steering_callback(self, msg):
        """Store steering angle (60Hz)."""
        steering_angle = float(msg.steer)
        timestamp = self.get_clock().now()
        
        with self.buffer_lock:
            self.steering_buffer.append(steering_angle)
            self.timestamps.append(timestamp)
    
    def lane_callback(self, msg):
        """Store lane offset (60Hz)."""
        lane_offset = float(msg.lane_offset)
        
        with self.buffer_lock:
            self.lane_buffer.append(lane_offset)
    
    def process_window(self):
        """Process 1-second window and publish metrics."""
        with self.buffer_lock:
            if len(self.steering_buffer) < self.window_size:
                # Not enough data yet
                self.get_logger().debug(
                    f"Buffer filling: {len(self.steering_buffer)}/{self.window_size}"
                )
                return
            
            # Convert to numpy arrays
            steering_data = np.array(list(self.steering_buffer))
            lane_data = np.array(list(self.lane_buffer)) if len(self.lane_buffer) > 0 else np.zeros_like(steering_data)
            
            try:
                # Calculate metrics using standardized function
                entropy, steering_rate, sdlp = vehicle_feature_extraction(
                    steering_data=steering_data,
                    lane_position_data=lane_data,
                    window=self.window_duration  # 1 second
                )
                
                # Convert steering rate to per-minute
                steering_rate_per_min = steering_rate * 60  # Convert from per-second to per-minute
                
                # Publish metrics
                self._publish_metrics(entropy, steering_rate_per_min, sdlp)
            except Exception as e:
                self.get_logger().error(f"Error processing metrics: {e}")
    
    def _publish_metrics(self, entropy, steering_rate, sdlp):
        """Publish computed metrics to /carla_metrics."""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "carla_steering"
        msg.vector.x = float(entropy)
        msg.vector.y = float(steering_rate)
        msg.vector.z = float(sdlp)
        
        self.metrics_pub.publish(msg)
        
        self.get_logger().info(
            f"Published metrics: E={entropy:.4f}, R={steering_rate:.1f}/min, SDLP={sdlp:.4f}",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = CarlaSteeringMetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping CARLA Steering Metrics Node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()