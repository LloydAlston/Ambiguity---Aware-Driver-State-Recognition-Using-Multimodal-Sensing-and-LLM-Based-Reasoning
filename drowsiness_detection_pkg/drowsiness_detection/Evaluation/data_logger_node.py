#!/usr/bin/env python3
"""
ROS2 Node: ML Data Logger
- Subscribes to /camera_predictions and /carla_predictions
- Logs every prediction to a CSV file for offline analysis.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
import csv
import os
import time
from datetime import datetime
import json

class MLDataLoggerNode(Node):
    def __init__(self):
        super().__init__("data_logger_node")

        # === Parameters ===
        self.declare_parameter("log_dir", "/home/karthik/DROWSINESS_DETECTION/ml_logs")
        self.log_dir = self.get_parameter("log_dir").value
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create/Open CSV file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = os.path.join(self.log_dir, f"ml_performance_log_{timestamp_str}.csv")
        self.csv_file = open(self.csv_file_path, mode='w', newline='', buffering=1) # Line buffering
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write Header
        self.header = ["timestamp", "datetime", "source", "class_prediction", "class_name", "confidence", "features_json"]
        self.csv_writer.writerow(self.header)
        
        self.class_names = {0: "Alert", 1: "Drowsy", 2: "Very Drowsy"}

        # === Subscribers ===
        self.create_subscription(
            Float64MultiArray, "/camera_predictions", self.camera_callback, 10
        )
        self.create_subscription(
            Float64MultiArray, "/carla_predictions", self.carla_callback, 10
        )
        
        self.get_logger().info(f"ML Data Logger initialized. Writing to: {self.csv_file_path}")

    def camera_callback(self, msg):
        self._log_data("Camera", msg.data)

    def carla_callback(self, msg):
        self._log_data("Carla", msg.data)
        
    def _log_data(self, source, data):
        if len(data) < 8:
            return

        try:
            # Extract common fields (Protocol from ML nodes)
            # [features..., prediction, alert_prob, drowsy_prob, very_drowsy_prob, confidence]
            prediction = int(data[3])
            confidence = float(data[7])
            class_name = self.class_names.get(prediction, "Unknown")
            
            # Pack rest as JSON features
            features = {
                "raw_values": list(data),
                "probs": {
                    "alert": data[4],
                    "drowsy": data[5],
                    "very_drowsy": data[6]
                }
            }
            
            row = [
                time.time(),
                datetime.now().isoformat(),
                source,
                prediction,
                class_name,
                f"{confidence:.4f}",
                json.dumps(features)
            ]
            
            self.csv_writer.writerow(row)
            # self.get_logger().info(f"Logged {source} prediction: {class_name}")
            
        except Exception as e:
            self.get_logger().error(f"Error logging data: {e}")

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MLDataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
