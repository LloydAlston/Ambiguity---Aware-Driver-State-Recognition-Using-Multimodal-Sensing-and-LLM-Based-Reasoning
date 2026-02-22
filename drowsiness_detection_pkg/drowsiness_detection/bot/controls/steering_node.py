#!/usr/bin/env python3

"""
This ROS2 Node controls the steering wheel vibration of a Logitech G29 with raw HID
commands.
The vibration pattern is manually encoded as byte sequences and written with `os.write`.
"""
import os
import time
from threading import Lock, Thread

import rclpy
from rclpy.node import Node
from drowsiness_detection_msg.msg import Vibration


class WheelControlVibration(Node):
    """
    ROS2 Node for controlling Logitech G29 steering wheel vibration
    via raw HID commands.
    """

    def __init__(self):
        super().__init__("wheel_vibration_node")

        self.declare_parameter("hid_device", "logitech_g29")
        self.HIDRAW_DEVICE = self.get_parameter("hid_device").value

        # Try opening the HIDRAW device
        self.raw_dev = None
        try:
            self.raw_dev = os.open(f"/dev/{self.HIDRAW_DEVICE}", os.O_RDWR)
            self.get_logger().info("Logitech G29 initialized successfully.")
        except OSError as e:
            self.get_logger().error(
                f"Failed to initialize device {self.HIDRAW_DEVICE}: {e}"
            )
            self.raw_dev = None

        self._steering_wheel_write_lock = Lock()
        self.vibration_active = False

        # ROS2 subscriber
        self.create_subscription(
            Vibration,
            "/wheel_vibration",  # topic name
            self.vibration_callback,  # callback
            10,  # QoS history depth
        )

    # --- Device commands ---
    def _send_vibration_command(self, force_altitude: int):
        try:
            os.write(
                self.raw_dev,
                bytearray(
                    [
                        0x21,
                        0x06,
                        128 + force_altitude,
                        128 - force_altitude,
                        8,
                        8,
                        0x0F,
                    ]
                ),
            )
        except OSError as e:
            self.get_logger().error(f"Error starting vibration: {e}")

    def _send_stop_command(self):
        try:
            os.write(
                self.raw_dev, bytearray([0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            )
        except OSError as e:
            self.get_logger().error(f"Error stopping vibration: {e}")

    def vibrate(self, duration=0.05, intensity=25):
        """
        Trigger vibration.

        Args:
            duration (float): Duration of vibration in seconds.
            intensity (int): Intensity (0-60).
        """
        if self.raw_dev is None:
            self.get_logger().warn("Raw device not initialized. Cannot vibrate.")
            return

        intensity = max(0, min(intensity, 60))

        def vibration_thread():
            with self._steering_wheel_write_lock:
                self._send_vibration_command(intensity)
                self.vibration_active = True
            time.sleep(duration)
            with self._steering_wheel_write_lock:
                self._send_stop_command()
                self.vibration_active = False

        Thread(target=vibration_thread, daemon=True).start()

    # --- ROS2 callback ---
    def vibration_callback(self, msg: Vibration):
        # Ensure duration and intensity are valid
        if msg.duration <= 0 or not (0 <= msg.intensity <= 60):
            self.get_logger().warn(
                f"Invalid vibration message received: duration={msg.duration}, intensity={msg.intensity}"
            )
            return

        # Trigger vibration
        self.vibrate(msg.duration, msg.intensity)

    def close(self):
        """Close HID device."""
        if self.raw_dev:
            try:
                os.close(self.raw_dev)
                self.get_logger().info("Raw device closed.")
                self.raw_dev = None
            except OSError as e:
                self.get_logger().error(f"Error closing raw device: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WheelControlVibration()
    rclpy.spin(node)
    node.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
