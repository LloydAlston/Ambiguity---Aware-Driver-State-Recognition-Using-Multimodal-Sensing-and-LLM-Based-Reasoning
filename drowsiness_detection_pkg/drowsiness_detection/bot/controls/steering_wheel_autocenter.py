#!/usr/bin/env python3

"""
This ROS2 Node controls the steering wheel force feedback (autocenter) of a Logitech G29
using evdev. It subscribes to the `/wheel_ffb` topic to receive strength value (0-65535)
and also sends a one-time initial FFB value on startup.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import evdev
from evdev import ecodes


class WheelFFBNode(Node):
    def __init__(self):
        super().__init__("wheel_ffb_node")

        # Initialize the device
        self.raw_device = None
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for device in devices:
            if device.name == "Logitech G29 Driving Force Racing Wheel":
                self.raw_device = device
                break

        if self.raw_device is None:
            self.get_logger().error("Logitech G29 not found.")
            raise RuntimeError("Wheel not found")

        self.get_logger().info("Logitech G29 initialized successfully.")

        # ROS Subscriber
        self.subscription = self.create_subscription(
            Int32, "/wheel_ffb", self.ffb_callback, 10
        )
        self.get_logger().info("Subscribed to /wheel_ffb topic.")

        # -------------------- Publish initial FFB once --------------------
        self.ffb_pub = self.create_publisher(Int32, "/wheel_ffb", 10)
        # Create a timer that runs once, then cancels itself
        self._initial_ffb_timer = self.create_timer(0.01, self.publish_initial_ffb)

    def publish_initial_ffb(self):
        """Publish a single FFB value of 30000 and cancel the timer."""
        msg = Int32()
        msg.data = 30000  # initial autocenter strength
        self.ffb_pub.publish(msg)
        self.get_logger().info(f"Initial /wheel_ffb value published: {msg.data}")

        # Cancel the timer so it does not run again
        self._initial_ffb_timer.cancel()

    def ffb_callback(self, msg: Int32):
        """Receive FFB strength (0-65535) and send autocenter force feedback."""
        strength = max(0, min(msg.data, 65535))  # Clamp
        try:
            self.raw_device.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, strength)
            self.raw_device.syn()  # Sync event
            self.get_logger().info(
                f"Autocenter force feedback sent with strength: {strength}"
            )
        except Exception as e:
            self.get_logger().error(f"Error sending force feedback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WheelFFBNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
