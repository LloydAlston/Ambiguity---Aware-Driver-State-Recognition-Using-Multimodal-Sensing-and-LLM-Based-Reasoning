#!/usr/bin/env python3

"""
ROS2 Node: FanController
Subscribes to /fan_speed (Int32) to control a fan via serial.
Fan levels: 0 (off), 1 (low), 2 (medium), 3 (high)
Fan runs for 20 seconds when a command is received, then turns off automatically.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial
import os


class FanController(Node):
    def __init__(self):
        super().__init__("fan_controller")
        self.subscription = self.create_subscription(
            Int32, "/fan_speed", self.fan_speed_callback, 10
        )

        # --- Configuration ---
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        SERIAL_PORT = self.get_parameter("serial_port").value
        BAUD_RATE = 115200

        self.declare_parameter("fan_l1_pct", 30.0)
        self.declare_parameter("fan_l2_pct", 60.0)

        # Fan levels (percentages)
        self.FAN_LEVELS = {
            0: 0.0,
            1: self.get_parameter("fan_l1_pct").value,
            2: self.get_parameter("fan_l2_pct").value,
            3: 100.0,
        }

        # --- Initialize Serial ---
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            self.get_logger().info(f"Connected to fan controller on {SERIAL_PORT}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to serial device: {e}")
            exit(1)

        # Timer for auto turn-off
        self.off_timer = None

    def fan_speed_callback(self, msg: Int32):
        """Callback function to handle incoming fan speed messages.

        Args:
            msg (Int32): The fan speed message.
        """
        level = msg.data
        self.get_logger().info(f"Received fan speed: {level}")
        self.set_fan(level)

        # Cancel any existing turn-off timer
        if self.off_timer:
            self.off_timer.cancel()

        # Set a timer to turn off the fan after 20 seconds
        self.off_timer = self.create_timer(20.0, self.turn_off_fan)

    def set_fan(self, level: int):
        """Set the fan speed.

        Args:
            level (int): The desired fan speed level (0-3).
        """
        if level not in self.FAN_LEVELS:
            self.get_logger().warn(f"Invalid fan level: {level}")
            return

        duty = self.FAN_LEVELS[level]
        command = f"set_duty {duty}\n"
        try:
            self.ser.write(command.encode())
            self.ser.flush()
            self.get_logger().info(f"Fan set to level {level} ({duty}%)")
        except Exception as e:
            self.get_logger().error(f"Failed to write to serial: {e}")

    def turn_off_fan(self):
        """Turn off the fan after finishing the desired duration.
        """
        self.set_fan(0)
        self.get_logger().info("Fan turned off automatically after 20 seconds")
        if self.off_timer:
            self.off_timer.cancel()
            self.off_timer = None


def main(args=None):
    rclpy.init(args=args)
    node = FanController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down FanController...")
    finally:
        if node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
