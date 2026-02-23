#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import carla
from carla_msgs.msg import CarlaEgoVehicleInfo, CarlaEgoVehicleInfoWheel
from geometry_msgs.msg import Vector3
import random

class SpawnAndGetVehicleInfo(Node):
    def __init__(self):
        super().__init__('get_vehicle_info')

        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # Publisher (optional)
        self.pub = self.create_publisher(CarlaEgoVehicleInfo, '/carla/hero/vehicle_info', 10)

        # Spawn and get info once
        self.timer = self.create_timer(1.0, self.spawn_and_print)
        self.done = False

    def spawn_and_print(self):
        if self.done:
            return

        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.find('vehicle.mercedes.coupe_2020')
        bp.set_attribute('role_name', 'hero')

        # Choose random color if available
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
            self.get_logger().info(f'Color: {color}')

        # Choose random spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        # Spawn vehicle
        vehicle = self.world.try_spawn_actor(bp, spawn_point)
        if not vehicle:
            self.get_logger().warn('Spawn failed (spot occupied). Try again.')
            return

        self.get_logger().info(f'Spawned {vehicle.type_id} (id={vehicle.id})')

        # Get physics info
        physics = vehicle.get_physics_control()

        # Print summary
        for i, wheel in enumerate(physics.wheels):
            self.get_logger().info(
                f"Wheel {i}: steer={wheel.max_steer_angle:.2f}°, "
                f"radius={wheel.radius:.2f}cm, friction={wheel.tire_friction:.2f}"
            )

        # Optionally publish the info message
        msg = CarlaEgoVehicleInfo()
        msg.id = vehicle.id
        msg.type = vehicle.type_id
        msg.rolename = 'hero'

        for w in physics.wheels:
            wheel_msg = CarlaEgoVehicleInfoWheel()
            wheel_msg.tire_friction = w.tire_friction
            wheel_msg.damping_rate = w.damping_rate
            wheel_msg.max_steer_angle = w.max_steer_angle
            wheel_msg.radius = w.radius
            wheel_msg.max_brake_torque = w.max_brake_torque
            wheel_msg.max_handbrake_torque = w.max_handbrake_torque
            wheel_msg.position = Vector3(
                x=w.position.x, y=w.position.y, z=w.position.z)
            msg.wheels.append(wheel_msg)

        self.pub.publish(msg)
        self.get_logger().info('Published /carla/hero/vehicle_info message.')

        self.done = True

def main(args=None):
    rclpy.init(args=args)
    node = SpawnAndGetVehicleInfo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
