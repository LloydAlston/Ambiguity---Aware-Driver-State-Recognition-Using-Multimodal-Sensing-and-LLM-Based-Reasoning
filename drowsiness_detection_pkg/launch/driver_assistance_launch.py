from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            # Feature Extraction Nodes
            Node(
                package="spinnaker_camera_cpp",
                executable="camera_node",
                name="spinnaker_camera_node",
                output="screen",
                parameters=[{"camera_index": 0}],
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="camera_mediapipe_node",
                name="camera_mediapipe_node",
                output="screen",
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="carla_manual_control",
                name="carla_manual_control",
                output="screen",
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="hybrid_vlm_node",
                name="hybrid_vlm_node",
                output="screen",
            ),
            # ML Classification Nodes
            Node(
                package="drowsiness_detection_pkg",
                executable="camera_ml_node",
                name="camera_ml_node",
                output="screen",
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="carla_ml_node",
                name="carla_ml_node",
                output="screen",
            ),
            # Decision & Reasoning Nodes
            Node(
                package="drowsiness_detection_pkg",
                executable="integrated_llm_node",
                name="integrated_llm_node",
                output="screen",
                parameters=[
                    {"driver_id": "maria"},
                    {"ollama_endpoint": "http://localhost:11434"},
                ],
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="drowsiness_alert_dispatcher",
                name="drowsiness_alert_dispatcher",
                output="screen",
                parameters=[
                    {"driver_id": "maria"},
                ],
            ),
            # Actuator Nodes
            Node(
                package="drowsiness_detection_pkg",
                executable="speaker_node",
                name="speaker_node",
                output="screen",
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="fan_node",
                name="fan_node",
                output="screen",
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="steering_node",
                name="steering_node",
                output="screen",
            ),
        ]
    )

