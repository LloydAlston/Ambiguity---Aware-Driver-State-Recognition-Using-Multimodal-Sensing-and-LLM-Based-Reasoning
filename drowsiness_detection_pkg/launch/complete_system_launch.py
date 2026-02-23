from launch import LaunchDescription
import os
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Advanced Launch File for Drowsiness Detection System
    """

    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================

    driver_id_arg = DeclareLaunchArgument(
        "driver_id",
        default_value="maria",
        description="Driver identifier for saving session data.",
    )

    driver_id = LaunchConfiguration("driver_id")

    # ========================================================================
    # FEATURE EXTRACTION NODES
    # ========================================================================

    spinnaker_camera_node = Node(
        package="spinnaker_camera_cpp",
        executable="camera_node",
        name="spinnaker_camera_node",
        output="screen",
        parameters=[{"camera_index": 0}],
    )

    mediapipe_node = Node(
        package="drowsiness_detection_pkg",
        executable="camera_mediapipe_node",
        name="camera_mediapipe_node",
        output="screen",
        parameters=[{"driver_id": driver_id}],
    )

    carla_node = Node(
        package="drowsiness_detection_pkg",
        executable="carla_manual_control",
        name="carla_manual_control",
        output="screen",
        parameters=[{"driver_id": driver_id}],
    )

    hybrid_vlm_node = Node(
        package="drowsiness_detection_pkg",
        executable="hybrid_vlm_node",
        name="hybrid_vlm_node",
        output="screen",
        parameters=[{"driver_id": driver_id}],
    )

    # ========================================================================
    # ML CLASSIFICATION NODES
    # ========================================================================

    camera_ml_node = Node(
        package="drowsiness_detection_pkg",
        executable="camera_ml_node",
        name="camera_ml_node",
        output="screen",
        parameters=[{"driver_id": driver_id}],
    )

    carla_ml_node = Node(
        package="drowsiness_detection_pkg",
        executable="carla_ml_node",
        name="carla_ml_node",
        output="screen",
        parameters=[{"driver_id": driver_id}],
    )

    # ========================================================================
    # CORE DECISION & ROUTING NODES
    # ========================================================================

    integrated_llm_node = Node(
        package="drowsiness_detection_pkg",
        executable="integrated_llm_node",
        name="integrated_llm_node",
        output="screen",
        parameters=[
            {"driver_id": driver_id},
            {"ollama_endpoint": "http://localhost:11434"},
            {"ollama_model": "llama3.1:8b"},
            {"temperature": 0.3},
            {"timeout_seconds": 90},
            {"decision_interval_seconds": 60},
        ],
    )

    drowsiness_alert_dispatcher = Node(
        package="drowsiness_detection_pkg",
        executable="drowsiness_alert_dispatcher",
        name="drowsiness_alert_dispatcher",
        output="screen",
        parameters=[
            {"driver_id": driver_id},
            {"audio_dir": "/home/karthik/DROWSINESS_DETECTION/audio_alerts"},
        ],
    )

    # ========================================================================
    # ACTUATOR CONTROL NODES
    # ========================================================================

    speaker_node = Node(
        package="drowsiness_detection_pkg",
        executable="speaker_node",
        name="speaker_node",
        output="screen",
    )

    fan_node = Node(
        package="drowsiness_detection_pkg",
        executable="fan_node",
        name="fan_node",
        output="screen",
    )

    steering_node = Node(
        package="drowsiness_detection_pkg",
        executable="steering_node",
        name="steering_node",
        output="screen",
    )

    # ========================================================================
    # EVALUATION NODES
    # ========================================================================
    
    data_logger_node = Node(
        package="drowsiness_detection_pkg",
        executable="data_logger_node",
        name="data_logger_node",
        output="screen",
        parameters=[{"log_dir": "./ml_logs"}],
    )

    carla_steering_metrics_node = Node(
        package="drowsiness_detection_pkg",
        executable="carla_steering_metrics_node",
        name="carla_steering_metrics_node",
        output="screen",
    )

    # ========================================================================
    # DELAYED STARTUP
    # ========================================================================

    delayed_nodes = TimerAction(
        period=5.0,
        actions=[
            mediapipe_node,
            spinnaker_camera_node,
            carla_node,
            carla_steering_metrics_node,
            hybrid_vlm_node,
            camera_ml_node,
            carla_ml_node,
            integrated_llm_node,
            drowsiness_alert_dispatcher,
            speaker_node,
            fan_node,
            steering_node,
            data_logger_node,
        ],
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================

    return LaunchDescription(
        [
            driver_id_arg,
            # Removed setup_network (no sudo, no enp130s0)
            delayed_nodes,
        ]
    )
