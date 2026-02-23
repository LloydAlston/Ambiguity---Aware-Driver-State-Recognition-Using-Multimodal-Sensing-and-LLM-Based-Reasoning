from setuptools import find_packages, setup
import os
from glob import glob


package_name = "drowsiness_detection_pkg"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    package_data={
        "drowsiness_detection_pkg.carla": ["wheel_config.ini"],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (
            "share/drowsiness_detection/bot/configs",
            [
                "drowsiness_detection/bot/configs/prompt.toml",
                "drowsiness_detection/bot/configs/schema.json",
            ],
        ),
        (
            os.path.join("share", package_name, "templates"),
            glob("drowsiness_detection/labelling/templates/*.html"),
        ),
        (
            os.path.join("share", package_name, "static"),
            glob("drowsiness_detection/labelling/static/**/*", recursive=True),
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="Driver assistance package for drowsiness detection",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            # FEATURE EXTRACTION
            # "pyspin_camera_node = drowsiness_detection.camera.camera_pyspin_node:main",
            "camera_mediapipe_node = drowsiness_detection.camera.camera_mediapipe_node:main",
            "carla_manual_control = drowsiness_detection.carla.carla_manual_control:main",
            "hybrid_vlm_node = drowsiness_detection.VLM.hybrid_vlm_node:main",

            # ML CLASSIFICATION
            "camera_ml_node = drowsiness_detection.ML_Models.camera_ml_node:main",
            "carla_ml_node = drowsiness_detection.ML_Models.carla_ml_node:main",

            # CORE DECISION & DISPATCH
            "integrated_llm_node = drowsiness_detection.LLM_Reasoning.integrated_llm_node:main",
            "drowsiness_alert_dispatcher = drowsiness_detection.bot.controls.drowsiness_alert_dispatcher:main",

            # ACTUATORS
            "speaker_node = drowsiness_detection.bot.controls.speaker_node:main",
            "fan_node = drowsiness_detection.bot.controls.fan_node:main",
            "steering_node = drowsiness_detection.bot.controls.steering_node:main",

            # OPTIONAL / LEGACY
            "driver_assistance_node = drowsiness_detection.main:main",
            "drowsiness_gui = drowsiness_detection.labelling.app_v2:main",
            "steering_wheel_ffb_node = drowsiness_detection.bot.controls.steering_wheel_autocenter:main",
            "vehicle_info_node = drowsiness_detection.carla.vehicle_info:main",
            "ml_aggregator_node = drowsiness_detection.ml_nodes.ml_aggregator_node:main",
            "data_logger_node = drowsiness_detection.Evaluation.data_logger_node:main",
            "carla_steering_metrics_node = drowsiness_detection.carla_steering_metrics_node:main",
        ],
    },
)
