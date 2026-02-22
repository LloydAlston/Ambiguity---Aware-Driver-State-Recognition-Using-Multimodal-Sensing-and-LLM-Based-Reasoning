#!/usr/bin/env python3

"""
This version implements a web application for real-time drowsiness detection labelling.
It uses Flask for the web interface and ROS2 for data communication.
The application allows multiple annotators to label drowsiness levels and actions,
with support for auto-submission of previous labels if no new input is provided.
It also includes tracking for the 'Save Video' action.
"""
from waitress import serve


import os
import time
import threading
from collections import deque
from copy import deepcopy
import json

from flask import Flask, render_template, Response, request, jsonify
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from drowsiness_detection_msg.msg import (
    EarMarValue,
    LanePosition,
    DrowsinessMetricsData,
    Vibration,
    CombinedAnnotations,
    AnnotatorLabels,
)

from drowsiness_detection_msg.srv import StoreLabels

from std_msgs.msg import Float32MultiArray, String, Int32
from carla_msgs.msg import CarlaEgoVehicleControl
from cv_bridge import CvBridge

# --- Data Manager ---
class DataManager:
    """Thread-safe storage for all ROS and Flask shared data for multiple annotators."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_image = None
        self.live_ear = deque(maxlen=360)
        self.live_mar = deque(maxlen=360)
        self.live_steering = deque(maxlen=360)
        self.latest_metrics = {}
        self.latest_phase_info = {}
        self.selected_labels_buffer = {}  # annotator -> labels
        self.last_submitted_labels = {}  # annotator -> labels

    def set_image(self, img):
        with self.lock:
            self.latest_image = img

    def set_ear_mar(self, ear, mar):
        with self.lock:
            self.live_ear.append(ear)
            self.live_mar.append(mar)

    def set_steering(self, steer):
        with self.lock:
            self.live_steering.append(steer)

    def set_metrics(self, metrics):
        with self.lock:
            self.latest_metrics.update(metrics)

    def set_phase_info(self, phase_info):
        with self.lock:
            self.latest_phase_info.update(phase_info)

    def get_all_live_data(self):
        with self.lock:
            return {
                "live_ear": list(self.live_ear),
                "live_mar": list(self.live_mar),
                "live_steering": list(self.live_steering),
                "latest_metrics": deepcopy(self.latest_metrics),
                "phase_info": deepcopy(self.latest_phase_info),
                "last_submitted_labels": deepcopy(self.last_submitted_labels),
                "selected_labels_buffer": deepcopy(self.selected_labels_buffer),
            }

    def get_image(self):
        with self.lock:
            return deepcopy(self.latest_image)


# --- ROS Bridge ---
class RosBridge(Node):
    # Voice commands & feedback constants
    VOICE_COMMANDS = "/home/user/voice_files/Trigger_1.mp3"

    VOICE_FEEDBACK = {
        "Yes": "/home/user/voice_files/joke.mp3",
        "No": "/home/user/voice_files/Feedback.mp3",
    }

    def __init__(self, data_manager):
        super().__init__("drowsiness_web_node")
        self.data_manager = data_manager
        self.bridge = CvBridge()
        self.get_logger().info("ROS Bridge Node Initialized.")

        # --- Subscriptions ---
        self.create_subscription(
            Image, "/camera/image_raw", self.cb_camera, qos_profile_sensor_data
        )
        self.create_subscription(
            EarMarValue, "/ear_mar", self.cb_earmar, qos_profile_sensor_data
        )
        self.create_subscription(
            CarlaEgoVehicleControl,
            "/carla/hero/vehicle_control_cmd",
            self.cb_steering,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Float32MultiArray,
            "/driver_assistance/window_phase",
            self.cb_window_phase,
            10,
        )
        
    
        self.store_labels_client = self.create_client(StoreLabels, "store_labels")


        # --- New subscriptions/publishers ---
        self.vibration_pub = self.create_publisher(Vibration, "/wheel_vibration", 10)

        self.fan_pub = self.create_publisher(Int32, "/fan_speed", 10)
        self.mp4_pub = self.create_publisher(String, "/audio_file", 10)
        
        self.last_published_window_id = -1


        self.get_logger().info("ROS Bridge Node ready.")

    # --- Steering vibration --

    def vibrate(self, duration: float, intensity: int):
        """Publish vibration activation message."""
        out = Vibration()
        out.duration = float(duration)
        out.intensity = int(intensity)
        self.vibration_pub.publish(out)

    # --- Voice commands ---
    def publish_mp4(self, file_path: str):
        if not file_path or not os.path.isfile(file_path):
            self.get_logger().warn(f"Invalid MP4 file path: {file_path}")
            return
        msg = String()
        msg.data = file_path
        self.mp4_pub.publish(msg)
        self.get_logger().info(f"Published MP4 file path: {file_path}")

    # --- Fan control ---
    def set_fan_speed(self, speed_level: int):
        """Set fan speed by publishing to the fan_speed topic."""

        if speed_level not in [0, 1, 2, 3]:
            self.get_logger().warn(f"Invalid fan speed level: {speed_level}")
            return
        self.get_logger().info(f"Setting fan speed to level {speed_level}")
        ros_msg = Int32()
        ros_msg.data = speed_level
        self.fan_pub.publish(ros_msg)

    # --- ROS Data callbacks ---
    def cb_camera(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.data_manager.set_image(cv_img.copy())
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def cb_earmar(self, msg):
        try:
            self.data_manager.set_ear_mar(msg.ear_value, msg.mar_value)
        except Exception as e:
            self.get_logger().error(f"Error in ear_mar_callback: {e}")

    def cb_steering(self, msg):
        self.data_manager.set_steering(msg.steer)

    def cb_window_phase(self, msg):
        if len(msg.data) < 3:
            return

        phase = int(msg.data[0])
        window_id = int(msg.data[1])
        remaining_time = msg.data[2]

        # Update internal state for the web UI
        self.data_manager.set_phase_info({
            "phase": phase,
            "window_id": window_id,
            "remaining_time": remaining_time,
        })

        # --- Trigger CombinedAnnotations publish when window just ended ---
        if remaining_time <= 0.0 and window_id != self.last_published_window_id:
            self.last_published_window_id = window_id
            self.publish_combined_annotations(window_id)

    def publish_combined_annotations(self, window_id: int):
        """Combine annotator labels and send them to DriverAssistanceNode via StoreLabels service."""
        try:
            # Ensure service exists
            if not hasattr(self, "store_labels_client"):
                self.store_labels_client = self.create_client(StoreLabels, "store_labels")

            if not self.store_labels_client.wait_for_service(timeout_sec=3.0):
                self.get_logger().error("Service 'store_labels' not available.")
                return

            req = StoreLabels.Request()
            req.window_id = window_id
            req.annotator_labels = []

            with self.data_manager.lock:
                # union of annotators that ever submitted or currently have buffered data
                annotators = set(self.data_manager.selected_labels_buffer.keys()) | set(
                    self.data_manager.last_submitted_labels.keys()
                )
                if not annotators:
                    annotators = {"default_annotator"}

                for annotator in annotators:
                    # --- Determine which label set to use ---
                    if annotator in self.data_manager.selected_labels_buffer:
                        labels_to_use = deepcopy(self.data_manager.selected_labels_buffer[annotator])

                        # check if empty (no actual inputs)
                        is_empty_label = (
                            (not labels_to_use.get("drowsiness_level"))
                            or labels_to_use["drowsiness_level"] in [None, "", "None"]
                        ) and not labels_to_use.get("actions") and not labels_to_use.get("notes")

                        if is_empty_label:
                            # No new input → reuse last label if available
                            if annotator in self.data_manager.last_submitted_labels:
                                labels_to_use = deepcopy(self.data_manager.last_submitted_labels[annotator])
                                labels_to_use["auto_submitted"] = True
                                submission_type = "auto"
                            else:
                                # Brand new annotator with no previous label
                                labels_to_use = {
                                    "drowsiness_level": "None",
                                    "actions": [],
                                    "notes": "Auto-submitted: annotator inactive.",
                                    "auto_submitted": True,
                                    "voice_feedback": None,
                                    "video_save_requested": False,
                                }
                                submission_type = "auto"
                        else:
                            # user provided manual labels
                            submission_type = "manual"
                            labels_to_use["auto_submitted"] = False
                            self.data_manager.last_submitted_labels[annotator] = deepcopy(labels_to_use)

                        # remove buffer entry after using it
                        del self.data_manager.selected_labels_buffer[annotator]
                    else:
                        # no new buffer → reuse last known label
                        if annotator in self.data_manager.last_submitted_labels:
                            labels_to_use = deepcopy(self.data_manager.last_submitted_labels[annotator])
                            labels_to_use["auto_submitted"] = True
                            submission_type = "auto"
                        else:
                            # fallback to safe default
                            labels_to_use = {
                                "drowsiness_level": "None",
                                "actions": [],
                                "notes": "Auto-submitted: annotator inactive.",
                                "auto_submitted": True,
                                "voice_feedback": None,
                                "video_save_requested": False,
                            }
                            submission_type = "auto"

                    # ensure field exists before next step
                    labels_to_use["submission_type"] = submission_type

                    # convert to ROS message format
                    labels_prepared = prepare_labels_for_saving(labels_to_use)
                    ann_msg = AnnotatorLabels()
                    ann_msg.annotator_name = annotator
                    ann_msg.drowsiness_level = str(labels_prepared.get("drowsiness_level", "None"))
                    ann_msg.actions = labels_to_use.get("actions", [])
                    ann_msg.notes = labels_prepared.get("notes", "")
                    ann_msg.voice_feedback = str(labels_prepared.get("voice_feedback", ""))
                    ann_msg.submission_type = submission_type
                    # ann_msg.auto_submitted = bool(labels_prepared.get("auto_submitted", False))
                    # ann_msg.is_flagged = bool(labels_prepared.get("is_flagged", False))
                    ann_msg.action_fan = bool(labels_prepared.get("fan", False))
                    ann_msg.action_voice_command = bool(labels_prepared.get("voice_command", False))
                    ann_msg.action_steering_vibration = bool(labels_prepared.get("steering_vibration", False))
                    ann_msg.action_save_video = bool(labels_prepared.get("action_save_video", False))
                    req.annotator_labels.append(ann_msg)

            # async call
            future = self.store_labels_client.call_async(req)

            def _on_result(fut):
                try:
                    res = fut.result()
                    if res.success:
                        self.get_logger().info(
                            f"[SERVICE] Labels successfully stored for window {window_id} ({len(req.annotator_labels)} annotators)"
                        )
                    else:
                        self.get_logger().warn(
                            f"[SERVICE] Label store failed for window {window_id}: {res.message}"
                        )
                except Exception as e:
                    self.get_logger().error(f"Service call failed: {e}")

            future.add_done_callback(_on_result)

        except Exception as e:
            self.get_logger().error(f"Error preparing StoreLabels service request: {e}")



# MODIFIED: Added action_save_video conversion and removal of original key
def prepare_labels_for_saving(labels):
    prepared = deepcopy(labels)

    # One-hot encode actions (already separate columns)
    prepared["voice_command"] = 1 if "Voice Command" in labels.get("actions", []) else 0
    prepared["steering_vibration"] = (
        1 if "Steering Vibration" in labels.get("actions", []) else 0
    )
    prepared["fan"] = 1 if "Fan" in labels.get("actions", []) else 0

    # Voice feedback: force int or default 0
    vf = labels.get("voice_feedback")
    if vf == "Yes":
        prepared["voice_feedback"] = 1
    elif vf == "No":
        prepared["voice_feedback"] = 0
    else:
        prepared["voice_feedback"] = None  # safe default

    # NEW: Convert the video_save_requested boolean flag to action_save_video integer flag
    prepared["action_save_video"] = int(prepared.get("video_save_requested", False)) 

    # Auto-submitted: convert bool → int
    prepared["auto_submitted"] = int(prepared.get("auto_submitted", 0))

    # Remove raw 'actions' list and temporary keys to avoid h5py object dtype error
    for key in ["actions", "timestamp", "video_save_requested"]: 
        if key in prepared:
            del prepared[key]
    return prepared


# --- Flask ---
data_manager = DataManager()
app = Flask(__name__)

# Global reference to the RosBridge node for use in Flask routes
ros_bridge_node = None


def ros_spin():
    global ros_bridge_node
    rclpy.init()
    ros_bridge_node = RosBridge(data_manager)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_bridge_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if ros_bridge_node:
            ros_bridge_node.destroy_node()
        rclpy.shutdown()


threading.Thread(target=ros_spin, daemon=True).start()


@app.route("/")
def index():
    return render_template("index_v2.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = data_manager.get_image()
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_live_data")
def get_live_data():
    return jsonify(data_manager.get_all_live_data())


# --- NEW ROUTE: For real-time action activation ---
@app.route("/activate_action", methods=["POST"])
def activate_action():
    req = request.get_json() or {}
    action = req.get("action")
    feedback = req.get("feedback")

    if ros_bridge_node is None:
        return jsonify({"status": "error", "message": "ROS node not ready."}), 500

    def handle_action(action, feedback):
        """Run the requested action in a separate thread."""
        if action == "Steering Vibration":
            ros_bridge_node.get_logger().info(
                "Steering vibration action triggered via web interface."
            )
            ros_bridge_node.vibrate(duration=2.0, intensity=30)

        elif action == "Voice Command":
            ros_bridge_node.get_logger().info(
                "Voice command action triggered via web interface (Main command)."
            )
            # Play main voice command only
            voice_text = RosBridge.VOICE_COMMANDS
            ros_bridge_node.publish_mp4(voice_text)

        elif action == "Voice Feedback":
            if feedback in RosBridge.VOICE_FEEDBACK:
                ros_bridge_node.get_logger().info(
                    f"Voice command feedback triggered via web interface: {feedback}"
                )
                feedback_file = RosBridge.VOICE_FEEDBACK[feedback]
                ros_bridge_node.publish_mp4(feedback_file)
            else:
                ros_bridge_node.get_logger().warn(
                    f"Voice Feedback action triggered with invalid feedback: {feedback}"
                )

        elif action == "Fan":
            ros_bridge_node.get_logger().info("Fan action triggered via web interface.")
            ros_bridge_node.set_fan_speed(3)

        else:
            ros_bridge_node.get_logger().warn(f"Unknown action: {action}")

    # Run the action in a separate thread so Flask doesn't block
    threading.Thread(target=handle_action, args=(action, feedback), daemon=True).start()

    return jsonify({"status": "success", "message": f"Action '{action}' triggered."})


# MODIFIED: Added video_save_requested extraction and storage
@app.route("/store_selected_labels", methods=["POST"])
def store_selected_labels():
    req = request.get_json() or {}
    window_id = req.get("window_id")
    annotator = req.get("annotator_name")
    voice_feedback = req.get("voice_feedback", "")
    
    # Extract the video save status from the request
    video_save_requested = req.get("video_save_requested", False)

    if window_id is None or annotator is None:
        return (
            jsonify(
                {"status": "error", "message": "window_id and annotator_name required"}
            ),
            400,
        )

    drowsiness_level = req.get("drowsiness_level")
    notes = req.get("notes", "")

    # The actions sent from the frontend are now just for storage
    actions_to_store = req.get("actions", [])

    buffered = {
        "drowsiness_level": drowsiness_level,
        "actions": actions_to_store,
        "notes": notes,
        "voice_feedback": voice_feedback,
        "timestamp": time.time(),
        "auto_submitted": False,
        # Store the status in the buffer
        "video_save_requested": video_save_requested,
    }

    with data_manager.lock:
        data_manager.selected_labels_buffer[annotator] = buffered

    return jsonify({"status": "success", "annotator_used": annotator})


@app.route("/submit_labels", methods=["POST"])
def submit_labels():
    # The submit button now only stores the labels, it no longer triggers actions.
    # The actions are triggered by the individual button presses.
    return store_selected_labels()


@app.route("/save_video_segment", methods=["POST"])
def save_video_segment():
    """
    Placeholder route for the Save Video button.
    The actual flagging is handled in store_selected_labels,
    this route is primarily for frontend feedback.
    """
    req = request.get_json() or {}
    window_id = req.get("window_id")
    annotator = req.get("annotator_name")

    # The main logic for storing the video flag is in store_selected_labels.
    # This route just acknowledges the action was requested.
    
    if ros_bridge_node is not None:
         ros_bridge_node.get_logger().info(f"Video save requested by {annotator} for window {window_id}.")

    return jsonify({"status": "success", "message": "Video save request recorded."})



def main():
    try:
        serve(app, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("Shutting down web interface.")


if __name__ == "__main__":
    main() 
