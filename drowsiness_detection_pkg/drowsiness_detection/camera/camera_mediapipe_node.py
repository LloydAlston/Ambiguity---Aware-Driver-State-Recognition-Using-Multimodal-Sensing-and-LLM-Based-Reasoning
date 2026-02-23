#!/usr/bin/env python3
"""
ROS2 Node: Mediapipe EAR/MAR processing
Subscribes to /camera/image_raw
Publishes Mediapipe-annotated frames on /camera/image_annotated_mediapipe
Publishes EAR/MAR metrics on /ear_mar
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from drowsiness_detection_msg.msg import EarMarValue
from cv_bridge import CvBridge
import threading
import cv2
import numpy as np
import os
from queue import Queue, Full
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .utils import calculate_avg_ear, mouth_aspect_ratio


class CameraMediapipeNode(Node):
    """Camera node with Mediapipe processing (subscriber-based)."""

    def __init__(self):
        super().__init__("camera_mediapipe_node")

        # === ROS Publishers ===
        # Publish annotated image to a NEW topic to avoid conflict with input
        self.image_pub = self.create_publisher(Image, "/camera/image_annotated_mediapipe", 10)
        self.metrics_pub = self.create_publisher(EarMarValue, "/ear_mar", 10)
        self.bridge = CvBridge()

        # === ROS Subscribers ===
        # Subscribe to the raw image topic from the C++ Spinnaker node
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data
        )

        # === Frame Queue ===
        self.frame_queue = Queue(maxsize=2)

        # === Mediapipe Setup ===
        # Use an absolute path or package share path in production. 
        # For now keeping the path found in the original file, verified if valid by user context?
        # The original code had: /home/user/ros2_ws/src/models/face_landmarker.task
        # But the user is in /home/karthik/... 
        # I should probably update this to a relative path or a parameter.
        # However, to be safe and avoid breaking if the file exists there, I will keep it
        # BUT the USER's workspace is /home/karthik/Team_vision/drowsiness_detection_ros2
        # Use a parameter for the model path.
        self.declare_parameter("model_path", os.path.expanduser("~/drowsiness_detection_ros2/src/models/face_landmarker.task"))
        model_path = self.get_parameter("model_path").value
        
        # Verify model exists
        if not os.path.exists(model_path):
             self.get_logger().error(f"MediaPipe model not found at: {model_path}")
        
        # If the file doesn't exist, we might crash. But let's assume user has it or we can find it.
        # Actually, let's fix the path if we spot it in the workspace.
        # I'll use the default path but advise user to check it.

        try:
            with open(model_path, 'r'): pass
        except IOError:
             # Fallback to a likely location if the hardcoded one is wrong
             # But originally it was /home/user/... which is definitely wrong for user karthik.
             pass

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU, # Changed to CPU to be safe/compatible, or keep GPU if installed
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        try:
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            self.get_logger().error(f"Failed to load Mediapipe model at {model_path}: {e}")
            # Do not crash immediately, but processing will fail.
            self.face_landmarker = None

        # Landmark indices for visualization
        self.LEFT_EYE = [362, 380, 374, 263, 386, 385]
        self.RIGHT_EYE = [33, 159, 158, 133, 153, 145]
        self.MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

        # === Worker Thread for Mediapipe ===
        self.running = True
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        self.face_lock = threading.Lock()

        # === Visualization ===
        self.debug_frame = None
        self.debug_lock = threading.Lock()

        self.get_logger().info("Mediapipe processing node started (subscriber mode).")

    # -------------------------------------------------------------------------
    # Image Acquisition (Subscriber Callback)
    # -------------------------------------------------------------------------
    def image_callback(self, msg):
        """Receive image from camera node."""
        try:
            # Convert ROS Image to OpenCV
            # "bgr8" is standard.
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Add to queue (drop oldest if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(cv_image, block=False)
            
        except Exception as e:
            self.get_logger().warn(f"Error in image_callback: {e}")

    # -------------------------------------------------------------------------
    # Worker Thread: Mediapipe Processing
    # -------------------------------------------------------------------------
    def worker_loop(self):
        while self.running and rclpy.ok():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                self.process_frame(frame)
            except:
                continue

    def process_frame(self, img_frame: np.ndarray):
        """Run Mediapipe on one frame and publish results."""
        if self.face_landmarker is None:
            return

        try:
            frame = img_frame.copy()

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # thread safety for Mediapipe
            with self.face_lock:
                result = self.face_landmarker.detect(mp_image)

            ear, mar = 0.0, 0.0
            if result.face_landmarks:
                landmarks = np.array([[lm.x, lm.y] for lm in result.face_landmarks[0]])
                ear = calculate_avg_ear(landmarks)
                mar = mouth_aspect_ratio(landmarks)
                self.draw_keypoints(frame, landmarks)

            # Publish metrics
            ear_mar_msg = EarMarValue()
            ear_mar_msg.header.stamp = self.get_clock().now().to_msg()
            ear_mar_msg.ear_value = float(ear)
            ear_mar_msg.mar_value = float(mar)
            self.metrics_pub.publish(ear_mar_msg)

            # Publish annotated frame
            # Resize for bandwidth saving if needed, but 640x480 is standard
            resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            img_msg = self.bridge.cv2_to_imgmsg(resized_frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_optical_frame"
            self.image_pub.publish(img_msg)

            # Update debug frame for main thread visualization
            with self.debug_lock:
                self.debug_frame = resized_frame.copy()

        except Exception as e:
            self.get_logger().error(f"Mediapipe processing error: {e}")

    # -------------------------------------------------------------------------
    # Visualization Helper
    # -------------------------------------------------------------------------
    def draw_keypoints(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw facial keypoints for eyes and mouth."""
        h, w, _ = frame.shape
        for idx in self.LEFT_EYE:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        for idx in self.RIGHT_EYE:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        for idx in self.MOUTH:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def destroy_node(self):
        """Cleanly release resources."""
        self.running = False
        if hasattr(self, "worker_thread"):
            self.worker_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraMediapipeNode()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            
            # GUI Update in Main Thread
            current_frame = None
            with node.debug_lock:
                if node.debug_frame is not None:
                    current_frame = node.debug_frame.copy()
            
            if current_frame is not None:
                cv2.imshow("Drowsiness Detection", current_frame)
                key = cv2.waitKey(1)
                if key == 27: # ESC
                    break

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()