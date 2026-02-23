import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- EAR and MAR functions provided ---
def calculate_ear(landmarks: np.ndarray, eye: str) -> float:
    landmarks_indices = {
        "right_eye": [33, 159, 158, 133, 153, 145],
        "left_eye": [362, 380, 374, 263, 386, 385],
    }
    indices = landmarks_indices[eye]
    a = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
    b = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
    c = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
    return (a + b) / (2.0 * c)

def calculate_avg_ear(landmarks: np.ndarray) -> float:
    left_ear = calculate_ear(landmarks, "left_eye")
    right_ear = calculate_ear(landmarks, "right_eye")
    return (left_ear + right_ear) / 2.0

def mouth_aspect_ratio(landmarks: np.array) -> float:
    mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]
    a = np.linalg.norm(landmarks[mouth_indices[1]] - landmarks[mouth_indices[7]])
    b = np.linalg.norm(landmarks[mouth_indices[2]] - landmarks[mouth_indices[6]])
    c = np.linalg.norm(landmarks[mouth_indices[3]] - landmarks[mouth_indices[5]])
    d = np.linalg.norm(landmarks[mouth_indices[0]] - landmarks[mouth_indices[4]])
    return (a + b + c) / (2.0 * d)


# --- Landmark indices for visualization ---
LEFT_EYE = [362, 380, 374, 263, 386, 385]
RIGHT_EYE = [33, 159, 158, 133, 153, 145]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

def draw_keypoints(frame, landmarks):
    h, w, _ = frame.shape
    # Draw left eye in blue
    for idx in LEFT_EYE:
        x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    # Draw right eye in green
    for idx in RIGHT_EYE:
        x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    # Draw mouth in red
    for idx in MOUTH:
        x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
# --- Initialize FaceLandmarker ---
base_options = python.BaseOptions(model_asset_path=r"/home/user/ros2_ws/src/models/face_landmarker.task")#, delegate=python.BaseOptions.Delegate.GPU)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# --- Webcam loop ---
cap = cv2.VideoCapture(0)
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    ear, mar = 0.0, 0.0
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        landmark_array = np.array([[lm.x, lm.y] for lm in landmarks])
        ear = calculate_avg_ear(landmark_array)
        mar = mouth_aspect_ratio(landmark_array)
        draw_keypoints(frame, landmark_array)

    # Compute FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
    prev_time = curr_time

    # Display EAR, MAR, FPS
    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"MAR: {mar:.3f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Keypoints EAR/MAR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()