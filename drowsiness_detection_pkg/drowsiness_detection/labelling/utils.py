from cv_bridge import CvBridge
from drowsiness_detection_msg.msg import DrowsinessMetricsData


def extract_window_data(msg: "DrowsinessMetricsData", bridge: CvBridge) -> dict:
    """
    Extract metrics, raw arrays, and images from a DrowsinessMetricsData message
    and arrange them into a dictionary for saving or further processing.

    Args:
        msg: DrowsinessMetricsData ROS message.
        bridge: CvBridge instance to convert ROS Images to OpenCV format.

    Returns:
        dict containing:
            - metrics: dict of computed metrics
            - raw_data: dict of arrays
            - images: list of OpenCV images
            - window_id: int
    """
    window_id = msg.window_id

    # --- Metrics ---
    metrics = {}
    if len(msg.metrics) == 6:
        metrics = {
            "perclos": msg.metrics[0],
            "blink_rate": msg.metrics[1],
            "yawn_rate": msg.metrics[2],
            "steering_entropy": msg.metrics[3],
            "steering_rate": msg.metrics[4],
            "sdlp": msg.metrics[5],
        }

    # --- Raw arrays ---
    raw_data = {
        "ear_array": list(msg.ear_array),
        "mar_array": list(msg.mar_array),
        "steering_array": list(msg.steering_array),
        "lane_position_array": list(msg.lane_position_array),
    }

    # --- Decode ROS Images to OpenCV ---
    decoded_images = [img.data for img in msg.images]

    return {
        "metrics": metrics,
        "raw_data": raw_data,
        "images": decoded_images,
        "window_id": window_id,
    }
