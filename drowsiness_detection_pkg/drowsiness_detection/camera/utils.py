"""This module contains the utility functions for calculating the drowsiness metrics from
facial landmarks and vehicle features (steering angle, lane position)"""

import math
from typing import Tuple, List
import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "low_pass_filter",
    "approx_entropy",
    "count_reversals",
    "steering_reversals",
    "lane_position_std_dev",
    "calculate_ear",
    "calculate_avg_ear",
    "mouth_aspect_ratio",
    "calculate_perclos",
    "calculate_blink_frequency",
    "calculate_yawn_frequency",
]


def calculate_ear(landmarks: np.ndarray, eye: str) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.

    The EAR is calculated using the formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
    where p1, p2, p3, p4, p5, p6 are 2D landmark points.

    Args:
        landmarks (np.ndarray): Array of face landmarks.
        eye (str): Either 'left_eye' or 'right_eye'.

    Returns:
        float: The calculated Eye Aspect Ratio
    """
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
    """Calculate the average Eye Aspect Ratio (EAR) from the face landmarks.
    refer to the paper for more details: https://ieeexplore.ieee.org/document/10039811

    Args:
        landmarks (np.ndarray): Array of face landmarks.

    Returns:
        float: Average Eye Aspect Ratio.
    """
    left_ear = calculate_ear(landmarks, "left_eye")
    right_ear = calculate_ear(landmarks, "right_eye")
    return (left_ear + right_ear) / 2.0


def mouth_aspect_ratio(landmarks: np.array) -> float:
    """Calculate the Mouth Aspect Ratio (MAR) from the landmarks.

    Args:
        landmarks (np.array): Array of facial landmarks.

    Returns:
        float: The calculated Mouth Aspect Ratio.
    """
    # Define mouth landmarks indices
    mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]

    # Calculate distances between the landmarks
    a = np.linalg.norm(
        np.array(landmarks[mouth_indices[1]]) - np.array(landmarks[mouth_indices[7]])
    )
    b = np.linalg.norm(
        np.array(landmarks[mouth_indices[2]]) - np.array(landmarks[mouth_indices[6]])
    )
    c = np.linalg.norm(
        np.array(landmarks[mouth_indices[3]]) - np.array(landmarks[mouth_indices[5]])
    )

    d = np.linalg.norm(
        np.array(landmarks[mouth_indices[0]]) - np.array(landmarks[mouth_indices[4]])
    )

    return (a + b + c) / (2.0 * d)


def calculate_perclos(
    ear_values: np.ndarray, ear_threshold: float, min_consec_frames: int
) -> float:
    """
    Calculate PERCLOS (Percentage of Eye Closure) from stored EAR values.

    Args:
        ear_values (np.ndarray): Array of EAR values sampled over time.
        ear_threshold (float): EAR threshold below which eyes are considered closed.
        min_consec_frames (int): Minimum consecutive frames below threshold to count as closed.

    Returns:
        float: PERCLOS percentage over the EAR values window.
    """
    closed_frames = 0
    consec_count = 0

    for ear in ear_values:
        if ear is not None:  # frames with no mediapipe mesh records aa none
            if ear < ear_threshold:  # eye closed
                consec_count += 1
            else:
                if consec_count >= min_consec_frames:  # eye opened
                    closed_frames += (
                        consec_count  # add the consecutive frames to closed frames
                    )
                consec_count = 0  # restart the count

    # Account for closing at the end of the window
    if consec_count >= min_consec_frames:
        closed_frames += consec_count

    total_frames = np.sum([ear is not None for ear in ear_values])

    if total_frames == 0:
        return 0.0

    perclos = (closed_frames / total_frames) * 100.0
    return perclos


def calculate_blink_frequency(
    ear_values: np.ndarray, ear_threshold: float, fps: float, min_consec_frames: int = 3
) -> float:
    """
    Calculate blink frequency (blinks per minute) from stored EAR values.

    Args:
        ear_values (np.ndarray): Array of EAR values sampled over time.
        ear_threshold (float): EAR threshold below which eyes are considered closed.
        fps (float): Frames per second of the data sampling.
        min_consec_frames (int): Minimum consecutive frames below threshold to count as a blink.

    Returns:
        float: Blink frequency in blinks per minute.
    """
    blinks = 0
    consec_closed_frames = 0
    eyes_closed = False

    for ear in ear_values:
        if ear < ear_threshold:  # eyes closed
            consec_closed_frames += 1
            if consec_closed_frames >= min_consec_frames:
                eyes_closed = True
        else:  # eyes open
            if eyes_closed:
                blinks += 1
                eyes_closed = False
            consec_closed_frames = 0

    total_seconds = len(ear_values) / fps if fps > 0 else 1
    blink_freq = (blinks * 60) / total_seconds
    return blink_freq


def calculate_yawn_frequency(
    mar_values: np.ndarray, mar_threshold: float, min_consec_frames: int, fps: float
) -> float:
    """
    Calculate yawn frequency (yawns per minute) from stored MAR values.

    Args:
        mar_values (np.ndarray): Array of MAR values sampled over time.
        mar_threshold (float): MAR threshold above which mouth is considered open.
        min_consec_frames (int): Minimum consecutive frames above threshold to count as a
        yawn. yawning is a quick act of opening and closing the mouth, which lasts for
        around 4 to 6 s.
        fps (float): Frames per second of the data sampling.

    Returns:
        float: Yawn frequency in yawns per minute.
    """
    yawn_count = 0
    consec_frames = 0
    yawn_detected = False

    for mar in mar_values:
        if mar > mar_threshold:
            consec_frames += 1
            if consec_frames >= min_consec_frames and not yawn_detected:
                yawn_count += 1
                yawn_detected = True
        else:
            consec_frames = 0
            yawn_detected = False

    total_seconds = len(mar_values) / fps if fps > 0 else 1
    yawn_freq = (yawn_count * 60) / total_seconds
    return yawn_freq


## Vehicle Features Utils ##


def low_pass_filter(
    theta: np.array, cutoff_freq_hz: float, filter_order: float, sampling_rate: float
) -> np.array:
    """Low Pass Filter

    Args:
        theta (np.array): raw steering data

        cutoff_freq_hz (float): low-pass Butterworth filter cutoff frequency (Hz), 2Hz is
        recommended as the optimal parameter for cognitive load based on findings from the
        literature: "A Steering Wheel Reversal Rate Metric for Assessing Effects of Visual
        and Cognitive Secondary Task Load"

        filter_order (float): order of butterworth filter, 2nd order is recommended
        sampling_rate (float): sampling freq in Hz (samples per second)

    Returns:
        np.array: Filtered steering data
    """
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq_hz / nyquist_freq
    b, a = butter(filter_order, normal_cutoff, btype="low", analog=False)
    theta_filtered = filtfilt(b, a, theta)

    return theta_filtered


# === STEERING WHEEL MOVEMENT ANALYSIS ===
# reference paper: [A Steering Wheel Reversal Rate Metric for Assessing Effects of Visual
# and Cognitive Secondary Task Load](https://core.ac.uk/download/pdf/159068039.pdf)


def approx_entropy(time_series: np.array, run_length: int = 2) -> float:
    """Approximate entropy (2sec window) [https://www.mdpi.com/1424-8220/17/3/495]

    Args:
        time_series (np.array): steering movement data
        run_length (int): length of the run data (window with overlapping of the data,
        example x = [1,2,3], if runlength=2 then output will be [[1,2], [2,3]])

    Returns:
        float: regularity (close to 0 : no irregularity, close to 1: irregularity)
    """
    std_dev = np.std(time_series)
    filter_level = 0.2 * std_dev

    def _maxdist(x_i, x_j):
        return max(abs(ua - va) for ua, va in zip(x_i, x_j))

    def _phi(m):
        n = time_series_length - m + 1
        x = [
            [time_series[j] for j in range(i, i + m - 1 + 1)]
            for i in range(time_series_length - m + 1)
        ]
        counts = [
            sum(1 for x_j in x if _maxdist(x_i, x_j) <= filter_level) / n for x_i in x
        ]
        return sum(math.log(c) for c in counts) / n

    time_series_length = len(time_series)

    return abs(_phi(run_length + 1) - _phi(run_length))


def count_reversals(
    theta_vals: np.array, gap: float
) -> Tuple[int, List[Tuple[float, float]]]:
    """calculates steering reversal count

    Args:
        theta_vals (np.array): steering angles
        gap (float): threeshold

    Returns:
        Tuple[int, List[Tuple[float, float]]]: reversal count, list of reversal indices
    """

    k = 0
    nr = 0
    r = []
    n = len(theta_vals)
    for l in range(1, n):
        if theta_vals[l] - theta_vals[k] >= gap:
            nr += 1
            r.append((k, l))
            k = l
        elif theta_vals[l] < theta_vals[k]:
            k = l
    return nr, r


def steering_reversals(filtered_theta: np.array, theta_min: float = 0.1) -> int:
    """calculate the steering wheel reversals of both upward and downward

    Args:
        filtered_theta (np.array): filtered steering wheel data
        theta_min (float): gap size threeshold

    Returns:
        int: reversal count
    """

    # Calculate discrete derivative
    diff_x = np.diff(filtered_theta)
    sign_diff = np.sign(diff_x)

    stationary_points = [0]  # include first index

    for i in range(1, len(sign_diff)):
        if sign_diff[i] != sign_diff[i - 1]:
            stationary_points.append(i)

    stationary_points.append(len(filtered_theta) - 1)  # include last index

    nr_up, _ = count_reversals(filtered_theta, theta_min)
    # To count downward, repeat on negative signal
    nr_down, _ = count_reversals(-filtered_theta, theta_min)

    total_reversals = nr_up + nr_down

    return total_reversals


# === LANE POSITION ===


def lane_position_std_dev(deviations: np.array) -> float:
    """Calculate the standard deviation of lane position deviations.

    Args:
        deviations (np.array): Array of lane position deviations from the lane center.

    Returns:
        float: Standard deviation of the deviations.
    """

    deviations = np.array(deviations)
    std_dev = np.std(deviations, ddof=0)  # population standard deviation
    return std_dev


def vehicle_feature_extraction(
    steering_data: np.array,
    lane_position_data: np.array,
    window: int,
) -> Tuple[float, float, float]:
    """calculate the metric scores from vehicle steering angles and lane position data.

    Args:
        steering_data (np.array): vehicle steering angle data for x time stamps
        lane_position_data (np.array): lane position data for x timestamps
        window (int): time window
    Returns:
        Tuple[float, float, float]: steering entropy, steering rate, sdlp
    """
    sampling_frequency = len(steering_data) / window
    filtered_theta = low_pass_filter(
        theta=steering_data,
        cutoff_freq_hz=2,
        filter_order=2,
        sampling_rate=sampling_frequency,
    )
    entropy = approx_entropy(time_series=filtered_theta, run_length=2)
    steering_rate = steering_reversals(filtered_theta=filtered_theta)
    sdlp = lane_position_std_dev(lane_position_data)

    return entropy, steering_rate, sdlp
