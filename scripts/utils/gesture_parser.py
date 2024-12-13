import json
import numpy as np
from sklearn.cluster import DBSCAN

# MIN_BPM = 20
# MAX_BPM = 300


def load_gesture_data(gesture_filepath):
    """Load gesture data from a JSON file."""
    with open(gesture_filepath, "r") as file:
        return json.load(file)


def parse_gestures(results, beat_confidence_threshold=0.0):
    """Parse gestures from results and return beat positions and tempo adjustments."""
    # ("b", timestamp, confidence), ("db", timestamp, confidence)
    left_beat_positions = []
    right_beat_positions = []
    tempo_changes = [("constant", 0)]  # Initialize with a constant tempo

    current_tempo_change_status = "constant"
    for frame in results:
        timestamp = frame["frame_timestamp_ms"]
        for gesture in frame["gestures"]:
            if (
                gesture["gesture_category"] == "Pointing_Up"
                and gesture["gesture_score"] > beat_confidence_threshold
            ):
                # if gesture["handedness_category"] == "Left":
                #     left_beat_positions.append(
                #         (
                #             "b",
                #             timestamp,
                #             gesture["gesture_score"],
                #         )
                #     )
                if gesture["handedness_category"] == "Right":
                    right_beat_positions.append(
                        (
                            "b",
                            timestamp,
                            gesture["gesture_score"],
                        )
                    )
            elif (
                gesture["gesture_category"] == "Open_Palm"
                and gesture["gesture_score"] > beat_confidence_threshold
            ):
                # if gesture["handedness_category"] == "Left":
                #     left_beat_positions.append(
                #         (
                #             "db",
                #             timestamp,
                #             gesture["gesture_score"],
                #         )
                #     )
                if gesture["handedness_category"] == "Right":
                    right_beat_positions.append(
                        (
                            "db",
                            timestamp,
                            gesture["gesture_score"],
                        )
                    )
            elif (
                gesture["gesture_category"] == "Thumb_Up"
                and gesture["gesture_score"] > beat_confidence_threshold
            ):
                if current_tempo_change_status != "up":
                    tempo_changes.append(("up", timestamp))
                    current_tempo_change_status = "up"
            elif (
                gesture["gesture_category"] == "Thumb_Down"
                and gesture["gesture_score"] > beat_confidence_threshold
            ):
                if current_tempo_change_status != "down":
                    tempo_changes.append(("down", timestamp))
                    current_tempo_change_status = "down"
            elif (
                gesture["gesture_category"] == "Victory"
                and gesture["gesture_score"] > beat_confidence_threshold
            ):
                if current_tempo_change_status != "cut":
                    tempo_changes.append(("cut", timestamp))
                    current_tempo_change_status = "cut"

    return left_beat_positions, right_beat_positions, tempo_changes


def estimate_tempo_from_beats(beat_positions, tolerance_factor=2.0):
    """
    Estimate the tempo from beat positions by removing irregular beats.

    Parameters:
    - beat_positions (list of float): List of beat timestamps in seconds.
    - tolerance_factor (float): Factor for determining outliers (default is 1.5).

    Returns:
    - estimated_tempo (float): Estimated tempo in beats per minute (BPM).
    - filtered_beats (list of float): List of filtered beat positions.
    """
    beat_positions = np.array(beat_positions)
    # Calculate inter-beat intervals (IBIs) in milliseconds
    ibi_array = np.diff(beat_positions)

    # Calculate the median and median absolute deviation (MAD) for IBI filtering
    median_ibi = np.median(ibi_array)
    mad = np.median(np.abs(ibi_array - median_ibi))

    # Define a threshold to identify outliers
    threshold = tolerance_factor * mad

    # print(
    #     "Median IBI:",
    #     median_ibi,
    #     "MAD:",
    #     mad,
    #     "Threshold:",
    #     threshold,
    #     "IBI array:",
    #     ibi_array,
    # )

    # Get indexes of the regular IBIs
    regular_ibis_idx = np.where(np.abs(ibi_array - median_ibi) <= threshold)[0]
    regular_beats_idx = regular_ibis_idx + 1
    if 0 in regular_ibis_idx:
        regular_beats_idx = np.insert(regular_beats_idx, 0, 0)

    filtered_intervals = ibi_array[regular_ibis_idx]
    filtered_beats = beat_positions[regular_beats_idx]

    # Estimate the tempo range based on the filtered IBIs (min and max BPM)
    estimated_tempo = 60000 / np.array(
        [max(filtered_intervals), min(filtered_intervals)]
    )

    return estimated_tempo, filtered_beats


def filter_beats(beat_positions, tempo_changes=None, tolerance=1.0):
    """
    Filters beat positions based on tempo change data.

    Parameters:
    - beat_positions (list of float): List of beat timestamps in seconds.
    - tempo_changes (list of tuples): List of tuples with tempo status
                    ("constant", "up", "down") and corresponding time.
    - tolerance (float): Allowed deviation ratio for stable beat intervals
                         in "constant" regions (default is 0.1, or 10%).

    Returns:
    - filtered_beats (list of float): Filtered list of beat timestamps.
    """
    # Initialize variables
    filtered_beats = []
    tempo_events = []  # (start_time, end_time, value_range)

    # Process each tempo change region
    for i, _ in enumerate(tempo_changes):
        # Define the region's start and end based on tempo change timestamps
        current_status, start_time = tempo_changes[i]
        end_time = (
            tempo_changes[i + 1][1] if i + 1 < len(tempo_changes) else float("inf")
        )

        # Filter beat positions within the current tempo region
        region_beats = [b for b in beat_positions if start_time <= b < end_time]
        if len(region_beats) < 2:
            # Skip if there are not enough beats to analyze
            filtered_beats.extend(region_beats)
            continue

        # Calculate beat intervals in the region
        intervals = np.diff(region_beats)

        if current_status == "constant" or current_status == "cut":
            # estimate the tempo for the constant region
            estimated_tempo_range, stable_beats = estimate_tempo_from_beats(
                region_beats, tolerance_factor=100.0
            )
            # add the tempo range to the tempo_events
            tempo_events.append((start_time, end_time, estimated_tempo_range))
            filtered_beats.extend(stable_beats)
            # print(
            #     "Region beats:",
            #     start_time,
            #     end_time,
            #     current_status,
            #     estimated_tempo_range,
            # )

        elif current_status == "down":
            # in this region, the tempo should be slower than the previous region
            previous_tempo_range = tempo_events[-1][2]
            # remove beats that are faster than the previous region
            valid_beats = [region_beats[0]]
            for j in range(1, len(region_beats)):
                # if 60000 / (intervals[j - 1]) <= previous_tempo_range[0] * (
                #     1 + tolerance
                # ):
                valid_beats.append(region_beats[j])
            filtered_beats.extend(valid_beats)
            # calculate the tempo range for the valid beats
            valid_intervals = np.diff(valid_beats)
            valid_tempo_range = (
                [
                    60000 / max(valid_intervals),
                    60000 / min(valid_intervals),
                ]
                if len(valid_intervals) > 0
                else previous_tempo_range
            )
            # add the tempo range to the tempo_events
            tempo_events.append((start_time, end_time, valid_tempo_range))
        elif current_status == "up":
            # in this region, the tempo should be faster than the previous region
            previous_tempo_range = tempo_events[-1][2]
            # in this case, if the tempo is slower than the previous region,
            # interpolate more beats to make the tempo faster
            valid_beats = [region_beats[0]]
            for j in range(1, len(region_beats)):
                # if 60000 / (intervals[j - 1]) >= previous_tempo_range[1] * (
                #     1 - tolerance
                # ):
                valid_beats.append(region_beats[j])
                # else:
                #     # interpolate more beats
                #     num_interpolated = int(
                #         intervals[j - 1] / (previous_tempo_range[1] * (1 - tolerance))
                #     )
                #     interpolated_beats = np.linspace(
                #         region_beats[j - 1], region_beats[j], num_interpolated + 2
                #     )[1:-1]
                #     valid_beats.extend(interpolated_beats)
            filtered_beats.extend(valid_beats)
            # calculate the tempo range for the valid beats
            valid_intervals = np.diff(valid_beats)
            valid_tempo_range = [
                60000 / max(valid_intervals),
                60000 / min(valid_intervals),
            ]
            # add the tempo range to the tempo_events
            tempo_events.append((start_time, end_time, valid_tempo_range))

    filtered_beats = np.array(filtered_beats)
    return filtered_beats, tempo_events


def find_gathering_points(gesture_positions, eps=200, min_samples=2):
    """
    Find gathering points using DBSCAN clustering to identify dense regions of gestures.
    Parameters:
        gesture_positions: Array of gesture timestamps with confidence scores.
        eps: Maximum distance (ms) between points in a cluster.
        min_samples: Minimum number of points to form a cluster.
    Returns:
        Array of cluster center timestamps representing the detected beat positions.
    """
    # Filter gestures by confidence threshold
    filtered_timestamps = np.array([t for _, t, _ in gesture_positions if t > 0])

    # Reshape data for DBSCAN (DBSCAN expects a 2D array)
    X = filtered_timestamps.reshape(-1, 1)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    # Calculate cluster centers (use median or mean of each cluster)
    beat_positions = []
    for cluster in set(clustering.labels_):
        if cluster == -1:
            continue  # Ignore noise points
        cluster_points = X[clustering.labels_ == cluster]
        # use the first point as the beat position
        beat_position = cluster_points[0][0]
        beat_positions.append(beat_position)

    return np.array(beat_positions)


def smooth_tempo_contour(tempo_contour, window_size=3):
    """Smooth the tempo contour using a moving average."""
    if len(tempo_contour) < window_size:
        return tempo_contour  # No smoothing if the contour is too short

    smoothed_contour = []
    for i in range(len(tempo_contour)):
        # Calculate the window range
        start = max(0, i - window_size // 2)
        end = min(len(tempo_contour), i + window_size // 2 + 1)

        # Compute the average over the window
        window_avg = np.mean(tempo_contour[start:end])
        smoothed_contour.append(window_avg)

    return smoothed_contour


def calculate_peaks(beat_positions, window_size=5, start_time=0, end_time=0):
    """
    Smooth peaks in a binary list using a specified kernel.

    Args:
        beat_positions (np.ndarray): a list of beat positions, in ms.
        window_size (int): The size of the smoothing window (should be odd).
        smoothing_type (str): Type of smoothing.

    Returns:
        np.ndarray: Smoothed list with values between 0 and 1.
    """
    # convert ms into fps=100
    beat_positions = beat_positions * 100 / 1000

    min_time = int(start_time * 100)
    max_time = int(end_time * 100) + 1
    binary_list = np.zeros(int(max_time - min_time + 1))
    binary_list[beat_positions.astype(int) - int(min_time)] = 1

    # Ensure the window size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    binary_array = np.array(binary_list)
    smoothed_array = np.zeros_like(binary_array, dtype=float)
    half_window = window_size // 2

    # Create the smoothing kernel
    x = np.linspace(-half_window, half_window, window_size)
    kernel = np.exp(-((x / (half_window / 2)) ** 2))

    # Apply smoothing
    for i, value in enumerate(binary_array):
        if value == 1:
            start = max(0, i - half_window)
            end = min(len(binary_array), i + half_window + 1)
            kernel_start = half_window - (i - start)
            kernel_end = half_window + (end - i)
            smoothed_array[start:end] += kernel[kernel_start:kernel_end]

    smoothed_array = np.clip(smoothed_array, 0.2, 1)

    return smoothed_array


def modify_beat_activations_by_gesture(beat_peaks, beat_activations, mode="*"):
    """
    Modify the beat activations based on the detected beat peaks.

    Args:
        beat_peaks (np.ndarray): Smoothed beat peaks.
        beat_activations (np.ndarray): Beat activations from the audio analysis.

    Returns:
        np.ndarray: Modified beat activations.
    """
    modified_beat_activations = beat_activations.copy()

    # make them the same length so that they can multiply
    common_length = min(len(beat_peaks), len(beat_activations))
    beat_peaks = beat_peaks[:common_length]
    beat_activations = beat_activations[:common_length]

    if mode == "*":
        # Multiply the beat activations by the beat peaks
        modified_beat_activations[:common_length] = beat_activations * beat_peaks  # * 2
        # for i in range(common_length):
        #     if beat_peaks[i] > 0.8:
        #         modified_beat_activations[i] = beat_activations[i] * 2 * beat_peaks[i]
        #     else:
        #         modified_beat_activations[i] = beat_activations[i] * beat_peaks[i]
    elif mode == "+":
        # Modify the beat activations based on the beat peaks
        modified_beat_activations[:common_length] = (
            beat_activations * 0.9 + beat_peaks * 0.1
        )

    return modified_beat_activations


def prepare_tempo_list(tempo_events, start_time, end_time):
    """
    Prepare tempo list for beat tracking based on tempo events.

    Args:
        tempo_events (list of tuples): List of tempo change events.
        start_time (float): Start time of the audio file, in seconds.
        end_time (float): End time of the audio file, in seconds.

    Returns:
        beats_per_bar_list (list of int): List of beats per bar for each tempo region.
        min_bpm_list (list of float): List of minimum BPM values for each tempo region.
        max_bpm_list (list of float): List of maximum BPM values for each tempo region.
    """

    beats_per_bar_list = []
    min_bpm_list = []
    max_bpm_list = []

    for i, tempo_event in enumerate(tempo_events):
        start, end, tempo_range = tempo_event
        start /= 1000  # convert ms to s
        end /= 1000  # convert ms to s
        if i == 0:
            start = start_time
        if i == len(tempo_events) - 1:
            end = end_time
        # calculate how many bars are in this region
        duration = end - start
        min_bpm = tempo_range[0]
        max_bpm = tempo_range[1]
        min_beat_num = duration * min_bpm / 60
        max_beat_num = duration * max_bpm / 60

        beats_per_bar = 3
        min_bar_num = min_beat_num / beats_per_bar
        max_bar_num = max_beat_num / beats_per_bar

        for bar_num in range(int(max_bar_num)):
            beats_per_bar_list.append(beats_per_bar)
            min_bpm_list.append(min_bpm)
            max_bpm_list.append(max_bpm)

    return beats_per_bar_list, min_bpm_list, max_bpm_list
