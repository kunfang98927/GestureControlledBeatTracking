import numpy as np
from madmom_bt_demo import get_joint_db_b_activations, db_b_tracking
from utils.visualization import visualize_with_db, visualize_with_db_for_slide
from utils.gesture_parser import (
    load_gesture_data,
    parse_gestures,
    find_gathering_points,
    filter_beats,
    smooth_tempo_contour,
    calculate_peaks,
    modify_beat_activations_by_gesture,
    prepare_tempo_list,
)


def get_madmom_downbeats_beats(file_path):
    """Read the beat annotations from a txt file."""

    with open(file_path, "r") as file:
        annotation = file.read()
    annotation = annotation.split("\n")
    annotation = [line.split(" ") for line in annotation if line]
    beats = [float(line[0]) for line in annotation]
    downbeats = [float(line[0]) for line in annotation if float(line[1]) == 1.0]
    beats = np.array(beats)
    downbeats = np.array(downbeats)

    return beats, downbeats


if __name__ == "__main__":

    # Define the paths
    gesture_filepath = "../demo_1204_R_17/R_17-ziyu-lazy_ges.json"
    wav_filepath = "../dataset/audio/R_17_MAPS.wav"
    madmom_activation_path = "../demo_1204_R_17/R_17_db_b_activation.txt"
    madmom_downbeat_beat_path = "../demo_1204_R_17/R_17_beats.txt"

    output_beats_path = "../demo_1204_R_17/experiments_for_report/sample2-lazy/R_17-ziyu-lazy_beats_1206_envelope-win-b51-db101_npclip02.txt"
    output_downbeats_path = "../demo_1204_R_17/experiments_for_report/sample2-lazy/R_17-ziyu-lazy_downbeats_1206_envelope-win-b51-db101_npclip02.txt"
    image_path = "../demo_1204_R_17/experiments_for_report/sample2-lazy/R_17-ziyu-lazy_1206_envelope-win-b51-db101_npclip02.png"

    # Time range for audio analysis, aligned with gesture data, in seconds
    audio_time_range = (49.80, 147.80)
    process_duration = audio_time_range[1] - audio_time_range[0]  # Duration in seconds

    # Get beat activations by madmom from the audio file
    joint_activations = get_joint_db_b_activations(
        wav_filepath, madmom_activation_path=madmom_activation_path
    )
    beat_activations = joint_activations[:, 0]
    downbeat_activations = joint_activations[:, 1]
    beat_activations = beat_activations[
        int(audio_time_range[0] * 100) : int(audio_time_range[1] * 100)
    ]
    downbeat_activations = downbeat_activations[
        int(audio_time_range[0] * 100) : int(audio_time_range[1] * 100)
    ]

    # Get beat positions by madmom from the txt file
    madmom_beats, madmom_downbeats = get_madmom_downbeats_beats(
        madmom_downbeat_beat_path
    )
    madmom_beats = madmom_beats[
        (madmom_beats >= audio_time_range[0]) & (madmom_beats <= audio_time_range[1])
    ]
    madmom_downbeats = madmom_downbeats[
        (madmom_downbeats >= audio_time_range[0])
        & (madmom_downbeats <= audio_time_range[1])
    ]

    # Load and parse data
    gesture_data = load_gesture_data(gesture_filepath)
    gesture_results = gesture_data["results"]
    # according to the process_duration, only keep the gesture_data within the range
    gesture_results = [
        r for r in gesture_results if r["frame_timestamp_ms"] <= process_duration * 1000
    ]

    # Parse gestures to find beat positions and tempo changes
    left_gesture_positions, right_gesture_positions, tempo_changes = parse_gestures(
        gesture_results
    )
    left_gesture_positions = [
        (v, t + audio_time_range[0] * 1000, c) for v, t, c in left_gesture_positions
    ]
    right_gesture_positions = [
        (v, t + audio_time_range[0] * 1000, c) for v, t, c in right_gesture_positions
    ]
    tempo_changes = [(v, t + audio_time_range[0] * 1000) for v, t in tempo_changes]
    print("tempo_changes:", tempo_changes)

    # Find beat positions based on gesture data
    left_beat_positions = []
    right_beat_positions = []
    if len(left_gesture_positions) != 0:
        left_beat_positions = find_gathering_points(left_gesture_positions)
    if len(right_gesture_positions) != 0:
        right_beat_positions = find_gathering_points(right_gesture_positions)
    beat_positions = np.sort(
        np.concatenate([left_beat_positions, right_beat_positions])
    )

    # Find downbeat positions based on gesture data
    left_db_positions = []
    right_db_positions = []
    left_gesture_positions_db = [
        (v, t, c) for v, t, c in left_gesture_positions if v == "db"
    ]
    right_gesture_positions_db = [
        (v, t, c) for v, t, c in right_gesture_positions if v == "db"
    ]
    if len(left_gesture_positions_db) != 0:
        left_db_positions = find_gathering_points(left_gesture_positions_db)
    if len(right_gesture_positions_db) != 0:
        right_db_positions = find_gathering_points(right_gesture_positions_db)
    downbeat_positions = np.sort(
        np.concatenate([left_db_positions, right_db_positions])
    )

    # Filter beat positions based on tempo changes
    filtered_beat_positions, tempo_events = filter_beats(beat_positions, tempo_changes)

    # Clean up the beat positions
    tempo_contour = 60000 / np.diff(
        filtered_beat_positions
    )  # BPM = 60,000 ms / beat interval
    smoothed_tempo_contour = smooth_tempo_contour(tempo_contour, window_size=10)

    # make each peak smoother
    beat_peaks = calculate_peaks(
        filtered_beat_positions,
        window_size=51,
        start_time=audio_time_range[0],
        end_time=audio_time_range[1],
    )
    downbeat_peaks = calculate_peaks(
        downbeat_positions,
        window_size=101,
        start_time=audio_time_range[0],
        end_time=audio_time_range[1],
    )

    # Calculate new activations based on the gesture data
    modified_beat_activations = modify_beat_activations_by_gesture(
        beat_peaks, beat_activations
    )
    modified_downbeat_activations = modify_beat_activations_by_gesture(
        downbeat_peaks, downbeat_activations
    )
    modified_joint_activations = np.column_stack(
        (modified_beat_activations, modified_downbeat_activations)
    )
    print("modified_joint_activations:", modified_joint_activations.shape)

    # Beat tracking based on the modified activations
    # beats_per_bar_list, min_bpm_list, max_bpm_list = prepare_tempo_list(
    #     tempo_events, audio_time_range[0], audio_time_range[1]
    # )
    # beats_per_bar_list = [3, 3, 3, 3]
    # min_bpm_list = [40, 40, 66, 31]
    # max_bpm_list = [79, 75, 95, 73]
    modified_joint_positions = db_b_tracking(
        modified_joint_activations,
        beats_per_bar=[3],
        min_bpm=30,  # min(smoothed_tempo_contour) + 20,
        max_bpm=82,  # max(smoothed_tempo_contour) + 100,
    )
    if len(modified_joint_positions) != 0:
        modified_joint_positions[:, 0] = (
            modified_joint_positions[:, 0] + audio_time_range[0]
        )
    modified_downbeat_positions = []
    modified_beat_positions = []
    for time, phase in modified_joint_positions:
        if phase == 1.0:
            modified_downbeat_positions.append(time)
        modified_beat_positions.append(time)
    modified_downbeat_positions = np.array(modified_downbeat_positions)
    modified_beat_positions = np.array(modified_beat_positions)

    # Print the results
    print("Tempo Events:", tempo_events)
    print(
        "Final tempo range:", min(smoothed_tempo_contour), max(smoothed_tempo_contour)
    )

    visualize_with_db_for_slide(
        left_gesture_positions,
        right_gesture_positions,
        tempo_changes,
        left_beat_positions,
        right_beat_positions,
        left_db_positions,
        right_db_positions,
        filtered_beat_positions,
        smoothed_tempo_contour,
        beat_peaks,
        downbeat_peaks,
        beat_activations,
        downbeat_activations,
        madmom_beats,
        madmom_downbeats,
        modified_beat_activations,
        modified_downbeat_activations,
        modified_beat_positions,
        modified_downbeat_positions,
        start_time=audio_time_range[0],
        end_time=audio_time_range[1],
        image_path=image_path,
    )

    # save the beat positions in txt file
    np.savetxt(output_beats_path, modified_beat_positions, fmt="%f")
    np.savetxt(output_downbeats_path, modified_downbeat_positions, fmt="%f")

    print("The beat positions are saved in", output_beats_path)
