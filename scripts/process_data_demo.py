import numpy as np
from madmom_bt_demo import get_beat_activations, read_txt_beats, beat_tracking
from utils.visualization import visualize, visualize_beats_for_slide
from utils.gesture_parser import (
    load_gesture_data,
    parse_gestures,
    find_gathering_points,
    filter_beats,
    smooth_tempo_contour,
    calculate_peaks,
    modify_beat_activations_by_gesture,
)

if __name__ == "__main__":

    # For kun's data, audio_time_range = (1.2, 45.0)
    # For ziyu's data, audio_time_range = (1.0, 45.0)

    # Define the paths
    gesture_filepath = "../demo_1203_R_22/R_22-kun_ges.json"
    wav_filepath = "../dataset/audio/R_22_MAPS.wav"
    madmom_activation_path = "../dataset/madmom_outputs/R_22_MAPS_act.txt"
    madmom_beat_path = "../dataset/madmom_outputs/R_22_MAPS_b.txt"

    # Define the output paths
    output_beats_path = "../demo_1203_R_22/experiments_for_report/sample4-kun-x4/R_22-kun_beat_1206_envelope-win-31.txt"
    output_image_path = "../demo_1203_R_22/experiments_for_report/sample4-kun-x4/R_22-kun_1206_envelope-win-31.png"

    # Time range for audio analysis, aligned with gesture data, in seconds
    audio_time_range = (1.2, 45.0)
    process_duration = audio_time_range[1] - audio_time_range[0]  # Duration in seconds

    # Get beat activations from the audio file
    beat_activations = get_beat_activations(
        wav_filepath, madmom_activation_path=madmom_activation_path
    )
    beat_activations = np.array(beat_activations)
    beat_activations = beat_activations[
        int(audio_time_range[0] * 100) : int(audio_time_range[1] * 100)
    ]

    # Get beat positions from the txt file
    madmom_beat_positions = read_txt_beats(madmom_beat_path)
    madmom_beat_positions = np.array(madmom_beat_positions)
    madmom_beat_positions = madmom_beat_positions[
        (madmom_beat_positions >= audio_time_range[0])
        & (madmom_beat_positions <= audio_time_range[1])
    ]

    # Load and parse data
    gesture_data = load_gesture_data(gesture_filepath)
    gesture_results = gesture_data["results"]
    # according to the process_duration, only keep the gesture_data within the range
    gesture_results = [
        r for r in gesture_results if r["frame_timestamp_ms"] <= process_duration * 1000
    ]

    # Parse gestures to find beat positions and tempo adjustments
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
        window_size=31,
        start_time=audio_time_range[0],
        end_time=audio_time_range[1],
    )
    modified_beat_activations = modify_beat_activations_by_gesture(
        beat_peaks, beat_activations
    )
    modified_beat_positions = beat_tracking(
        modified_beat_activations,
        min_bpm=min(smoothed_tempo_contour),
        max_bpm=max(smoothed_tempo_contour),
    )
    if len(modified_beat_positions) != 0:
        modified_beat_positions += audio_time_range[0]

    # Print the results
    print("Tempo Events:", tempo_events)
    print(
        "Final tempo range:",
        min(smoothed_tempo_contour),
        max(smoothed_tempo_contour),
    )

    visualize_beats_for_slide(
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
        beat_activations,
        madmom_beat_positions,
        modified_beat_activations,
        modified_beat_positions,
        start_time=audio_time_range[0],
        end_time=audio_time_range[1],
        image_path=output_image_path,
    )

    # save the beat positions in txt file
    np.savetxt(output_beats_path, modified_beat_positions, fmt="%f")
