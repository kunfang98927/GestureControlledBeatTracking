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
    gesture_filepath = "../demo_1205_May/May-ziyu1_ges.json"
    wav_filepath = "../demo_1205_May/May.mp3"
    madmom_activation_path = (
        "../demo_1205_May/madmom_downbeats_beats/May_db_b_activation.txt"
    )
    madmom_downbeat_beat_path = "../demo_1205_May/madmom_downbeats_beats/May_beats.txt"

    output_beats_path = "../demo_1205_May/experiments_for_report/sample1/May-ziyu1_beats_win-b21-db51.txt"
    output_downbeats_path = "../demo_1205_May/experiments_for_report/sample1/May-ziyu1_downbeats_win-b21-db51.txt"
    image_path = (
        "../demo_1205_May/experiments_for_report/sample1/May-ziyu1_win-b21-db51.png"
    )

    # Time range for audio analysis, aligned with gesture data, in seconds
    audio_time_range = (0.6, 146.0)
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

    # Initialize variables for beat tracking
    total_filtered_beat_positions = []
    total_smoothed_tempo_contour = []
    total_beat_peaks = np.zeros(int(process_duration * 100))
    total_downbeat_peaks = np.zeros(int(process_duration * 100))
    total_modified_beat_activations = np.zeros(int(process_duration * 100))  # 10ms
    total_modified_downbeat_activations = np.zeros(int(process_duration * 100))  # 10ms
    total_modified_beat_positions = []
    total_modified_downbeat_positions = []
    segment_overlap = 5 * 1000  # 3 seconds

    # Process beat tracking based on segmentations
    for seg_start_line, seg_end_line, tempo_range in tempo_events:

        seg_start = max(seg_start_line - segment_overlap, audio_time_range[0] * 1000)
        seg_end = min(seg_end_line + segment_overlap, audio_time_range[1] * 1000)

        print("Tempo change at", seg_start, "to", seg_end, ":", tempo_range)

        seg_beat_positions = filtered_beat_positions[
            (filtered_beat_positions >= seg_start)
            & (filtered_beat_positions <= seg_end)
        ]
        total_filtered_beat_positions.extend(seg_beat_positions)

        seg_downbeat_positions = downbeat_positions[
            (downbeat_positions >= seg_start) & (downbeat_positions <= seg_end)
        ]

        # Calculate tempo contour
        seg_tempo_contour = 60000 / np.diff(
            seg_beat_positions
        )  # BPM = 60,000 ms / beat interval
        # copy the last tempo to the end
        seg_tempo_contour = np.append(seg_tempo_contour, seg_tempo_contour[-1])

        seg_smoothed_tempo_contour = smooth_tempo_contour(
            seg_tempo_contour, window_size=10
        )
        seg_db_tempo_contour = 60000 / np.diff(seg_downbeat_positions)
        seg_smoothed_db_tempo_contour = smooth_tempo_contour(
            seg_db_tempo_contour, window_size=10
        )
        print("seg_tempo_contour", len(seg_tempo_contour))
        print("seg_smoothed_tempo_contour", len(seg_smoothed_tempo_contour))
        print("seg_beat_positions", len(seg_beat_positions))

        total_smoothed_tempo_contour.extend(seg_smoothed_tempo_contour)

        # Calculate the number of beats per bar
        seg_mean_bpm = np.mean(seg_smoothed_tempo_contour)
        seg_db_mean_bpm = np.mean(seg_smoothed_db_tempo_contour)
        seg_beats_per_bar = round(seg_mean_bpm / seg_db_mean_bpm)
        seg_std_bpm = np.std(seg_smoothed_tempo_contour)
        transition_lambda = 100 / seg_std_bpm

        print(
            "seg_beats_per_bar:",
            seg_beats_per_bar,
            "seg_db_mean_bpm:",
            seg_db_mean_bpm,
            "seg_mean_bpm:",
            seg_mean_bpm,
        )
        print(
            "min_seg_tempo:",
            min(seg_smoothed_tempo_contour),
            "max_seg_tempo:",
            max(seg_smoothed_tempo_contour),
        )
        print("seg_std_bpm:", seg_std_bpm, "transition_lambda:", transition_lambda)

        # make each peak smoother
        seg_beat_peaks = calculate_peaks(
            seg_beat_positions,
            window_size=21,
            start_time=seg_start / 1000,
            end_time=seg_end / 1000,
        )
        seg_downbeat_peaks = calculate_peaks(
            seg_downbeat_positions,
            window_size=51,  # (31 - 1) * seg_beats_per_bar + 1,
            start_time=seg_start / 1000,
            end_time=seg_end / 1000,
        )

        start_index = int(seg_start / 10) - int(audio_time_range[0] * 100)
        end_index = min(
            start_index + len(seg_beat_peaks),
            int(process_duration * 100),
        )
        total_beat_peaks[start_index:end_index] += seg_beat_peaks[
            : end_index - start_index
        ]
        total_downbeat_peaks[start_index:end_index] += seg_downbeat_peaks[
            : end_index - start_index
        ]

        seg_beat_activations = beat_activations[start_index:end_index]
        seg_downbeat_activations = downbeat_activations[start_index:end_index]
        seg_beat_peaks = seg_beat_peaks[: end_index - start_index]
        seg_downbeat_peaks = seg_downbeat_peaks[: end_index - start_index]

        print("seg_beat_activations:", seg_beat_activations.shape)
        print("seg_downbeat_activations:", seg_downbeat_activations.shape)
        print("seg_beat_peaks:", seg_beat_peaks.shape)
        print("seg_downbeat_peaks:", seg_downbeat_peaks.shape)

        # Calculate new activations based on the gesture data
        seg_modified_beat_activations = modify_beat_activations_by_gesture(
            seg_beat_peaks, seg_beat_activations, mode="*"
        )
        # seg_modified_beat_activations = modify_beat_activations_by_gesture(
        #     seg_downbeat_peaks, seg_beat_activations, mode="+"
        # )
        seg_modified_downbeat_activations = modify_beat_activations_by_gesture(
            seg_downbeat_peaks, seg_downbeat_activations, mode="*"
        )
        seg_modified_joint_activations = np.column_stack(
            (seg_modified_beat_activations, seg_modified_downbeat_activations)
        )
        total_modified_beat_activations[
            start_index:end_index
        ] += seg_modified_beat_activations[: end_index - start_index]
        total_modified_downbeat_activations[
            start_index:end_index
        ] += seg_modified_downbeat_activations[: end_index - start_index]
        print("seg_modified_joint_activations:", seg_modified_joint_activations.shape)

        # Beat tracking based on the modified activations
        seg_modified_joint_positions = db_b_tracking(
            seg_modified_joint_activations,
            beats_per_bar=seg_beats_per_bar,
            min_bpm=min(seg_smoothed_tempo_contour),
            max_bpm=max(seg_smoothed_tempo_contour),
            transition_lambda=transition_lambda,  # default 100
        )
        if len(seg_modified_joint_positions) != 0:
            seg_modified_joint_positions[:, 0] = (
                seg_modified_joint_positions[:, 0] + seg_start / 1000
            )
        seg_modified_downbeat_positions = []
        seg_modified_beat_positions = []
        for time, phase in seg_modified_joint_positions:
            if time < seg_start_line / 1000 or time > seg_end_line / 1000:
                continue
            if phase == 1.0:
                seg_modified_downbeat_positions.append(time)
            seg_modified_beat_positions.append(time)
        seg_modified_downbeat_positions = np.array(seg_modified_downbeat_positions)
        seg_modified_beat_positions = np.array(seg_modified_beat_positions)

        total_modified_beat_positions.extend(seg_modified_beat_positions)
        total_modified_downbeat_positions.extend(seg_modified_downbeat_positions)

    total_filtered_beat_positions = np.array(total_filtered_beat_positions)

    visualize_with_db_for_slide(
        left_gesture_positions,
        right_gesture_positions,
        tempo_changes,
        left_beat_positions,
        right_beat_positions,
        left_db_positions,
        right_db_positions,
        total_filtered_beat_positions,
        total_smoothed_tempo_contour,
        total_beat_peaks,
        total_downbeat_peaks,
        beat_activations,
        downbeat_activations,
        madmom_beats,
        madmom_downbeats,
        total_modified_beat_activations,
        total_modified_downbeat_activations,
        total_modified_beat_positions,
        total_modified_downbeat_positions,
        start_time=audio_time_range[0],
        end_time=audio_time_range[1],
        image_path=image_path,
    )

    # save the beat positions in txt file
    np.savetxt(output_beats_path, total_modified_beat_positions, fmt="%f")
    np.savetxt(output_downbeats_path, total_modified_downbeat_positions, fmt="%f")

    print("The beat positions are saved in", output_beats_path)
