"""Beat tracking using madmom library."""

import os
import numpy as np
import madmom
import mir_eval
import matplotlib.pyplot as plt
from dirs import MADMOM_OUTPUT_DIR, AUDIO_DIR, ANNOTATION_DIR

# beat_tracker = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)


def get_joint_db_b_activations(file_path, madmom_activation_path=None):
    """Get joint downbeat and beat activations using madmom."""

    if madmom_activation_path:
        with open(madmom_activation_path, "r") as file:
            lines = file.readlines()
        joint_activations = [line.strip().split(" ") for line in lines]
        joint_activations = np.array(joint_activations, dtype=float)

        return joint_activations

    proc = madmom.features.downbeats.RNNDownBeatProcessor()
    # Step 1: Process the audio file to detect downbeat and beat activations
    joint_activations = proc(file_path)

    return joint_activations


def get_beat_activations(file_path, madmom_activation_path=None):
    """Get beat activations using madmom."""

    if madmom_activation_path:
        return read_txt_activations(madmom_activation_path)

    proc = madmom.features.beats.RNNBeatProcessor()
    # Step 1: Process the audio file to detect beat activations
    beat_activations = proc(file_path)

    return beat_activations


def db_b_tracking(
    db_beat_activations,
    beats_per_bar=None,
    min_bpm=None,
    max_bpm=None,
    transition_lambda=100,
):
    """Perform downbeat and beat tracking on the joint activations."""

    # Step 2: Track downbeats and beats based on the joint activations
    print("[madmom] Tracking downbeats and beats...")
    if min_bpm and max_bpm and beats_per_bar:
        dbn_beat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=beats_per_bar,
            fps=100,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            transition_lambda=transition_lambda,
        )
    else:
        dbn_beat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 2],
            fps=100,
            correct=True,
        )
    db_beats = dbn_beat_tracker(db_beat_activations)

    return db_beats


def beat_tracking(beat_activations, min_bpm=None, max_bpm=None):
    """Perform beat tracking on the beat activations."""

    # Step 2: Track beats based on the beat activations
    print("[madmom] Tracking beats...")
    if min_bpm and max_bpm:
        dbn_beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
            fps=100, min_bpm=min_bpm, max_bpm=max_bpm, correct=True
        )
    else:
        dbn_beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    beats = dbn_beat_tracker(beat_activations)

    return beats


def beat_tracking_metrics(detected_beats, annotated_beats, tolerance=0.07):
    """
    Calculate precision, recall, and F-measure for beat tracking.

    Parameters:
    - detected_beats (np.ndarray): Array of detected beat times in seconds.
    - annotated_beats (np.ndarray): Array of annotated beat times in seconds.
    - tolerance (float): Time window (in seconds) within which a detected beat
                         is considered correct (default is 0.07 seconds).

    Returns:
    - precision (float): Precision of the beat tracking.
    - recall (float): Recall of the beat tracking.
    - f_measure (float): F-measure of the beat tracking.
    """
    # Sort both arrays to ensure proper matching
    detected_beats = np.sort(detected_beats)
    annotated_beats = np.sort(annotated_beats)

    # Keep track of matched beats
    matched_detected = np.zeros(len(detected_beats), dtype=bool)
    matched_annotated = np.zeros(len(annotated_beats), dtype=bool)

    # Match detected beats to annotated beats within the tolerance window
    for i, det_beat in enumerate(detected_beats):
        # Find annotated beats within tolerance window of the current detected beat
        within_tolerance = np.abs(annotated_beats - det_beat) <= tolerance
        if np.any(within_tolerance):
            matched_detected[i] = True
            matched_annotated[np.argmax(within_tolerance)] = (
                True  # Mark the first matching annotated beat
            )

    # Calculate precision, recall, and F-measure
    true_positives = np.sum(matched_detected)
    precision = true_positives / len(detected_beats) if len(detected_beats) > 0 else 0
    recall = true_positives / len(annotated_beats) if len(annotated_beats) > 0 else 0
    f_measure = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f_measure


def evaluate_beats(beats, annotations):
    """Evaluate the beat tracking results."""

    # Evaluate the beat tracking results precision, recall, and F-measure
    precision, recall, f_measure = beat_tracking_metrics(beats, annotations)
    print(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-measure: {f_measure:.4f}"
    )

    # mir_eval evaluation
    scores = mir_eval.beat.evaluate(annotations, beats)
    print(scores)
    f_measure_mir_eval = scores["F-measure"]
    print(f"F-measure (mir_eval): {f_measure_mir_eval:.4f}")

    return precision, recall, f_measure, f_measure_mir_eval


def inference_madmom(audio_path, activation_path=None, beats_path=None):

    # get beat activations
    beat_activations = get_beat_activations(audio_path)

    # perform beat tracking
    beats = beat_tracking(beat_activations)

    # save the beat activations and beats for sonic visualizer
    np.savetxt(activation_path, beat_activations, fmt="%.6f")
    np.savetxt(beats_path, beats, fmt="%.6f")

    return beat_activations, beats


def inference_madmom_with_db(
    audio_path, activation_path=None, beats_path=None, downbeats_path=None
):

    # get beat activations
    beat_activations = get_joint_db_b_activations(audio_path)

    # perform beat tracking
    beats = db_b_tracking(beat_activations)

    downbeats = [time for time, _ in beats if _ == 1]

    # save the beat activations and beats for sonic visualizer
    np.savetxt(activation_path, beat_activations, fmt="%.6f")
    np.savetxt(beats_path, beats, fmt="%.6f")
    np.savetxt(downbeats_path, downbeats, fmt="%.6f")

    return beat_activations, beats, downbeats


def inference_madmom_control(audio_path, activation_path=None, beats_path=None):

    # get beat activations
    beat_activations = get_beat_activations(audio_path)
    # beat_activation_path = activation_path
    # beat_activations = np.loadtxt(beat_activation_path)

    # # perform beat tracking (49.667s) for R_38
    # # (0s, 45.033s) – [20, 25] bpm
    # # (45.033s, 70.000s) – [18, 20] bpm
    # # (70.000s, end) – [40, 139] bpm
    # beats_0 = beat_tracking(beat_activations[: (4966 + 4503)], min_bpm=18, max_bpm=25)
    # beats_1 = beat_tracking(
    #     beat_activations[(4966 + 4503) : (4966 + 7000)], min_bpm=18, max_bpm=25
    # )
    # beats_2 = beat_tracking(beat_activations[(4966 + 7000) :], min_bpm=40, max_bpm=139)
    # beats = np.concatenate([beats_0, beats_1 + 49.66 + 45.03, beats_2 + 49.66 + 70.00])

    # perform beat tracking for R_48
    beats = beat_tracking(beat_activations, min_bpm=89, max_bpm=100)

    # save the beat activations and beats for sonic visualizer
    np.savetxt(activation_path, beat_activations, fmt="%.6f")
    np.savetxt(beats_path, beats, fmt="%.6f")

    return beat_activations, beats


def read_txt_beats(file_path):
    """Read the beat annotations from a txt file."""

    with open(file_path, "r") as file:
        annotation = file.read()
    annotation = annotation.split("\n")
    annotation = [line.split("\t") for line in annotation if line]
    beat_annotations = [float(line[0]) for line in annotation]
    beat_annotations = np.array(beat_annotations)

    return beat_annotations


def read_txt_activations(beat_activation_path):
    """Read beat activations from a text file."""
    with open(beat_activation_path, "r") as file:
        return [float(line.strip()) for line in file]


def evaluate_madmom_beats(annotation_file):

    # define the file paths
    annotation_path = os.path.join(ANNOTATION_DIR, annotation_file)
    perf_id = annotation_file.replace("_b.txt", "")
    madmom_beats_path = os.path.join(MADMOM_OUTPUT_DIR, f"{perf_id}_MAPS_crf_b.txt")

    # read the beat annotations
    beat_annotations = read_txt_beats(annotation_path)
    madmom_beats = read_txt_beats(madmom_beats_path)

    print(f"Performance ID: {perf_id}")
    print(f"Beat annotations: {len(beat_annotations)}")
    print(f"Madmom beats: {len(madmom_beats)}")

    # evaluate the beat tracking results
    precision, recall, f_measure, f_measure_mir_eval = evaluate_beats(
        madmom_beats, beat_annotations
    )

    # save the evaluation results
    report_path = os.path.join(MADMOM_OUTPUT_DIR, "evaluation_report_crf_1.txt")
    with open(report_path, "a") as file:
        file.write(
            f"{perf_id}, p={precision:.4f}, r={recall:.4f}, f={f_measure:.4f}, (mir_eval)f={f_measure_mir_eval:.4f}\n"
        )


def plot_beats(
    beat_activations, beats, beat_annotations, beats_modified, file_name="beat_tracking"
):
    """Plot the beat tracking results and annotations as vertical lines."""

    # three subplots sharing both x/y axes
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(100, 10))

    # plot as vertical lines
    axs[0].vlines(beat_annotations, 0, 1, colors="g", label="Annotations")
    axs[0].set_title("Beat Annotations")
    axs[1].vlines(beats, 0, 1, colors="b", label="Beat Tracking")
    axs[1].set_title("Beat Tracking")
    axs[2].vlines(beats_modified, 0, 1, colors="r", label="Modified Beat Tracking")
    axs[2].set_title("Modified Beat Tracking")

    # beat activations: fps = 100
    time = np.arange(0, len(beat_activations) / 100, 0.01)
    axs[3].plot(time, beat_activations, label="Beat Activations")
    axs[3].set_title("Beat Activations")

    # x-axis should have more ticks, 1000 ms = 1 second, 1 second a tick
    axs[0].set_xticks(np.arange(0, max(beats), 2000))
    axs[1].set_xticks(np.arange(0, max(beats), 2000))
    axs[2].set_xticks(np.arange(0, max(beats), 2000))

    plt.tight_layout()

    # save the plot
    plt.savefig(f"{file_name}.png")


def boost_activations_sigmoid(activations, c=0.5, k=8):
    """Boost activations using a sigmoid function for smooth scaling."""
    return 1 / (1 + np.exp(-(activations - c) * k))


def evaluate_modified_beat_activations(beat_activations, beats, perf_id="R_35"):

    annotation_path = os.path.join(ANNOTATION_DIR, f"{perf_id}_b.txt")
    beat_annotations = read_txt_beats(annotation_path)

    # evaluate the beat tracking results
    f_measure = evaluate_beats(beats, beat_annotations)
    print(
        f"Performance ID: {perf_id}, (Original Estimation) F-measure: {f_measure:.4f}"
    )

    # modify the beat activations: correct the beat activations
    # goal: boost big beats and reduce small beats
    beat_activations[11000:12700] = boost_activations_sigmoid(
        beat_activations[11000:12700]
    )
    beat_activations[18300:20400] = boost_activations_sigmoid(
        beat_activations[18300:20400]
    )
    beat_activations[24700:25500] = boost_activations_sigmoid(
        beat_activations[24700:25500]
    )
    beat_activations[26600:28200] = boost_activations_sigmoid(
        beat_activations[26600:28200]
    )
    beat_activations[29500:31500] = boost_activations_sigmoid(
        beat_activations[29500:31500]
    )

    # perform beat tracking
    beats_modified = beat_tracking(beat_activations)

    plot_beats(
        beat_activations,
        beats,
        beat_annotations,
        beats_modified,
        file_name=f"{perf_id}_modified",
    )

    # evaluate the beat tracking results
    f_measure = evaluate_beats(beats_modified, beat_annotations)
    print(f"Performance ID: {perf_id}, (Modified) F-measure: {f_measure:.4f}")

    # # save the beats
    # activation_path = os.path.join(
    #     MADMOM_OUTPUT_DIR, f"{perf_id}_MAPS_act_modified.txt"
    # )
    # beats_path = os.path.join(MADMOM_OUTPUT_DIR, f"{perf_id}_MAPS_b_modified.txt")
    # np.savetxt(activation_path, beat_activations, fmt="%.6f")
    # np.savetxt(beats_path, beats, fmt="%.6f")


def plot_all_annotations(annotations):
    """Plot all annotations in one plot."""

    num_plots = len(annotations)
    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(100, 50))

    for i, annotation_file in enumerate(annotations):
        annotation_path = os.path.join(ANNOTATION_DIR, annotation_file)
        beat_annotations = read_txt_beats(annotation_path)
        axs[i].vlines(beat_annotations, 0, 1, colors="g", label="Annotations")
        axs[i].set_title(f"Beat Annotations ({annotation_file})")

    plt.tight_layout()
    plt.savefig("annotations.png")


if __name__ == "__main__":

    # Inference madmom without any control
    # beat_activations, beats, dbs = inference_madmom_with_db(
    #     "../demo_1205_May/May.mp3",
    #     "../demo_1205_May/May_db_b_activation.txt",
    #     "../demo_1205_May/May_beats.txt",
    #     "../demo_1205_May/May_downbeats.txt",
    # )
    beat_activations, beats = inference_madmom(
        "../demo_1205_May/May.mp3",
        "../demo_1205_May/madmom_only_beats/May_b_activation.txt",
        "../demo_1205_May/madmom_only_beats/May_beats.txt",
    )


#     # Inference madmom with tempo control
#     beat_activations_control, beats_control = inference_madmom_control(
#         "../demo_1106_R_48/R_48_4-4_db_30s.wav",
#         "../demo_1106_R_48/R_48_4-4_db_30s_act_control.txt",
#         "../demo_1106_R_48/R_48_4-4_db_30s_b_control.txt",
#     )

#     # beat_activations_control, beats_control = inference_madmom_control(
#     #     "../dataset/audio/R_48_MAPS.wav",
#     #     "../demo_1106_R_48/R_48_2-4_db_30s_act_control.txt",
#     #     "../demo_1106_R_48/R_48_2-4_db_30s_b_control.txt",
#     # )
