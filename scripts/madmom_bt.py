"""Beat tracking using madmom library."""

import os
import numpy as np
from tqdm import tqdm
import madmom
import mir_eval
import matplotlib.pyplot as plt
from dirs import MADMOM_OUTPUT_DIR, AUDIO_DIR, ANNOTATION_DIR

proc = madmom.features.beats.RNNBeatProcessor()
beat_tracker = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)


def get_beat_activations(file_path):
    """Get beat activations using madmom."""

    # Step 1: Process the audio file to detect beat activations
    beat_activations = proc(file_path)

    return beat_activations


def beat_tracking(beat_activations):
    """Perform beat tracking on the beat activations."""

    # Step 2: Track beats based on the beat activations
    print("Tracking beats...")
    beats = beat_tracker(beat_activations)

    return beats


import numpy as np


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


def read_txt_beats(file_path):
    """Read the beat annotations from a txt file."""

    with open(file_path, "r") as file:
        annotation = file.read()
    annotation = annotation.split("\n")
    annotation = [line.split("\t") for line in annotation if line]
    beat_annotations = [float(line[0]) for line in annotation]
    beat_annotations = np.array(beat_annotations)

    return beat_annotations


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

    # audio_files = os.listdir(AUDIO_DIR)
    # audio_files = [file for file in audio_files if file.endswith(".wav")]

    # annotations = os.listdir(ANNOTATION_DIR)
    # annotations = [file for file in annotations if file.endswith("_b.txt")]

    # for audio_file in tqdm(audio_files):
    #     audio_path = os.path.join(AUDIO_DIR, audio_file)
    #     activation_path = os.path.join(
    #         MADMOM_OUTPUT_DIR, audio_file.replace(".wav", "_crf_act.txt")
    #     )
    #     beats_path = os.path.join(
    #         MADMOM_OUTPUT_DIR, audio_file.replace(".wav", "_crf_b.txt")
    #     )
    #     beat_activations, beats = inference_madmom(
    #         audio_path, activation_path, beats_path
    #     )

    #     evaluate_madmom_beats(audio_file.replace("_MAPS.wav", "_b.txt"))

    # evaluate_madmom_beats()
    # evaluate_modified_beat_activations(beat_activations, beats)

    beat_activations, beats = inference_madmom(
        "../demo_1106_R_38/R_38_cut.wav",
        "../demo_1106_R_38/R_38_cut_act.txt",
        "../demo_1106_R_38/R_38_cut_b.txt",
    )
