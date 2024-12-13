import matplotlib.pyplot as plt
import numpy as np


def visualize_with_db(
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
    start_time=0,
    end_time=0,
    image_path="output.png",
):
    """Visualize beat positions with the tempo contour."""
    fig, axs = plt.subplots(9, 1, figsize=(20, 18), sharex=True)

    if len(left_gesture_positions) != 0:
        gesture_timestamps_l = [t for _, t, _ in left_gesture_positions]
        confidences_l = [c for _, _, c in left_gesture_positions]
        beat_types_l = [t for t, _, _ in left_gesture_positions]
        gesture_timestamps_l = np.array(gesture_timestamps_l)

        gesture_timestamps_b_l = gesture_timestamps_l[np.array(beat_types_l) == "b"]
        gesture_timestamps_db_l = gesture_timestamps_l[np.array(beat_types_l) == "db"]
        confidences_b_l = np.array(confidences_l)[np.array(beat_types_l) == "b"]
        confidences_db_l = np.array(confidences_l)[np.array(beat_types_l) == "db"]

    if len(right_gesture_positions) != 0:
        gesture_timestamps_r = [t for _, t, _ in right_gesture_positions]
        confidences_r = [c for _, _, c in right_gesture_positions]
        beat_types_r = [t for t, _, _ in right_gesture_positions]
        gesture_timestamps_r = np.array(gesture_timestamps_r)

        gesture_timestamps_b_r = gesture_timestamps_r[np.array(beat_types_r) == "b"]
        gesture_timestamps_db_r = gesture_timestamps_r[np.array(beat_types_r) == "db"]
        confidences_b_r = np.array(confidences_r)[np.array(beat_types_r) == "b"]
        confidences_db_r = np.array(confidences_r)[np.array(beat_types_r) == "db"]

    # Plot raw gesture with confidence and tempo changes
    if len(left_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_l / 1000,  # Convert to seconds
            confidences_b_l,
            color="gray",
            label="Raw Gesture (Beat; Left)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_l / 1000,  # Convert to seconds
            confidences_db_l,
            color="purple",
            label="Raw Gesture (Downbeat; Left)",
            alpha=0.6,
            s=5,
        )
    if len(right_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_r / 1000,  # Convert to seconds
            confidences_b_r,
            color="blue",
            label="Raw Gesture (Beat; Right)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_r / 1000,  # Convert to seconds
            confidences_db_r,
            color="green",
            label="Raw Gesture (Downbeat; Right)",
            alpha=0.6,
            s=5,
        )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "up"],
        0,
        1,
        color="orange",
        alpha=1.0,
        linewidth=3,
        label="Tempo Up",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "down"],
        0,
        1,
        color="red",
        alpha=0.7,
        linewidth=3,
        label="Tempo Down",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "cut"],
        0,
        1,
        color="black",
        alpha=0.7,
        linewidth=3,
        label="Tempo Cut",
    )
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Confidence")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Gesture with Confidence and Tempo changes")
    # place the legend outside the plot
    axs[0].legend(loc="lower right", fontsize="x-small")
    axs[0].grid(True)

    # Plot original and smoothed tempo contour
    axs[1].vlines(
        np.array(left_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="grey",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(right_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="blue",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(left_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="purple",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(right_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="green",
        linestyle="--",
        alpha=0.8,
    )

    axs[2].vlines(
        filtered_beat_positions / 1000,  # Convert to seconds
        0,
        max(smoothed_tempo_contour) + 20,
        color="blue",
        linestyle="--",
        alpha=0.8,
        label="Filtered beats",
    )
    axs[2].plot(
        filtered_beat_positions[1:] / 1000,  # Convert to seconds
        smoothed_tempo_contour,
        label="Filtered tempo (BPM)",
        linestyle="-",
        marker="o",
        alpha=0.5,
        markersize=5,
        color="purple",
    )
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Extracted Beat Positions")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Tempo (BPM)")
    axs[2].set_yticks(np.arange(0, 160, 20))
    axs[2].set_title("Filtered Beat Positions and Tempo Contour")
    axs[2].legend(fontsize="x-small")
    axs[2].grid(True)
    # x-axis ticks every 10 seconds
    axs[0].set_xticks(
        np.arange(
            start_time,
            end_time + 10,
            10,
        )
    )

    # Plot smoothed beat peaks
    axs[3].plot(
        np.arange(len(beat_peaks)) / 100 + start_time,
        beat_peaks,
        label="Smoothed Beat Peaks",
        color="blue",
    )
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Peak")
    axs[3].set_title("Smoothed Beat Peaks")
    axs[3].legend(fontsize="x-small")
    axs[3].grid(True)

    # Plot smoothed downbeat_peaks
    axs[4].plot(
        np.arange(len(downbeat_peaks)) / 100 + start_time,
        downbeat_peaks,
        label="Smoothed Downbeat Peaks",
        color="green",
    )
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Peak")
    axs[4].set_title("Smoothed Downbeat Peaks")
    axs[4].legend(fontsize="x-small")
    axs[4].grid(True)

    # Plot beat activations
    axs[5].plot(
        np.arange(len(beat_activations)) / 100 + start_time,  # fps for madmom is 100
        beat_activations,
        label="Beat Activations",
        color="grey",
    )
    axs[5].vlines(
        madmom_beats,
        0,
        max(beat_activations) + 0.1,
        color="blue",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Madmom Beat Positions",
    )
    axs[5].set_xlabel("Time (s)")
    axs[5].set_ylabel("Activation")
    axs[5].set_title("Beat Activations")
    axs[5].legend(fontsize="x-small")
    axs[5].grid(True)

    # Plot downbeat activations
    axs[7].plot(
        np.arange(len(downbeat_activations)) / 100
        + start_time,  # fps for madmom is 100
        downbeat_activations,
        label="Downbeat Activations",
        color="grey",
    )
    axs[7].vlines(
        madmom_downbeats,
        0,
        max(downbeat_activations) + 0.1,
        color="green",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Madmom Downbeat Positions",
    )
    axs[7].set_xlabel("Time (s)")
    axs[7].set_ylabel("Activation")
    axs[7].set_title("Downbeat Activations")
    axs[7].legend(fontsize="x-small")
    axs[7].grid(True)

    # Plot modified beat activations
    axs[6].plot(
        np.arange(len(modified_beat_activations)) / 100 + start_time,
        modified_beat_activations,
        label="Modified Beat Activations",
        color="red",
    )
    axs[6].vlines(
        modified_beat_positions,
        0,
        max(modified_beat_activations) + 0.1,
        color="purple",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Modified Beat Positions",
    )
    axs[6].set_xlabel("Time (s)")
    axs[6].set_ylabel("Activation")
    axs[6].set_title("Beat Activations Modified by Gesture")
    axs[6].legend(fontsize="x-small")
    axs[6].grid(True)

    # Plot modified downbeat activations
    axs[8].plot(
        np.arange(len(modified_downbeat_activations)) / 100 + start_time,
        modified_downbeat_activations,
        label="Modified Downbeat Activations",
        color="red",
    )
    axs[8].vlines(
        modified_downbeat_positions,
        0,
        max(modified_downbeat_activations) + 0.1,
        color="purple",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Modified Downbeat Positions",
    )
    axs[8].set_xlabel("Time (s)")
    axs[8].set_ylabel("Activation")
    axs[8].set_title("Downbeat Activations Modified by Gesture")
    axs[8].legend(fontsize="x-small")
    axs[8].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig(image_path)


def visualize_with_db_for_slide(
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
    start_time=0,
    end_time=0,
    image_path="output.png",
):
    """Visualize beat positions with the tempo contour."""
    fig, axs = plt.subplots(14, 1, figsize=(30, 18), sharex=True)

    if len(left_gesture_positions) != 0:
        gesture_timestamps_l = [t for _, t, _ in left_gesture_positions]
        confidences_l = [c for _, _, c in left_gesture_positions]
        beat_types_l = [t for t, _, _ in left_gesture_positions]
        gesture_timestamps_l = np.array(gesture_timestamps_l)

        gesture_timestamps_b_l = gesture_timestamps_l[np.array(beat_types_l) == "b"]
        gesture_timestamps_db_l = gesture_timestamps_l[np.array(beat_types_l) == "db"]
        confidences_b_l = np.array(confidences_l)[np.array(beat_types_l) == "b"]
        confidences_db_l = np.array(confidences_l)[np.array(beat_types_l) == "db"]

    if len(right_gesture_positions) != 0:
        gesture_timestamps_r = [t for _, t, _ in right_gesture_positions]
        confidences_r = [c for _, _, c in right_gesture_positions]
        beat_types_r = [t for t, _, _ in right_gesture_positions]
        gesture_timestamps_r = np.array(gesture_timestamps_r)

        gesture_timestamps_b_r = gesture_timestamps_r[np.array(beat_types_r) == "b"]
        gesture_timestamps_db_r = gesture_timestamps_r[np.array(beat_types_r) == "db"]
        confidences_b_r = np.array(confidences_r)[np.array(beat_types_r) == "b"]
        confidences_db_r = np.array(confidences_r)[np.array(beat_types_r) == "db"]

    # Plot raw gesture with confidence and tempo changes
    if len(left_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_l / 1000,  # Convert to seconds
            confidences_b_l,
            color="gray",
            label="Raw Gesture (Beat; Left)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_l / 1000,  # Convert to seconds
            confidences_db_l,
            color="purple",
            label="Raw Gesture (Downbeat; Left)",
            alpha=0.6,
            s=5,
        )
    if len(right_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_r / 1000,  # Convert to seconds
            confidences_b_r,
            color="blue",
            label="Raw Gesture (Beat; Right)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_r / 1000,  # Convert to seconds
            confidences_db_r,
            color="green",
            label="Raw Gesture (Downbeat; Right)",
            alpha=0.6,
            s=5,
        )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "up"],
        0,
        1,
        color="orange",
        alpha=1.0,
        linewidth=3,
        label="Tempo Up",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "down"],
        0,
        1,
        color="red",
        alpha=0.7,
        linewidth=3,
        label="Tempo Down",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "cut"],
        0,
        1,
        color="black",
        alpha=0.7,
        linewidth=3,
        label="Tempo Cut",
    )
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Confidence")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Gesture with Confidence and Tempo changes")
    # place the legend outside the plot
    axs[0].legend(loc="lower right", fontsize="x-small")
    axs[0].grid(True)

    # Plot original and smoothed tempo contour
    axs[1].vlines(
        np.array(left_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="grey",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(right_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="blue",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(left_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="purple",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(right_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="green",
        linestyle="--",
        alpha=0.8,
    )

    axs[2].vlines(
        filtered_beat_positions / 1000,  # Convert to seconds
        0,
        max(smoothed_tempo_contour) + 20,
        color="blue",
        linestyle="--",
        alpha=0.8,
        label="Filtered beats",
    )
    axs[3].plot(
        filtered_beat_positions[1:] / 1000,  # Convert to seconds
        smoothed_tempo_contour[: len(filtered_beat_positions) - 1],
        label="Filtered tempo (BPM)",
        linestyle="-",
        marker="o",
        alpha=0.5,
        markersize=5,
        color="purple",
    )
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Extracted Beat Positions")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(fontsize="x-small")
    axs[2].grid(True)

    axs[3].set_title("Filtered Beat Positions")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Tempo (BPM)")
    axs[3].set_yticks(np.arange(0, 160, 20))
    axs[3].set_title("Tempo Contour")
    axs[3].legend(fontsize="x-small")
    axs[3].grid(True)
    # x-axis ticks every 10 seconds
    axs[0].set_xticks(
        np.arange(
            start_time,
            end_time + 10,
            10,
        )
    )

    # Plot smoothed beat peaks
    axs[4].plot(
        np.arange(len(beat_peaks)) / 100 + start_time,
        beat_peaks,
        label="Smoothed Beat Peaks",
        color="blue",
    )
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Peak")
    axs[4].set_title("Smoothed Beat Peaks")
    axs[4].legend(fontsize="x-small")
    axs[4].grid(True)

    # Plot smoothed downbeat_peaks
    axs[5].plot(
        np.arange(len(downbeat_peaks)) / 100 + start_time,
        downbeat_peaks,
        label="Smoothed Downbeat Peaks",
        color="green",
    )
    axs[5].set_xlabel("Time (s)")
    axs[5].set_ylabel("Peak")
    axs[5].set_title("Smoothed Downbeat Peaks")
    axs[5].legend(fontsize="x-small")
    axs[5].grid(True)

    # Plot beat activations
    axs[6].plot(
        np.arange(len(beat_activations)) / 100 + start_time,  # fps for madmom is 100
        beat_activations,
        label="Madmom Beat Activations",
        color="grey",
    )
    axs[6].set_xlabel("Time (s)")
    axs[6].set_ylabel("Activation")
    axs[6].set_title("Madmom Beat Activations")
    axs[6].legend(fontsize="x-small")
    axs[6].grid(True)

    axs[7].vlines(
        madmom_beats,
        0,
        max(beat_activations) + 0.1,
        color="blue",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Madmom Beat Positions",
    )
    axs[7].set_xlabel("Time (s)")
    axs[7].set_title("Madmom Beat Positions")
    axs[7].legend(fontsize="x-small")
    axs[7].grid(True)

    # Plot modified beat activations
    axs[8].plot(
        np.arange(len(modified_beat_activations)) / 100 + start_time,
        modified_beat_activations,
        label="Modified Beat Activations",
        color="red",
    )
    axs[8].set_xlabel("Time (s)")
    axs[8].set_ylabel("Activation")
    axs[8].set_title("Beat Activations Modified by Gesture")
    axs[8].legend(fontsize="x-small")
    axs[8].grid(True)

    axs[9].vlines(
        modified_beat_positions,
        0,
        max(modified_beat_activations) + 0.1,
        color="purple",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Modified Beat Positions",
    )
    axs[9].set_xlabel("Time (s)")
    axs[9].set_ylabel("Activation")
    axs[9].set_title("Beat Positions Modified by Gesture")
    axs[9].legend(fontsize="x-small")
    axs[9].grid(True)

    # Plot downbeat activations
    axs[10].plot(
        np.arange(len(downbeat_activations)) / 100
        + start_time,  # fps for madmom is 100
        downbeat_activations,
        label="Madmom Downbeat Activations",
        color="grey",
    )
    axs[10].set_xlabel("Time (s)")
    axs[10].set_ylabel("Activation")
    axs[10].set_title("Madmom Downbeat Activations")
    axs[10].legend(fontsize="x-small")
    axs[10].grid(True)

    axs[11].vlines(
        madmom_downbeats,
        0,
        max(downbeat_activations) + 0.1,
        color="green",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Madmom Downbeat Positions",
    )
    axs[11].set_xlabel("Time (s)")
    axs[11].set_ylabel("Activation")
    axs[11].set_title("Madmom Downbeat Activations")
    axs[11].legend(fontsize="x-small")
    axs[11].grid(True)

    # Plot modified downbeat activations
    axs[12].plot(
        np.arange(len(modified_downbeat_activations)) / 100 + start_time,
        modified_downbeat_activations,
        label="Modified Downbeat Activations",
        color="red",
    )
    axs[12].set_xlabel("Time (s)")
    axs[12].set_ylabel("Activation")
    axs[12].set_title("Downbeat Activations Modified by Gesture")
    axs[12].legend(fontsize="x-small")
    axs[12].grid(True)

    axs[13].vlines(
        modified_downbeat_positions,
        0,
        max(modified_downbeat_activations) + 0.1,
        color="purple",
        linestyle="--",
        alpha=0.8,
        linewidth=1.0,
        label="Modified Downbeat Positions",
    )
    axs[13].set_xlabel("Time (s)")
    axs[13].set_ylabel("Activation")
    axs[13].set_title("Downbeat Activations Modified by Gesture")
    axs[13].legend(fontsize="x-small")
    axs[13].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig(image_path)


def visualize(
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
    start_time=0,
    end_time=0,
    image_path="output.png",
):
    """Visualize beat positions with the tempo contour."""
    fig, axs = plt.subplots(6, 1, figsize=(20, 10), sharex=True)

    if len(left_gesture_positions) != 0:
        gesture_timestamps_l = [t for _, t, _ in left_gesture_positions]
        confidences_l = [c for _, _, c in left_gesture_positions]
        beat_types_l = [t for t, _, _ in left_gesture_positions]
        gesture_timestamps_l = np.array(gesture_timestamps_l)

        gesture_timestamps_b_l = gesture_timestamps_l[np.array(beat_types_l) == "b"]
        gesture_timestamps_db_l = gesture_timestamps_l[np.array(beat_types_l) == "db"]
        confidences_b_l = np.array(confidences_l)[np.array(beat_types_l) == "b"]
        confidences_db_l = np.array(confidences_l)[np.array(beat_types_l) == "db"]

    if len(right_gesture_positions) != 0:
        gesture_timestamps_r = [t for _, t, _ in right_gesture_positions]
        confidences_r = [c for _, _, c in right_gesture_positions]
        beat_types_r = [t for t, _, _ in right_gesture_positions]
        gesture_timestamps_r = np.array(gesture_timestamps_r)

        gesture_timestamps_b_r = gesture_timestamps_r[np.array(beat_types_r) == "b"]
        gesture_timestamps_db_r = gesture_timestamps_r[np.array(beat_types_r) == "db"]
        confidences_b_r = np.array(confidences_r)[np.array(beat_types_r) == "b"]
        confidences_db_r = np.array(confidences_r)[np.array(beat_types_r) == "db"]

    # Plot raw gesture with confidence and tempo changes
    if len(left_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_l / 1000,  # Convert to seconds
            confidences_b_l,
            color="gray",
            label="Raw Gesture (Beat; Left)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_l / 1000,  # Convert to seconds
            confidences_db_l,
            color="purple",
            label="Raw Gesture (Downbeat; Left)",
            alpha=0.6,
            s=5,
        )
    if len(right_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_r / 1000,  # Convert to seconds
            confidences_b_r,
            color="blue",
            label="Raw Gesture (Beat; Right)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_r / 1000,  # Convert to seconds
            confidences_db_r,
            color="green",
            label="Raw Gesture (Downbeat; Right)",
            alpha=0.6,
            s=5,
        )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "up"],
        0,
        1,
        color="orange",
        alpha=1.0,
        linewidth=3,
        label="Tempo Up",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "down"],
        0,
        1,
        color="red",
        alpha=0.7,
        linewidth=3,
        label="Tempo Down",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "cut"],
        0,
        1,
        color="black",
        alpha=0.7,
        linewidth=3,
        label="Tempo Cut",
    )
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Confidence")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Gesture with Confidence and Tempo changes")
    # place the legend outside the plot
    axs[0].legend(loc="lower right", fontsize="x-small")
    axs[0].grid(True)

    # Plot original and smoothed tempo contour
    axs[1].vlines(
        np.array(left_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="grey",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(right_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="blue",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(left_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="purple",
        linestyle="--",
        alpha=0.8,
    )
    axs[1].vlines(
        np.array(right_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="green",
        linestyle="--",
        alpha=0.8,
    )

    axs[2].vlines(
        filtered_beat_positions / 1000,  # Convert to seconds
        0,
        max(smoothed_tempo_contour) + 20,
        color="green",
        linestyle="--",
        alpha=0.8,
        label="Filtered beats",
    )
    axs[2].plot(
        filtered_beat_positions[1:] / 1000,  # Convert to seconds
        smoothed_tempo_contour,
        label="Filtered tempo (BPM)",
        linestyle="-",
        marker="o",
        alpha=0.5,
        markersize=5,
        color="purple",
    )
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Extracted Beat Positions")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Tempo (BPM)")
    axs[2].set_yticks(np.arange(0, 160, 20))
    axs[2].set_title("Filtered Beat Positions and Tempo Contour")
    axs[2].legend(fontsize="x-small")
    axs[2].grid(True)
    # x-axis ticks every 10 seconds
    axs[0].set_xticks(
        np.arange(
            start_time,
            end_time + 10,
            10,
        )
    )

    # Plot smoothed beat peaks
    axs[3].plot(
        np.arange(len(beat_peaks)) / 100 + start_time,
        beat_peaks,
        label="Smoothed Beat Peaks",
        color="red",
    )
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Peak")
    axs[3].set_title("Smoothed Beat Peaks")
    axs[3].legend(fontsize="x-small")
    axs[3].grid(True)

    # Plot beat activations
    axs[4].plot(
        np.arange(len(beat_activations)) / 100 + start_time,  # fps for madmom is 100
        beat_activations,
        label="Beat Activations",
        color="green",
    )
    axs[4].vlines(
        madmom_beat_positions,
        0,
        max(beat_activations) + 0.1,
        color="blue",
        linestyle="--",
        alpha=0.8,
        linewidth=0.7,
        label="Madmom Beat Positions",
    )
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Activation")
    axs[4].set_title("Beat Activations")
    axs[4].legend(fontsize="x-small")
    axs[4].grid(True)

    # Plot modified beat activations
    axs[5].plot(
        np.arange(len(modified_beat_activations)) / 100 + start_time,
        modified_beat_activations,
        label="Modified Beat Activations",
        color="red",
    )
    axs[5].vlines(
        modified_beat_positions,
        0,
        max(modified_beat_activations) + 0.1,
        color="purple",
        linestyle="--",
        alpha=0.8,
        linewidth=0.7,
        label="Modified Beat Positions",
    )
    axs[5].set_xlabel("Time (s)")
    axs[5].set_ylabel("Activation")
    axs[5].set_title("Beat Activations Modified by Gesture")
    axs[5].legend(fontsize="x-small")
    axs[5].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig(image_path)


def visualize_beats_for_slide(
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
    start_time=0,
    end_time=0,
    image_path="output.png",
):
    """Visualize beat positions with the tempo contour."""
    fig, axs = plt.subplots(9, 1, figsize=(15, 12), sharex=True)

    if len(left_gesture_positions) != 0:
        gesture_timestamps_l = [t for _, t, _ in left_gesture_positions]
        confidences_l = [c for _, _, c in left_gesture_positions]
        beat_types_l = [t for t, _, _ in left_gesture_positions]
        gesture_timestamps_l = np.array(gesture_timestamps_l)

        gesture_timestamps_b_l = gesture_timestamps_l[np.array(beat_types_l) == "b"]
        gesture_timestamps_db_l = gesture_timestamps_l[np.array(beat_types_l) == "db"]
        confidences_b_l = np.array(confidences_l)[np.array(beat_types_l) == "b"]
        confidences_db_l = np.array(confidences_l)[np.array(beat_types_l) == "db"]

    if len(right_gesture_positions) != 0:
        gesture_timestamps_r = [t for _, t, _ in right_gesture_positions]
        confidences_r = [c for _, _, c in right_gesture_positions]
        beat_types_r = [t for t, _, _ in right_gesture_positions]
        gesture_timestamps_r = np.array(gesture_timestamps_r)

        gesture_timestamps_b_r = gesture_timestamps_r[np.array(beat_types_r) == "b"]
        gesture_timestamps_db_r = gesture_timestamps_r[np.array(beat_types_r) == "db"]
        confidences_b_r = np.array(confidences_r)[np.array(beat_types_r) == "b"]
        confidences_db_r = np.array(confidences_r)[np.array(beat_types_r) == "db"]

    # Plot raw gesture with confidence and tempo changes
    if len(left_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_l / 1000,  # Convert to seconds
            confidences_b_l,
            color="gray",
            label="Raw Gesture (Beat; Left)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_l / 1000,  # Convert to seconds
            confidences_db_l,
            color="purple",
            label="Raw Gesture (Downbeat; Left)",
            alpha=0.6,
            s=5,
        )
    if len(right_gesture_positions) != 0:
        axs[0].scatter(
            gesture_timestamps_b_r / 1000,  # Convert to seconds
            confidences_b_r,
            color="blue",
            label="Raw Gesture (Beat; Right)",
            alpha=0.6,
            s=5,
        )
        axs[0].scatter(
            gesture_timestamps_db_r / 1000,  # Convert to seconds
            confidences_db_r,
            color="green",
            label="Raw Gesture (Downbeat; Right)",
            alpha=0.6,
            s=5,
        )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "up"],
        0,
        1,
        color="orange",
        alpha=1.0,
        linewidth=3,
        label="Tempo Up",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "down"],
        0,
        1,
        color="red",
        alpha=0.7,
        linewidth=3,
        label="Tempo Down",
    )
    axs[0].vlines(
        [t / 1000 for _, t in tempo_changes if _ == "cut"],
        0,
        1,
        color="black",
        alpha=0.7,
        linewidth=3,
        label="Tempo Cut",
    )
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Confidence")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Gesture with Confidence and Tempo changes")
    # place the legend outside the plot
    axs[0].legend(loc="lower right", fontsize="x-small")
    axs[0].grid(True)

    # Plot original and smoothed tempo contour
    axs[1].vlines(
        np.array(left_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="grey",
        linestyle="--",
        alpha=0.8,
        linewidth=3.0,
    )
    axs[1].vlines(
        np.array(right_beat_positions) / 1000,  # Convert to seconds
        0,
        0.6,
        color="blue",
        linestyle="--",
        alpha=0.8,
        linewidth=3.0,
    )
    axs[1].vlines(
        np.array(left_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="purple",
        linestyle="--",
        alpha=0.8,
        linewidth=3.0,
    )
    axs[1].vlines(
        np.array(right_db_positions) / 1000,  # Convert to seconds
        0,
        1,
        color="green",
        linestyle="--",
        alpha=0.8,
        linewidth=3.0,
    )

    axs[2].vlines(
        filtered_beat_positions / 1000,  # Convert to seconds
        0,
        max(smoothed_tempo_contour) + 20,
        color="green",
        linestyle="--",
        alpha=0.8,
        label="Filtered beats",
        linewidth=3.0,
    )
    axs[3].plot(
        filtered_beat_positions[1:] / 1000,  # Convert to seconds
        smoothed_tempo_contour,
        label="Filtered tempo (BPM)",
        linestyle="-",
        marker="o",
        alpha=0.5,
        markersize=8,
        color="purple",
        linewidth=2.0,
    )
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Extracted Beat Positions")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Tempo (BPM)")
    axs[3].set_yticks(np.arange(0, 160, 40))
    axs[2].set_title("Filtered Beat Positions by Gestures")
    axs[3].set_title("Filtered Tempo Contour by Gestures")
    axs[3].legend(fontsize="x-small")
    axs[3].grid(True)
    # x-axis ticks every 10 seconds
    axs[0].set_xticks(
        np.arange(
            start_time,
            end_time + 10,
            10,
        )
    )

    # Plot smoothed beat peaks
    axs[4].plot(
        np.arange(len(beat_peaks)) / 100 + start_time,
        beat_peaks,
        label="Smoothed Beat Peaks",
        color="red",
        linewidth=1.0,
    )
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Peak")
    axs[4].set_title("Smoothed Beat Peaks")
    axs[4].legend(fontsize="x-small")
    axs[4].grid(True)

    # Plot beat activations
    axs[5].plot(
        np.arange(len(beat_activations)) / 100 + start_time,  # fps for madmom is 100
        beat_activations,
        label="Madmom Beat Activations",
        color="green",
        linewidth=1.0,
    )
    axs[6].vlines(
        madmom_beat_positions,
        0,
        max(beat_activations) + 0.1,
        color="blue",
        linestyle="--",
        alpha=0.8,
        linewidth=3.0,
        label="Madmom Beat Positions",
    )
    axs[5].set_xlabel("Time (s)")
    axs[5].set_ylabel("Activation")
    axs[5].set_title("Beat Activations by Madmom")
    axs[5].legend(fontsize="x-small")
    axs[5].grid(True)
    axs[6].set_xlabel("Time (s)")
    axs[6].set_title("Beat Positions by Madmom")
    axs[6].grid(True)

    # Plot modified beat activations
    axs[7].plot(
        np.arange(len(modified_beat_activations)) / 100 + start_time,
        modified_beat_activations,
        label="Modified Beat Activations",
        color="red",
        linewidth=1.0,
    )
    axs[8].vlines(
        modified_beat_positions,
        0,
        max(modified_beat_activations) + 0.1,
        color="purple",
        linestyle="--",
        alpha=0.8,
        linewidth=3.0,
        label="Modified Beat Positions",
    )
    axs[7].set_xlabel("Time (s)")
    axs[7].set_ylabel("Activation")
    axs[7].set_title("Beat Activations Modified by Gesture")
    axs[7].legend(fontsize="x-small")
    axs[7].grid(True)
    axs[8].set_xlabel("Time (s)")
    axs[8].set_title("Beat Positions Modified by Gesture")
    axs[8].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig(image_path)
