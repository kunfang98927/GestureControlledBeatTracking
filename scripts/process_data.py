import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN

MIN_BPM = 20
MAX_BPM = 300


def load_gesture_data(filepath):
    """Load gesture data from a JSON file."""
    with open(filepath, "r") as file:
        return json.load(file)


def parse_gestures(results, beat_confidence_threshold=0.5):
    """Parse gestures from results and return beat positions and speed adjustments."""
    beat_positions = []
    confidences = []

    for frame in results:
        timestamp = frame["frame_timestamp_ms"]
        for gesture in frame["gestures"]:
            if (
                gesture["gesture_category"] == "Pointing_Up"
                and gesture["gesture_score"] > beat_confidence_threshold
            ):
                beat_positions.append(timestamp)
                confidences.append(gesture["gesture_score"])

    return beat_positions, confidences


def plot_smoothed_beats(timestamps, confidences, smoothed_timestamps):
    """Plot original confidence scores, original beat positions, and smoothed beat positions."""
    plt.figure(figsize=(10, 3))

    # Scatter plot for confidence scores
    plt.scatter(timestamps, confidences, color="gray", label="Raw Gesture", alpha=0.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Confidence Score")
    plt.title("Gesture Confidence and Beat Positions Over Time")
    plt.xticks(np.arange(0, max(timestamps), 2000))
    plt.ylim(0, 1)  # Assuming confidence is normalized between 0 and 1

    plt.vlines(
        smoothed_timestamps,
        0,
        1,
        color="blue",
        linestyle="--",
        alpha=0.8,
        label="Beats",
    )

    # Labels and legend
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.tight_layout()
    # save the plot
    plt.savefig("../demo_1106/extracted_beats.png")


def find_gathering_points(timestamps, eps=100, min_samples=2):
    """
    Find gathering points using DBSCAN clustering to identify dense regions of gestures.
    Parameters:
        timestamps: Array of gesture timestamps.
        eps: Maximum distance (ms) between points in a cluster.
        min_samples: Minimum number of points to form a cluster.
    Returns:
        Array of cluster center timestamps representing the detected beat positions.
    """
    # Filter gestures by confidence threshold
    filtered_timestamps = np.array([t for t in timestamps if t > 0])

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
        beat_position = np.median(cluster_points)  # Use median as a stable measure
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


def visualize_beat_positions_and_tempo(
    gesture_positions,
    confidences,
    beat_positions,
    smoothed_tempo_contour,
    madmom_beats,
    madmom_tempo_contour,
    beat_annotations=None,
    annotation_tempo_contour=None,
):
    """Visualize beat positions with the tempo contour."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    # Plot beat positions as vertical lines
    axs[0].vlines(
        beat_positions,
        0,
        1,
        color="blue",
        linestyle="--",
        alpha=0.8,
        label="Beats (Gesture)",
    )
    axs[0].vlines(
        madmom_beats,
        0,
        1,
        color="red",
        linestyle="--",
        alpha=0.8,
        label="Beats (Madmom)",
    )
    axs[0].scatter(
        gesture_positions,
        confidences,
        color="gray",
        label="Raw Gesture",
        alpha=0.6,
        s=5,
    )
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Confidence Score")
    axs[0].set_title(
        "Gesture Confidence and Beat Positions (Blue: Gesture, Red: Madmom)"
    )
    # place the legend outside the plot
    axs[0].legend(loc="lower right", fontsize="x-small")

    # Plot original and smoothed tempo contour
    axs[1].plot(
        beat_positions[1:],
        smoothed_tempo_contour,
        label="Tempo (Gesture)",
        linestyle="-",
        marker="o",
        alpha=0.5,
        markersize=5,
        color="blue",
    )
    axs[1].plot(
        madmom_beats[1:],
        madmom_tempo_contour,
        label="Tempo (Madmom)",
        linestyle="-",
        marker="o",
        alpha=0.5,
        markersize=5,
        color="red",
    )
    if beat_annotations is not None and annotation_tempo_contour is not None:
        axs[1].plot(
            beat_annotations[1:],
            annotation_tempo_contour,
            label="Tempo (Annotation)",
            linestyle="-",
            marker="o",
            alpha=0.5,
            markersize=5,
            color="green",
        )
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Tempo (BPM)")
    axs[1].set_title("Tempo Contour")
    axs[1].legend()

    # x-axis should have more ticks, 1000 ms = 1 second, 1 second a tick
    axs[0].set_xticks(np.arange(0, max(madmom_beats), 2000))
    axs[1].set_xticks(np.arange(0, max(madmom_beats), 2000))

    plt.tight_layout()

    # save the plot
    plt.savefig("../demo_1106/beat_tempo.png")


if __name__ == "__main__":

    # Define the file path
    filepath = "../demo_1106/R_38_cut_ges.json"
    madmom_beats_path = "../demo_1106/madmom_beats.npy"
    annotation_path = "../demo_1106/annotation.tsv"

    # Load and parse data
    data = load_gesture_data(filepath)
    results = data["results"]

    # Get the maximum time of the gesture data
    max_time_gesture = results[-1]["frame_timestamp_ms"]

    # Read the annotation file
    with open(annotation_path, "r") as file:
        annotation = file.read()
    annotation = annotation.split("\n")
    annotation = [line.split("\t") for line in annotation if line]
    beat_annotations = [float(line[0]) * 1000 for line in annotation]
    beat_annotations = np.array(beat_annotations)
    beat_annotations = beat_annotations[beat_annotations < max_time_gesture]

    # Load the madmom beat positions
    madmom_beats = np.load(madmom_beats_path)
    madmom_beats = madmom_beats * 1000  # Convert to milliseconds
    madmom_beats = madmom_beats[madmom_beats < max_time_gesture]
    madmom_tempo_contour = 60000 / np.diff(
        madmom_beats
    )  # BPM = 60,000 ms / beat interval
    madmom_smoothed_tempo_contour = smooth_tempo_contour(
        madmom_tempo_contour, window_size=3
    )

    # Parse gestures to find beat positions and speed adjustments
    gesture_positions, confidences = parse_gestures(results)
    beat_positions = find_gathering_points(gesture_positions, eps=100, min_samples=2)
    # plot_smoothed_beats(gesture_positions, confidences, beat_positions)

    # Clean up the beat positions
    tempo_contour = 60000 / np.diff(beat_positions)  # BPM = 60,000 ms / beat interval
    smoothed_tempo_contour = smooth_tempo_contour(tempo_contour, window_size=3)

    # Annotation tempo contour
    annotation_tempo_contour = 60000 / np.diff(beat_annotations)
    annotation_smoothed_tempo_contour = smooth_tempo_contour(
        annotation_tempo_contour, window_size=3
    )

    # Print the results
    print("Estimated Beat Positions (ms):", beat_positions)
    print("Smoothed Tempo Contour (BPM):", smoothed_tempo_contour)

    visualize_beat_positions_and_tempo(
        gesture_positions,
        confidences,
        beat_positions,
        smoothed_tempo_contour,
        madmom_beats,
        madmom_tempo_contour,
        beat_annotations=None,
        annotation_tempo_contour=None,
    )

    # save the beat positions in txt file
    np.savetxt("../demo_1106/extracted_beats.txt", beat_positions / 1000)
