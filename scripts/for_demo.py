import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    madmom_activation_path = "../dataset/madmom_outputs/R_17_MAPS_act.txt"
    madmom_beat_path = "../dataset/madmom_outputs/R_17_MAPS_b.txt"

    activations = np.loadtxt(madmom_activation_path)
    beats = np.loadtxt(madmom_beat_path)

    print(activations.shape, beats.shape)
    start_time = 20  # seconds
    end_time = 30  # seconds
    activations = activations[start_time * 100 : end_time * 100]
    # 20 - 30 seconds
    beats = beats[(beats >= start_time) & (beats <= end_time)]

    # plot the activations
    plt.figure(figsize=(6, 2))
    plt.plot(activations, label="Beat activations")
    plt.title("Beat activations")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xticks(
        np.arange(0, len(activations) + 100, 100),
        np.arange(start_time, end_time + 1, 1),
    )
    plt.tight_layout()
    # save the plot
    plt.savefig("../example_imgs/example_activation.png")

    # plot the beats
    plt.figure(figsize=(6, 2))
    plt.vlines(beats, 0, 1, colors="r", linestyles="dashed", label="Beats")
    plt.title("Beats")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    # save the plot
    plt.savefig("../example_imgs/example_beats.png")
