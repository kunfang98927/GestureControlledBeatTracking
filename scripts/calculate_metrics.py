import mir_eval
import numpy as np


def read_beats(file_path):
    """Read the beat annotations from a txt file."""
    with open(file_path, "r") as file:
        annotation = file.read()
    annotation = annotation.split("\n")
    annotation = [line.split(" ") for line in annotation if line]
    beats = [float(line[0]) for line in annotation]
    beats = np.array(beats)

    return beats


if __name__ == "__main__":

    # only beats
    beats_label_path_1 = "../dataset/annotations/R_22_b.txt"
    madmom_beats_path_1 = "../dataset/madmom_outputs/R_22_MAPS_b.txt"
    ours_beats_path_1_1 = "../demo_1203_R_22/experiments_for_report/sample1-kun/R_22-kun_beat_1206_envelope-win-61.txt"
    ours_beats_path_1_2 = "../demo_1203_R_22/experiments_for_report/sample2-kun-2x/R_22-kun-2x_beat_1206_envelope-win-31.txt"
    ours_beats_path_1_3 = "../demo_1203_R_22/experiments_for_report/sample3-ziyu/R_22-ziyu_beat_1206_envelope-win-21.txt"
    ours_beats_path_1_4 = "../demo_1203_R_22/experiments_for_report/sample4-kun-x4/R_22-kun_beat_1206_envelope-win-31.txt"
    time_range_1_1 = [1.2, 45]
    time_range_1_2 = [1.2, 45]
    time_range_1_3 = [1.0, 45]
    time_range_1_4 = [1.2, 45]

    # beats and downbeats
    beats_label_path_2 = "../dataset/annotations/R_17_b.txt"
    downbeats_label_path_2 = "../dataset/annotations/R_17_db.txt"
    madmom_beats_path_2 = "../demo_1204_R_17/R_17_beats.txt"
    madmom_downbeats_path_2 = "../demo_1204_R_17/R_17_downbeats.txt"

    ours_beats_path_2_1 = "../demo_1204_R_17/experiments_for_report/sample1/R_17-ziyu1_beats_1206_envelope-win-b51-db101_npclip02.txt"
    ours_downbeats_path_2_1 = "../demo_1204_R_17/experiments_for_report/sample1/R_17-ziyu1_downbeats_1206_envelope-win-b51-db101_npclip02.txt"
    time_range_2_1 = [49.80, 140.50]  # [49.60, 156.60]

    ours_beats_path_2_2 = "../demo_1204_R_17/experiments_for_report/sample2-lazy/R_17-ziyu-lazy_beats_1206_envelope-win-b51-db101_npclip02.txt"
    ours_downbeats_path_2_2 = "../demo_1204_R_17/experiments_for_report/sample2-lazy/R_17-ziyu-lazy_downbeats_1206_envelope-win-b51-db101_npclip02.txt"
    time_range_2_2 = [49.80, 140.50]  # [49.80, 147.80]

    ours_beats_path_2_3 = "../demo_1204_R_17/experiments_for_report/sample3-noisy/R_17-ziyu-noisy_beats_1206_envelope-win-b51-db101_npclip02.txt"
    ours_downbeats_path_2_3 = "../demo_1204_R_17/experiments_for_report/sample3-noisy/R_17-ziyu-noisy_downbeats_1206_envelope-win-b51-db101_npclip02.txt"
    time_range_2_3 = [49.80, 140.50]  # [47.50, 140.50]  #

    # calculate the beat tracking evaluation metrics
    beats_label_1 = read_beats(beats_label_path_1)
    madmom_beats_1 = read_beats(madmom_beats_path_1)
    ours_beats_1_1 = read_beats(ours_beats_path_1_1)
    ours_beats_1_2 = read_beats(ours_beats_path_1_2)
    ours_beats_1_3 = read_beats(ours_beats_path_1_3)
    ours_beats_1_4 = read_beats(ours_beats_path_1_4)

    beats_label_1 = beats_label_1[
        (beats_label_1 >= time_range_1_1[0]) & (beats_label_1 <= time_range_1_1[1])
    ]
    madmom_beats_1 = madmom_beats_1[
        (madmom_beats_1 >= time_range_1_1[0]) & (madmom_beats_1 <= time_range_1_1[1])
    ]
    ours_beats_1_1 = ours_beats_1_1[
        (ours_beats_1_1 >= time_range_1_1[0]) & (ours_beats_1_1 <= time_range_1_1[1])
    ]
    ours_beats_1_2 = ours_beats_1_2[
        (ours_beats_1_2 >= time_range_1_2[0]) & (ours_beats_1_2 <= time_range_1_2[1])
    ]
    ours_beats_1_3 = ours_beats_1_3[
        (ours_beats_1_3 >= time_range_1_3[0]) & (ours_beats_1_3 <= time_range_1_3[1])
    ]
    ours_beats_1_4 = ours_beats_1_4[
        (ours_beats_1_4 >= time_range_1_4[0]) & (ours_beats_1_4 <= time_range_1_4[1])
    ]

    beats_label_2 = read_beats(beats_label_path_2)
    downbeats_label_2 = read_beats(downbeats_label_path_2)
    madmom_beats_2 = read_beats(madmom_beats_path_2)
    ours_beats_2_1 = read_beats(ours_beats_path_2_1)
    ours_beats_2_2 = read_beats(ours_beats_path_2_2)
    ours_beats_2_3 = read_beats(ours_beats_path_2_3)
    madmom_downbeats_2 = read_beats(madmom_downbeats_path_2)
    ours_downbeats_2_1 = read_beats(ours_downbeats_path_2_1)
    ours_downbeats_2_2 = read_beats(ours_downbeats_path_2_2)
    ours_downbeats_2_3 = read_beats(ours_downbeats_path_2_3)

    beats_label_2 = beats_label_2[
        (beats_label_2 >= time_range_2_1[0]) & (beats_label_2 <= time_range_2_1[1])
    ]
    downbeats_label_2 = downbeats_label_2[
        (downbeats_label_2 >= time_range_2_1[0])
        & (downbeats_label_2 <= time_range_2_1[1])
    ]
    madmom_beats_2 = madmom_beats_2[
        (madmom_beats_2 >= time_range_2_1[0]) & (madmom_beats_2 <= time_range_2_1[1])
    ]
    ours_beats_2_1 = ours_beats_2_1[
        (ours_beats_2_1 >= time_range_2_1[0]) & (ours_beats_2_1 <= time_range_2_1[1])
    ]
    ours_beats_2_2 = ours_beats_2_2[
        (ours_beats_2_2 >= time_range_2_2[0]) & (ours_beats_2_2 <= time_range_2_2[1])
    ]
    ours_beats_2_3 = ours_beats_2_3[
        (ours_beats_2_3 >= time_range_2_3[0]) & (ours_beats_2_3 <= time_range_2_3[1])
    ]
    madmom_downbeats_2 = madmom_downbeats_2[
        (madmom_downbeats_2 >= time_range_2_1[0])
        & (madmom_downbeats_2 <= time_range_2_1[1])
    ]
    ours_downbeats_2_1 = ours_downbeats_2_1[
        (ours_downbeats_2_1 >= time_range_2_1[0])
        & (ours_downbeats_2_1 <= time_range_2_1[1])
    ]
    ours_downbeats_2_2 = ours_downbeats_2_2[
        (ours_downbeats_2_2 >= time_range_2_2[0])
        & (ours_downbeats_2_2 <= time_range_2_2[1])
    ]
    ours_downbeats_2_3 = ours_downbeats_2_3[
        (ours_downbeats_2_3 >= time_range_2_3[0])
        & (ours_downbeats_2_3 <= time_range_2_3[1])
    ]

    ####### calculate the f-measure for beat tracking #######
    # for R_22, madmom and ours
    madmom_f_measure_1 = mir_eval.beat.f_measure(
        beats_label_1, madmom_beats_1, f_measure_threshold=0.07
    )
    ours_f_measure_1_1 = mir_eval.beat.f_measure(
        beats_label_1, ours_beats_1_1, f_measure_threshold=0.07
    )
    ours_f_measure_1_2 = mir_eval.beat.f_measure(
        beats_label_1, ours_beats_1_2, f_measure_threshold=0.07
    )
    ours_f_measure_1_3 = mir_eval.beat.f_measure(
        beats_label_1, ours_beats_1_3, f_measure_threshold=0.07
    )
    ours_f_measure_1_4 = mir_eval.beat.f_measure(
        beats_label_1, ours_beats_1_4, f_measure_threshold=0.07
    )
    # for R_17, madmom and ours (beats & downbeats)
    madmom_f_measure_2 = mir_eval.beat.f_measure(
        beats_label_2, madmom_beats_2, f_measure_threshold=0.07
    )
    ours_f_measure_2_1 = mir_eval.beat.f_measure(
        beats_label_2, ours_beats_2_1, f_measure_threshold=0.07
    )
    ours_f_measure_2_2 = mir_eval.beat.f_measure(
        beats_label_2, ours_beats_2_2, f_measure_threshold=0.07
    )
    ours_f_measure_2_3 = mir_eval.beat.f_measure(
        beats_label_2, ours_beats_2_3, f_measure_threshold=0.07
    )
    madmom_f_measure_2_db = mir_eval.beat.f_measure(
        downbeats_label_2, madmom_downbeats_2, f_measure_threshold=0.07
    )
    ours_f_measure_2_1_db = mir_eval.beat.f_measure(
        downbeats_label_2, ours_downbeats_2_1, f_measure_threshold=0.07
    )
    ours_f_measure_2_2_db = mir_eval.beat.f_measure(
        downbeats_label_2, ours_downbeats_2_2, f_measure_threshold=0.07
    )
    ours_f_measure_2_3_db = mir_eval.beat.f_measure(
        downbeats_label_2, ours_downbeats_2_3, f_measure_threshold=0.07
    )

    print("F-measure for beat tracking:")
    print(f"R_22, madmom: {madmom_f_measure_1}")
    print(f"R_22, ours_1: {ours_f_measure_1_1}")
    print(f"R_22, ours_2: {ours_f_measure_1_2}")
    print(f"R_22, ours_3: {ours_f_measure_1_3}")
    print(f"R_22, ours_4: {ours_f_measure_1_4}")
    print("----------------------")
    print(f"R_17, madmom: {madmom_f_measure_2}")
    print(f"R_17, ours_1: {ours_f_measure_2_1}")
    print(f"R_17, ours_2: {ours_f_measure_2_2}")
    print(f"R_17, ours_3: {ours_f_measure_2_3}")
    print(f"R_17, madmom (downbeats): {madmom_f_measure_2_db}")
    print(f"R_17, ours (downbeats): {ours_f_measure_2_1_db}")
    print(f"R_17, ours (downbeats): {ours_f_measure_2_2_db}")
    print(f"R_17, ours (downbeats): {ours_f_measure_2_3_db}")
