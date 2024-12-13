import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dirs import METADATA_DIR, ANNOTATION_DIR, RAW_DATA_DIR


def main():
    metadata_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    metadata = pd.read_csv(metadata_path)
    perf_ids = metadata["performance_id"].tolist()
    for perf_id in tqdm(perf_ids):
        perf_id_short = "_".join(perf_id.split("_")[:2])
        row = metadata[metadata["performance_id"] == perf_id_short]
        anno_path = os.path.join(
            RAW_DATA_DIR,
            "ACPAS-dataset",
            row["folder"].iloc[0],
            row["performance_annotation"].iloc[0],
        )

        try:
            with open(anno_path, "r") as f:
                beats_anno = []
                downbeats_anno = []
                for line in f:
                    if line.strip() == "":
                        continue
                    beat, beat_type = line.strip().split("\t")
                    beat = float(beat)
                    beats_anno.append(beat)
                    if beat_type == "db":
                        downbeats_anno.append(beat)
                beats_anno = np.array(beats_anno)
                downbeats_anno = np.array(downbeats_anno)
                # save the beat annotations to .txt
                with open(os.path.join(ANNOTATION_DIR, f"{perf_id}_b.txt"), "w") as f:
                    for beat in beats_anno:
                        f.write(f"{beat}\n")
                with open(os.path.join(ANNOTATION_DIR, f"{perf_id}_db.txt"), "w") as f:
                    for beat in downbeats_anno:
                        f.write(f"{beat}\n")
                # # save the beat annotations to .npy
                # np.save(
                #     os.path.join(ANNOTATION_DIR, f"{perf_id}_b.npy"),
                #     beats_anno,
                # )
                # np.save(
                #     os.path.join(ANNOTATION_DIR, f"{perf_id}_db.npy"),
                #     downbeats_anno,
                # )

        except Exception as e:
            print(f"Error reading annotation file {anno_path}: {e}")
            continue


if __name__ == "__main__":
    main()
