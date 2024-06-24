import glob
import pandas as pd
import numpy as np
from pathlib import Path
import os

CWD = Path(os.getcwd())
NORMAL_DATASET = CWD / "mafaulda_mel" / "normal"
ANOMALY_DATASET = CWD / "mafaulda_mel" / "abnormal"

if __name__ == "__main__":
    classes = [
        "horizontal-misalignment",
        "imbalance",
        "overhang",
        "underhang",
        "vertical-misalignment",
    ]

    normal_files = glob.glob(str(NORMAL_DATASET / "**" / "*.bortoni"), recursive=True)
    anomaly_files = glob.glob(str(ANOMALY_DATASET / "**" / "*.bortoni"), recursive=True)
    dataset = []

    all_files = normal_files + anomaly_files
    for filepath in all_files:
        filename = Path(filepath).stem
        subsample_group = int(filename.split("_")[-1])
        if subsample_group == 1:
            label = 0
            for i in range(len(classes)):
                if classes[i] in filename:
                    label = i + 1
            dataset.append([filename, filepath, str(label)])

    pd.DataFrame(dataset, columns=["filename", "filepath", "label"]).to_csv(
        "mel_dataset.csv",
        sep=",",
        index=False,
    )
