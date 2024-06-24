import glob
import pandas as pd
import numpy as np
import pathlib
import os
import feature_extraction

CWD = pathlib.Path(os.getcwd())
NORMAL_DATASET = CWD / "mafaulda" / "normal"
ANOMALY_DATASET = CWD / "mafaulda" / "anomaly"
IMBALANCE_DATASET = ANOMALY_DATASET / "imbalance"
HMISALIGMENT_DATASET = ANOMALY_DATASET / "horizontal-misalignment"
OVERHANG_DATASET = ANOMALY_DATASET / "overhang"
UNDERHANG_DATASET = ANOMALY_DATASET / "underhang"
VMISALIGMENT_DATASET = ANOMALY_DATASET / "vertical-misalignment"


def get_attributes_from_signal(signal):
    attributes = []
    attributes.append(feature_extraction.rms_value(signal))
    attributes.append(feature_extraction.sra_value(signal))
    attributes.append(feature_extraction.kv_value(signal))
    attributes.append(feature_extraction.sv_value(signal))
    attributes.append(feature_extraction.ppv_value(signal))
    attributes.append(feature_extraction.cf_value(signal))
    attributes.append(feature_extraction.if_value(signal))
    attributes.append(feature_extraction.mf_value(signal))
    attributes.append(feature_extraction.sf_value(signal))
    attributes.append(feature_extraction.kf_value(signal))
    attributes.append(feature_extraction.fc_value(signal))
    attributes.append(feature_extraction.rmsf_value(signal))
    attributes.append(feature_extraction.rvf_value(signal))
    return attributes


global_dataset_index = 0


def fill_matrix_from_file(matrix, files, label):
    global global_dataset_index
    for file in files:
        signals = pd.read_csv(file).to_numpy()
        signals_amount = signals.shape[1]
        attributes = []
        for s in range(signals_amount):
            attributes += get_attributes_from_signal(signals[:, s])
        attributes += [label]
        matrix[global_dataset_index, :] = np.array(attributes)[:]
        global_dataset_index += 1


if __name__ == "__main__":

    anomaly_files_dir = [
        IMBALANCE_DATASET,
        HMISALIGMENT_DATASET,
        OVERHANG_DATASET,
        UNDERHANG_DATASET,
        VMISALIGMENT_DATASET,
    ]
    all_files = glob.glob(str(CWD / "mafaulda" / "**" / "*.csv"), recursive=True)
    amount_files = len(all_files)

    new_dataset = np.zeros((amount_files, (8 * 13) + 1), dtype=np.float32)

    label = 0
    normal_files = glob.glob(str(NORMAL_DATASET / "*.csv"))
    print(len(normal_files))
    assert len(normal_files) != 0
    fill_matrix_from_file(
        matrix=new_dataset,
        files=normal_files,
        label=label,
    )
    print(global_dataset_index)

    for dir in anomaly_files_dir:
        label += 1
        files = glob.glob(str(dir / "**" / "*.csv"), recursive=True)
        assert len(files) != 0
        fill_matrix_from_file(
            matrix=new_dataset,
            files=files,
            label=label,
        )
        print(global_dataset_index)

    np.savetxt("stat_dataset.csv", new_dataset, delimiter=",")
