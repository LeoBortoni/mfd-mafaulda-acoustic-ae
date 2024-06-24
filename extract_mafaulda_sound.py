import pandas as pd
import numpy as np
import scipy.io.wavfile as wavf
from multiprocessing import Pool
import glob
import os
import pathlib

CURRENT_DIR = pathlib.Path(os.getcwd())
ORIGINAL = CURRENT_DIR / "mafaulda_raw"
OUTPUT = CURRENT_DIR / "mafaulda_sound"
files = glob.glob(str(ORIGINAL / "**" / "*.csv"), recursive=True)


def is_normal(file_path: str) -> bool:
    split = file_path.split("\\")
    for path in split:
        if path == "normal":
            return True


def get_anomaly_name(file_path: str) -> str:
    split = file_path.split("\\")
    concat = False
    filename = ""
    for path in split:
        if concat == True:
            filename += path + "_"
        if path == "mafaulda_raw":
            concat = True
    return filename.replace(".csv_", ".wav")


def create_wav_file(file):
    fs = 50000
    sound_array = pd.read_csv(file).to_numpy()[:, -1].astype(np.float32)
    if is_normal(file):
        file_output_name = str(
            OUTPUT / "normal" / file.split("\\")[-1].replace(".csv", ".wav")
        )
        os.makedirs(os.path.dirname(file_output_name), exist_ok=True)
        wavf.write(file_output_name, fs, sound_array)
        print(file_output_name)
    else:
        file_output_name = get_anomaly_name(file)
        file_output_name = str(OUTPUT / "abnormal" / file_output_name)
        os.makedirs(os.path.dirname(file_output_name), exist_ok=True)
        wavf.write(file_output_name, fs, sound_array)
        print(file_output_name)


if __name__ == "__main__":
    with Pool(25) as p:
        p.map(create_wav_file, files)
