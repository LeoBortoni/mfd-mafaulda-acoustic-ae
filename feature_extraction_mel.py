import yaml
from storage_model import FeatureExtractionParameters, AudioFileFeature
from pathlib import Path
import os
import librosa
import glob
import numpy as np
import pickle
from multiprocessing import Pool


class IOFiles:
    def __init__(
        self, camsd_folder: str, output_feature_folder: str, databases: list[str]
    ):
        self.camsd_folder = Path(camsd_folder)
        self.output_feature_folder = Path(output_feature_folder)
        self.database_folders = databases

        self._check_exist_folder(self.camsd_folder)
        for database in self.database_folders:
            self._check_database_exist(database, self.camsd_folder)

    def _check_exist_folder(self, folder: Path):
        if not folder.exists():
            raise Exception(f"{folder} Is not an existing Path")

    def _check_database_exist(self, database: str, folder: Path):
        database_path = folder / database
        if not database_path.exists():
            raise Exception(f"Database {database_path} Is not an existing Path")

    def get_databases_path(self) -> list[Path]:
        return [self.camsd_folder / database for database in self.database_folders]


def get_all_sound_files_from_a_dir(dir: Path) -> list[str]:
    return glob.glob((dir / "**" / "*.wav").__str__(), recursive=True)


def _generate_stack_of_frames_feature_matrix(
    log_mel_energy: np.ndarray, feature_parameters: FeatureExtractionParameters
) -> np.ndarray:
    assert len(log_mel_energy.shape) == 2
    assert log_mel_energy.shape[0] == feature_parameters.mel_bins

    feature_vector_size = (
        feature_parameters.frame_stack_size * feature_parameters.mel_bins
    )

    amount_of_objects = (
        len(log_mel_energy[0, :]) - feature_parameters.frame_stack_size + 1
    )

    assert amount_of_objects >= 1

    matrix_stack_of_frames = np.zeros((amount_of_objects, feature_vector_size))

    mel_bins = feature_parameters.mel_bins
    for i in range(feature_parameters.frame_stack_size):
        init_stack = i * mel_bins
        end_stack = mel_bins * (i + 1)
        time_init = i
        time_end = amount_of_objects + i
        matrix_stack_of_frames[:, init_stack:end_stack] = log_mel_energy[
            :, time_init:time_end
        ].T

    return matrix_stack_of_frames


def _filter_by_hop_frames(
    matrix_stack_frames: np.ndarray, feature_parameters: FeatureExtractionParameters
) -> np.ndarray:
    return matrix_stack_frames[:: feature_parameters.frame_hop_size]


def get_feature_vector_from_audio_file(
    file: str, feature_parameters: FeatureExtractionParameters
) -> AudioFileFeature:
    data, sr = librosa.load(file, sr=None, mono=True)

    mel_energy = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_mels=feature_parameters.mel_bins,
        n_fft=feature_parameters.stft_window_size,
        hop_length=feature_parameters.stft_hop_size,
        power=1,
    )

    log_mel_energy = librosa.amplitude_to_db(mel_energy)

    matrix_stack_frames = _generate_stack_of_frames_feature_matrix(
        log_mel_energy, feature_parameters
    )

    filtered_stack_frames = _filter_by_hop_frames(
        matrix_stack_frames, feature_parameters
    )

    return AudioFileFeature(
        stack_frames_matrix=filtered_stack_frames.astype(np.float32),
        sample_rate=sr,
        filename=file,
        size_audio_file_in_elements=len(data),
        feature_extraction_parameters=feature_parameters,
    )


def get_output_filepath(audio_file: str, database: Path, output_folder: str):
    database_name = database.name
    position = audio_file.find(database_name)
    if position != -1:  # Check if the word is found in the text
        result = audio_file[position:]
    else:
        assert False
    output_file = result.replace(database.name, database.name)
    return Path(output_folder) / output_file


def task_generate_audio_feature_file(
    audio_file, feature_parameters, database_path, output_feature_folder
):
    audio_file_feature_vector = get_feature_vector_from_audio_file(
        audio_file, feature_parameters
    )
    output_filepath = get_output_filepath(
        audio_file, database_path, output_feature_folder
    )
    os.makedirs(os.path.dirname(output_filepath.__str__()), exist_ok=True)
    output = output_filepath.parent / (output_filepath.stem + ".bortoni")
    filehandler = open(output, "wb")
    pickle.dump(audio_file_feature_vector, filehandler)
    filehandler.close()


CONFIG_FILE = "./config_featuring.yaml"

if __name__ == "__main__":

    with open(CONFIG_FILE) as f:
        file = yaml.load(f, Loader=yaml.SafeLoader)
    feature_parameters = FeatureExtractionParameters(
        **file["feature_extraction_parameters"]
    )
    io_manager = IOFiles(
        file["camsd_folder"], file["output_feature_folder"], file["databases"]
    )

    for database in io_manager.get_databases_path():
        all_sound_files = get_all_sound_files_from_a_dir(database)
        task_args = [
            (file, feature_parameters, database, io_manager.output_feature_folder)
            for file in all_sound_files
        ]
        # task_generate_audio_feature_file(*task_args[0])
        with Pool(25) as p:
            p.starmap(task_generate_audio_feature_file, task_args)
