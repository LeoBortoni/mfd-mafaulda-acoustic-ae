import numpy as np
from dataclasses import dataclass


@dataclass
class FeatureExtractionParameters:
    stft_window_size: int = 0
    stft_hop_size: int = 0
    mel_bins: int = 0
    frame_stack_size: int = 0
    frame_hop_size: int = 0


class AudioFileFeature:
    def __init__(
        self,
        stack_frames_matrix: np.ndarray,
        sample_rate: int,
        filename: str,
        size_audio_file_in_elements: int,
        feature_extraction_parameters: FeatureExtractionParameters,
    ):
        self.stack_frames_matrix = stack_frames_matrix
        self.sample_rate = sample_rate
        self.filename = filename
        self.size_audio_file_in_elements = size_audio_file_in_elements
        self.feature_extraction_parameters = feature_extraction_parameters
