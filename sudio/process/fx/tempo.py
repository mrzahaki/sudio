# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


from sudio.process.fx import FX
from sudio.io import SampleFormat
import numpy as np
from typing import TYPE_CHECKING
from  sudio.process.fx._tempo import tempo_cy

class Tempo(FX):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Tempo audio effect processor for time stretching.

        Configures time stretching with support for both streaming and offline 
        audio processing, optimized for 32-bit floating-point precision.

        Notes:
        ------
        Implements advanced time stretching using WSOLA (Waveform Similarity 
        Overlap-Add) algorithm to modify audio tempo without altering pitch.
        """
        features = {
            'streaming_feature': True, 
            'offline_feature': True,
            'preferred_datatype': SampleFormat.FLOAT32
            }
        super().__init__(*args, **kwargs, **features)

    def process(self, data: np.ndarray, tempo:float=1.0, envelope:np.ndarray=[], **kwargs):
        """
        Perform time stretching on the input audio data without altering pitch.

        This method allows tempo modification through uniform or dynamic tempo changes,
        utilizing an advanced Waveform Similarity Overlap-Add (WSOLA) algorithm to 
        manipulate audio duration while preserving sound quality and spectral characteristics.

        Parameters:
        -----------
        data : np.ndarray
            Input audio data as a NumPy array. Supports mono and multi-channel audio.
            Recommended data type is float32.

        tempo : float, optional
            Tempo scaling factor for time stretching.
            - 1.0 means no change in tempo/duration
            - < 1.0 slows down audio (increases duration)
            - > 1.0 speeds up audio (decreases duration)
            Default is 1.0.

            Examples:
            - 0.5: doubles audio duration
            - 2.0: halves audio duration

        envelope : np.ndarray, optional
            Dynamic tempo envelope for time-varying tempo modifications.
            Allows non-uniform tempo changes across the audio signal.
            Default is an empty list (uniform tempo modification).

            Example:
            - A varying array of tempo ratios can create complex time-stretching effects

        **kwargs : dict
            Additional keyword arguments passed to the underlying tempo algorithm.
            Allows fine-tuning of advanced parameters such as:
            - sequence_ms: Sequence length for time-stretching window
            - seekwindow_ms: Search window for finding similar waveforms
            - overlap_ms: Crossfade overlap between segments
            - enable_spline: Enable spline interpolation for envelope
            - spline_sigma: Gaussian smoothing parameter for envelope

        Returns:
        --------
        np.ndarray
            Time-stretched audio data with the same number of channels and original data type
            as the input.

        Examples:
        ---------
        >>> slow_audio = tempo_processor.process(audio_data, tempo=0.5)  # Slow down audio
        >>> fast_audio = tempo_processor.process(audio_data, tempo=1.5)  # Speed up audio
        >>> dynamic_tempo = tempo_processor.process(audio_data, envelope=[0.5, 1.0, 2.0])  # Dynamic tempo

        Notes:
        ------
        - Preserves audio quality with minimal artifacts
        - Uses advanced WSOLA algorithm for smooth time stretching
        - Supports both uniform and dynamic tempo modifications
        - Computationally efficient implementation
        - Does not change the pitch of the audio

        Warnings:
        ---------
        - Extreme tempo modifications (very low or high values) may introduce 
        audible artifacts or sound distortions
        - Performance and quality may vary depending on audio complexity
        """

        dtype = data.dtype
        data = tempo_cy(
            data, 
            np.asarray(envelope, dtype=np.double), 
            self._sample_rate, 
            default_tempo=tempo, 
            **kwargs
            )
        return data.astype(dtype)
    
    