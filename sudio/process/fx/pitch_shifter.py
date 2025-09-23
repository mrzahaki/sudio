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
from  sudio.process.fx._pitch_shifter import pitch_shifter_cy
from sudio.io import SampleFormat
import numpy as np

class PitchShifter(FX):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the PitchShifter audio effect processor.

        This method configures the PitchShifter effect with specific processing features,
        setting up support for both streaming and offline audio processing.
        """
        
        features = {
            'streaming_feature': True, 
            'offline_feature': True,
            'preferred_datatype': SampleFormat.FLOAT32
            }
        super().__init__(*args, **kwargs, **features)

    def process(
            self, 
            data: np.ndarray, 
            semitones:np.float32=0.0, 
            cent:np.float32=0.0, 
            ratio:np.float32=1.0, 
            envelope:np.ndarray=[],
            **kwargs
            ):

        """
        Perform pitch shifting on the input audio data.

        This method allows pitch modification through multiple parametrization approaches:
        1. Semitone and cent-based pitch shifting
        2. Direct ratio-based pitch shifting
        3. Envelope-based dynamic pitch shifting

        Parameters:
        -----------
        data : np.ndarray
            Input audio data as a NumPy array. Supports mono and multi-channel audio.
            Recommended data type is float32.

        semitones : np.float32, optional
            Number of semitones to shift the pitch. 
            Positive values increase pitch, negative values decrease pitch.
            Default is 0.0 (no change).
            
            Example:
            - 12.0 shifts up one octave
            - -12.0 shifts down one octave

        cent : np.float32, optional
            Fine-tuning pitch adjustment in cents (1/100th of a semitone).
            Allows precise micro-tuning between semitones.
            Default is 0.0.

            Example:
            - 50.0 shifts up half a semitone
            - -25.0 shifts down a quarter semitone

        ratio : np.float32, optional
            Direct pitch ratio modifier. 
            - 1.0 means no change
            - > 1.0 increases pitch
            - < 1.0 decreases pitch
            Default is 1.0.

            Note: When semitones or cents are used, this ratio is multiplicative.

        envelope : np.ndarray, optional
            Dynamic pitch envelope for time-varying pitch shifting.
            If provided, allows non-uniform pitch modifications across the audio.
            Default is an empty list (uniform pitch shifting).

            Example:
            - A varying array of ratios can create complex pitch modulations

        **kwargs : dict
            Additional keyword arguments passed to the underlying pitch shifting algorithm.
            Allows fine-tuning of advanced parameters like:
            - sample_rate: Audio sample rate
            - frame_length: Processing frame size
            - converter_type: Resampling algorithm

        Returns:
        --------
        np.ndarray
            Pitch-shifted audio data with the same number of channels as input.

        Examples:
        ---------
        >>> record = record.afx(PitchShifter, start=30, envelope=[1, 3, 1, 1]) # Dynamic pitch shift
        >>> record = record.afx(PitchShifter, semitones=4)  # Shift up 4 semitones
        
        Notes:
        ------
        - Uses high-quality time-domain pitch shifting algorithm
        - Preserves audio quality with minimal artifacts
        - Supports both uniform and dynamic pitch modifications
        """
       
        # pitch ratio based on semitones and cents
        if semitones != 0.0 or cent != 0.0:
            pitch_ratio = pow(2.0, (semitones + cent / 100) / 12.0) * ratio
        else:
            pitch_ratio =  ratio

        res = pitch_shifter_cy(
            data,
            np.asarray(envelope, dtype=np.double),
            sample_rate=self._sample_rate,
            ratio=pitch_ratio,
            **kwargs
        )

        return res
    
    
    