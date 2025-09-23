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


from typing import Union
from sudio.process.fx import FX
from sudio.io import SampleFormat
from sudio.utils.math import db2amp
import numpy as np

class Gain(FX):
    def __init__(self, *args, **kwargs) -> None:
        features = {
            'streaming_feature': True, 
            'offline_feature': True,
            'preferred_datatype': SampleFormat.FLOAT32
        }
        """
        Initialize the Gain audio effect processor.

        Configures gain processing with streaming and offline capabilities,
        optimized for 32-bit floating-point audio processing.

        Parameters:
        -----------
        *args : Variable positional arguments
            Arguments for parent FX class initialization.

        **kwargs : Variable keyword arguments
            Additional configuration parameters.
        """
        super().__init__(*args, **kwargs, **features)

    def process(
            self, 
            data: np.ndarray, 
            gain_db: Union[float, int] = 0.0, 
            channel:int=None,
            **kwargs
            ) -> np.ndarray:
        """
        Apply dynamic gain adjustment to audio signals with soft clipping.

        Modify audio amplitude using decibel-based gain control, featuring built-in
        soft clipping to prevent harsh distortion and maintain signal integrity.

        Parameters:
        -----------
        data : numpy.ndarray
            Input audio data to be gain-processed. Supports single and multi-channel inputs.

        gain_db : float or int, optional
            Gain adjustment in decibels:
            - 0.0 (default): No volume change
            - Negative values: Reduce volume
            - Positive values: Increase volume

        Additional keyword arguments are ignored in this implementation.

        Returns:
        --------
        numpy.ndarray
            Gain-adjusted audio data with preserved dynamic range and minimal distortions

        Examples:
        ---------
            >>> from sudio.process.fx import Gain
            >>> su = sudio.Master()        
            >>> rec = su.add('file.mp3')
            >>> rec.afx(Gain, gain_db=-30, start=2.7, stop=7)
        """

        gain = db2amp(gain_db)
        processed_data  = np.tanh(data * gain)
        return processed_data 
    
