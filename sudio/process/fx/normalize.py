# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


import numpy as np
from sudio.process.fx import FX
from sudio.io import SampleFormat
from sudio.utils.math import normalize


class Normalize(FX):
    def __init__(self, *args, **kwargs) -> None:
        features = {
            'streaming_feature': True, 
            'offline_feature': True,
            'preferred_datatype': SampleFormat.FLOAT32
        }
        """
        Initialize the Normalize audio effect processor.
        """
        super().__init__(*args, **kwargs, **features)


    def process(
        self, 
        data: np.ndarray,
        peak_level:float=0.0, 
        equalize_channels:bool=False, 
        dc_offset:float=0.0,
        **kwargs
    ) -> np.ndarray:
        """
        applies normalization techniques to audio signals, offering 
        control over signal levels, channel balance, and overall dynamics.

        Parameters:
        -----------

        data : numpy.ndarray
            input audio data. Supports single and multi-channel configurations.
        
        peak_level : float, optional
            target peak amplitude in decibels. 
            - Default (0.0): Uses maximum existing amplitude as reference
            - Allows precise output signal strength control
        
        equalize_channels : bool, optional
            channel normalization behavior:
            - False (default): Normalizes each channel independently
            - True: Equalizes maximum amplitude across all channels
        
        dc_offset : float, optional
            Amplitude offset adjustment:
            - Ranges typically between -1 and 1
            - Addresses minor signal imbalances or subtle audio artifacts
        
        Returns:
        --------
        numpy.ndarray
            normalized audio data, maintaining original input dimensionality
        """        

        processed_data = normalize(
            data, 
            float(peak_level),
            bool(equalize_channels),
            float(dc_offset)
            )
        return processed_data
