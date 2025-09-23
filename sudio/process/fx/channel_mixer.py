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

from typing import Union, List, Optional
import numpy as np
from sudio.process.fx import FX
from sudio.io import SampleFormat
from sudio.process.fx._channel_mixer import channel_mixer


class ChannelMixer(FX):
    def __init__(
            self, *args, 
            **kwargs) -> None:
        """
        initialize the ChannelMixer audio effect processor.

        Parameters:
        -----------
        *args : Variable positional arguments
            Arguments to be passed to the parent FX class initializer.
        **kwargs : dict, optional
            Additional keyword arguments for configuration.

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
        correlation: Optional[Union[List[List[float]], np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        apply channel mixing to the input audio signal based on a correlation matrix.

        manipulates multi-channel audio by applying inter-channel correlation 
        transformations while preserving signal characteristics.

        Parameters:
        -----------
        data : numpy.ndarray
            Input multi-channel audio data. Must have at least 2 dimensions.
            Shape expected to be (num_channels, num_samples).

        correlation : Union[List[List[float]], numpy.ndarray], optional
            Correlation matrix defining inter-channel relationships.
            - If None, returns input data unchanged
            - Must be a square matrix matching number of input channels
            - Values must be between -1 and 1
            - Matrix shape: (num_channels, num_channels)

        **kwargs : dict, optional
            Additional processing parameters (currently unused).

        Returns:
        --------
        numpy.ndarray
            Channel-mixed audio data with the same shape as input.

        Raises:
        -------
        ValueError
            - If input data has fewer than 2 channels
            - If correlation matrix is incorrectly shaped
            - If correlation matrix contains values outside [-1, 1]

        Examples:
        ---------
            >>> from sudio.process.fx import ChannelMixer
            >>> su = sudio.Master()        
            >>> rec = su.add('file.mp3')
            >>> newrec = rec.afx(ChannelMixer, correlation=[[.4,-.6], [0,1]]) #for two channel
        """

        if data.ndim < 2:
            raise ValueError("Input data must be multichannel")
        
        nchannels = self._nchannels

        if correlation is None:
            return data
        else:
            correlation = np.asarray(correlation, dtype=np.float32)

        if (correlation.shape != (nchannels, nchannels) or 
            not np.all((-1 <= correlation) & (correlation <= 1))):
            raise ValueError(f"Invalid correlation matrix: shape {correlation.shape}, expected {(nchannels, nchannels)}")

        return channel_mixer(data, correlation)
    
