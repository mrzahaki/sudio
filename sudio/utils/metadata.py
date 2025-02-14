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


from sudio.metadata import AudioMetadata
from sudio.io import SampleFormat
from sudio.utils.typeconversion import dtype_descriptor
from sudio.process.fx import Normalize
from typing import Union
import numpy as np


def audio_metadata(
        data:Union[list, tuple, np.ndarray, bytes], 
        sample_rate:int,
        sample_format:SampleFormat=SampleFormat.FLOAT32,
        nchannels:int=None,
        interleave:bool=False,
        normalize:bool=False, 
        peak_level:float=0.0, 
        equalize_channels:bool=False, 
        dc_offset:float=0.0,
        name:str='',
        )->AudioMetadata:
    
    """
    Converts and processes audio data into an AudioMetadata object.

    This function takes raw audio data and transforms it into a standardized metadata 
    representation. It handles various input types, channel configurations, and optional 
    preprocessing like normalization.

    Args:
        data: Audio data as a list, tuple, NumPy array, or bytes
        sample_rate: Audio sampling rate in Hz
        sample_format: Format of audio samples (default is 32-bit float)
        nchannels: Number of audio channels (auto-detected if not specified)
        interleave: Whether audio data is interleaved
        normalize: Apply amplitude normalization
        peak_level: Target peak amplitude for normalization
        equalize_channels: Balance volume across channels during normalization
        dc_offset: DC offset adjustment
        name: Optional name for the audio metadata

    Returns:
        An AudioMetadata object with processed audio information

    Raises:
        TypeError: For invalid input types
        ValueError: For invalid parameter values
    """

    dtype = dtype_descriptor(sample_format)    
    assert isinstance(sample_rate, int), TypeError('sample_rate must be an integer')
    assert sample_rate > 0, ValueError('sample_rate must be positive')

    if isinstance(data, (list, tuple, np.ndarray)):
        data = np.array(data, dtype=dtype)
            
    elif isinstance(data, bytes):
        data = np.frombuffer(data, dtype=dtype)
        assert isinstance(nchannels, int), ValueError('missed nchannels')
    else:
        raise TypeError('data must be a list, tuple, ndarray or bytes')

    if nchannels is None:
        nchannels = 1 if data.ndim < 2 else data.shape[1] if interleave else data.shape[0]
    else:
        assert isinstance(nchannels, int), TypeError('nchannels must be an integer')
        assert nchannels > 0, ValueError('nchannels must be positive')

    if interleave:
        data = data.reshape((-1, nchannels)).T
    else:
        data = data.reshape((nchannels, -1))

    if normalize:
        common_args = {
            'data_size': None,
            'sample_rate': sample_rate,
            'nchannels': nchannels,
            'sample_format': sample_format,
            'data_nperseg': None,
            'sample_type': None,
            'sample_width': None,
        }
        norm = Normalize(**common_args)
        data = norm.typesafe_process(
            data, 
            float(peak_level),
            bool(equalize_channels),
            float(dc_offset)
            )

    assert data.size % nchannels == 0, ValueError('data size must be divisible by nchannels')
    if nchannels == 1:
        data = data.flatten()
    data = data.T.tobytes()
    record = AudioMetadata(name, **{
                'size': len(data),
                'frameRate': sample_rate,
                'o': data,
                'sampleFormat': sample_format,
                'nchannels': nchannels,
                'duration': None,
            }
        )
    
    return record



__all__ = [
    'audio_metadata',
]
