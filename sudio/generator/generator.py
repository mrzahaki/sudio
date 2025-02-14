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

from sudio.io import SampleFormat
from sudio.utils import audio_metadata
from sudio.utils.functional import filter_kwargs as fw
from sudio.utils.functional import exclude_kwargs as ekw
from sudio.utils.functional import type_check
from sudio.utils.typeconversion import dtype_converter, dtype_descriptor
from sudio.process.fx import Normalize

class Generator:
    
    @type_check()
    def __init__(
                self, 
                sample_rate: int = None,
                nchannels: int = None,
                target_sample_format: SampleFormat = SampleFormat.FLOAT32,
                process_sample_format: SampleFormat = SampleFormat.FLOAT32,
                data_nperseg:int=None,
                sample_width: int = None,
                streaming_feature: bool = True, 
                offline_feature: bool = True, 
                eps:float=1e-7,
                interleave: bool = False,
                normalize:bool=False, 
                peak_level:float=0.0, 
                equalize_channels:bool=False, 
                dc_offset:float=0.0,
                name:str='',
                **kwargs,
                ) -> None:
            """
            Base class for audio signal generators.

            initializes core audio generation parameters and validates configuration.
            Designed for inheritance by specialized generator implementations.

            Parameters:
            -----------
            sample_rate : int
                Audio sampling rate in Hz. Must match system capabilities.
            nchannels : int
                Number of audio channels (1=mono, 2=stereo, etc.).
            target_sample_format : SampleFormat, default=FLOAT32
                Output format for final audio data.
            process_sample_format : SampleFormat, default=FLOAT32
                Internal processing format for audio calculations.
            data_nperseg : int, optional
                Segment length for spectral processing.
            sample_width : int, optional
                Bytes per sample, derived from format if not specified.
            streaming_feature : bool, default=True
                Enable real-time audio streaming capability.
            offline_feature : bool, default=True
                Enable file-based/batch processing.
            eps : float, default=1e-7
                Numerical stability constant for signal processing.
            interleave : bool, default=False
                Store multi-channel data in interleaved format.
            normalize : bool, default=False
                Apply peak normalization during finalization.
            peak_level : float, default=0.0
                Target peak amplitude in dBFS for normalization.
            equalize_channels : bool, default=False
                Balance peak levels across all channels.
            dc_offset : float, default=0.0
                DC bias adjustment in amplitude units.
            name : str, default=''
                Identifier for audio metadata.

            Raises:
            -------
            TypeError
                If invalid keyword arguments are provided.
            """
            self._sample_rate = sample_rate
            self._nchannels = nchannels
            self._sample_format = process_sample_format
            self._sample_type_descriptor = dtype_descriptor(process_sample_format)
            self._target_sample_format = target_sample_format
            self._target_sample_type_descriptor = dtype_descriptor(target_sample_format)
            self._data_nperseg = data_nperseg
            self._sample_width = sample_width
            self._streaming_feature = streaming_feature
            self._offline_feature = offline_feature
            self._eps = eps
            self._interleave = interleave
            self._normalize = normalize
            self._peak_level = peak_level
            self._equalize_channels = equalize_channels
            self._dc_offset = dc_offset
            self._name = name


    def get_sample_rate(self)->int:
        """
        Retrieve the audio sampling rate.
        """
        return self._sample_rate
    
    def get_nchannels(self)->int:
        """
        Get the number of audio channels.
        """
        return self._nchannels
    
    def get_sample_format(self)->SampleFormat:
        """
        Retrieve the audio sample format.
        """
        return self._sample_format
    
    def get_sample_type_descriptor(self)->str:
        """
        Get additional sample type information.
        """
        return self._sample_type_descriptor
    
    def get_sample_width(self):
        """
        Retrieve the bit depth or bytes per sample.
        """
        return self._sample_width
    
    
    def is_streaming_supported(self) -> bool:
        """
        Determine if audio streaming is supported for this generator.
        """
        return self._streaming_feature
    
    def is_offline_supported(self) -> bool:
        """
        Check if file/batch audio processing is supported.
        """
        return self._offline_feature
    
    def generate(self, *args, **kwargs):
        """
        Base method for audio signal generation.
        
        This method should be implemented by specific generator classes
        to define their unique audio generation logic.
        """
        ...

    def __call__(self, *args, **kwargs):
        """
        Generate audio data and extract metadata.
        
        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        
        Returns:
            An AudioMetadata object with processed audio information
        
        Raises:
            TypeError: For invalid input types
            ValueError: For invalid parameter values
        """
        if ekw(self.generate, audio_metadata, **kwargs):
            raise ValueError('Invalid keyword arguments')
        
        data = self.generate(
            *args, 
            **fw(self.generate, **kwargs)
            )

        data = dtype_converter(data, self._target_sample_format, self._sample_format)
        metadata = audio_metadata(
            data=data,
            sample_rate=self._sample_rate,
            sample_format=self._target_sample_format,
            nchannels=self._nchannels,
            interleave=self._interleave,
            normalize=self._normalize,
            peak_level=self._peak_level,
            equalize_channels=self._equalize_channels,
            dc_offset=self._dc_offset,
            name=self._name,
            **fw(audio_metadata, **kwargs),
            )        

        return metadata
    



__all__ = [
    'Generator',
]
