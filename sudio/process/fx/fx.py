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
from sudio.utils.typeconversion import dtype_converter

class FX:
    def __init__(
            self, 
            *args,
            data_size: int = None,
            sample_rate: int = None,
            nchannels: int = None,
            sample_format: SampleFormat = SampleFormat.UNKNOWN,
            data_nperseg:int=None,
            sample_type: str = '',
            sample_width: int = None,
            streaming_feature: bool = True, 
            offline_feature: bool = True, 
            preferred_datatype: SampleFormat = SampleFormat.UNKNOWN,
            **kwargs,
            ) -> None:
        
        """
        Initialize the base Effects (FX) processor with audio configuration and processing features.

        This method sets up the fundamental parameters and capabilities for audio signal processing,
        providing a flexible foundation for various audio effects and transformations.

        Parameters:
        -----------
        data_size : int, optional
            Total size of the audio data in samples. Helps in memory allocation and processing planning.

        sample_rate : int, optional
            Number of audio samples processed per second. Critical for time-based effects and analysis.

        nchannels : int, optional
            Number of audio channels (mono, stereo, etc.). Determines multi-channel processing strategies.

        sample_format : SampleFormat, optional
            Represents the audio data's numeric representation and precision. 
            Defaults to UNKNOWN if not specified.

        data_nperseg : int, optional
            Number of samples per segment, useful for segmented audio processing techniques.

        sample_type : str, optional
            Additional type information about the audio samples.

        sample_width : int, optional
            Bit depth or bytes per sample, influencing audio resolution and dynamic range.

        streaming_feature : bool, default True
            Indicates if the effect supports real-time, streaming audio processing.

        offline_feature : bool, default True
            Determines if the effect can process entire audio files or large datasets.

        preferred_datatype : SampleFormat, optional
            Suggested sample format for optimal processing. Defaults to UNKNOWN.

        Notes:
        ------
        This base class provides a standardized interface for audio effect processors,
        enabling consistent configuration and feature detection across different effects.
        """
        self._streaming_feature = streaming_feature
        self._offline_feature = offline_feature
        self._preferred_datatype = preferred_datatype
        self._data_size = data_size
        self._sample_rate = sample_rate
        self._nchannels = nchannels
        self._sample_format = sample_format
        self._data_nperseg = data_nperseg
        self._sample_type = sample_type
        self._sample_width = sample_width

    def is_streaming_supported(self) -> bool:
        """
        Determine if audio streaming is supported for this effect.
        """
        return self._streaming_feature

    def is_offline_supported(self) -> bool:
        """
        Check if file/batch audio processing is supported.
        """
        return self._offline_feature

    def get_preferred_datatype(self)->SampleFormat:
        """
        Retrieve the recommended sample format for optimal processing.
        """
        return self._preferred_datatype
    
    
    def get_data_size(self)->int:
        """
        Get the total size of audio data in samples.
        """
        return self._data_size
    
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
    
    def get_sample_type(self)->str:
        """
        Get additional sample type information.
        """
        return self._sample_type
    
    def get_sample_width(self):
        """
        Retrieve the bit depth or bytes per sample.
        """
        return self._sample_width

    def typesafe_process(self, dt, *args, **kwargs):
        """
        Process audio data with automatic type conversion.

        Converts the input data type to the preferred type (and normalize data for target) before processing and reverts it back after processing.

        Parameters:
        dt : any
            The audio data to be processed.
        *args : tuple
            Additional positional arguments for the processing method.
        **kwargs : dict
            Additional keyword arguments for the processing method.

        Returns:
        any
            The processed audio data with the original data type.
        """

        dt = dtype_converter(dt, self._preferred_datatype, self._sample_format)
        dt = self.process(dt, *args, **kwargs)
        dt = dtype_converter(dt, self._sample_format, self._preferred_datatype)
        return dt

    def process(*args, **kwargs):
        """
        Base method for audio signal processing.
        
        This method should be implemented by specific effect classes
        to define their unique audio transformation logic.
        """
        ...

        