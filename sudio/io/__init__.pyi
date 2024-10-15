
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


from typing import List, Tuple, Callable, Optional, Union
import enum

class FileFormat(enum.Enum):
    """
    Supported audio file formats.

    Enum values:
    - UNKNOWN
    - WAV
    - FLAC
    - VORBIS
    - MP3
    """
    UNKNOWN: FileFormat
    WAV: FileFormat
    FLAC: FileFormat
    VORBIS: FileFormat
    MP3: FileFormat

class SampleFormat(enum.Enum):
    """
    Audio sample formats.

    Enum values:
    - UNKNOWN
    - UNSIGNED8
    - SIGNED16
    - SIGNED24
    - SIGNED32
    - FLOAT32
    """
    UNKNOWN: SampleFormat
    UNSIGNED8: SampleFormat
    SIGNED16: SampleFormat
    SIGNED24: SampleFormat
    SIGNED32: SampleFormat
    FLOAT32: SampleFormat

class DitherMode(enum.Enum):
    """
    Dithering modes for audio processing.

    Enum values:
    - NONE
    - RECTANGLE
    - TRIANGLE
    """
    NONE: DitherMode
    RECTANGLE: DitherMode
    TRIANGLE: DitherMode

class AudioFileInfo:
    """
    Holds information about an audio file.

    Attributes:
    - name (str): File name
    - file_format (FileFormat): Format of the file (FileFormat enum)
    - nchannels (int): Number of audio channels
    - sample_rate (int): Sampling rate in Hz
    - sample_format (SampleFormat): Format of audio samples (SampleFormat enum)
    - num_frames (int): Total number of audio frames
    - duration (float): Duration of the audio in seconds
    """
    name: str
    file_format: FileFormat
    nchannels: int
    sample_rate: int
    sample_format: SampleFormat
    num_frames: int
    duration: float

class AudioDeviceInfo:
    """
    Information about an audio device.

    Attributes:
    - index (int): Device index
    - name (str): Device name
    - max_input_channels (int): Maximum number of input channels
    - max_output_channels (int): Maximum number of output channels
    - default_sample_rate (int): Default sampling rate of the device
    - is_default_input (bool): True if this is the default input device
    - is_default_output (bool): True if this is the default output device
    """
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool
    is_default_output: bool

class AudioStream:
    """
    Handles real-time audio streaming for input and output.

    Methods:
    - open(): Set up the audio stream
    - start(): Begin audio streaming
    - stop(): Halt audio streaming
    - close(): Clean up and close the stream
    - read_stream(): Read audio data from the stream
    - write_stream(): Write audio data to the stream
    - get_stream_read_available(): Check available frames for reading
    - get_stream_write_available(): Check available frames for writing

    Static methods:
    - get_input_devices(): List all input devices
    - get_output_devices(): List all output devices
    - get_default_input_device(): Get info on the default input device
    - get_default_output_device(): Get info on the default output device
    - get_device_count(): Count all audio devices
    - get_device_info_by_index(): Get info on a specific device by index
    """
    def __init__(self) -> None: ...
    def open(self,
             input_dev_index: Optional[int] = None,
             output_dev_index: Optional[int] = None,
             sample_rate: Optional[float] = None,
             format: Optional[SampleFormat] = None,
             input_channels: Optional[int] = None,
             output_channels: Optional[int] = None,
             frames_per_buffer: Optional[int] = None,
             enable_input: Optional[bool] = None,
             enable_output: Optional[bool] = None,
             stream_flags: Optional[int] = None,
             input_callback: Optional[Callable[[bytes, int, SampleFormat], Tuple[bytes, bool]]] = None,
             output_callback: Optional[Callable[[int, SampleFormat], Tuple[bytes, bool]]] = None
             ) -> None: 
        """
        Open an audio stream.

        Args:
        - input_dev_index (int, optional): Input device index
        - output_dev_index (int, optional): Output device index
        - sample_rate (float, optional): Audio sample rate
        - format (SampleFormat, optional): Audio sample format
        - input_channels (int, optional): Number of input channels
        - output_channels (int, optional): Number of output channels
        - frames_per_buffer (int, optional): Number of frames per buffer
        - enable_input (bool, optional): Enable input streaming
        - enable_output (bool, optional): Enable output streaming
        - stream_flags (int, optional): PortAudio stream flags
        - input_callback (Callable, optional): Input callback function
        - output_callback (Callable, optional): Output callback function
        """
        pass
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def read_stream(self, frames: int) -> Tuple[bytes, int]: 
        """
        Read audio data from the stream.

        Args:
        - frames (int): Number of frames to read

        Returns:
        - Tuple[bytes, int]: Raw audio bytes and number of frames read
        """
        pass
    def write_stream(self, data: bytes) -> int: 
        """
        Write audio data to the stream.

        Args:
        - data (bytes): Raw audio data to write

        Returns:
        - int: Number of frames written
        """
        pass
    def get_stream_read_available(self) -> int: ...
    def get_stream_write_available(self) -> int: ...
    
    @staticmethod
    def get_input_devices() -> List[AudioDeviceInfo]: ...
    @staticmethod
    def get_output_devices() -> List[AudioDeviceInfo]: ...
    @staticmethod
    def get_default_input_device() -> AudioDeviceInfo: ...
    @staticmethod
    def get_default_output_device() -> AudioDeviceInfo: ...
    @staticmethod
    def get_device_count() -> int: ...
    @staticmethod
    def get_device_info_by_index(index: int) -> AudioDeviceInfo: ...

def write_to_default_output(
    data: bytes,
    format: SampleFormat = SampleFormat.FLOAT32,
    channels: Optional[int] = None,
    sample_rate: Optional[float] = None
) -> None: 
    """
    Write raw audio data to the default output device.

    Args:
    - data (bytes): Raw PCM audio data
    """
    pass

def get_sample_size(format: SampleFormat) -> int: ...

class codec:
    """
    Audio codec submodule.

    Methods:
    - decode_audio_file(): Decode an audio file to raw PCM data
    - encode_wav_file(): Encode raw PCM data to a WAV file
    - get_file_info(): Get information about an audio file
    - stream_audio_file(): Stream audio data from a file
    """

    class PyAudioStreamIterator:
        """
        An iterator that reads audio data from a stream.

        Methods:
        - __iter__: Returns the iterator instance
        - __next__: Returns the next chunk of audio data

        Args:
        - stream (AudioFileStream): The audio stream object
        - frames_to_read (int): Number of frames to read at each step
        """
        def __iter__(self) -> 'codec.PyAudioStreamIterator': ...
        def __next__(self) -> bytes: ...

    @staticmethod
    def decode_audio_file(
        filename: str,
        output_format: SampleFormat = SampleFormat.SIGNED16,
        nchannels: int = 2,
        sample_rate: int = 44100,
        dither: DitherMode = DitherMode.NONE
    ) -> bytes: 
        """
        Decode an audio file to raw PCM data.

        Args:
        - filename (str): Path to the audio file
        - output_format (SampleFormat, optional): Desired output sample format
        - nchannels (int, optional): Number of channels for output
        - sample_rate (int, optional): Desired output sample rate
        - dither (DitherMode, optional): Dithering mode to use

        Returns:
        - bytes: Raw PCM audio data
        """
        pass

    @staticmethod
    def encode_wav_file(
        filename: str,
        data: bytes,
        format: SampleFormat,
        nchannels: int,
        sample_rate: int
    ) -> None: 
        """
        Encode raw PCM data to a WAV file.

        Args:
        - filename (str): Path to save the WAV file
        - data (bytes): Raw PCM audio data
        - format (SampleFormat): Audio sample format
        - nchannels (int): Number of audio channels
        - sample_rate (int): Audio sample rate
        """
        pass

    @staticmethod
    def get_file_info(filename: str) -> AudioFileInfo: 
        """
        Get information about an audio file.

        Args:
        - filename (str): Path to the audio file

        Returns:
        - AudioFileInfo: Information about the file
        """
        pass

    @staticmethod
    def stream_audio_file(
        filename: str,
        output_format: SampleFormat = SampleFormat.SIGNED16,
        nchannels: int = 2,
        sample_rate: int = 44100,
        frames_to_read: int = 1024,
        dither: DitherMode = DitherMode.NONE,
        seek_frame: int = 0
    ) -> PyAudioStreamIterator: 
        """
        Stream audio data from a file.

        Args:
        - filename (str): Path to the audio file
        - output_format (SampleFormat, optional): Desired output sample format
        - nchannels (int, optional): Number of channels for output
        - sample_rate (int, optional): Desired output sample rate
        - frames_to_read (int, optional): Number of frames to read at a time
        - dither (DitherMode, optional): Dithering mode to use
        - seek_frame (int, optional): Starting frame for reading

        Returns:
        - PyAudioStreamIterator: Iterator yielding chunks of audio data
        """
        pass

    class AudioFileStream:
        """
        Audio file stream for reading PCM data.

        Methods:
        - read_frames(): Read audio frames from the stream

        Args:
        - filename (str): Path to the audio file
        - output_format (SampleFormat): Desired output sample format
        - nchannels (int): Number of channels for output
        - sample_rate (int): Desired output sample rate
        - frames_to_read (int): Number of frames to read per call
        - dither (DitherMode): Dithering mode to use
        - seek_frame (int): Starting frame for reading
        """
        def __init__(self,
                     filename: str,
                     output_format: SampleFormat = SampleFormat.SIGNED16,
                     nchannels: int = 2,
                     sample_rate: int = 44100,
                     frames_to_read: int = 1024,
                     dither: DitherMode = DitherMode.NONE,
                     seek_frame: int = 0) -> None: ...
        def read_frames(self, frames_to_read: int = 0) -> bytes: ...