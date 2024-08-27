from _miniaudio import ffi, lib
from sudio.audiosys.typeconversion import get_sample_width_from_format
from sudio.types import LibSampleFormat, FileFormat

class AudioFileInfo:
    """
    Contains various properties of an audio file.

    Attributes:
    - name (str): The name or path of the audio file.
    - file_format (FileFormat): The file format of the audio file (Enum).
    - nchannels (int): The number of audio channels.
    - sample_rate (int): The sample rate of the audio file.
    - sample_format (LibSampleFormat): The sample format in memory (Enum).
    - sample_format_name (str): The human-readable name of the sample format.
    - sample_width (int): The sample width in bytes.
    - num_frames (int): The total number of frames in the audio file.
    - duration (float): The duration of the audio file in seconds.
    """

    def __init__(self, name: str, file_format: FileFormat, nchannels: int, sample_rate: int,
                 sample_format: LibSampleFormat, duration: float, num_frames: int) -> None:
        """
        Initializes an AudioFileInfo object with specified audio file properties.

        Parameters:
        - name (str): The name or path of the audio file.
        - file_format (FileFormat): The file format of the audio file (Enum).
        - nchannels (int): The number of audio channels.
        - sample_rate (int): The sample rate of the audio file.
        - sample_format (LibSampleFormat): The sample format in memory (Enum).
        - duration (float): The duration of the audio file in seconds.
        - num_frames (int): The total number of frames in the audio file.
        """
        self.name = name
        self.nchannels = nchannels
        self.sample_rate = sample_rate
        self.sample_format = sample_format
        self.sample_format_name = ffi.string(lib.ma_get_format_name(sample_format.value)).decode()
        self.sample_width = get_sample_width_from_format(sample_format)
        self.num_frames = num_frames
        self.duration = duration
        self.file_format = file_format

    def __str__(self) -> str:
        """
        Returns a string representation of the AudioFileInfo object.

        Returns:
        - str: String representation of the object.
        """
        return "<{clazz}: '{name}' {nchannels} ch, {sample_rate} Hz, {sample_format_name}, " \
               "{num_frames} frames={duration:.2f} sec.>".format(clazz=self.__class__.__name__, **(vars(self)))

    def __repr__(self) -> str:
        """
        Returns a string representation of the AudioFileInfo object.

        Returns:
        - str: String representation of the object.
        """
        return str(self)
