from enum import Enum
from _miniaudio import lib

class StreamMode(Enum):
    """Enumeration representing different stream modes."""
    normal = 0
    optimized = 1


class SampleFormat(Enum):
    """Enumeration representing different audio sample formats."""
    formatFloat32 = 1
    formatInt32 = 2
    # formatInt24 = 4
    formatInt16 = 8
    formatInt8 = 16
    formatUInt8 = 32
    formatUnknown = None


class FileFormat(Enum):
    """Enumeration representing different audio file formats."""
    UNKNOWN = 0
    WAV = 1
    FLAC = 2
    VORBIS = 3
    MP3 = 4


class LibSampleFormat(Enum):
    """Enumeration representing different sample formats in memory."""
    UNKNOWN = lib.ma_format_unknown
    UNSIGNED8 = lib.ma_format_u8
    SIGNED16 = lib.ma_format_s16
    SIGNED24 = lib.ma_format_s24
    SIGNED32 = lib.ma_format_s32
    FLOAT32 = lib.ma_format_f32


class DitherMode(Enum):
    """Enumeration representing different dithering modes when converting."""
    NONE = lib.ma_dither_mode_none
    RECTANGLE = lib.ma_dither_mode_rectangle
    TRIANGLE = lib.ma_dither_mode_triangle
