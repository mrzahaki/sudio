"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""

from enum import Enum
from _miniaudio import lib

class StreamMode(Enum):
    normal = 0
    optimized = 1


class RefreshError(Exception):
    pass

class MiniaudioError(Exception):
    """When a miniaudio specific error occurs."""
    pass


class DecodeError(MiniaudioError):
    """When something went wrong during decoding an audio file."""
    pass

class StreamError(Exception):
    pass

class Name:
    def __get__(self, instance, owner):
        return instance._rec.name

    def __set__(self, instance, value):
        instance._rec.name = value



class SampleFormat(Enum):
    formatFloat32 = 1
    formatInt32 = 2
    # formatInt24 = 4
    formatInt16 = 8
    formatInt8 = 16
    formatUInt8 = 32
    formatUnknown = None


class FileFormat(Enum):
    """Audio file format"""
    UNKNOWN = 0
    WAV = 1
    FLAC = 2
    VORBIS = 3
    MP3 = 4


class LibSampleFormat(Enum):
    """Sample format in memory"""
    UNKNOWN = lib.ma_format_unknown
    UNSIGNED8 = lib.ma_format_u8
    SIGNED16 = lib.ma_format_s16
    SIGNED24 = lib.ma_format_s24
    SIGNED32 = lib.ma_format_s32
    FLOAT32 = lib.ma_format_f32


class DitherMode(Enum):
    """How to dither when converting"""
    NONE = lib.ma_dither_mode_none
    RECTANGLE = lib.ma_dither_mode_rectangle
    TRIANGLE = lib.ma_dither_mode_triangle

SampleMap = {
    SampleFormat.formatInt16 : LibSampleFormat.SIGNED16,
    SampleFormat.formatFloat32 : LibSampleFormat.FLOAT32,
    SampleFormat.formatInt32 : LibSampleFormat.SIGNED32,
    SampleFormat.formatUInt8 : LibSampleFormat.UNSIGNED8,
    SampleFormat.formatInt8 : LibSampleFormat.UNSIGNED8,
    SampleFormat.formatUnknown : LibSampleFormat.UNKNOWN,
}

ISampleMap = {value : key for (key, value) in SampleMap.items()}

SampleMapValue = {key.value : value for (key, value) in SampleMap.items()}

ISampleFormat = {key.value : key for (key, value) in SampleMap.items()}        