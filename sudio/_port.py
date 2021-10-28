"""
with the name of ALLAH:
 SUDIO (https://github.com/MrZahaki/sudio)

 audio processing platform

 Author: hussein zahaki (hossein.zahaki.mansoor@gmail.com)

 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""

from ._register import Members as Mem
import sys
import os
import array
from enum import Enum
from typing import Generator, Dict, Set, Optional, Union, Any, Callable
from _miniaudio import ffi, lib
import pyaudio
try:
    import numpy
except ImportError:
    numpy = None

lib.init_miniaudio()



#
# @Mem.process.add
# def get_default_input_device_info(self):
#     return self._paudio.get_default_input_device_info()
#
#
# @Mem.process.add
# def get_default_output_device_info(self):
#     return self._paudio.get_default_output_device_info()
#
# @Mem.process.add
# def get_default_output_device_info(self, filename):
#     return miniaudio.get_file_info(filename)


class Audio(pyaudio.PyAudio):
    pass

    @staticmethod
    def get_sample_size(format):
        return pyaudio.get_sample_size(format)


@Mem.process.add
def get_default_input_device_info(self):
    au = Audio()
    data = au.get_default_input_device_info()
    au.terminate()
    return data


@Mem.process.add
def get_device_count(self):
    au = Audio()
    data = au.get_device_count()
    au.terminate()
    return data


@Mem.process.add
def get_device_info_by_index(self, index):
    au = Audio()
    data = au.get_device_info_by_index(index)
    au.terminate()
    return data


@Mem.process.add
def get_input_devices(self):
    p = Audio()
    dev = {}
    default_dev = p.get_default_input_device_info()['index']
    for i in range(p.get_device_count()):
        tmp = p.get_device_info_by_index(i)
        if tmp['maxInputChannels'] > 0:
            del tmp['index']
            del tmp['maxOutputChannels']
            dev[i] = tmp
            if i == default_dev:
                dev['defaultDevice'] = True
            else:
                dev['defaultDevice'] = False
    p.terminate()
    assert len(dev) > 0
    return dev


@Mem.process.add
def get_output_devices(self):
    p = Audio()
    dev = {}
    default_dev = p.get_default_output_device_info()['index']
    for i in range(p.get_device_count()):
        tmp = p.get_device_info_by_index(i)
        if tmp['maxOutputChannels'] > 0:
            del tmp['index']
            del tmp['maxInputChannels']
            dev[i] = tmp
            if i == default_dev:
                dev['defaultDevice'] = True
            else:
                dev['defaultDevice'] = False
    p.terminate()
    assert len(dev) > 0
    return dev


# # @Mem.process.add
# def file(self, file, name=None):
#     return self._File(self, file, name)
#
# # @Mem.process.add
# class _File:
#     pass

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

class SoundFileInfo:
    """Contains various properties of an audio file."""
    def __init__(self, name: str, file_format: FileFormat, nchannels: int, sample_rate: int,
                 sample_format: LibSampleFormat, duration: float, num_frames: int) -> None:
        self.name = name
        self.nchannels = nchannels
        self.sample_rate = sample_rate
        self.sample_format = sample_format
        self.sample_format_name = ffi.string(lib.ma_get_format_name(sample_format.value)).decode()
        self.sample_width = _width_from_format(sample_format)
        self.num_frames = num_frames
        self.duration = duration
        self.file_format = file_format

    def __str__(self) -> str:
        return "<{clazz}: '{name}' {nchannels} ch, {sample_rate} hz, {sample_format.name}, " \
               "{num_frames} frames={duration:.2f} sec.>".format(clazz=self.__class__.__name__, **(vars(self)))

    def __repr__(self) -> str:
        return str(self)


class MiniaudioError(Exception):
    """When a miniaudio specific error occurs."""
    pass


class DecodeError(MiniaudioError):
    """When something went wrong during decoding an audio file."""
    pass


@Mem.process.add
def get_file_info(filename: str) -> SoundFileInfo:
    """Fetch some information about the audio file."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".ogg", ".vorbis"):
        return vorbis_get_file_info(filename)
    elif ext == ".mp3":
        return mp3_get_file_info(filename)
    elif ext == ".flac":
        return flac_get_file_info(filename)
    elif ext == ".wav":
        return wav_get_file_info(filename)
    raise DecodeError("unsupported file format")


# @Mem.sudio.add
def vorbis_get_file_info(filename: str) -> SoundFileInfo:
    """Fetch some information about the audio file (vorbis format)."""
    filenamebytes = _get_filename_bytes(filename)
    with ffi.new("int *") as error:
        vorbis = lib.stb_vorbis_open_filename(filenamebytes, error, ffi.NULL)
        if not vorbis:
            raise DecodeError("could not open/decode file")
        try:
            info = lib.stb_vorbis_get_info(vorbis)
            duration = lib.stb_vorbis_stream_length_in_seconds(vorbis)
            num_frames = lib.stb_vorbis_stream_length_in_samples(vorbis)
            return SoundFileInfo(filename, FileFormat.VORBIS, info.channels, info.sample_rate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.stb_vorbis_close(vorbis)


# @Mem.sudio.add
def vorbis_get_info(data: bytes) -> SoundFileInfo:
    """Fetch some information about the audio data (vorbis format)."""
    with ffi.new("int *") as error:
        vorbis = lib.stb_vorbis_open_memory(data, len(data), error, ffi.NULL)
        if not vorbis:
            raise DecodeError("could not open/decode data")
        try:
            info = lib.stb_vorbis_get_info(vorbis)
            duration = lib.stb_vorbis_stream_length_in_seconds(vorbis)
            num_frames = lib.stb_vorbis_stream_length_in_samples(vorbis)
            return SoundFileInfo("<memory>", FileFormat.VORBIS, info.channels, info.sample_rate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.stb_vorbis_close(vorbis)


# @Mem.sudio.add
def flac_get_file_info(filename: str) -> SoundFileInfo:
    """Fetch some information about the audio file (flac format)."""
    filenamebytes = _get_filename_bytes(filename)
    flac = lib.drflac_open_file(filenamebytes, ffi.NULL)
    if not flac:
        raise DecodeError("could not open/decode file")
    try:
        duration = flac.totalPCMFrameCount / flac.sampleRate
        sample_width = flac.bitsPerSample // 8
        return SoundFileInfo(filename, FileFormat.FLAC, flac.channels, flac.sampleRate,
                             _format_from_width(sample_width), duration, flac.totalPCMFrameCount)
    finally:
        lib.drflac_close(flac)

# @Mem.sudio.add
def flac_get_info(data: bytes) -> SoundFileInfo:
    """Fetch some information about the audio data (flac format)."""
    flac = lib.drflac_open_memory(data, len(data), ffi.NULL)
    if not flac:
        raise DecodeError("could not open/decode data")
    try:
        duration = flac.totalPCMFrameCount / flac.sampleRate
        sample_width = flac.bitsPerSample // 8
        return SoundFileInfo("<memory>", FileFormat.FLAC, flac.channels, flac.sampleRate,
                             _format_from_width(sample_width), duration, flac.totalPCMFrameCount)
    finally:
        lib.drflac_close(flac)


# @Mem.sudio.add
def mp3_get_file_info(filename: str) -> SoundFileInfo:
    """Fetch some information about the audio file (mp3 format)."""
    filenamebytes = _get_filename_bytes(filename)
    with ffi.new("drmp3 *") as mp3:
        if not lib.drmp3_init_file(mp3, filenamebytes, ffi.NULL):
            raise DecodeError("could not open/decode file")
        try:
            num_frames = lib.drmp3_get_pcm_frame_count(mp3)
            duration = num_frames / mp3.sampleRate
            return SoundFileInfo(filename, FileFormat.MP3, mp3.channels, mp3.sampleRate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.drmp3_uninit(mp3)


# @Mem.sudio.add
def mp3_get_info(data: bytes) -> SoundFileInfo:
    """Fetch some information about the audio data (mp3 format)."""
    with ffi.new("drmp3 *") as mp3:
        if not lib.drmp3_init_memory(mp3, data, len(data), ffi.NULL):
            raise DecodeError("could not open/decode data")
        try:
            num_frames = lib.drmp3_get_pcm_frame_count(mp3)
            duration = num_frames / mp3.sampleRate
            return SoundFileInfo("<memory>", FileFormat.MP3, mp3.channels, mp3.sampleRate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.drmp3_uninit(mp3)


def wav_get_file_info(filename: str) -> SoundFileInfo:
    """Fetch some information about the audio file (wav format)."""
    filenamebytes = _get_filename_bytes(filename)
    with ffi.new("drwav*") as wav:
        if not lib.drwav_init_file(wav, filenamebytes, ffi.NULL):
            raise DecodeError("could not open/decode file")
        try:
            duration = wav.totalPCMFrameCount / wav.sampleRate
            sample_width = wav.bitsPerSample // 8
            is_float = wav.translatedFormatTag == lib.DR_WAVE_FORMAT_IEEE_FLOAT
            return SoundFileInfo(filename, FileFormat.WAV, wav.channels, wav.sampleRate,
                                 _format_from_width(sample_width, is_float), duration, wav.totalPCMFrameCount)
        finally:
            lib.drwav_uninit(wav)


def wav_get_info(data: bytes) -> SoundFileInfo:
    """Fetch some information about the audio data (wav format)."""
    with ffi.new("drwav*") as wav:
        if not lib.drwav_init_memory(wav, data, len(data), ffi.NULL):
            raise DecodeError("could not open/decode data")
        try:
            duration = wav.totalPCMFrameCount / wav.sampleRate
            sample_width = wav.bitsPerSample // 8
            is_float = wav.translatedFormatTag == lib.DR_WAVE_FORMAT_IEEE_FLOAT
            return SoundFileInfo("<memory>", FileFormat.WAV, wav.channels, wav.sampleRate,
                                 _format_from_width(sample_width, is_float), duration, wav.totalPCMFrameCount)
        finally:
            lib.drwav_uninit(wav)

@Mem.sudio.add
def wav_write_file(filename: str, data, nchannels, sample_rate, sample_width) -> None:
    """Writes the pcm sound to a WAV file"""
    with ffi.new("drwav_data_format*") as fmt, ffi.new("drwav*") as pwav:
        fmt.container = lib.drwav_container_riff
        fmt.format = lib.DR_WAVE_FORMAT_PCM
        fmt.channels = nchannels
        fmt.sampleRate = sample_rate
        fmt.bitsPerSample = sample_width * 8
        # what about floating point format?
        filename_bytes = filename.encode(sys.getfilesystemencoding())
        buffer_len = int(len(data) / sample_width)
        if not lib.drwav_init_file_write_sequential(pwav, filename_bytes,
                                                    fmt, buffer_len, ffi.NULL):
            raise IOError("can't open file for writing")
        try:
            lib.drwav_write_pcm_frames(pwav, int(buffer_len / nchannels), data)
        finally:
            lib.drwav_uninit(pwav)


def _create_int_array(itemsize: int) -> array.array:
    for typecode in "Bhilq":
        a = array.array(typecode)
        if a.itemsize == itemsize:
            return a
    raise ValueError("cannot create array")


def _get_filename_bytes(filename: str) -> bytes:
    filename2 = os.path.expanduser(filename)
    if not os.path.isfile(filename2):
        raise FileNotFoundError(filename)
    return filename2.encode(sys.getfilesystemencoding())


def _width_from_format(sampleformat: LibSampleFormat) -> int:
    widths = {
        LibSampleFormat.UNSIGNED8: 1,
        LibSampleFormat.SIGNED16: 2,
        LibSampleFormat.SIGNED24: 3,
        LibSampleFormat.SIGNED32: 4,
        LibSampleFormat.FLOAT32: 4
    }
    if sampleformat in widths:
        return widths[sampleformat]
    raise MiniaudioError("unsupported sample format", sampleformat)


def _array_proto_from_format(sampleformat: LibSampleFormat) -> array.array:
    arrays = {
        LibSampleFormat.UNSIGNED8: _create_int_array(1),
        LibSampleFormat.SIGNED16: _create_int_array(2),
        LibSampleFormat.SIGNED32: _create_int_array(4),
        LibSampleFormat.FLOAT32: array.array('f')
    }
    if sampleformat in arrays:
        return arrays[sampleformat]
    raise MiniaudioError("the requested sample format can not be used directly: "
                         + sampleformat.name + " (convert it first)")


def _format_from_width(sample_width: int, is_float: bool = False) -> LibSampleFormat:
    if is_float:
        return LibSampleFormat.FLOAT32
    elif sample_width == 1:
        return LibSampleFormat.UNSIGNED8
    elif sample_width == 2:
        return LibSampleFormat.SIGNED16
    elif sample_width == 3:
        return LibSampleFormat.SIGNED24
    elif sample_width == 4:
        return LibSampleFormat.SIGNED32
    else:
        raise MiniaudioError("unsupported sample width", sample_width)


def decode_file(filename: str, output_format: LibSampleFormat = LibSampleFormat.SIGNED16,
                nchannels: int = 2, sample_rate: int = 44100, dither: DitherMode = DitherMode.NONE):
    """Convenience function to decode any supported audio file to raw PCM samples in your chosen format."""
    sample_width = _width_from_format(output_format)
    filenamebytes = _get_filename_bytes(filename)
    with ffi.new("ma_uint64 *") as frames, ffi.new("void **") as memory:
        decoder_config = lib.ma_decoder_config_init(output_format.value, nchannels, sample_rate)
        decoder_config.ditherMode = dither.value
        result = lib.ma_decode_file(filenamebytes, ffi.addressof(decoder_config), frames, memory)
        if result != lib.MA_SUCCESS:
            raise DecodeError("failed to decode file", result)
        buflen = frames[0] * nchannels * sample_width
        buffer = ffi.buffer(memory[0], buflen)
        # byte = bytearray(buflen)
        # byte[:] = buffer
        byte = bytes(buffer)
        lib.ma_free(memory[0], ffi.NULL)
        return byte

def _samples_stream_generator(frames_to_read: int, nchannels: int, output_format: LibSampleFormat,
                              decoder: ffi.CData, data: Any,
                              on_close: Optional[Callable] = None) -> Generator[array.array, int, None]:
    _reference = data    # make sure any data passed in is not garbage collected
    sample_width = _width_from_format(output_format)
    samples_proto = _array_proto_from_format(output_format)
    allocated_buffer_frames = max(frames_to_read, 16384)
    try:
        with ffi.new("int8_t[]", allocated_buffer_frames * nchannels * sample_width) as decodebuffer:
            buf_ptr = ffi.cast("void *", decodebuffer)
            want_frames = (yield samples_proto) or frames_to_read
            while True:
                num_frames = lib.ma_decoder_read_pcm_frames(decoder, buf_ptr, want_frames)
                if num_frames <= 0:
                    break
                buffer = ffi.buffer(decodebuffer, num_frames * sample_width * nchannels)
                samples = array.array(samples_proto.typecode)
                samples.frombytes(buffer)
                want_frames = (yield samples) or frames_to_read
    finally:
        if on_close:
            on_close()
        lib.ma_decoder_uninit(decoder)


def stream_file(filename: str, output_format: LibSampleFormat = LibSampleFormat.SIGNED16, nchannels: int = 2,
                sample_rate: int = 44100, frames_to_read: int = 1024,
                dither: DitherMode = DitherMode.NONE, seek_frame: int = 0) -> Generator[array.array, int, None]:
    """
    Convenience generator function to decode and stream any supported audio file
    as chunks of raw PCM samples in the chosen format.
    If you send() a number into the generator rather than just using next() on it,
    you'll get that given number of frames, instead of the default configured amount.
    This is particularly useful to plug this stream into an audio device callback that
    wants a variable number of frames per call.
    """
    filenamebytes = _get_filename_bytes(filename)
    decoder = ffi.new("ma_decoder *")
    decoder_config = lib.ma_decoder_config_init(output_format.value, nchannels, sample_rate)
    decoder_config.ditherMode = dither.value
    result = lib.ma_decoder_init_file(filenamebytes, ffi.addressof(decoder_config), decoder)
    if result != lib.MA_SUCCESS:
        raise DecodeError("failed to init decoder", result)
    if seek_frame > 0:
        result = lib.ma_decoder_seek_to_pcm_frame(decoder, seek_frame)
        if result != lib.MA_SUCCESS:
            raise DecodeError("failed to seek to frame", result)
    g = _samples_stream_generator(frames_to_read, nchannels, output_format, decoder, None)
    dummy = next(g)
    assert len(dummy) == 0
    return g
