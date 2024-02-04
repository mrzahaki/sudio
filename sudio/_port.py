"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""

import pyaudio
import sys
import array
from typing import Generator, Optional, Any, Callable
from _miniaudio import ffi, lib
lib.init_miniaudio()

from sudio._register import Members as Mem
from sudio.types import LibSampleFormat, DitherMode, DecodeError
from sudio.extras.arraytool import get_array_proto_from_format
from sudio.audioutils.typeconversion import get_sample_width_from_format
from sudio.extras.io import get_encoded_filename_bytes



class Audio(pyaudio.PyAudio):

    def open_stream(self,
                    *args,
                    **kwargs
                    ):
        to_int = Audio.to_int
        to_int(kwargs, 'format')
        to_int(kwargs, 'channels')
        to_int(kwargs, 'rate')
        to_int(kwargs, 'frames_per_buffer')
        to_int(kwargs, 'input_device_index')
        to_int(kwargs, 'output_device_index')

        return self.open(*args, **kwargs)

    @staticmethod
    def get_sample_size(format: int):
        return pyaudio.get_sample_size(int(format))

    @staticmethod
    def to_int(obj, key):
        try:
            if not obj[key] is None:
                obj[key] = int(obj[key])
        except KeyError:
            pass

@Mem.master.add
def get_default_input_device_info(self):
    au = Audio()
    data = au.get_default_input_device_info()
    au.terminate()
    return data


@Mem.master.add
def get_device_count(self):
    au = Audio()
    data = au.get_device_count()
    au.terminate()
    return data


@Mem.master.add
def get_device_info_by_index(self, index: int):
    au = Audio()
    data = au.get_device_info_by_index(int(index))
    au.terminate()
    return data


@Mem.master.add
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


@Mem.master.add
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



def decode_file(filename: str, output_format: LibSampleFormat = LibSampleFormat.SIGNED16,
                nchannels: int = 2, sample_rate: int = 44100, dither: DitherMode = DitherMode.NONE):
    """Convenience function to decode any supported audio file to raw PCM samples in your chosen format."""
    sample_width = get_sample_width_from_format(output_format)
    filenamebytes = get_encoded_filename_bytes(filename)
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
    sample_width = get_sample_width_from_format(output_format)
    samples_proto = get_array_proto_from_format(output_format)
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
    filenamebytes = get_encoded_filename_bytes(filename)
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
