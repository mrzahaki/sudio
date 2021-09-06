# the name of allah
from ._register import Members as Mem
from ._tools import Tools
import wave
import pyaudio
import time
import numpy as np
import struct
# from scipy.signal import upfirdn, firwin, get_window
# import threading
# import queue

# output can be a 'array','wave' or standard wave 'frame'
# note:params sampwidth, nchannels, framerate just work in frame and array type
@Mem.sudio.add
def play(inp: object, inp_format: str = 'wave', sampwidth: int = 2, nchannels: int = 1, framerate: int = 44100) -> object:
    if inp_format == 'wave':
        inp.rewind()
        ding_data = inp.readframes(inp.getnframes())
        sampwidth = inp.getsampwidth()
        nchannels = inp.getnchannels()
        framerate = inp.getframerate()

    elif inp_format == 'frame':
        ding_data = inp

    else:
        ding_data = inp.astype(f'int{sampwidth * 8}').tostring()


    audio = pyaudio.PyAudio()
    stream_out = audio.open(
        format=audio.get_format_from_width(sampwidth),
        channels=nchannels,
        rate=framerate, input=False, output=True)
    # print(audio.get_format_from_width(inp.getsampwidth()))
    stream_out.start_stream()
    stream_out.write(ding_data)

    # time.sleep(0.2)

    stream_out.stop_stream()
    stream_out.close()
    audio.terminate()


# Format: h is short in C type
# Format: l is long in C type
# '?' -> _BOOL , 'h' -> short, 'i' -> int
# c -> char,
# s -> string example '3s'-> b'sbs'
# f -> float, q-> long long
# @  Byte order->native, Size->native, Alignment->native,
# =  Byte order->native, Size->standard, Alignment->none,
# <  Byte order->little-endian, Size->standard, Alignment->none,
# >  Byte order->big-endian, Size->standard, Alignment->none,
# !  Byte order->network (= big-endian), Size->standard, Alignment->none,
# A format character may be preceded by an integral repeat count.
#  For example, the format string '4h' means exactly the same as 'hhhh'.
# https://stackoverflow.com/questions/23154400/read-the-data-of-a-single-channel-from-a-stereo-wave-file-in-python
# output can be a 'array','wave' or standard wave 'frame'
# output ncahnnels is equal to input ncahnnels
# get all channels
@Mem.sudio.add
def get_channel(wav, ch_index=None, output='wave', out_wave_name='sound0'):

    nframes = wav.getnframes()
    ncahnnels = wav.getnchannels()

    frame_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(wav.getsampwidth())]  # 1 2 4 8
    wav.rewind()

    frame = wav.readframes(nframes)
    # convert the byte string into a list of ints, little-endian 16-bit samples
    signal = list(struct.unpack(f'<{nframes * ncahnnels}{frame_width}', frame))
    if not ch_index is None:
        ch_index %= ncahnnels
        # np.fromstring(signal, )
        signal = signal[ch_index::ncahnnels]

    if output == 'array':
        if ch_index is None:
            retval = []
            for i in range(ncahnnels):
                retval += [signal[i::ncahnnels]]
            return np.array(retval)
        return np.array(signal)

    elif output == 'frame':
            return frame
    else:
        obj = wave.open(f'{out_wave_name}.wav', 'w')
        if ch_index is None:
            obj.setnchannels(ncahnnels)
        else:
            obj.setnchannels(1)  # mono
        obj.setsampwidth(wav.getsampwidth())
        obj.setframerate(wav.getframerate())
        obj.writeframes(frame)
        obj.close()


# output/inp can be a 'array','wave' or standard wave 'frame' types
# output ncahnnels is equal to wav ncahnnels
# set inp as channel ch_index of wav
# inp must be mono(nchannels == 1)
@Mem.sudio.add
def set_channel(wav, ch_index, inp, inp_format='wave', output='wave', out_wave_name='sound0'):
    wav_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(wav.getsampwidth())]  # 1 2 4 8

    if inp_format == 'wave':
        assert (inp.getsampwidth() == wav.getsampwidth() and
                inp.getnframes() == wav.getnframes() and
                inp.getnchannels() == 1 and
                inp.getframerate == wav.getframerate())
        inp.rewind()
        signal = inp.readframes(inp.getnframes())
        frame_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(inp.getsampwidth())]  # 1 2 4 8
        assert wav_width == frame_width

        # convert the byte string into a list of ints, little-endian 16-bit samples
        signal = list(struct.unpack(f'<{inp.getnframes()}{frame_width}',
                                    signal))

    else:
        assert (len(inp) == wav.getnframes())

        if inp_format == 'frame':
            # convert the byte string into a list of ints, little-endian 16-bit samples
            signal = list(struct.unpack(f'<{wav.getnframes()}{wav_width}',
                                        inp))

        else:  # np array
            signal = inp.tolist()

    nchannels = wav.getnchannels()
    assert nchannels > 1
    ch_index %= nchannels

    wav.rewind()
    wav_signal = wav.readframes(wav.getnframes())
    # convert the byte string into a list of ints, little-endian 16-bit samples
    wav_signal = list(struct.unpack(f'<{wav.getnframes() * nchannels}{wav_width}',
                                    wav_signal))

    print(len(wav_signal), len(signal))
    for i, j in zip(range(ch_index, len(wav_signal), nchannels), range(ch_index, len(signal))):
        wav_signal[i] = signal[j]

    if output == 'array':
        return np.array(wav_signal)

    else:
        signal = struct.pack(f'<{len(wav_signal)}{wav_width}', *wav_signal)
        if output == 'frame':
            return signal
        else:
            obj = wave.open(f'./sounds/{out_wave_name}.wav', 'w')
            obj.setnchannels(nchannels)
            obj.setsampwidth(wav.getsampwidth())
            obj.setframerate(wav.getframerate())
            obj.writeframes(signal)
            obj.close()
            return wave.open(f'./sounds/{out_wave_name}.wav', 'r')

# output can be a 'array','wave' or standard wave 'frame' types
# output ncahnnels is equal to input ncahnnels
@Mem.sudio.add
def del_channel(wav, ch_index, output='wave', out_wave_name='sound0'):
    return set_channel(wav, ch_index, np.zeros(wav.getnframes(), dtype='int'), inp_format='array',
                             output=output, out_wave_name=out_wave_name)


@Mem.sudio.add
def bark_scale(freq):
    return 13 * np.atan(freq * 76e-5) + 3.5 * np.atan((freq / 75e2) ** 2)


@Mem.sudio.add
def mel_scale(freq):
    return 2595 * np.log10(1 + freq / 700)

@Mem.sudio.add
def erb_scale(freq):
    return 24.7 * (1 + 4.37 * freq / 1000)

@Mem.sudio.add
def win_parser_mono(window, win_num):
    return window[win_num]


@Mem.sudio.add
def win_parser(window, win_num, nchannel):
    return window[nchannel][win_num]

