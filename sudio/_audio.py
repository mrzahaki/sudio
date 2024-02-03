"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""
import io
import os
from builtins import ValueError
import pandas as pd
from typing import Union

from sudio._register import Members as Mem
from sudio._port import Audio
import wave
# import pyaudio
# import time
import numpy as np
import struct
from sudio.extras.timed_indexed_string import TimedIndexedString
from sudio.types import DecodeError


def cache_write(*head,
                head_dtype: str = 'u8',
                data: Union[bytes, io.BufferedRandom] = None,
                file_name: str = None,
                file_mode: str = 'wb+',
                buffered_random: io.BufferedRandom = None,
                pre_truncate: bool = False,
                pre_seek: tuple = None,
                after_seek: tuple = None,
                pre_flush: bool = False,
                after_flush: bool = False,
                sizeon_out: bool = False,
                data_chunck: int = int(1e6)) -> Union[io.BufferedRandom, tuple]:

    if buffered_random:
        file: io.BufferedRandom = buffered_random
    elif file_name:
        file = open(file_name, file_mode)
    else:
        raise ValueError
    size = 0

    if pre_seek:
        file.seek(*pre_seek)

    if pre_truncate:
        file.truncate()

    if pre_flush:
        file.flush()

    if head:
        size = file.write(np.asarray(head, dtype=head_dtype))

    if data:
        if type(data) is io.BufferedRandom:
            buffer = data.read(data_chunck)
            while buffer:
                size += file.write(buffer)
                buffer = data.read(data_chunck)

        else:
            size += file.write(data)

    if after_seek:
        file.seek(*after_seek)

    if after_flush:
        file.flush()

    if sizeon_out:
        return file, size
    return file


def smart_cache(record: Union[pd.Series, dict],
                path_server: TimedIndexedString,
                master_obj,
                decoder: callable = None,
                sync_sample_format_id: int = None,
                sync_nchannels: int = None,
                sync_sample_rate: int = None,
                safe_load: bool = True) -> pd.Series:

    sync_sample_format_id = master_obj._sample_format if sync_sample_format_id is None else sync_sample_format_id
    sync_nchannels = master_obj.nchannels if sync_nchannels is None else sync_nchannels
    sync_sample_rate = master_obj._sample_rate if sync_sample_rate is None else sync_sample_rate

    path: str = path_server()
    cache = master_obj._cache()
    try:
        if path in cache:
            try:
                os.rename(path, path)
                f = record['o'] = open(path, 'rb+')
            except OSError:
                while 1:
                    new_path: str = path_server()
                    try:
                        if os.path.exists(new_path):
                            os.rename(new_path, new_path)
                            f = record['o'] = open(new_path, 'rb+')
                        else:
                            # write new file based on the new path
                            data_chunck = int(1e7)
                            with open(path, 'rb+') as pre_file:
                                f = record['o'] = open(new_path, 'wb+')
                                data = pre_file.read(data_chunck)
                                while data:
                                    f.write(data)
                                    data = pre_file.read(data_chunck)
                        break
                    except OSError:
                        # if data already opened in another process
                        continue

            f.seek(0, 0)
            cache_info = f.read(master_obj.__class__.CACHE_INFO)
            try:
                csize, cframe_rate, csample_format, cnchannels = np.frombuffer(cache_info, dtype='u8').tolist()
            except ValueError:
                # bad cache error
                f.close()
                os.remove(f.name)
                raise DecodeError

            csample_format = csample_format if csample_format else None
            record['size'] = csize
            # if os.path.getsize(path) > csize:
            #     f.seek(csize, 0)
            #     f.truncate()
            #     f.flush()
            #     f.seek(master_obj.__class__.CACHE_INFO, 0)

            record['frameRate'] = cframe_rate
            record['nchannels'] = cnchannels
            record['sampleFormat'] = csample_format

            if (cnchannels == sync_nchannels and
                    csample_format == sync_sample_format_id and
                    cframe_rate == sync_sample_rate):
                pass

            elif safe_load:
                # printiooi
                # print('noooooo')
                # print_en
                # print('path in cache safe load add file')
                record['o'] = f.read()
                record = master_obj.__class__._sync(record,
                                                    sync_nchannels,
                                                    sync_sample_rate,
                                                    sync_sample_format_id)

                f = record['o'] = (
                    cache_write(record['size'],
                                record['frameRate'],
                                record['sampleFormat'] if record['sampleFormat'] else 0,
                                record['nchannels'],
                                buffered_random=f,
                                data=record['o'],
                                pre_seek=(0, 0),
                                pre_truncate=True,
                                pre_flush=True,
                                after_seek=(master_obj.__class__.CACHE_INFO, 0),
                                after_flush=True)
                )

        else:
            raise DecodeError

    except DecodeError:

        if decoder is not None:
            record['o'] = decoder()
        if type(record['o']) is io.BufferedRandom:
            record['size'] = os.path.getsize(record['o'].name)
        else:
            record['size'] = len(record['o']) + master_obj.CACHE_INFO

        f = record['o'] = (
            cache_write(record['size'],
                        record['frameRate'],
                        record['sampleFormat'] if record['sampleFormat'] else 0,
                        record['nchannels'],
                        file_name=path,
                        data=record['o'],
                        pre_seek=(0, 0),
                        pre_truncate=True,
                        pre_flush=True,
                        after_seek=(master_obj.__class__.CACHE_INFO, 0),
                        after_flush=True)
        )

    record['duration'] = record['size'] / (record['frameRate'] *
                                           record['nchannels'] *
                                           Audio.get_sample_size(record['sampleFormat']))
    if type(record) is pd.Series:
        post = record['o'].name.index(master_obj.__class__.BUFFER_TYPE)
        pre = max(record['o'].name.rfind('\\'), record['o'].name.rfind('/'))
        pre = (pre + 1) if pre > 0 else 0
        record.name = record['o'].name[pre: post]

    return record

# output can be a 'array','wave' or standard wave 'frame'
# note:params sampwidth, nchannels, framerate just work in frame and array type
@Mem.sudio.add
def play(inp: object, sample_format: int = 2, nchannels: int = 1, framerate: int = 44100):
    audio = Audio()
    stream_out = audio.open_stream(
                        format=int(sample_format),
                        channels=int(nchannels),
                        rate=int(framerate),
                        input=False,
                        output=True)
    stream_out.start_stream()
    stream_out.write(inp)

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


# @Mem.sudio.add
# def bark_scale(freq):
#     return 13 * np.atan(freq * 76e-5) + 3.5 * np.atan((freq / 75e2) ** 2)
#
#
# @Mem.sudio.add
# def mel_scale(freq):
#     return 2595 * np.log10(1 + freq / 700)
#
#
# @Mem.sudio.add
# def erb_scale(freq):
#     return 24.7 * (1 + 4.37 * freq / 1000)


@Mem.sudio.add
def win_parser_mono(window, win_num):
    return window[win_num]


@Mem.sudio.add
def win_parser(window, win_num, nchannel):
    return window[nchannel][win_num]

