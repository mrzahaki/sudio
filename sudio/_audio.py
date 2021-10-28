"""
with the name of ALLAH:
 SUDIO (https://github.com/MrZahaki/sudio)

 audio processing platform

 Author: hussein zahaki (hossein.zahaki.mansoor@gmail.com)

 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""
import io
import os
from builtins import ValueError
from contextlib import contextmanager
import pandas as pd

from ._register import Members as Mem
from ._tools import Tools
from ._port import *
import wave
# import pyaudio
# import time
import numpy as np
import struct
import scipy.signal as scisig


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
                path_server: Tools.IndexedName,
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
    stream_out = audio.open(
        format=sample_format,
        channels=nchannels,
        rate=framerate, input=False, output=True)
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


class Name:
    def __get__(self, instance, owner):
        return instance._rec.name

    def __set__(self, instance, value):
        instance._rec.name = value


@Mem.sudio.add
class WrapGenerator:
    name = Name()
    def __init__(self, other, record):

        rec_type = type(record)
        if rec_type is pd.Series:
            self._rec: pd.Series = record
            if type(record['o']) is bytes:
                record['o'] = (
                    cache_write(record['size'],
                                record['frameRate'],
                                record['sampleFormat'] if record['sampleFormat'] else 0,
                                record['nchannels'],
                                file_name=other.__class__.DATA_PATH + record.name + other.__class__.BUFFER_TYPE,
                                data=record['o'],
                                pre_truncate=True,
                                after_seek=(other.__class__.CACHE_INFO, 0),
                                after_flush=True)
                )
                record['size'] += other.__class__.CACHE_INFO

        elif rec_type is str:
            self._rec: pd.Series = other.load(record, series=True)

        self._parent = other
        self._file: io.BufferedRandom = self._rec['o']
        self._name = Tools.IndexedName(self._file.name,
                                       seed='wrrapped',
                                       start_before=self._parent.__class__.BUFFER_TYPE)
        self._size = self._rec['size']
        self._duration = record['duration']
        self._frame_rate = self._rec['frameRate']
        self._nchannels = self._rec['nchannels']
        self._sample_format = self._rec['sampleFormat']
        self._nperseg = self._rec['nperseg']
        self._sample_type = self._parent._constants[0]
        self.sample_width = Audio.get_sample_size(self._rec['sampleFormat'])
        self._data = b''
        self._seek = 0

    def __call__(self,
                 *args,
                 sync_sample_format_id: int = None,
                 sync_nchannels: int = None,
                 sync_sample_rate: int = None,
                 safe_load: bool = True
                 ):

        # create new cache file and new record based on new cache
        record = (
            smart_cache(self._rec.copy(),
                        self._name,
                        self._parent,
                        sync_sample_format_id=sync_sample_format_id,
                        sync_nchannels=sync_nchannels,
                        sync_sample_rate=sync_sample_rate,
                        safe_load=safe_load)
        )

        return Wrap(self._parent,
                    record,
                    self)

    def get_data(self):
        return self._rec.copy()

    @contextmanager
    def get(self, offset=None, whence=None):
        try:
            # self._file.flush()
            if offset is not None and whence is not None:
                self._seek = self._file.seek(offset, whence)
            else:
                self._seek = self._file.tell()

            yield self._file
        finally:
            self._file.seek(self._seek, 0)

    def set_data(self, data):
        self._data = data

    def get_sample_format(self):
        return ISampleFormat[self._sample_format]

    def get_sample_width(self):
        return self.sample_width

    def get_master(self):
        return self._parent

    def get_size(self):
        return os.path.getsize(self._file.name) - self._file.tell()

    def get_cache_size(self):
        return os.path.getsize(self._file.name)

    def get_frame_rate(self):
        return self._frame_rate

    def get_nchannels(self):
        return self._nchannels

    def get_duration(self):
        return self._duration

    def join(self,
             *other,
             sync_sample_format_id: int = None,
             sync_nchannels: int = None,
             sync_sample_rate: int = None,
             safe_load: bool = True
             ):

        return (self.__call__(sync_sample_format_id=sync_sample_format_id,
                             sync_nchannels=sync_nchannels,
                             sync_sample_rate=sync_sample_rate,
                             safe_load=safe_load).join(*other))

    def __getitem__(self, item):
        return self.__call__().__getitem__(item)

    def __del__(self):
        if not self._file.closed:
            self._file.close()
        try:
            self._parent.del_record(self.name)
        except ValueError:
            # 404 dont found /:
            pass
        if os.path.exists(self._file.name):
            os.remove(self._file.name)

    def __str__(self):
        return 'wrap generator object of {}'.format(self._rec.name)

    def __mul__(self, scale):
        return self.__call__().__mul__(scale)

    def __truediv__(self, scale):
        return self.__call__().__truediv__(scale)

    def __pow__(self, power, modulo=None):
        return self.__call__().__pow__(power, modulo=modulo)

    def __add__(self, other):
        return self.__call__().__add__(other)

    def __sub__(self, other):
        return self.__call__().__sub__(other)


@Mem.sudio.add
class Wrap:
    name = Name()

    def __init__(self, other, record, generator):
        """
        :param other: parent object
        :param record: preloaded record or pandas series

        \nSlicing : \n
        The Wrapped object can be sliced using the standard Python x[start: stop: step]
        syntax, where x is the wrapped object.

         Slicing the time domain:  \n
         The basic slice syntax is [i: j: k, i(2): j(2): k(2), i(n): j(n): k(n)] where i is the
         start time, j is the stop time in integer or float types and k is the step(negative number for inversing).
         This selects the nXm seconds with index times
         i, i+1, i+2, ..., j, i(2), i(2)+1, ..., j(2), i(n), ..., j(n)
         j where m = j - i (j > i).

          Note: for i < j, i is the stop time and j is the start time,
          means that audio data read inversely.

          Filtering(Slicing the frequency domain):  \n
         The basic slice syntax is ['i': 'j': 'filtering options',
         'i(2)': 'j(2)': 'options(2)', ..., 'i(n)': 'j(n)': 'options(n)']
         where i is the starting frequency and j is the stopping frequency with type of string
         in the same units as fs that fs is 2 half-cycles/sample.
         This activates n number of iir filters with specified frequencies and options.

          For the slice syntax [x: y: options] we have:
           - x= None, y= 'j': low pass filter with a cutoff frequency of the j
           - x= 'i', y= None: high pass filter with a cutoff frequency of the i
           - x= 'i', y= 'j': bandpass filter with the critical frequencies of the i, j
           - x= 'i', y= 'j', options='scale=[Any negative number]': bandstop filter with the critical frequencies of the i, j

          Filtering options:
           - ftype: optional; The type of IIR filter to design:\n
             - Butterworth : ‘butter’(default)
             - Chebyshev I : ‘cheby1’
             - Chebyshev II : ‘cheby2’
             - Cauer/elliptic: ‘ellip’
             - Bessel/Thomson: ‘bessel’
           - rs: float, optional:\n
             For Chebyshev and elliptic filters, provides the minimum attenuation in the stop band. (dB)
           - rp: float, optional:\n
             For Chebyshev and elliptic filters, provides the maximum ripple in the passband. (dB)
           - order: The order of the filter.(default 5)
           - scale: [float, int] optional; The attenuation or Amplification factor,
             that in the bandstop filter must be a negative number.

          Complex slicing:
           The basic slice syntax is [a: b, 'i': 'j': 'filtering options', ..., 'i(n)': 'j(n)': 'options(n)', ...
           , a(n): b(n), 'i': 'j': 'options', ..., 'i(n)': 'j(n)': 'options(n)'] or
           [a: b, [Filter block 1)], a(2): b(2), [Filter block 2]  ... , a(n): b(n), [Filter block n]]
           Where i is the starting frequency, j is the stopping frequency, a is the starting
           time and b is the stopping time in seconds. This activates n number of filter blocks [described in
           the filtering section] that each of them operates within a predetermined time range.

        note:
         The sliced object is stored statically so calling the original wrapped returns The sliced object.

        Dynamic and static memory management:
         Wrapped objects normally stored statically, so all of the calculations need additional IO read/write time,
         This decrese dynaamic memory usage specically for big audio data.
         All of the operations (mathematical, sclicing, etc) can be done faster in dynamic memory using unpack method.

        \nExamples
          --------
         Slicing wrap object snd remove 10 to 36 seconds
        >>> master = Master()
        >>> wrap = master.add('file.mp3')
        >>> master.echo(wrap[5: 10, 36:90] * .5)

         Bandstop filter:
        >>> wrap['200': '1000': 'order=6, scale=-.8']
        >>> master.echo(wrap)

         Complex slicing and inversing:
        >>> wrap[: 10, :'100': 'scale=1.1', 5: 15: -1, '100': '5000': 'order=10, scale=.8']
        >>> master.echo(wrap - .7)

         Simple two band EQ:
        >>> wrap[20:30,: '100': 'order=4, scale=.7', '100'::'order=5, scale=.4']
        >>> master.echo(wrap)

        """

        self._rec = record

        self._seek = 0
        self._data = self._rec['o']
        self._generator: WrapGenerator = generator
        self._file: io.BufferedRandom = self._rec['o']
        self._size = self._rec['size']
        self._duration = record['duration']
        self._frame_rate = self._rec['frameRate']
        self._nchannels = self._rec['nchannels']
        self._sample_format = self._rec['sampleFormat']
        self._nperseg = self._rec['nperseg']
        self._parent = other
        self._sample_type = self._parent._constants[0]
        self.sample_width = Audio.get_sample_size(self._rec['sampleFormat'])
        self._time_calculator = lambda t: int(self._frame_rate *
                                              self._nchannels *
                                              self.sample_width *
                                              t)
        self._itime_calculator = lambda byte: byte / (self._frame_rate *
                                                      self._nchannels *
                                                      self.sample_width)
        self._packed = True

    def get_sample_format(self):
        return ISampleFormat[self._sample_format]

    def get_sample_width(self):
        return self.sample_width

    def get_master(self):
        return self._parent

    def get_size(self):
        return os.path.getsize(self._file.name)

    def get_frame_rate(self):
        return self._frame_rate

    def get_nchannels(self):
        return self._nchannels

    def get_duration(self):
        return self._itime_calculator(self.get_size())

    def join(self, *other):
        other = self._parent.sync(*other,
                                  nchannels=self._nchannels,
                                  sample_rate=self._frame_rate,
                                  sample_format=ISampleFormat[self._sample_format],
                                  output='ndarray_data')
        # print(other)
        with self.unpack() as main_data:
            axis = 0
            if len(main_data.shape) > 1 and main_data.shape[0] > 1:
                axis = 1

            for series in other:
                # assert self._frame_rate == other._frame_rate and
                # print(series['o'], other)
                main_data = np.append(main_data, series['o'], axis=axis)
            self.set_data(main_data)
        return self

    def _time_slice(self, item: tuple):
        if item is None:
            item = [None, None, None]

        record_time = [None, None]
        if item[0] is not None:
            assert item[0] < self._generator.get_duration(), OverflowError('input time must be less than the record duration')
            record_time[0] = self._time_calculator(item[0])
        if item[1] is not None:
            assert item[1] < self._generator.get_duration(), OverflowError('input time must be less than the record duration')
            record_time[1] = self._time_calculator(item[1])

        time_buffer = [i if i else 0 for i in record_time]

        step = item[2]
        if time_buffer[0] > time_buffer[1]:
            time_buffer[0], time_buffer[1] = time_buffer[1], time_buffer[0]
            if step is not None:
                step = -1 * item[2]
            else:
                step = -1

        if self._packed:
            with self._generator.get() as generator:
                generator.seek(time_buffer[0], 0)
                time_buffer = abs(time_buffer[1] - time_buffer[0])
                time_buffer = time_buffer if time_buffer else self._generator._size
                data = generator.read(time_buffer)

            # filter process
            data = self._from_buffer(data)
            if self._nchannels > 1:
                return data[:, :: step]
            return data[:: step]

        else:
            record_time = [(None if i is None else i // self._nchannels) for i in record_time]
            if self._nchannels > 1:
                return self._data[:, record_time[0]: record_time[1]: step]
            else:
                return self._data[record_time[0]: record_time[1]: step]

    def _parse(self, item, buffer=None, last_item=[]):

        if not buffer:
            buffer = {-1: -1}
            last_item = [None]

        if type(item) is slice:
            obj_type = Wrap._slice_type(item)
            if obj_type is int:
                # time
                last_item[0] = (item.start, item.stop, item.step)
                buffer[last_item[0]] = []
                # print('time', buffer)

            elif obj_type is None:
                last_item[0] = None
                buffer[None] = []

            elif obj_type is str:
                # Butterworth: ‘butter’

                # Chebyshev
                # I: ‘cheby1’

                # Chebyshev
                # II: ‘cheby2’

                # Cauer / elliptic: ‘ellip’

                # Bessel / Thomson: ‘bessel’
                filt = {'ftype': 'butter',
                        'rs': None,
                        'rp': None,
                        'order': 5,
                        'scale': None}

                if item.step:
                    parsed = Tools.str2dict(item.step, item_sep=',', dict_eq='=')
                    for i in parsed:
                        if i in filt:
                            filt[i] = parsed[i]

                if item.start is not None and item.stop is not None and \
                        filt['scale'] and float(filt['scale']) < 0:
                    btype = 'bandstop'
                    freq = float(item.start), float(item.stop)
                    assert freq[1] > freq[0], ValueError('{freq0} is bigger than {freq1}'.format(freq0=freq[0],
                                                                                                 freq1=freq[1]))

                elif item.start is not None and item.stop is not None:
                    btype = 'bandpass'
                    freq = float(item.start), float(item.stop)
                    assert freq[1] > freq[0], ValueError('{freq0} is bigger than {freq1}'.format(freq0=freq[0],
                                                                                                 freq1=freq[1]))
                elif item.start is not None:
                    btype = 'highpass'
                    freq = float(item.start)

                elif item.stop is not None:
                    btype = 'lowpass'
                    freq = float(item.stop)

                else:
                    return buffer

                # iir = scisig.firwin(50, freq, fs=self.frame_rate)

                # print(btype)
                iir = scisig.iirfilter(filt['order'], freq, btype=btype, fs=self._frame_rate, output='sos',
                                       rs=filt['rs'], rp=filt['rp'], ftype=filt['ftype'])
                if last_item[0] is None:
                    buffer[None] = []
                buffer[last_item[0]].append((iir,
                                             *[abs(float(i)) for i in (filt['scale'],) if i is not None]))

        elif type(item) is list or type(item) is tuple:
            for item in item:
                assert type(item) is slice
                # print(buffer, last_item[0])
                self._parse(item, buffer=buffer, last_item=last_item)

        return buffer

    def __getitem__(self, item):
        item = self._parse(item)
        tim = list(item.keys())[1:]
        data = buffer = []
        byte = b''

        for obj in tim:
            buffer.append(self._time_slice(obj))
        for idx, obj in enumerate(tim):
            frq, data = item[obj], buffer[idx]
            tmp = []
            if len(frq):
                # print(data.shape)
                for i in frq:
                    tmp.append(Wrap._freq_slice(data, i))
                data = np.sum(tmp, axis=0)
                # data[data > 2**self.sample_width-100] *= .5
            if self._packed:
                byte += self._to_buffer(data)

        if self._packed:
            with self.get(self._parent.__class__.CACHE_INFO, 0) as file:
                file.truncate()
                file.write(byte)
                file.flush()
                self._data = file

        else:
            self._data = data

        return self

    def __del__(self):
        if not self._file.closed:
            self._file.close()
        try:
            self._parent.del_record(self.name)
        except ValueError:
            # 404 dont found /:
            pass
        if os.path.exists(self._file.name):
            os.remove(self._file.name)

    def __str__(self):
        return 'wrap object of {}'.format(self._rec.name)

    def __mul__(self, scale):
        assert type(scale) is float or type(scale) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data * scale
        return self

    def __truediv__(self, scale):
        assert type(scale) is float or type(scale) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data // scale
        return self

    def __pow__(self, power, modulo=None):
        assert type(power) is float or type(power) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data ** power
        return self

    def __add__(self, other):
        assert type(other) is float or type(other) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data + other
        return self

    def __sub__(self, other):
        assert type(other) is float or type(other) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data // other
        return self

    def _from_buffer(self, data: bytes):
        data = np.frombuffer(data, self._sample_type)
        if self._nchannels > 1:
            data = np.append(*[[data[i::self._nchannels]] for i in range(self._nchannels)],
                             axis=0)
        return data

    def _to_buffer(self, data: np.array):
        if self._nchannels > 1:
            data = data.T.reshape(np.prod(data.shape))
        return data.astype(self._sample_type).tobytes()

    @contextmanager
    def unpack(self, reset=False):
        '''
        Audio data unpacked from cached files to the dynamic memory by calling unpacker with (flags).
        note:
         All calculations in the unpacked block are performed on the precached
         files (not the original audio data).

        :param reset: Reset the audio pointer to time 0 (Equivalent to slice '[:]').
        :return: audio data in ndarray format with shape(number of audio channels, block size).

        EXAMPLES
        --------
        >>> master = Master()
        >>> wrap = master.add('file.mp3')
        >>> with wrap.unpack() as data:
        >>>     wrap.set_data(data * .7)
        >>> master.echo(wrap)

        '''
        try:
            self._packed = False
            if reset:
                self.reset()

            with self.get() as f:
                data = self._from_buffer(f.read())
                self._data = data
            yield data

        finally:
            self._packed = True
            data = self._to_buffer(self._data)

            with self.get(self._parent.__class__.CACHE_INFO, 0) as file:
                file.truncate()
                file.write(data)
                file.flush()

            self._data = self._file

    def get_data(self):
        if self._packed:
            record = self._rec.copy()
            size = record['size'] = os.path.getsize(self._file.name)
            record['duration'] = self._itime_calculator(size)
            # print(record['duration'])
            return record
        else:
            return self._data

    def is_packed(self):
        return self._packed

    @contextmanager
    def get(self, offset=None, whence=None):
        try:
            # self._file.flush()
            if offset is not None and whence is not None:
                self._seek = self._file.seek(offset, whence)
            else:
                self._seek = self._file.tell()

            yield self._file
        finally:
            self._file.seek(self._seek, 0)

    def set_data(self, data):
        assert not self.is_packed(), 'just used in unpacked mode'
        self._data = data

    @staticmethod
    def _slice_type(item: slice):
        item = [i for i in [item.start, item.stop, item.step] if i is not None]
        if len(item) == 0: return None

        item_type = list(map(lambda x: int if x is float else x, map(type, item)))
        assert item_type.count(item_type[0]) == len(item), TypeError('unknown item type')
        return item_type[0]

    @staticmethod
    def _freq_slice(buffer: np.array, item: tuple):
        if len(item) > 1:
            return scisig.sosfilt(item[0], buffer) * item[1]
        return scisig.sosfilt(item[0], buffer)




