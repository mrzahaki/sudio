#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


import io
import os
import numpy as np
import scipy.signal as scisig
from contextlib import contextmanager
from typing import Union

from sudio.types.name import Name
from sudio.io import SampleFormat, get_sample_size
from sudio.audiosys.typeconversion import convert_array_type
from sudio.utils.strtool import parse_dictionary_string
from sudio.metadata import AudioMetadata

class Wrap:
    name = Name()

    def __init__(self, other, record, generator):
        """
        Initialize the Wrap object.

        :param other: The parent object.
        :param record: Preloaded record or AudioMetadata.
        :param generator: The audio generator associated with the record.

        Slicing:
        The Wrapped object can be sliced using standard Python slice syntax x[start: stop: step],
        where x is the wrapped object.

        Slicing the time domain:
        Use [i: j: k, i(2): j(2): k(2), i(n): j(n): k(n)] syntax, where i is start time, j is stop time,
        and k is step (negative for inversing). This selects nXm seconds with index times
        i, i+1, i+2, ..., j, i(2), i(2)+1, ..., j(2), i(n), ..., j(n) where m = j - i (j > i).

        Note: for i < j, i is stop time, j is start time, meaning audio data is read inversely.

        Filtering (Slicing the frequency domain):
        Use ['i': 'j': 'filtering options', 'i(2)': 'j(2)': 'options(2)', ..., 'i(n)': 'j(n)': 'options(n)']
        where i is starting frequency, j is stopping frequency (string type, same units as fs).
        Activates n iir filters with specified frequencies and options.

        For slice syntax [x: y: options]:
        - x=None, y='j': low pass filter with a cutoff frequency of j
        - x='i', y=None: high pass filter with a cutoff frequency of i
        - x='i', y='j': bandpass filter with critical frequencies i, j
        - x='i', y='j', options='scale=[Any negative number]': bandstop filter with critical frequencies i, j

        Filtering options:
        - ftype: optional; The type of IIR filter to design: 'butter' (default), 'cheby1', 'cheby2', 'ellip', 'bessel'
        - rs: float, optional: For Chebyshev and elliptic filters, provides minimum attenuation in stop band (dB)
        - rp: float, optional: For Chebyshev and elliptic filters, provides maximum ripple in passband (dB)
        - order: The order of the filter (default 5)
        - scale: [float, int] optional; Attenuation or amplification factor, must be negative in bandstop filter.

        Complex slicing:
        Use [a: b, 'i': 'j': 'filtering options', ..., 'i(n)': 'j(n)': 'options(n)', ..., a(n): b(n), ...]
        or [a: b, [Filter block 1)], a(2): b(2), [Filter block 2] ..., a(n): b(n), [Filter block n]].
        i is starting frequency, j is stopping frequency, a is starting time, b is stopping time in seconds.
        Activates n filter blocks described in the filtering section, each operating within a predetermined time range.

        Note:
        The sliced object is stored statically, so calling the original wrapped returns the sliced object.

        Dynamic and static memory management:
        Wrapped objects are stored statically; all calculations need additional IO read/write time.
        This reduces dynamic memory usage, especially for large audio data.
        All operations (mathematical, slicing, etc.) can be done faster in dynamic memory using the unpack method.

        Examples:
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

        Simple two-band EQ:
        >>> wrap[20:30,: '100': 'order=4, scale=.7', '100'::'order=5, scale=.4']
        >>> master.echo(wrap)
        """


        self._rec = record

        self._seek = 0
        self._data = self._rec['o']
        self._generator = generator
        self._file: io.BufferedRandom = self._rec['o']
        self._size = self._rec['size']
        self._duration = record['duration']
        self._sample_rate = self._rec['frameRate']
        self._nchannels = self._rec['nchannels']
        self._sample_format = self._rec['sampleFormat']
        self._nperseg = self._rec['nperseg']
        self._parent = other
        self._sample_type = self._parent._sample_width_format_str
        self.sample_width = get_sample_size(self._rec['sampleFormat'])
        self._time_calculator = lambda t: int(self._sample_rate *
                                              self._nchannels *
                                              self.sample_width *
                                              t)
        self._itime_calculator = lambda byte: byte / (self._sample_rate *
                                                      self._nchannels *
                                                      self.sample_width)
        self._packed = True

    def get_sample_format(self) -> SampleFormat:
        """
        Get the sample format of the audio data.

        :return: The sample format enumeration.
        """
        return self._sample_format

    def get_sample_width(self) -> int:
        """
        Get the sample width (in bytes) of the audio data.

        :return: The sample width.
        """
        return self.sample_width

    def get_master(self):
        """
        Get the parent object (Master) associated with this Wrap object.

        :return: The parent Master object.
        """
        return self._parent

    def get_size(self) -> int:
        """
        Get the size of the audio data file.

        :return: The size of the audio data file in bytes.
        """
        return os.path.getsize(self._file.name)

    def get_sample_rate(self) -> int:
        """
        Get the frame rate of the audio data.

        :return: The frame rate of the audio data.
        """
        return self._sample_rate

    def get_nchannels(self) -> int:
        """
        Get the number of channels in the audio data.

        :return: The number of channels.
        """
        return self._nchannels

    def get_duration(self) -> float:
        """
        Get the duration of the audio data in seconds.

        :return: The duration of the audio data.
        """
        return self._itime_calculator(self.get_size())

    def join(self, *other):
        """
        Join the current audio data with other audio data.

        :param other: Other audio data to be joined.
        :return: The current Wrap object after joining with other audio data.
        """
        other = self._parent.sync(*other,
                                  nchannels=self._nchannels,
                                  sample_rate=self._sample_rate,
                                  sample_format=self._sample_format,
                                  output='ndarray_data')
        # print(other)
        with self.unpack() as main_data:
            axis = 0
            if len(main_data.shape) > 1 and main_data.shape[0] > 1:
                axis = 1

            for series in other:
                # assert self._sample_rate == other._sample_rate and
                # print(series['o'], other)
                main_data = np.concatenate((main_data, series['o']), axis=axis)
            self.set_data(main_data)
        return self

    def _time_slice(self, item: tuple):
        """
        Slice the audio data based on the specified time range.

        :param item: A tuple specifying the start, stop, and step values for slicing.
        :return: The sliced audio data.
        """
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
        """
        Parse the input item to determine filtering options and apply filtering.

        :param item: The input item to be parsed.
        :param buffer: A dictionary to store the parsed information.
        :param last_item: A list to store the last parsed item.
        :return: The parsed buffer with filtering information.
        """
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
                    parsed = parse_dictionary_string(item.step, item_sep=',', dict_eq='=')
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

                # iir = scisig.firwin(50, freq, fs=self.sample_rate)

                # print(btype)
                iir = scisig.iirfilter(filt['order'], freq, btype=btype, fs=self._sample_rate, output='sos',
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
        """
        Implement the slicing behavior for the Wrap object.

        :param item: The slicing item specifying time and frequency ranges.
        :return: The modified Wrap object after slicing.
        """
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
        """
        Handle the deletion of the Wrap object.

        :return: None
        """
        if not self._file.closed:
            self._file.close()
        try:
            self._parent.del_record(self.name)
        except ValueError:
            # 404 dont found /:
            pass
        except AssertionError:
            pass

        if os.path.exists(self._file.name):
            os.remove(self._file.name)

    def __str__(self):
        """
        Return a string representation of the Wrap object.

        :return: A string representation of the Wrap object.
        """
        return 'wrap object of {}'.format(self._rec.name)

    def __mul__(self, scale):
        """
        Multiply the audio data in the Wrap object by a scale factor.

        :param scale: The scale factor (float or int).
        :return: The modified Wrap object after multiplication.
        """
        assert type(scale) is float or type(scale) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data * scale
        return self

    def __truediv__(self, scale):
        """
        Divide the audio data in the Wrap object by a scale factor.

        :param scale: The scale factor (float or int).
        :return: The modified Wrap object after division.
        """
        assert type(scale) is float or type(scale) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data // scale
        return self

    def __pow__(self, power, modulo=None):
        """
        Raise the audio data in the Wrap object to a power.

        :param power: The exponent (float or int).
        :param modulo: Not used.
        :return: The modified Wrap object after exponentiation.
        """
        assert type(power) is float or type(power) is int
        assert self._packed, AttributeError('The Wrap object must be packed')
        with self.unpack() as data:
            self._data = data ** power
        return self

    def __add__(self, other):
        """
        Add the audio data in the Wrap object to another object or a constant.

        :param other: The other object or constant to be added.
        :return: The modified Wrap object after addition.
        """
        assert self._packed, AttributeError('The Wrap object must be packed')
        
        if type(other) is float or type(other) is int:
            with self.unpack() as data:
                self._data = data + other
        else:
            assert self._nchannels == other._nchannels, ValueError('channels must be equal')
            assert other._packed, AttributeError('The Wrap object must be packed')

            with self.unpack() as data:
                with other.unpack() as otherdata:
                    if data.shape[-1] > otherdata.shape[-1]:
                        common_data = data[:, :otherdata.shape[-1]] + otherdata
                        self._data = np.concatenate((common_data, data[:, otherdata.shape[-1]:]), axis=1)

                    else:
                        common_data = data + otherdata[:, :data.shape[-1]]
                        self._data = np.concatenate((common_data, otherdata[:, data.shape[-1]:]), axis=1)
        return self

    def __sub__(self, other):
        """
        Subtract a constant from the audio data in the Wrap object.

        :param other: The constant to be subtracted.
        :return: The modified Wrap object after subtraction.
        """
        assert self._packed, AttributeError('The Wrap object must be packed')
        
        if type(other) is float or type(other) is int:
            with self.unpack() as data:
                self._data = data - other
        else:
            assert self._nchannels == other._nchannels, ValueError('channels must be equal')
            assert other._packed, AttributeError('The Wrap object must be packed')

            with self.unpack() as data:
                with other.unpack() as otherdata:
                    if data.shape[-1] > otherdata.shape[-1]:
                        common_data = data[:, :otherdata.shape[-1]] - otherdata
                        self._data = np.concatenate((common_data, data[:, otherdata.shape[-1]:]), axis=1)

                    else:
                        common_data = data - otherdata[:, :data.shape[-1]]
                        self._data = np.concatenate((common_data, otherdata[:, data.shape[-1]:]), axis=1)
        return self

    def _from_buffer(self, data: bytes):
        """
        Convert binary data to a NumPy array.

        :param data: Binary data to be converted.
        :return: The NumPy array representing the data.
        """
        data = np.frombuffer(data, self._sample_type)
        if self._nchannels > 1:
            data = np.concatenate([[data[i::self._nchannels]] for i in range(self._nchannels)],
                             axis=0)
        return data

    def _to_buffer(self, data: np.array):
        """
        Convert a NumPy array to binary data.

        :param data: The NumPy array to be converted.
        :return: Binary data representing the array.
        """
        if self._nchannels > 1:
            data = data.T.reshape(np.prod(data.shape))
        return data.astype(self._sample_type).tobytes()

    @contextmanager
    def unpack(self, reset=False, astype:SampleFormat=SampleFormat.UNKNOWN) -> np.ndarray:
        '''
        Unpacks audio data from cached files to dynamic memory.

        :param reset: Resets the audio pointer to time 0 (Equivalent to slice '[:]').
        :return: Audio data in ndarray format with shape (number of audio channels, block size).

        Notes:
        - All calculations within the unpacked block are performed on the pre-cached
        files, not on the original audio data.

        EXAMPLES
        --------
        >>> master = Master()
        >>> wrap = master.add('file.mp3')
        >>> with wrap.unpack() as data:
        >>>     wrap.set_data(data * 0.7)
        >>> master.echo(wrap)
        '''
        astype_backup = None

        try:
            self._packed = False
            if reset:
                self.reset()

            with self.get() as f:
                data = self._from_buffer(f.read())
                self._data = data
            if not astype == SampleFormat.UNKNOWN:
                astype_backup = self._sample_type
                data = convert_array_type(data, astype)
            yield data

        finally:
            self._packed = True
            if astype_backup is not None:
                data = data.astype(astype_backup)

            data = self._to_buffer(self._data)

            with self.get(self._parent.__class__.CACHE_INFO, 0) as file:
                file.truncate()
                file.write(data)
                file.flush()

            self._data = self._file

    def get_data(self) -> Union[AudioMetadata, np.ndarray]:
        """
        Get the audio data either from cached files or dynamic memory.

        :return: If packed, returns record information. If unpacked, returns the audio data.
        """
        if self._packed:
            record = self._rec.copy()
            size = record['size'] = os.path.getsize(self._file.name)
            record['duration'] = self._itime_calculator(size)
            # print(record['duration'])
            return record
        else:
            return self._data

    def is_packed(self) -> bool:
        """
        Check if the Wrap object is in packed mode.

        :return: True if the Wrap object is in packed mode, False otherwise.
        """
        return self._packed

    @contextmanager
    def get(self, offset=None, whence=None):
        """
        Context manager for getting a file handle and managing seek position.

        :param offset: Offset to seek within the file.
        :param whence: Reference point for the seek operation.
        :return: File handle for reading or writing.
        """
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
        """
        Set the audio data when the Wrap object is in unpacked mode.

        :param data: Audio data to be set.
        :return: None
        """
        assert not self.is_packed(), 'just used in unpacked mode'
        self._data = data

    @staticmethod
    def _slice_type(item: slice):
        """
        Determine the type of slice (int, float, or None) based on the provided slice object.

        :param item: The slice object.
        :return: Type of slice (int, float, or None).
        """
        item = [i for i in [item.start, item.stop, item.step] if i is not None]
        if len(item) == 0: return None

        item_type = list(map(lambda x: int if x is float else x, map(type, item)))
        assert item_type.count(item_type[0]) == len(item), TypeError('unknown item type')
        return item_type[0]

    @staticmethod
    def _freq_slice(buffer: np.array, item: tuple):
        """
        Apply frequency domain slicing to the audio data.

        :param buffer: Audio data.
        :param item: Tuple representing filter parameters (sos and scale).
        :return: Processed audio data.
        """
        if len(item) > 1:
            return scisig.sosfilt(item[0], buffer) * item[1]
        return scisig.sosfilt(item[0], buffer)

