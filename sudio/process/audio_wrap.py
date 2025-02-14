
# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


import os
import numpy as np
import scipy.signal as scisig
from contextlib import contextmanager
from typing import Union
import warnings
from enum import Enum, auto

from sudio.types import Name
from sudio.io import SampleFormat, get_sample_size
from sudio.utils.typeconversion import dtype_converter
from sudio.utils.strtool import parse_dictionary_string
from sudio.audiosys import write_to_cached_file
from sudio.utils.timed_indexed_string import TimedIndexedString
from sudio.metadata import AudioMetadata
from sudio.process.fx._fade_envelope import generate_envelope
from sudio.process.fx import FadePreset
from sudio.process.fx import  FX
from sudio.utils.math import db2amp, find_nearest_divisible


class AFXChannelFillMode(Enum):
    """
    Enumeration of different modes for handling length discrepancies 
    when applying effects to audio channels.
    """
    NONE = auto()      # No length adjustment
    PAD = auto()       # Pad the shorter array to match length
    TRUNCATE = auto()  # Truncate to the shorter array's length
    REFERENCE_REFINED = auto()  # Truncate or pad depend on the refined data 
    REFERENCE_RAW = auto()  # Truncate or pad depend on the original data 


class AudioWrap:
    """
    A class that handles audio data processing with caching capabilities.
    """
    name = Name()
    
    def __init__(self, master, record):
        """
        Initialize the AudioWrap object.

        Parameters
        ----------

        master : object
            The master instance.
        record : AudioMetadata or str
            An instance of `AudioMetadata`, a string representing audio metadata, 
            or an `AudioWrap` object itself.

            
        Slicing
        -------

        The wrapped object can be sliced using standard Python slice syntax `x[start: stop: speed_ratio]`, 
        where `x` is the wrapped object.

        Time Domain Slicing
        ~~~~~~~~~~~~~~~~~~~

        Use `[i: j: k, i(2): j(2): k(2), i(n): j(n): k(n)]` syntax, where:
        - `i` is the start time,
        - `j` is the stop time,
        - `k` is the `speed_ratio`, which adjusts the playback speed.

        This selects `nXm` seconds with index times:
        `i, i+1, ..., j`, `i(2), i(2)+1, ..., j(2)`, ..., `i(n), ..., j(n)` where `m = j - i` (`j > i`).

        Notes:
        - For `i < j`, `i` is the stop time and `j` is the start time, meaning audio data is read inversely.

        Speed Adjustment
        ~~~~~~~~~~~~~~~~~

        - `speed_ratio` > 1 increases playback speed (reduces duration).
        - `speed_ratio` < 1 decreases playback speed (increases duration).
        - Default `speed_ratio` is 1.0 (original speed).
        - Speed adjustments preserve pitch. (support this project bro)

        Frequency Domain Slicing (Filtering)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Use `['i': 'j': 'filtering options', 'i(2)': 'j(2)': 'options(2)', ..., 'i(n)': 'j(n)': 'options(n)']` syntax, where:
        - `i` is the starting frequency,
        - `j` is the stopping frequency (string type, in the same units as `fs`).

        This activates `n` IIR filters with specified frequencies and options.

        Slice Syntax for Filtering
        ---------------------------

        - `x=None`, `y='j'`: Low-pass filter with a cutoff frequency of `j`.
        - `x='i'`, `y=None`: High-pass filter with a cutoff frequency of `i`.
        - `x='i'`, `y='j'`: Band-pass filter with critical frequencies `i`, `j`.
        - `x='i'`, `y='j'`, `options='scale=[negative value]'`: Band-stop filter with critical frequencies `i`, `j`.

        Filtering Options
        -----------------

        - `ftype` : str, optional
            Type of IIR filter to design. Options: `'butter'` (default), `'cheby1'`, `'cheby2'`, `'ellip'`, `'bessel'`.
        - `rs` : float, optional
            Minimum attenuation in the stop band (dB) for Chebyshev and elliptic filters.
        - `rp` : float, optional
            Maximum ripple in the passband (dB) for Chebyshev and elliptic filters.
        - `order` : int, optional
            The order of the filter. Default is 5.
        - `scale` : float or int, optional
            Attenuation or amplification factor. Must be negative for a band-stop filter.

        Complex Slicing
        ---------------

        Use `[a: b, 'i': 'j': 'filtering options', ..., 'i(n)': 'j(n)': 'options(n)', ..., a(n): b(n), ...]`
        or `[a: b, [Filter block 1]], a(2): b(2), [Filter block 2] ..., a(n): b(n), [Filter block n]]`.

        - `i`, `j` are starting and stopping frequencies.
        - `a`, `b` are starting and stopping times in seconds.

        This activates `n` filter blocks described in the filtering section, each operating within a specific time range.

        Notes
        -----

        - Wrapped objects use static memory to reduce dynamic memory usage for large audio data.

        Examples
        --------

            >>> import sudio
            >>> master = Master()
            >>> audio = master.add('audio.mp3')
            >>> master.echo(audio[5: 10, 36:90] * -10)

        In this example, after creating an instance of the Master class, an audio file in MP3 format is loaded. 
        Then, the AudioWrap object is sliced from 5 to 10 and from 36 to 90, and these two sliced segments are joined. 
        The result is then gained with -10 dB.

            >>> filtered_audio = audio[5: 10, '200': '1000': 'order=6, scale=-.8']
            >>> master.echo(filtered_audio)

        Here we apply a 6th-order band-stop filter to the audio segment from 5 to 10 seconds, targeting frequencies between 200 Hz and 1000 Hz. 
        The filter, with a -0.8 dB attenuation, effectively suppresses this range. 
        Finally, the filtered audio is played through the standard output using the master.echo method.  
        
            >>> filtered_audio = audio[: 10, :'400': 'scale=1.1', 5: 15: 1.25, '100': '5000': 'order=10, scale=.8']
            >>> master.echo(filtered_audio)

        First, the audio is sliced into 10-second segments and enhanced using a low-pass filter (LPF) with a 400 Hz cutoff and a 1.1 scaling factor, boosting lower frequencies in the initial segment.
        Next, a 5 to 15-second slice is processed at 1.25x playback speed, adjusting the tempo.
        A 10th-order band-pass filter is then applied to this segment, isolating frequencies between 100 Hz and 5000 Hz with a 0.8 scale factor.
        Finally, the two processed segments are combined, and the fully refined audio is played through the standard output using master.echo.

        Simple two-band EQ:

            >>> filtered_audio = audio[40:60, : '200': 'order=4, scale=.8', '200'::'order=5, scale=.5'] * 1.7
            >>> master.echo(audio)
    
        Here, a two-band EQ tweaks specific frequencies within a 40-60 second audio slice. 
        First, a 4th-order low-pass filter reduces everything below 200 Hz, scaled by 0.8 to lower low frequencies. 
        Next, a 5th-order high-pass filter handles frequencies above 200 Hz, scaled by 0.5 to soften the highs. 
        After filtering, the overall volume is boosted by 1.7 times to balance loudness. 
        Finally, the processed audio is played using master.echo(), revealing how these adjustments shape the 
        sound—perfect for reducing noise or enhancing specific frequency ranges.
        """

        self._cache_info_size = master.__class__.CACHE_INFO_SIZE
        self._buffer_type = master.__class__.BUFFER_TYPE
        if isinstance(record, AudioMetadata):
            self._rec = record
            # bytes to cache if needed
            if isinstance(record.o, bytes):
                record.o = write_to_cached_file(
                    record.size,
                    record.frameRate,
                    record.sampleFormat if record.sampleFormat else 0,
                    record.nchannels,
                    file_name=master._audio_data_directory + record.name + self._buffer_type,
                    data=record.o,
                    pre_truncate=True,
                    after_seek=(self._cache_info_size, 0),
                    after_flush=True
                )
                record.size += self._cache_info_size
        elif isinstance(record, str):
            self._rec = master.load(record, series=True)
        elif not isinstance(record, self.__class__):
            raise ValueError("Record must be AudioMetadata, string path, or AudioWrap instances")

        # initialize basic attributes
        self._file = self._rec.o
        self._id_generator = TimedIndexedString(self._file.name,
                                seed='wrapped',
                                start_before=self._buffer_type)
        self._size = self._rec.size
        self._sample_rate = self._rec.frameRate
        self._nchannels = self._rec.nchannels
        self._sample_format = self._rec.sampleFormat
        self._nperseg = master._nperseg
        self._sample_type = master._sample_width_format_str
        self.sample_width = get_sample_size(self._rec.sampleFormat)
        self._data = self._rec.o
        
        # state tracking
        self._packed = True
        self._seek = 0
        self._master = master


    def __call__(self):
        """
        Create a new instance with same parameters.        
        """
        self._master.prune_cache()
        newrec = self._rec.copy()
        return self._master.add(newrec)

    def join(self, *other):
        """
        Join multiple audio segments together.

        :param other: Other audio data to be joined.
        :return: The current AudioWrap object after joining with other audio data.
        """
        
        other_data = self._master.sync(*other,
                                nchannels=self._nchannels,
                                sample_rate=self._sample_rate,
                                sample_format=self._sample_format,
                                output='ndarray_data')
        
        with self.unpack() as main_data:
            axis = 1 if len(main_data.shape) > 1 and main_data.shape[0] > 1 else 0
            
            for series in other_data:
                main_data = np.concatenate((main_data, series.o), axis=axis)
            self.set_data(main_data)
        return self
        

    def set_data(self, data):
        """
        Set the audio data when the object is in unpacked mode.

        :param data: Audio data to be set.
        :return: None
        """
        assert not self.is_packed(), 'just used in unpacked mode'
        self._data = data

    def __getitem__(self, item):
        """
        Implement slicing behavior with time and frequency domain operations.
        
        Args:
            item: Slice specification for time and/or frequency operations
            
        Returns:
            AudioWrap: Modified instance after slicing
        """
        item = self._parse(item)
        tim = list(item.keys())[1:]
        buffer = []
        data = []
        byte = b''

        for obj in tim:
            buffer.append(self._time_slice(obj))

        for idx, obj in enumerate(tim):
            frq, data = item[obj], buffer[idx]
            
            tmp = []
            if len(frq):
                # print(data.shape)
                for i in frq:
                    tmp.append(AudioWrap._freq_slice(data, i))
                data = np.sum(tmp, axis=0)
                # data[data > 2**self.sample_width-100] *= .5
            if self._packed:
                byte += self._to_buffer(data)


        other = self()
        # Write results
        if other._packed:
            with other.get(other._cache_info_size, 0) as file:
                file.truncate()
                file.write(byte)
                file.flush()
                other._data = file
        else:
            other._data = data

        return other

    @contextmanager
    def unpack(
        self, 
        reset=False, 
        astype:SampleFormat=SampleFormat.UNKNOWN, 
        start:float=None,
        stop:float=None,
        truncate:bool=True,
        ) -> np.ndarray:
        '''
        Unpacks audio data from cached files to dynamic memory.

        :param reset: Resets the audio pointer to time 0 (Equivalent to slice '[:]').
        :param astype: Target sample format for conversion and normalization
        :param start: start time. (assert start < stop) (also accept negative values) 
        :param stop: stop time. (assert start < stop) (also accept negative values)
        :param truncate: Whether to truncate the file after writing
        :return: Audio data in ndarray format with shape (number of audio channels, block size).

        EXAMPLE
        --------

        >>> import sudio
        >>> import numpy as np
        >>> from sudio.io import SampleFormat
        >>> 
        >>> ms = sudio.Master()
        >>> song = ms.add('file.ogg')
        >>> 
        >>> fade_length = int(song.get_sample_rate() * 5) 
        >>> fade_in = np.linspace(0, 1, fade_length) 
        >>> 
        >>> with song.unpack(astype=SampleFormat.FLOAT32, start=2.4, stop=20.8) as data:
        >>>     data[:, :fade_length] *= fade_in    
        >>>     song.set_data(data)
        >>> 
        >>> master.echo(wrap)

        This example shows how to use the sudio library to apply a 5-second fade-in effect to an audio file. 
        We load the audio file (file.ogg), calculate the number of samples for 5 seconds, and use NumPy's linspace to create a smooth volume increase.

        We then unpack the audio data between 2.4 and 20.8 seconds in FLOAT32 format, normalizing it to avoid clipping. 
        The fade-in is applied by multiplying the initial samples by the fade array. 
        Finally, the modified audio is repacked and played back with an echo effect. 
        This demonstrates how sudio handles fades and precise audio format adjustments.
        '''

        astype_backup = None
        bstop = None
        bstart = None

        if start is not None:
            assert isinstance(start, (float, int)), 'invalid type'
            if start < 0:
                start = self.get_duration() - start
            bstart = self.time2byte(start)
        else:
            bstart = 0

        if stop is not None:
            assert isinstance(stop, (float, int)), 'invalid type'
            if stop < 0:
                stop = self.get_duration() - stop
            bstop = self.time2byte(stop)
        else:
            bstop = self._size

        if start is not None and stop is not None:
            assert stop > start, "invalid parameter"

        bsize = bstop - bstart
        bsize = find_nearest_divisible(bsize, self.sample_width)

        assert bsize >= 0, 'bstop should be greater than bstart'

        try:
            self._packed = False
            if reset:
                self.reset()

            with self.get(self._cache_info_size + bstart, 0) as f:
                data = f.read(bsize)
                remained_data = f.read()
                data = self._from_buffer(data)
                self._data = data
            if not astype == SampleFormat.UNKNOWN:
                astype_backup = self._sample_format
                data = dtype_converter(data, astype, source_format=self._sample_format)
            yield data

        finally:
            self._packed = True

            data = self._data
            if astype_backup is not None:
                data = dtype_converter(data, astype_backup, source_format=astype)

            data = self._to_buffer(data)
            with self.get(self._cache_info_size + bstart, 0) as file:
                if truncate:
                    file.truncate()
                
                file.write(data + remained_data)
                file.flush()

            self._file.seek(self._cache_info_size, 0)
            self._size = self.get_size()
            self._rec.size = self._size
            self._data = self._file

    def _from_buffer(self, data: bytes) -> np.ndarray:
        """
        Convert binary data to a NumPy array.

        :param data: Binary data to be converted.
        :return: The NumPy array representing the data.
        """
        data = np.frombuffer(data, self._sample_type)
    
        if self._nchannels > 1:
            samples_per_channel = len(data) // self._nchannels
            data = data[:samples_per_channel * self._nchannels]  # Trim any excess samples            
            data = data.reshape(-1, self._nchannels).T
        
        return data

    def _to_buffer(self, data: np.ndarray) -> bytes:
        """
        Convert a NumPy array to binary data.

        :param data: The NumPy array to be converted.
        :return: Binary data representing the array.
        """
        if self._nchannels > 1:
            if data.ndim == 1:
                samples = len(data) // self._nchannels
                data = data[:samples * self._nchannels].reshape(self._nchannels, -1)
            
            data = data.T.reshape(-1)
        
        return data.astype(self._sample_type).tobytes()

    def __del__(self):
        """
        Handle the deletion of the AudioWrap object.

        :return: None
        """
        if not self._file.closed:
            self._file.close()
            
        try:
            self._master.del_record(self.name)
        except (ValueError, AssertionError):
            pass
            
        if os.path.exists(self._file.name):
            try:
                os.remove(self._file.name)
            except PermissionError:
                pass

    def __mul__(self, scale):
        """
        Adjust the audio volume by a decibel scale factor.

        Converts the input decibel value to an amplitude multiplier and applies 
        soft clipping using hyperbolic tangent to prevent digital distortion.

        :param scale: Volume adjustment in decibels (dB)
        :return: A new AudioWrap instance with volume-adjusted audio data

        Examples:
        ---------
        >>> wrap = AudioWrap('audio.wav')
        >>> loud_wrap = wrap * 6     # Increase volume by 6 dB
        >>> soft_wrap = wrap * -6    # Reduce volume by 6 dB
        """
        assert isinstance(scale, (float, int)), TypeError('Unknown type')
        assert self._packed, AttributeError('must be packed')

        scale = db2amp(scale)
        other = self()
        with other.unpack(astype=SampleFormat.FLOAT32) as data:
            other._data = np.tanh(data * scale)
        return other

    def invert(self):
        """
        Invert the audio signal by flipping its polarity.

        Multiplies the audio data by -1.0, effectively reversing the 
        signal's waveform around the zero axis. This changes the phase 
        of the audio while maintaining its original magnitude.

        :return: A new AudioWrap instance with inverted audio data

        Examples:
        ---------
        >>> wrap = AudioWrap('audio.wav')
        >>> inverted_wrap = wrap.invert()  # Flip the audio signal's polarity
        """
        assert self._packed, AttributeError('must be packed')

        other = self()
        with other.unpack(astype=SampleFormat.FLOAT32) as data:
            other._data = data * -1.0
        return other

    # def __truediv__(self, scale:Union[int,float]):
    #     """
    #     Divide the audio data by provided scale factor.

    #     :param scale: The scale factor (float or int).
    #     :return: The modified object after division.
    #     """
    #     assert isinstance(scale, (float, int))
    #     assert self._packed, AttributeError('must be packed')

    #     other = self()
    #     with other.unpack() as data:
    #         other._data = data // scale
    #     return other

    # def __pow__(self, power, modulo=None):
    #     """
    #     Raise the audio data to a power.

    #     :param power: The exponent (float or int).
    #     :param modulo: Not used.
    #     :return: The modified object after exponentiation.
    #     """
    #     assert isinstance(power, (float, int))
    #     assert self._packed, AttributeError('must be packed')

    #     other = self()
    #     with other.unpack() as data:
    #         other._data = data ** power
    #     return other

    
    def __add__(self, other):
        """
        Add the audio data of the current instance to another instance's data.

        :param other: The other AudioWrap instance object to be added.
        :return: The new AudioWrap object after subtraction.
        """
        assert self._packed, AttributeError('The AudioWrap object must be packed')
        
        newself = self()
        if isinstance(other, AudioWrap):
            assert newself._nchannels == other._nchannels, ValueError('channels must be equal')
            assert other._packed, AttributeError('The AudioWrap object must be packed')

            with newself.unpack() as data:
                with other.unpack() as otherdata:
                    if data.shape[-1] > otherdata.shape[-1]:
                        common_data = data[:, :otherdata.shape[-1]] + otherdata
                        newself._data = np.concatenate((common_data, data[:, otherdata.shape[-1]:]), axis=1)

                    else:
                        common_data = data + otherdata[:, :data.shape[-1]]
                        newself._data = np.concatenate((common_data, otherdata[:, data.shape[-1]:]), axis=1)
        else:
            raise TypeError(f'{type(other)} is not supported')
        return newself

    def __sub__(self, other):
        """
        Subtract the audio data of the current instance from another instance's data.

        :param other: The other AudioWrap instance.
        :return: The new AudioWrap object after subtraction.
        """
        assert self._packed, AttributeError('The AudioWrap object must be packed')
        
        newself = self()
        if isinstance(other, AudioWrap):
            assert newself._nchannels == other._nchannels, ValueError('channels must be equal')
            assert other._packed, AttributeError('The AudioWrap object must be packed')

            with newself.unpack() as data:
                with other.unpack() as otherdata:
                    if data.shape[-1] > otherdata.shape[-1]:
                        common_data = data[:, :otherdata.shape[-1]] - otherdata
                        newself._data = np.concatenate((common_data, data[:, otherdata.shape[-1]:]), axis=1)

                    else:
                        common_data = data - otherdata[:, :data.shape[-1]]
                        newself._data = np.concatenate((common_data, otherdata[:, data.shape[-1]:]), axis=1)
        else:
            raise TypeError(f'{type(other)} is not supported')                        
        return newself

    def validate_and_convert(self, t):

        duration = self.get_duration()
        if t is None:
            return None
        if not isinstance(t, (int, float)):
            raise ValueError(f"Time value must be numeric, got {type(t)}")
        if abs(t) >= duration:
            raise OverflowError(f'Input time ({t}) must be less than the record duration ({duration})')
        if t < 0:
            t += duration

        return self.time2byte(t)

    def _time_slice(self, item):
        """
        Slice and manipulate audio data with time-based range selection and speed adjustment.

        This method provides flexible audio slicing and time-stretching capabilities:
        - Selects a specific time range from the audio
        - Adjusts playback speed while preserving original pitch
        - Maintains frame and sample alignment
        - Works with both packed and unpacked audio data

        Parameters:
        -----------
        item : tuple or None, optional
            A tuple containing (start, stop, speed_ratio) with the following properties:
            - start : float or None
                Starting time point of the slice (None defaults to beginning)
            - stop : float or None
                Ending time point of the slice (None defaults to end)
            - speed_ratio : float or None, default 1.0
                Speed adjustment factor:
                - 1.0: Original speed (no change)
                - < 1.0: Slower playback
                - > 1.0: Faster playback

        Returns:
        --------
        numpy.ndarray
            Sliced and speed-adjusted audio data

        Raises:
        -------
        ValueError
            If speed ratio is not positive (≤ 0)

        Notes:
        ------
        - Preserves multi-channel audio structure
        - Ensures frame-aligned byte slicing
        - Uses tempo adjustment algorithm for speed changes
        - Maintains original data type after processing

        Examples:
        ---------
        # Slice first 5 seconds
        data = audio_wrap[0:5]

        # Speed up audio by 1.5x
        fast_data = audio_wrap[None, None, 1.5]

        # Slice and speed up simultaneously
        modified_data = audio_wrap[2:10, 0.8]
        """
        from  sudio.process.fx._tempo import tempo_cy

        item = item or (None, None, None)
        start, stop, step = item
        speed_ratio = step if step is not None else 1.0

        if speed_ratio <= 0:
            raise ValueError("Speed ratio must be positive")

        bytes_per_sample = len(np.dtype(self._sample_type).str)
        samples_per_frame = bytes_per_sample * self._nchannels
        
        start_byte = self.validate_and_convert(start) or 0
        stop_byte = self.validate_and_convert(stop) or -1

        if start_byte > 0:
            start_byte = (start_byte // samples_per_frame) * samples_per_frame
        if stop_byte > 0:
            stop_byte = ((stop_byte + samples_per_frame - 1) // samples_per_frame) * samples_per_frame

        if start_byte > stop_byte and stop_byte != -1:
            start_byte, stop_byte = stop_byte, start_byte
            step = -1
        else:
            step = 1

        if self._packed:
            with self.get() as generator:
                generator.seek(start_byte, 0)
                size = abs(stop_byte - start_byte) if stop_byte != -1 else self._size
                # size is aligned with frame size
                size = (size // samples_per_frame) * samples_per_frame
                data = generator.read(size)
                proceed_data = self._from_buffer(data)
                proceed_data = proceed_data[:, ::step] if self._nchannels > 1 else proceed_data[::step]
        else:
            start_idx = None if start_byte == 0 else start_byte // samples_per_frame
            stop_idx = None if stop_byte == -1 else stop_byte // samples_per_frame
            proceed_data = self._data[:, start_idx:stop_idx:step] if self._nchannels > 1 else self._data[start_idx:stop_idx:step]

        if abs(speed_ratio - 1.0) < 1e-6:
            return proceed_data

        dtype = proceed_data.dtype
        proceed_data = tempo_cy(proceed_data, np.asarray([]), self._sample_rate, default_tempo=speed_ratio)
        return proceed_data.astype(dtype)


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

        # Handle single slice or multiple slices
        if isinstance(item, slice):
            items = [item]
        elif isinstance(item, (list, tuple)):
            items = item
        else:
            return buffer

        for current_item in items:
            if not isinstance(current_item, slice):
                continue

            obj_type = AudioWrap._slice_type(current_item)
            
            if obj_type is int:
                # time slice
                last_item[0] = (current_item.start, current_item.stop, current_item.step)
                buffer[last_item[0]] = []

            elif obj_type is None:
                last_item[0] = None
                buffer[None] = []

            elif obj_type is str:
                # Filter slice
                filt = {
                    'ftype': 'butter',
                    'rs': None,
                    'rp': None,
                    'order': 5,
                    'scale': None
                }

                # detectparameters
                if current_item.step:
                    parsed = parse_dictionary_string(current_item.step, item_sep=',', dict_eq='=')
                    for key, value in parsed.items():
                        if key in filt:
                            filt[key] = value

                # filter type and frequencies
                if current_item.start is not None and current_item.stop is not None:
                    if filt['scale'] and float(filt['scale']) < 0:
                        btype = 'bandstop'
                    else:
                        btype = 'bandpass'
                    freq = float(current_item.start), float(current_item.stop)
                    assert freq[1] > freq[0], ValueError(f'{freq[0]} is bigger than {freq[1]}')

                elif current_item.start is not None:
                    btype = 'highpass'
                    freq = float(current_item.start)

                elif current_item.stop is not None:
                    btype = 'lowpass'
                    freq = float(current_item.stop)

                else:
                    continue

                iir = scisig.iirfilter(
                    filt['order'], 
                    freq, 
                    btype=btype, 
                    fs=self._sample_rate, 
                    output='sos',
                    rs=filt['rs'], 
                    rp=filt['rp'], 
                    ftype=filt['ftype']
                )

                #  buffer has an appropriate key
                buffer_key = last_item[0] if last_item[0] is not None else None
                if buffer_key not in buffer:
                    buffer[buffer_key] = []

                # filter to the buffer
                buffer[buffer_key].append((
                    iir, 
                    *[abs(float(i)) for i in (filt['scale'],) if i is not None]
                ))

        return buffer


    def __str__(self):
        """
        :return: string representation of the object.
        """
        return 'AudioWrap instance of {}'.format(self._rec.name)

    # Getters
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
        Get the parent object (Master) associated with this AudioWrap object.

        :return: The parent Master object.
        """
        return self._master

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
        return self.byte2time(self.get_size())

    def get_data(self) -> Union[AudioMetadata, np.ndarray]:
        """
        Get the audio data either from cached files or dynamic memory.

        :return: If packed, returns record information. If unpacked, returns the audio data.
        """
        if self._packed:
            record = self._rec.copy()
            size = os.path.getsize(self._file.name)
            record.size = size
            record.duration = self.byte2time(size)
            return self._rec
        return self._data

    def is_packed(self) -> bool:
        """
        :return: True if the object is in packed mode, False otherwise.
        """
        return self._packed

    @contextmanager
    def get(self, offset=None, whence=None):
        """
        Context manager for getting a file handle and managing data.

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

    def time2byte(self, t:float) -> int:
        """
        Convert time in seconds to byte offset in audio data.

        :param t: Time in seconds
        :return: Byte index corresponding to the specified time
        """
        idx = self._sample_rate * self._nchannels * self.sample_width * t
        return int(idx)

    def byte2time(self, byte:int) -> float:
        """
        Convert byte offset to time in seconds.

        :param byte: Byte index in audio data
        :return: Time in seconds corresponding to the byte index
        """
        return byte / (self._sample_rate * self._nchannels * self.sample_width)

    def afx(self, cls:FX, *args, 
           start:np.float32=None, 
           stop:np.float32=None, 
           duration:np.float32=None, 
           input_gain_db:np.float32=0.0, 
           output_gain_db:np.float32=0.0, 
           wet_mix:np.float32=None, 
           channel:int=None,
           channel_fill_mode:AFXChannelFillMode=AFXChannelFillMode.REFERENCE_REFINED,
           channel_fill_value:np.float32=0.0,
           transition:FadePreset =None,
           **kwargs):
        """
        Apply an audio effect to the audio data with advanced channel and gain controls.

        :param cls: Effect class to apply (must be a subclass of FX)
        :type cls: type[FX]

        :param start: Start time for effect application (optional)
        :type start: float, optional

        :param stop: Stop time for effect application (optional)
        :type stop: float, optional

        :param input_gain_db: Input gain in decibels, defaults to 0.0
        :type input_gain_db: float, optional

        :param output_gain_db: Output gain in decibels, defaults to 0.0
        :type output_gain_db: float, optional

        :param wet_mix: Effect mix ratio (0.0 to 1.0), optional

            - Blends original and processed signals
            - Note: Not supported for all effects

        :type wet_mix: float, optional

        :param channel: Specific channel to apply effect to in multi-channel audio

            - Only applicable for multi-channel audio (>1 channel)
            - Raises TypeError if used in mono mode

        :type channel: int, optional

        :param channel_fill_mode: Strategy for handling length mismatches between original and processed audio

            - AFXChannelFillMode.PAD: Pad shorter audio with specified fill value
            - AFXChannelFillMode.TRUNCATE: Truncate to shortest audio length
            - AFXChannelFillMode.REFERENCE_RAW: 

                * If refined audio is shorter, pad with fill value
                * If refined audio is longer, truncate to original audio length

            - AFXChannelFillMode.REFERENCE_REFINED: 

                * If refined audio is shorter, truncate original audio to refined length
                * If refined audio is longer, pad original audio with fill value

        :type channel_fill_mode: AFXChannelFillMode, optional


        :param channel_fill_value: Value used for padding when channel_fill_mode is PAD
        :type channel_fill_value: float, optional

        :return: New AudioWrap instance with applied effect
        :rtype: AudioWrap

        :raises TypeError: 

            - If effect class is not supported
            - If channel parameter is invalid
            - If attempting to use channel in mono mode

        :raises AttributeError: If audio data is not packed
        :raises RuntimeError: If channel dimensions are inconsistent

        """
        assert issubclass(cls, FX), TypeError('unsupported parameter')
        
        other = self()

        nchannels = other._nchannels
        if channel is not None:
            assert isinstance(channel, int), TypeError('Unknown channel type')
            assert self._nchannels > 1, TypeError('feature not available in MONO mode')
            assert channel < self._nchannels, TypeError('should be lower than number of audio channels')
            nchannels = 1

        common_args = {
            'data_size': other._size,
            'sample_rate': other._sample_rate,
            'nchannels': nchannels,
            'sample_format': other._sample_format,
            'data_nperseg': other._nperseg,
            'sample_type': other._sample_type,
            'sample_width': other.sample_width,
        }
        
        fx = cls(
            *args, 
            **common_args,
            **kwargs
        )
        
        assert fx.is_offline_supported(), 'fx is not compatible'
        
        if start is not None: 
            assert isinstance(start, (float, int)), 'Type Error'
        if stop is not None: 
            assert isinstance(stop, (float, int)), 'Type Error'
        if duration is not None:
            assert isinstance(duration, (float, int)), 'Type Error'
            assert duration > 0, ValueError('negative duration')
            if stop is None:
                stop = start + duration
        if transition is not None:
                assert isinstance(transition, FadePreset), TypeError('unknown transition')
                transition = transition.value

        assert other._packed, AttributeError('must be packed')

        dtype = fx.get_preferred_datatype()

        input_gain = dtype_converter(db2amp(input_gain_db), dtype)
        output_gain = dtype_converter(db2amp(output_gain_db), dtype)


        with other.unpack(
            astype=dtype, 
            start=start, 
            stop=stop,
            truncate=True,
        ) as pdata:
            
            data = pdata
            if channel is not None:
                data = pdata[channel]

            data_with_input_gain = data * input_gain
            preshape = data_with_input_gain.shape
            refined = fx.process(data_with_input_gain, *args, **kwargs)
            refined = refined * output_gain
            if transition is not None:
                envelope = generate_envelope(
                    refined.shape[-1],
                    preset=transition,
                )
                refined = refined * envelope
                
            assert len(preshape) == refined.ndim, RuntimeError("different number of channels")
            if wet_mix is not None:
                if preshape[-1] ==  refined.shape[-1]:
                    refined = data * (1 - wet_mix) + refined * wet_mix
                else:
                    warnings.warn("wet_mix is not supported for this fx")

            if channel is not None:
                
                diff = refined.shape[-1] - pdata.shape[-1]

                if diff == 0:
                    pass
                elif channel_fill_mode == AFXChannelFillMode.PAD:

                    pad_length = abs(diff)
                    
                    if  diff < 0:
                        # refined is shorter
                        refined = np.pad(refined, 
                                         (0, pad_length), 
                                         mode='constant', 
                                         constant_values=channel_fill_value)                        
                    else:
                        # pdata is shorter
                        pdata = np.pad(pdata, 
                                         ((0, 0), (0, pad_length)),
                                         mode='constant', 
                                         constant_values=channel_fill_value)    
                
                elif channel_fill_mode == AFXChannelFillMode.TRUNCATE:
                    if  diff < 0:
                        # refined is shorter
                        pdata = pdata[..., :refined.shape[-1]]
                    else:
                        # pdata is shorter
                        refined = refined[:pdata.shape[-1]]

                elif channel_fill_mode == AFXChannelFillMode.REFERENCE_RAW:
                    pad_length = abs(diff)

                    if  diff < 0:
                        # refined is shorter
                        refined = np.pad(refined, 
                                         (0, pad_length), 
                                         mode='constant', 
                                         constant_values=channel_fill_value)    
                    else:
                        # pdata is shorter
                        refined = refined[:pdata.shape[-1]]

                elif channel_fill_mode == AFXChannelFillMode.REFERENCE_REFINED:
                    pad_length = abs(diff)

                    if  diff < 0:
                        # refined is shorter
                        pdata = pdata[..., :refined.shape[-1]] 
                    else:
                        # pdata is shorter
                        pdata = np.pad(pdata, 
                                         ((0, 0), (0, pad_length)),
                                         mode='constant', 
                                         constant_values=channel_fill_value)    
                
                pdata[channel] = refined
                refined = pdata

            other._data = refined
            other._size = os.path.getsize(other._file.name)
            
        return other

    @staticmethod
    def _slice_type(item: slice):
        """
        Determine the type of slice (int, float, or None) based on the provided slice object.

        :param item: The slice object.
        :return: Type of slice (int, float, or None).
        """
        items = [i for i in (item.start, item.stop, item.step) if i is not None]
        if not items:
            return None
        item_type = list(map(lambda x: int if x is float else x, map(type, items)))
        assert item_type.count(item_type[0]) == len(items)
        return item_type[0]

    @staticmethod
    def _freq_slice(buffer: np.ndarray, item: tuple) -> np.ndarray:
        """
        Apply frequency domain slicing to the audio data.

        :param buffer: Audio data.
        :param item: Tuple representing filter parameters (sos and scale).
        :return: Processed audio data.
        """
        sos, *scale = item
        result = scisig.sosfilt(sos, buffer)
        return result * scale[0] if scale else result
    

    
