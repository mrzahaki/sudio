#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


import io
import os
from contextlib import contextmanager

from sudio.wrap.wrap import Wrap
from sudio.types.name import Name
from sudio.io import get_sample_size
from sudio.utils.cacheutil import write_to_cached_file, handle_cached_record
from sudio.audiosys.typeconversion import convert_array_type
from sudio.utils.timed_indexed_string import TimedIndexedString
from sudio.io import SampleFormat
from sudio.metadata import AudioMetadata
import numpy as np

class WrapGenerator:
    # Class variable to store the name
    name = Name()

    def __init__(self, master, record):
        '''
        Initialize the WrapGenerator instance.

        Args:
            master: The master instance.
            record: The record to be wrapped.
        '''
        # Check the type of record and handle accordingly
        rec_type = type(record)
        if rec_type is AudioMetadata:
            self._rec: AudioMetadata = record
            # Convert bytes to cache and update the record
            if type(record['o']) is bytes:
                record['o'] = (
                    write_to_cached_file(record['size'],
                                record['frameRate'],
                                record['sampleFormat'] if record['sampleFormat'] else 0,
                                record['nchannels'],
                                file_name=master._audio_data_directory + record.name + master.__class__.BUFFER_TYPE,
                                data=record['o'],
                                pre_truncate=True,
                                after_seek=(master.__class__.CACHE_INFO, 0),
                                after_flush=True)
                )
                record['size'] += master.__class__.CACHE_INFO

        elif rec_type is str:
            # Load record if it's a string
            self._rec: AudioMetadata = master.load(record, series=True)

        self._parent = master
        self._file: io.BufferedRandom = self._rec['o']
        self._name = TimedIndexedString(self._file.name,
                                seed='wrapped',
                                start_before=self._parent.__class__.BUFFER_TYPE)
        self._size = self._rec['size']
        self._duration = record['duration']
        self._sample_rate = self._rec['frameRate']
        self._nchannels = self._rec['nchannels']
        self._sample_format = self._rec['sampleFormat']
        self._nperseg = self._rec['nperseg']
        self._sample_type = self._parent._sample_width_format_str
        self.sample_width = get_sample_size(self._rec['sampleFormat'])
        self._seek = 0
        self._data = self._rec['o']
        self._packed = True


    def __call__(self,
                 *args,
                 sync_sample_format_id: int = None,
                 sync_nchannels: int = None,
                 sync_sample_rate: int = None,
                 safe_load: bool = True
                 ):
        '''
        Call the WrapGenerator instance, creating a new cache file and a new record.

        Args:
            sync_sample_format_id: Sync the sample format ID.
            sync_nchannels: Sync the number of channels.
            sync_sample_rate: Sync the sample rate.
            safe_load: Use safe loading.

        Returns:
            Wrap: A Wrap instance.
        '''
        record = (
            handle_cached_record(self._rec.copy(),
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

    def get_data(self) -> AudioMetadata:
        '''
        Get the data of the WrapGenerator instance.

        Returns:
            AudioMetadata: The data.
        '''
        return self._rec.copy()

    @contextmanager
    def get(self, offset=None, whence=None):
        '''
        Get the context manager for the file.

        Args:
            offset: Offset for seeking.
            whence: Reference for seeking.

        Yields:
            io.BufferedRandom: The file.
        '''
        try:
            # Seek and yield the file
            if offset is not None and whence is not None:
                self._seek = self._file.seek(offset, whence)
            else:
                self._seek = self._file.tell()

            yield self._file
        finally:
            # Seek back to the original position after the block
            self._file.seek(self._seek, 0)

    def set_data(self, data):
        '''
        Set the data of the WrapGenerator instance.

        Args:
            data: The data to be set.
        '''
        self._data = data

    def get_sample_format(self) -> SampleFormat:
        '''
        Get the sample format of the WrapGenerator instance.

        Returns:
            SampleFormat: The sample format.
        '''
        return self._sample_format

    def get_sample_width(self) -> int:
        '''
        Get the sample width of the WrapGenerator instance.

        Returns:
            int: The sample width.
        '''
        return self.sample_width

    def get_master(self):
        '''
        Get the master of the WrapGenerator instance.

        Returns:
            The master.
        '''
        return self._parent

    def get_size(self) -> int:
        '''
        Get the size of the WrapGenerator instance.

        Returns:
            int: The size.
        '''
        return os.path.getsize(self._file.name) - self._file.tell()

    def get_cache_size(self) -> int:
        '''
        Get the cache size of the WrapGenerator instance.

        Returns:
            int: The cache size.
        '''
        return os.path.getsize(self._file.name)

    def get_sample_rate(self) -> int:
        '''
        Get the frame rate of the WrapGenerator instance.

        Returns:
            int: The frame rate.
        '''
        return self._sample_rate

    def get_nchannels(self) -> int:
        '''
        Get the number of channels of the WrapGenerator instance.

        Returns:
            int: The number of channels.
        '''
        return self._nchannels

    def get_duration(self) -> float:
        '''
        Get the duration of the WrapGenerator instance.

        Returns:
            float: The duration.
        '''
        return self._duration

    def join(self,
             *other,
             sync_sample_format: SampleFormat = None,
             sync_nchannels: int = None,
             sync_sample_rate: int = None,
             safe_load: bool = True
             ):
        '''
        Join multiple WrapGenerators.

        Args:
            *other: Other WrapGenerators.
            sync_sample_format: Sync the sample format.
            sync_nchannels: Sync the number of channels.
            sync_sample_rate: Sync the sample rate.
            safe_load: Use safe loading.

        Returns:
            Wrap: A Wrap instance.
        '''
        return (self.__call__(sync_sample_format_id=sync_sample_format,
                             sync_nchannels=sync_nchannels,
                             sync_sample_rate=sync_sample_rate,
                             safe_load=safe_load).join(*other))

    def __getitem__(self, item):
        '''
        Get an item from the WrapGenerator.

        Args:
            item: The item to get.

        Returns:
            Wrap: A Wrap instance.
        '''
        return self.__call__().__getitem__(item)

    def __del__(self):
        '''
        Delete the WrapGenerator instance.
        '''
        if not self._file.closed:
            self._file.close()
        try:
            self._parent.del_record(self.name)
        except ValueError:
            # 404: Record not found
            pass
        if os.path.exists(self._file.name):
            try:
                os.remove(self._file.name)
            except PermissionError:
                pass
            

    def __str__(self):
        '''
        Get a string representation of the WrapGenerator instance.

        Returns:
            str: The string representation.
        '''
        return 'WrapGenerator object of {}'.format(self._rec.name)

    def __mul__(self, scale):
        '''
        Multiply the WrapGenerator instance by a scale.

        Args:
            scale: The scale.

        Returns:
            Wrap: A Wrap instance.
        '''
        return self.__call__().__mul__(scale)

    def __truediv__(self, scale):
        '''
        Divide the WrapGenerator instance by a scale.

        Args:
            scale: The scale.

        Returns:
            Wrap: A Wrap instance.
        '''
        return self.__call__().__truediv__(scale)

    def __pow__(self, power, modulo=None):
        '''
        Raise the WrapGenerator instance to the power.

        Args:
            power: The power.
            modulo: The modulo.

        Returns:
            Wrap: A Wrap instance.
        '''
        return self.__call__().__pow__(power, modulo=modulo)

    def __add__(self, other):
        '''
        Add another object to the WrapGenerator.

        Args:
            other: The other object.

        Returns:
            Wrap: A Wrap instance.
        '''
        return self.__call__().__add__(other)

    def __sub__(self, other):
        '''
        Subtract another object from the WrapGenerator.

        Args:
            other: The other object.

        Returns:
            Wrap: A Wrap instance.
        '''
        return self.__call__().__sub__(other)

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