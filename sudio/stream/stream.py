import threading
import queue
import numpy as np

from sudio.extras.arraytool import push
from sudio._register import Members as Mem
from sudio.types import StreamError


@Mem.master.add
class Stream:
    def __init__(self, other, obj, name=None, stream_type='multithreading', channel=None, process_type='main'):

        '''
        Initialize the Stream class for managing audio data streams.

        Args:
        - other: An instance of the main audio processing class.
        - obj: An object representing the data stream (e.g., list, queue).
        - name: A name for the stream (optional).
        - stream_type: Type of streaming operation ('multithreading' or reserved for future versions).
        - channel: Channel index for controlling multi-channel streams (optional).
        - process_type: Type of stream processing ('main', 'multi_stream', 'branch', 'queue').

        Raises:
        - StreamError: If there are issues with the provided parameters based on the selected processing mode.

        Note:
        - This class provides a flexible mechanism for managing different types of audio data streams based on the specified
          processing mode and stream type.
        '''

        self.stream_type = stream_type
        self.process_obj = other
        self.process_type = process_type
        self.pip = obj
        self.channel = channel
        self.name = name
        self._data_indexer = lambda data: data[channel]
        self._lock = threading.Lock()

        if process_type == 'main' and type(obj) == list and len(obj) == other.nchannels and channel is None:
            self.process_type = 'multi_stream'

        if process_type == 'multi_stream':
            if not channel is None:
                raise StreamError('In multi_stream mode, channel value must have None value (see documentation)')
            elif not type(obj) == list or not len(obj) == other.nchannels:
                raise StreamError('In multi_stream mode, pip argument must be a list with the length of the number of data channels')
            elif other.nchannels < 2 or other._mono_mode:
                raise StreamError('The multi_stream mode works only when the number of input channels is set to be more than 1')

        elif process_type == 'main':
            if not channel is None and not (type(channel) is int and channel <= other.nchannels):
                raise StreamError('Invalid control channel data')

        if stream_type == 'multithreading':
            if process_type == 'main':
                self.put = self._main_put
                self.clear = obj.clear
                self.sget = self._main_get
                self.get = self._main_get

            elif process_type == 'multi_stream':
                self.put = self._multi_main_put
                self.clear = self._multi_main_clear
                self.sget = self._multi_main_get
                self.get = self._multi_main_get

            elif process_type == 'branch':
                self.put = obj.put
                self.clear = obj.clear
                self.sget = obj.get
                self.get = obj.get

            elif process_type == 'queue':
                self.put = self._main_put
                self.clear = obj.queue.clear
                self.sget = self._main_get
                self.get = self._main_get

            if channel is None or self.process_obj._mono_mode or (self.process_obj.nchannels == 1):
                self._data_indexer = lambda data: data
        else:
            pass
            # obj.insert(0, np.vectorize(other._win, signature=''))
            # pickle.dumps(obj._config['authkey'])
            # reserved for next versions
            # self.put = obj.put_nowait
            # self.clear = obj.queue.clear
            # self.get = lambda: other._win(obj.get())
            # self.sget = obj.get

    def acquire(self):
        '''
        Acquire the lock for thread-safe operations.

        Note:
        - This method is used for thread synchronization in multithreading scenarios.
        '''
        self._lock.acquire()

    def release(self):
        '''
        Release the lock after thread-safe operations.

        Note:
        - This method is used for releasing the lock acquired during thread-safe operations.
        '''
        self._lock.release()

    def locked(self):
        '''
        Check if the lock is currently held.

        Returns:
        - bool: True if the lock is currently held, False otherwise.

        Note:
        - This method is used to check the status of the lock for thread synchronization.
        '''
        return self._lock.locked()

    def _main_put(self, data):
        '''
        Put data into the main stream and distribute it to all connected branches.

        Args:
        - data: Numpy array containing audio data.

        Notes:
        - Applies windowing to the data if specified in the main audio processing instance.
        - Distributes data to connected branches using their respective data indexers.
        '''

        data = self.process_obj._win(data)
        # print(data.shape)
        for branch in self.process_obj.branch_pipe_database:
            try:
                branch.put(branch._data_indexer(data))
            except queue.Full:
                pass
        # print(self.process_obj.branch_pipe_database)
        self.pip.put(data)

    def _main_get(self, *args):
        '''
        Get data from the main stream.

        Returns:
        - Numpy array containing audio data retrieved from the main stream.

        Notes:
        - Reverses windowing if specified in the main audio processing instance.
        '''
        data = self.pip.get(*args)
        if self.process_obj._window_type:
            data = self.process_obj._iwin(data)
        # print(data.shape)
        return data.astype(self.process_obj._constants[0])

    def _multi_main_put(self, data):
        '''
        Put data into the multi-stream, applying windowing if specified.

        Args:
        - data: Numpy array containing audio data.

        Notes:
        - Applies windowing to the data if specified in the main audio processing instance.
        - Distributes data to connected branches using their respective data indexers.
        '''
        # windowing
        if self.process_obj._window_type:
            final = []
            # Channel 0
            win = np.vstack((self.process_obj._win_buffer[0][1], np.hstack(
                (self.process_obj._win_buffer[0][1][self.process_obj._nhop:],
                 self.process_obj._win_buffer[0][0][:self.process_obj._nhop])))) * self.process_obj._window

            push(self.process_obj._win_buffer[0], data[0])

            try:
                self.pip[0].put(win)
            except queue.Full:
                pass

            final.append(win)
            # range(self.process_obj.nchannels)[1:]
            for i in self.process_obj._constants[2]:
                win = np.vstack((self.process_obj._win_buffer[i][1], np.hstack(
                    (self.process_obj._win_buffer[i][1][self.process_obj._nhop:],
                     self.process_obj._win_buffer[i][0][:self.process_obj._nhop])))) * self.process_obj._window

                push(self.process_obj._win_buffer[i], data[i])

                try:
                    self.pip[i].put(win)
                except queue.Full:
                    pass

                final.append(win)
        else:
            final = data.astype('float64')
            for i in self.process_obj._constants[4]:
                try:
                    self.pip[i].put(final[i])
                except queue.Full:
                    pass

        # for 2 channel win must be an 2, 2, self.process_obj._data_chunk(e.g. 256)
        # reshaping may create some errors
        # return data
        for branch in self.process_obj.branch_pipe_database:
            branch.put(branch._data_indexer(final))

    def _multi_main_get(self, *args):
        '''
        Get data from the multi-stream.

        Returns:
        - Numpy array containing audio data retrieved from the multi-stream.

        Notes:
        - Reverses windowing if specified in the main audio processing instance.
        '''
        # channel 0
        if self.process_obj._window_type:
            win = self.pip[0].get(*args)
            retval = np.hstack(
                (self.process_obj._iwin_buffer[0][self.process_obj._nhop:], win[1][:self.process_obj._nhop])) + \
                     win[0]
            self.process_obj._iwin_buffer[0] = win[1]

            for i in self.process_obj._constants[2]:
                win = self.pip[i].get(*args)
                tmp = np.hstack(
                    (self.process_obj._iwin_buffer[i][self.process_obj._nhop:], win[1][:self.process_obj._nhop])) + \
                      win[0]
                retval = np.vstack((retval, tmp))
                self.process_obj._iwin_buffer[i] = win[1]
        else:
            retval = self.pip[0].get(*args)
            for i in self.process_obj._constants[2]:
                tmp = self.pip[i].get(*args)
                retval = np.vstack((retval, tmp))

        return retval.astype(self.process_obj._constants[0])

    def _multi_main_clear(self):
        '''
        Clear all connected pipes in the multi-stream.
        '''
        for i in self.pip:
            i.clear()

    def set(self, other):
        '''
        Set the attributes of the current stream based on another stream.

        Args:
        - other: Another instance of the Stream class.

        Note:
        - This method allows setting the attributes of the current stream to match those of another stream.
        '''
        if isinstance(other, self.__class__):
            assert ('main' in self.process_type or
                    'multi_stream' in self.process_type or
                    'queue' in self.process_type) \
                   and \
                   ('main' in other.process_type or
                    'multi_stream' in other.process_type or
                    'queue' in other.process_type), \
                'set is enabled only for "main or multi_stream" modes'
            (self.put,
             self.clear,
             self.get,
             self.sget,
             self.stream_type,
             self.process_obj,
             self.process_type,
             self.pip,
             self.channel,
             self.name) = (other.put,
                           other.clear,
                           other.get,
                           other.sget,
                           other.stream_type,
                           other.process_obj,
                           other.process_type,
                           other.pip,
                           other.channel,
                           other.name)

    def __eq__(self, other):
        '''
        Check if two Stream instances are equal.

        Returns:
        - bool: True if the two instances are equal, False otherwise.

        Note:
        - This method compares key attributes to determine equality between two Stream instances.
        '''
        if isinstance(other, self.__class__):
            return (self.put, self.clear, self.get) == (other.put, other.clear, other.get)
        return False

    def __ne__(self, other):
        '''
        Check if two Stream instances are not equal.

        Returns:
        - bool: True if the two instances are not equal, False otherwise.

        Note:
        - This method is the negation of the __eq__ method.
        '''
        return not self.__eq__(other)

    def type(self):
        '''
        Get the type of stream processing.

        Returns:
        - str: The type of stream processing ('main', 'multi_stream', 'branch', 'queue').

        Note:
        - This method returns the type of stream processing used by the current instance.
        '''
        return self.process_type
