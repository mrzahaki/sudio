"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""
from ._register import Members as Mem
from ._register import static_vars
from ._tools import Tools
from ._audio import *
from ._pipeline import Pipeline
import scipy.signal as scisig
import numpy as np
import threading
import os
import samplerate

# ________________________________________________________________________________

@Mem.process.add
def _win_mono(self, data):
    if self._window_type:
        retval = np.vstack((self._win_buffer[1], np.hstack((self._win_buffer[1][self._nhop:],
                                                            self._win_buffer[0][:self._nhop])))) * self._window
        Tools.push(self._win_buffer, data)
    else:
        retval = data.astype('float64')
    return retval


@Mem.process.add
def _win_nd(self, data):
    # data = data.astype('float64')
    # retval frame consists of two window
    # that each window have the shape same as 'data' param shape(e.g. for two channel stream:(2, 2, chunk_size))
    # note when primary_filter is enabled retval retval shape changes depend on scisig.upfirdn filter.
    # In general form retval shape is
    # (number of channels, number of windows(2), size of data chunk depend on primary_filter activity).
    if self._window_type:
        final = []
        for i in self._constants[4]:
            final.append(np.vstack((self._win_buffer[i][1], np.hstack(
                (self._win_buffer[i][1][self._nhop:], self._win_buffer[i][0][:self._nhop])))) * self._window)
            Tools.push(self._win_buffer[i], data[i])
        # for 2 channel win must be an 2, 2, self._data_chunk(e.g. 256)
        final = np.array(final)
    else:
        final = data.astype('float64')
    return final


@Mem.process.add
def _iwin_mono(self, win):
    retval = np.hstack((self._iwin_buffer[0][self._nhop:], win[1][:self._nhop])) + win[0]
    self._iwin_buffer[0] = win[1]
    # return data
    # plt.plot(retval, label='iwin')
    # plt.legend()
    # plt.show()
    return retval


# start from index1 data
@Mem.process.add
def _iwin_nd(self, win):
    # win.shape =>(number of channels, number of windows(2), size of data chunk depend on primary_filter activity).
    # _iwin_buffer => [buffer 0, buffer1, buffer(number of channels)]
    # for 2 channel win must be an 2[two ], 2, self._data_chunk(e.g. 256)
    # pre post, data,
    # 2 window per frame

    # retval = np.hstack((win[n-1][self._nhop:], current_win[n+1][:self._nhop])) + current_win[n]
    # win[n-1] =  current_win[n+1]
    retval = np.hstack((self._iwin_buffer[0][self._nhop:], win_parser(win, 1, 0)[:self._nhop])) + \
             win_parser(win, 0, 0)
    self._iwin_buffer[0] = win_parser(win, 1, 0)

    for i in self._constants[2]:
        tmp = np.hstack((self._iwin_buffer[i][self._nhop:], win_parser(win, 1, i)[:self._nhop])) + \
              win_parser(win, 0, i)
        retval = np.vstack((retval, tmp))
        self._iwin_buffer[i] = win_parser(win, 1, i)

    return retval


# ____________________________________________________ pipe line
@Mem.process.add
def add_pipeline(self, name, pip, process_type='main', channel=None):
    '''
    :param name: string; Indicates the name of the pipeline
    :param pip: obj; Pipeline object/s
                Note:   In the multi_stream process type, pip argument must be an array of defined pipelines.
                        the size of the array must be same as the number of input channels.
    :param process_type: string; 'main', 'branch', 'multi_stream'
        The DSP core process input data and pass it to all of the activated pipelines[if exist] and then
        takes output from the main pipeline/s[multi_stream pipeline mode]

    :param channel: obj; None or [0 to self.nchannel]
        The input  data passed to pipeline can be an numpy array in
        (self.nchannel, 2[2 windows in 1 frame], self._nperseg) dimension [None]
        or mono (2, self._nperseg) dimension.
        Note in mono mode[self.nchannel = 1 or mono mode activated] channel must have None value.

    :param stream_type: string; 'multithreading'(used in next versions)

    :note:  multi_stream pipeline mode: If the process_type has a value of  'multi_stream', then multi_stream pipeline
            mode must be activated. In this mode there's most be self.nchannel pipelines
            as an array in the pip argument.


    :return: the pipeline must process data and return it to the core with the  dimensions as same as the input.
    '''
    stream_type = 'multithreading'

    # n-dim main pip
    if process_type == 'main' or process_type == 'multi_stream':
        self.main_pipe_database.append(
            self.__class__._Stream(self, pip, name, stream_type=stream_type, process_type=process_type))

    elif process_type == 'branch':
        self.branch_pipe_database.append(
            self.__class__._Stream(self, pip, name, stream_type=stream_type, channel=channel,
                                   process_type=process_type))

    # reserved for the next versions
    else:
        pass


@Mem.process.add
def set_pipeline(self, name, enable=True):
    if not enable:
        self._main_stream.clear()
        self._main_stream.set(self._normal_stream)
        return
    for obj in self.main_pipe_database:
        if name == obj.name:
            assert obj.process_type == 'main' or obj.process_type == 'multi_stream'
            if obj.process_type == 'multi_stream':
                for i in obj.pip:
                    assert i.is_alive(), f'Error: {name} pipeline dont alive!'
                    i.sync(self._master_sync)

            self._main_stream.clear()
            if self._main_stream.process_type == 'multi_stream':
                for i in self._main_stream.pip:
                    i.aasync()
            self._main_stream.set(obj)


@Mem.process.add
def set_window(self,
               window: object = 'hann',
               noverlap: int = None,
               NOLA_check: bool = True):
    if self._window_type == window and self._noverlap == noverlap:
        return
    else:
        self._window_type = window
        if noverlap is None:
            noverlap = self._nperseg // 2

        if type(window) is str or \
                type(window) is tuple or \
                type(window) == float:
            window = scisig.get_window(window, self._nperseg)

        elif window:
            assert len(window) == self._nperseg, 'control size of window'
            if NOLA_check:
                assert scisig.check_NOLA(window, self._nperseg, noverlap)

        elif self._window_type is None:
            window = None
            self._window_type = None

    self._window = window
    # refresh
    self._noverlap = noverlap
    self._nhop = self._nperseg - self._noverlap
    if self._mono_mode:
        self._iwin_buffer = [np.zeros(self._nperseg)]
        self._win_buffer = [np.zeros(self._nperseg), np.zeros(self._nperseg)]
    else:
        self._iwin_buffer = [np.zeros(self._nperseg) for i in range(self.nchannels)]
        self._win_buffer = [[np.zeros(self._nperseg), np.zeros(self._nperseg)] for i in range(self.nchannels)]
    self._main_stream.clear()


@Mem.process.add
def get_window(self):
    if self._window_type:
        return {'type': self._window_type,
                'window': self._window,
                'overlap': self._noverlap,
                'size': len(self._window)}
    else:
        return None


@Mem.process.add
def disable_std_input(self):
    self._main_stream.acquire()


@Mem.process.add
def enable_std_input(self):
    self._main_stream.clear()
    self._main_stream.release()


# arr format: [[data0:[ch0] [ch1]]  [data1: [ch0] [ch1]], ...]
@Mem.process.add
def shuffle3d_channels(self, arr):
    arr = self.__class__._shuffle3d_channels(arr)
    # res = np.array([])
    # for i in arr:
    #     res = np.append(res, Process.shuffle2d_channels(i))
    return arr.reshape(np.prod(arr.shape))
    # return res


@Mem.process.add
def shuffle2d_channels(arr):
    return arr.T.reshape(np.prod(arr.shape))


@Mem.process.add_static
def _sync(rec,
         nchannels: int,
         sample_rate: int,
         sample_format_id: int,
          output_data='byte'):
    '''

    :param rec:
    :param nchannels:
    :param sample_rate:
    :param sample_format_id:
    :param output_data: can be 'byte' or 'ndarray'
    :return:
    '''

    form = Audio.get_sample_size(rec['sampleFormat'])
    if rec['sampleFormat'] == SampleFormat.formatFloat32.value:
        form = '<f{}'.format(form)
    else:
        form = '<i{}'.format(form)
    data = np.frombuffer(rec['o'], form)
    if rec['nchannels'] == 1:
        if nchannels > rec['nchannels']:
            data = np.vstack([data for i in range(nchannels)])
            rec['nchannels'] = nchannels
    # elif nchannels == 1 or _mono_mode:
    #     data = np.mean(data.reshape(int(data.shape[-1:][0] / rec['nchannels']),
    #                                 rec['nchannels']),
    #                    axis=1)
    else:
        data = [[data[i::rec['nchannels']]] for i in range(nchannels)]
        data = np.append(*data, axis=0)

    if not sample_rate == rec['frameRate']:
        scale = sample_rate / rec['frameRate']

        # firwin = scisig.firwin(23, fc)
        # firdn = lambda firwin, data,  scale: samplerate.resample(data, )
        if len(data.shape) == 1:
            # mono
            data = samplerate.resample(data, scale, converter_type='sinc_fastest')
        else:
            # multi channel
            res = samplerate.resample(data[0], scale, converter_type='sinc_fastest')
            for i in data[1:]:
                res = np.vstack((res,
                                 samplerate.resample(i, scale, converter_type='sinc_fastest')))
            data = res

    if output_data.startswith('b') and rec['nchannels'] > 1:
        data = shuffle2d_channels(data)

    rec['nchannels'] = nchannels
    rec['sampleFormat'] = sample_format_id

    form = Audio.get_sample_size(sample_format_id)
    if sample_format_id == SampleFormat.formatFloat32.value:
        form = '<f{}'.format(form)
    else:
        form = '<i{}'.format(form)

    if output_data.startswith('b'):
        rec['o'] = data.astype(form).tobytes()
    else:
        rec['o'] = data.astype(form)

    rec['size'] = len(rec['o'])
    rec['frameRate'] = sample_rate

    rec['duration'] = rec['size'] / (rec['frameRate'] *
                                     rec['nchannels'] *
                                     Audio.get_sample_size(rec['sampleFormat']))

    return rec


@Mem.process.add
class _Stream:

    def __init__(self, other, obj, name=None, stream_type='multithreading', channel=None, process_type='main'):

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
                raise Error('In multi_stream  mode channel value must have None value(see documentation)')
            elif not type(obj) == list or not len(obj) == other.nchannels:
                raise Error('In multi_stream  mode pip argument must be a'
                            ' list with length of number of data channels')
            elif other.nchannels < 2 or other._mono_mode:
                raise Error('The multi_stream mode just works when the number of input channels '
                            'is set to be more than 1')

        elif process_type == 'main':
            if not channel is None and not (type(channel) is int and channel <= other.nchannels):
                raise Error('Control channel value')

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
        self._lock.acquire()

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def _main_put(self, data):
        data = self.process_obj._win(data)
        # print(data.shape)
        for branch in self.process_obj.branch_pipe_database:
            branch.put(branch._data_indexer(data))
        # print(self.process_obj.branch_pipe_database)
        self.pip.put(data)

    def _main_get(self, *args):
        data = self.pip.get(*args)
        if self.process_obj._window_type:
            data = self.process_obj._iwin(data)
        # print(data.shape)
        return data.astype(self.process_obj._constants[0])

    def _multi_main_put(self, data):

        # windowing
        if self.process_obj._window_type:
            final = []
            # Channel 0
            win = np.vstack((self.process_obj._win_buffer[0][1], np.hstack(
                (self.process_obj._win_buffer[0][1][self.process_obj._nhop:],
                 self.process_obj._win_buffer[0][0][:self.process_obj._nhop])))) * self.process_obj._window

            Tools.push(self.process_obj._win_buffer[0], data[0])
            self.pip[0].put(win)
            final.append(win)
            # range(self.process_obj.nchannels)[1:]
            for i in self.process_obj._constants[2]:
                win = np.vstack((self.process_obj._win_buffer[i][1], np.hstack(
                    (self.process_obj._win_buffer[i][1][self.process_obj._nhop:],
                     self.process_obj._win_buffer[i][0][:self.process_obj._nhop])))) * self.process_obj._window

                Tools.push(self.process_obj._win_buffer[i], data[i])
                self.pip[i].put(win)
                final.append(win)
        else:
            final = data.astype('float64')
            for i in self.process_obj._constants[4]:
                self.pip[i].put(final[i])

        # for 2 channel win must be an 2, 2, self.process_obj._data_chunk(e.g. 256)
        # reshaping may create some errors
        # return data
        for branch in self.process_obj.branch_pipe_database:
            branch.put(branch._data_indexer(final))

    def _multi_main_get(self, *args):
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
        for i in self.pip:
            i.clear()

    def set(self, other):
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
        if isinstance(other, self.__class__):
            return (self.put, self.clear, self.get) == (other.put, other.clear, other.get)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def type(self):
        return self.process_type


class _Time:
    def __get__(self, instance, owner):
        assert instance.isready(), PermissionError('current object is not streaming')
        return instance._itime_calculator(instance._stream_file.tell())

    def __set__(self, instance, tim):
        assert abs(tim) < instance.duration, OverflowError('input time must be less than the record duration')
        assert instance.isready(), PermissionError('current object is not streaming')
        seek = instance._time_calculator(abs(tim))
        if tim < 0:
            seek = instance._stream_file_size - seek
        instance._stream_file.seek(seek, 0)


@Mem.process.add
class StreamControl:
    time = _Time()

    def __init__(self, other, record, on_stop, loop_mode, stream_mode):
        '''
        The StreamControl class is defined to control the mainstream audio playback for special 'record'.

        '''
        self.other = other
        self._stream_type = stream_mode
        self._stream_file = record['o']
        self._stream_file_size = record['size']
        self._size_calculator = lambda: (self.other._data_chunk *
                                         self.other.nchannels *
                                         self.other._sampwidth)
        self._stream_data_size = self._size_calculator()
        self._stream_data_pointer = self._stream_file.tell()
        self._stream_on_stop = on_stop
        self._stream_loop_mode = loop_mode

        self.duration = record['duration']
        self._time_calculator = lambda t: int(self.other._frame_rate *
                                              self.other.nchannels *
                                              self.other._sampwidth *
                                              t)

        self._itime_calculator = lambda byte: byte / (self.other._frame_rate *
                                                      self.other.nchannels *
                                                      self.other._sampwidth)

    def _ready(self):
        assert not self._is_streaming(), PermissionError('Another file is streaming')

        if not self.isready():
            self.other._exstream_mode.clear()
            if self.other._stream_file is not None:
                self.other._stream_file.seek(self._stream_data_pointer, 0)

            self.other._stream_type = self._stream_type
            self.other._stream_file = self._stream_file
            self.other._stream_data_size = self._stream_data_size
            self.other._stream_data_pointer = self._stream_data_pointer
            self.other._stream_on_stop = self._stream_on_stop
            self.other._stream_loop_mode = self._stream_loop_mode

    def isready(self):
        return (self.other._stream_type == self._stream_type and
                self._stream_file == self.other._stream_file and
                self._stream_data_size == self.other._stream_data_size and
                self._stream_on_stop == self.other._stream_on_stop and
                self._stream_loop_mode == self.other._stream_loop_mode)

    def is_streaming(self):
        if self.isready():
            return self.other._exstream_mode.is_set()
        else:
            return False

    def _is_streaming(self):
        return self.other._exstream_mode.is_set()

    def start(self):
        self._ready()
        self.other._exstream_mode.set()
        self.other._main_stream.clear()

    def resume(self):
        assert self.isready(), PermissionError('Another file is streaming')
        self.other._exstream_mode.set()
        self.other._main_stream.clear()

    def stop(self):
        assert self.isready(), PermissionError('Another file is streaming')
        assert self._is_streaming(), PermissionError('The stream is empty')
        self.other._exstream_mode.clear()
        self.other._stream_file.seek(self._stream_data_pointer, 0)
        # self.other._main_stream.clear()

    def pause(self):
        assert self.isready(), PermissionError('Another file is streaming')
        assert self._is_streaming(), PermissionError('The stream is empty')
        self.other._exstream_mode.clear()
        # self.other._main_stream.clear()

    def enable_loop(self):
        assert self.isready(), PermissionError('Another file is streaming')
        assert self._is_streaming(), PermissionError('The stream is empty')
        self.other._stream_loop_mode = True
        self._stream_loop_mode = True

    def disable_loop(self):
        assert self.isready(), PermissionError('Another file is streaming')
        assert self._is_streaming(), PermissionError('The stream is empty')
        self.other._stream_loop_mode = False
        self._stream_loop_mode = False

# ________________________________________________________________________________
class Error(Exception):
    pass
