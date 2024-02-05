"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""
import scipy.signal as scisig
import numpy as np
import samplerate

from sudio._register import Members as Mem
from sudio.extras.arraytool import push
from sudio._audio import win_parser
from sudio.audioutils.audio import Audio
from sudio.types import SampleFormat
from sudio.stream.stream import Stream


# ________________________________________________________________________________

@Mem.master.add
def _win_mono(self, data):
    if self._window_type:
        retval = np.vstack((self._win_buffer[1], np.hstack((self._win_buffer[1][self._nhop:],
                                                            self._win_buffer[0][:self._nhop])))) * self._window
        push(self._win_buffer, data)
    else:
        retval = data.astype('float64')
    return retval


@Mem.master.add
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
            push(self._win_buffer[i], data[i])
        # for 2 channel win must be an 2, 2, self._data_chunk(e.g. 256)
        final = np.array(final)
    else:
        final = data.astype('float64')
    return final


@Mem.master.add
def _iwin_mono(self, win):
    retval = np.hstack((self._iwin_buffer[0][self._nhop:], win[1][:self._nhop])) + win[0]
    self._iwin_buffer[0] = win[1]
    # return data
    # plt.plot(retval, label='iwin')
    # plt.legend()
    # plt.show()
    return retval


# start from index1 data
@Mem.master.add
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
@Mem.master.add
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
            Stream(self, pip, name, stream_type=stream_type, process_type=process_type))

    elif process_type == 'branch':
        self.branch_pipe_database.append(
            Stream(self, pip, name, stream_type=stream_type, channel=channel,
                                   process_type=process_type))

    # reserved for the next versions
    else:
        pass


@Mem.master.add
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


@Mem.master.add
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


@Mem.master.add
def get_window(self):
    if self._window_type:
        return {'type': self._window_type,
                'window': self._window,
                'overlap': self._noverlap,
                'size': len(self._window)}
    else:
        return None


@Mem.master.add
def disable_std_input(self):
    self._main_stream.acquire()


@Mem.master.add
def enable_std_input(self):
    self._main_stream.clear()
    self._main_stream.release()


# arr format: [[data0:[ch0] [ch1]]  [data1: [ch0] [ch1]], ...]
@Mem.master.add
def shuffle3d_channels(self, arr):
    arr = self.__class__._shuffle3d_channels(arr)
    # res = np.array([])
    # for i in arr:
    #     res = np.append(res, Process.shuffle2d_channels(i))
    return arr.reshape(np.prod(arr.shape))
    # return res


@Mem.master.add
def shuffle2d_channels(arr):
    return arr.T.reshape(np.prod(arr.shape))


@Mem.master.add_static
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
    # print(nchannels)
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




