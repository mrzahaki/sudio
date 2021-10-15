"""
with the name of ALLAH:
 SUDIO (https://github.com/MrZahaki/sudio)

 audio processing platform

 Author: hussein zahaki (hossein.zahaki.mansoor@gmail.com)

 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""
import io
from ._register import Members as Mem
from ._register import static_vars
from ._tools import Tools
from ._audio import *
from ._port import *
from ._pipeline import Pipeline
from ._process_common import *
import tqdm
import scipy.signal as scisig
import threading
import queue
import pandas as pd
import numpy as np
from typing import Union
from enum import Enum
import warnings
import gc
import time
import os
from io import BufferedRandom


class StreamMode(Enum):
    normal = 0
    optimized = 1


@Mem.process.parent
@Mem.sudio.add
class Master:
    DATA_PATH = './data/'
    USER_PATH = DATA_PATH + 'user.sufile'
    SAVE_PATH = DATA_PATH + 'export/'
    SOUND_BUFFER_SIZE = 100
    SAMPLE_CATCH_TIME = 10  # s
    SAMPLE_CATCH_PRECISION = 0.1  # s
    SPEAK_LEVEL_GAP = 10  # dbu
    CACHE_INFO = 3 * 8

    _shuffle3d_channels = np.vectorize(lambda x: x.T.reshape(np.prod(x.shape)),
                                       signature='(m,n)->(c)')
    DATABASE_COLS = ['size', 'noise', 'frameRate', 'o', 'sampleFormat', 'nchannels', 'duration', 'nperseg']
    BUFFER_TYPE = '.bin'

    def __init__(self,
                 std_input_dev_id: int = None,
                 std_output_dev_id: int = None,
                 frame_rate: int = 48000,
                 nchannels: int = 2,
                 data_format: SampleFormat = SampleFormat.formatInt16,  # 16bit mode
                 mono_mode: bool = True,
                 ui_mode: bool = True,
                 nperseg: int = 1000,
                 noverlap: int = None,
                 window: object = 'hann',
                 NOLA_check: bool = True,
                 IO_mode: str = "stdInput2stdOutput",
                 input_dev_callback: callable = None,
                 output_dev_callback: callable = None,
                 master_mix_callback: callable = None):
        '''
        __init__(self,
                 std_input_dev_id: int = None,
                 std_output_dev_id: int = None,
                 frame_rate: int = 48000,
                 nchannels: int = 2,
                 data_format: SampleFormat = 16bit mode
                 mono_mode: bool = True,
                 ui_mode: bool = True,
                 nperseg: int = 256,
                 noverlap: int = None,
                 window: object = 'hann',
                 NOLA_check: bool = True,
                 IO_mode: str = "stdInput2stdOutput",
                 input_dev_callback: callable = None,
                 output_dev_callback: callable = None,
                 master_mix_callback: callable = None)

            Creating a Process object.

            Parameters
            ----------
            std_input_dev_id: int, optional
                os standard input device id. If not given, then the input device id will
                be selected automatically(ui_mode=False) or manually by the user(ui_mode=True)

            std_output_dev_id: int, optional
                os standard output device id. If not given, then the output device id will
                be selected automatically(ui_mode=False) or manually by the user(ui_mode=True)

            frame_rate:  int, optional
                Input channel sample rate. if std_input_dev_id selected as None then the value of
                frame_rate does not matter.

            nchannels: int, optional
                Number of input channels(inner channels)if std_input_dev_id selected as None then the value of
                nchannels does not matter.

            data_format: SampleFormat
                Specify the audio bit depths. Supported data format(from sudio):
                formatFloat32, formatInt32, formatInt24, formatInt16(default),
                formatInt8, formatUInt8

            mono_mode: bool, optional
                If True, then all of the input channels will be mixed to one channel.

            ui_mode: bool, optional
                If Enabled then user interface mode will be activated.

            nperseg: int,optional
                number of samples per each data frame(window)(in one channel)

            noverlap: int, default None
                When the length of a data set to be transformed is larger than necessary to
                provide the desired frequency resolution, a common practice is to subdivide
                it into smaller sets and window them individually.
                To mitigate the "loss" at the edges of the window, the individual sets may overlap in time.
                The noverlap defines the number of overlap between defined windows. If not given then it's value
                will be selected automatically.

            window:  string, float, tuple, ndarray, optional
                The type of window to create(string) or pre designed window in numpy array format.
                See below for more details.
                Default ("hann") Hanning function.
                None for disable windowing

            NOLA_check: bool, optional
                Check whether the Nonzero Overlap Add (NOLA) constraint is met.(If true)

            IO_mode: str, optional
                Input/Output processing mode can be:
                "stdInput2stdOutput":(default)
                    default mode; standard input stream (system audio or any
                     other windows defined streams) to standard defined output stream (system speakers)

                "usrInput2usrOutput":
                    user defined input stream (callback function defined with input_dev_callback)
                    to  user defined output stream (callback function defined with output_dev_callback)

                "usrInput2stdOutput":
                    user defined input stream (callback function defined with input_dev_callback)
                    to  user defined output stream (callback function defined with output_dev_callback)

                "stdInput2usrOutput":
                    standard input stream (system audio or any other windows defined streams)
                    to  user defined output stream (callback function defined with output_dev_callback)

            input_dev_callback: callable;
                callback function; default None;

                :inputs:    frame_count, time_info, status
                :outputs:   Numpy N-channel data frame in  (Number of channels, Number of data per segment)
                            dimensions.
                :note: In 1 channel mode data frame have  (Number of data per segment, ) shape

                can be used to define user input callback
                this function used in "usrInput2usrOutput" and "usrInput2usrOutput" IO modes
                and called by core every 1/frame_rate second and must returns frame data
                in (Number of channels, Number of data per segment) dimensions numpy array.
                each data must be in special format that defined in data_format section.

            output_dev_callback: callable;
                callback function; default None;

                :inputs:    Numpy N-channel data frame in  (Number of channels, Number of data per segment)
                            dimensions.
                :outputs:
                :note: In 1 channel mode data frame have  (Number of data per segment, ) shape

                Can be used to define user input callback.
                This function used in "usrInput2usrOutput" and "usrInput2usrOutput" IO modes
                and called by core every 1/frame_rate second and must returns frame data
                in (Number of channels, Number of data per segment) dimensions numpy array.
                each data must be in special format that defined in data_format section..

            master_mix_callback: callable, optional
                This callback is used before the main-processing stage in the master
                for controlling and mixing all slave channels to a ndarray of shape
                (master.nchannels, 2, master.nperseg).

                If this parameter is not defined, then the number of audio channels
                in all slaves must be the same as the master.


            Note
            -----
            If the window requires no parameters, then `window` can be a string.

            If the window requires parameters, then `window` must be a tuple
            with the first argument the string name of the window, and the next
            arguments the needed parameters.

            If `window` is a floating point number, it is interpreted as the beta
            parameter of the `~scipy.signal.windows.kaiser` window.

            Each of the window types listed above is also the name of
            a function that can be called directly to create a window of that type.

            window types:
                boxcar, triang , blackman, hamming, hann(default), bartlett, flattop, parzen, bohman, blackmanharris
                nuttall, barthann, cosine, exponential, tukey, taylor, kaiser (needs beta),
                gaussian (needs standard deviation), general_cosine (needs weighting coefficients),
                general_gaussian (needs power, width), general_hamming (needs window coefficient),
                dpss (needs normalized half-bandwidth), chebwin (needs attenuation)

            Note
            -----
            In order to enable inversion of an STFT via the inverse STFT in
            `istft`, the signal windowing must obey the constraint of "nonzero
            overlap add" (NOLA):

            .. math:: \sum_{t}w^{2}[n-tH] \ne 0

            for all :math:`n`, where :math:`w` is the window function, :math:`t` is the
            frame index, and :math:`H` is the hop size (:math:`H` = `nperseg` -
            `noverlap`).

            This ensures that the normalization factors in the denominator of the
            overlap-add inversion equation are not zero. Only very pathological windows
            will fail the NOLA constraint.

        Note
        -----
        Master/Slave mode details:
            Every Process object can operate in master or slave modes.
            Each input must be preprocessed before mixing different
            audio inputs, so the slave object is designed so that the
            post-processing stage is disabled and can be joined to master simply
            by the 'join' method.

        '''
        # _______________________________________________________First Initialization
        if ui_mode:
            print('Initialization...')

        try:
            os.mkdir(Master.DATA_PATH)
        except FileExistsError:
            pass
        try:
            os.mkdir(Master.SAVE_PATH)
        except:
            pass

        self._stream_type = StreamMode.optimized
        data_format = data_format.value
        self.io_mode = IO_mode
        self.input_dev_callback = input_dev_callback
        self.output_dev_callback = output_dev_callback

        self._exstream_mode = threading.Event()
        self._master_mute_mode = threading.Event()
        self._stream_loop_mode = False
        self._stream_on_stop = False
        self._stream_data_pointer = 0
        self._stream_data_size = None
        self._stream_file = None

        warnings.filterwarnings("ignore")
        # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        # np.warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._output_device_index = None
        p = self._paudio = Audio()  # Create an interface to PortAudio
        if std_input_dev_id is None:
            if ui_mode:
                dev = []
                for i in range(p.get_device_count()):
                    tmp = p.get_device_info_by_index(i)
                    if tmp['maxInputChannels'] > 0:
                        dev.append(tmp)
                assert len(dev) > 0
                print('\nplease choose input device from index :\n')
                default_dev = p.get_default_input_device_info()['index']
                for idx, j in enumerate(dev):
                    msg = f'Index {idx}: ' \
                          f' Name: {j["name"]},' \
                          f' Input Channels:{j["maxInputChannels"]},' \
                          f' Sample Rate:{j["defaultSampleRate"]},' \
                          f' Host Api:{j["hostApi"]}'
                    if default_dev == idx:
                        msg += '  <<default>>'
                    print(msg)

                while 1:
                    try:
                        dev_id = int(input('index for input dev: '))
                        break
                    except:
                        print('please enter valid index!')

                self._frame_rate = [int(dev[dev_id]['defaultSampleRate'])]
                frame_rate = self._frame_rate[0]
                self.nchannels = dev[dev_id]['maxInputChannels']
                self._input_dev_id = dev_id

            else:
                dev = p.get_default_input_device_info()
                self._frame_rate = [int(dev['defaultSampleRate'])]
                frame_rate = self._frame_rate[0]
                self.nchannels = dev['maxInputChannels']
                self._input_dev_id = dev['index']

        else:
            self._frame_rate = [frame_rate]
            self.nchannels = nchannels
            self._input_dev_id = std_input_dev_id

        self._sampwidth = Audio.get_sample_size(data_format)
        self._sample_format = data_format
        if self.io_mode.endswith("stdOutput"):
            if std_output_dev_id is None and ui_mode:
                dev = []

                for i in range(p.get_device_count()):
                    tmp = p.get_device_info_by_index(i)
                    if tmp['maxOutputChannels'] > 0:
                        dev.append(tmp)
                assert len(dev) > 0
                print('\nplease choose output device from index :\n')
                default_dev = p.get_default_output_device_info()['index']
                for idx, j in enumerate(dev):
                    msg = f'Index {idx}:' \
                          f'  Name: {j["name"]},' \
                          f' Channels:{j["maxOutputChannels"]},' \
                          f' Sample Rate:{j["defaultSampleRate"]},' \
                          f' Host Api:{j["hostApi"]}'
                    if default_dev == idx:
                        msg += '  <<default>>'
                    print(msg)

                while 1:
                    try:
                        self._output_device_index = int(input('index for output dev: '))
                        break
                    except:
                        print('please enter valid index!')

            else:
                self._output_device_index = std_output_dev_id

        if self.nchannels == 1:
            mono_mode = True

        self.branch_pipe_database_min_key = 0
        self._ui_mode = ui_mode
        self._window_type = window

        self._sample_rate = frame_rate
        self._nperseg = nperseg
        # check _win arguments
        if noverlap is None:
            noverlap = nperseg // 2
        if type(self._window_type) is str or \
                type(self._window_type) is tuple or \
                type(self._window_type) == float:
            window = scisig.get_window(window, nperseg)

        if self._window_type:
            assert len(window) == nperseg, 'control size of window'
            if NOLA_check:
                assert scisig.check_NOLA(window, nperseg, noverlap)
        elif self._window_type is None:
            window = None
            self._window_type = None

        # _______________________________________________________UI mode std_input_dev_id calculator
        record_period = nperseg / frame_rate
        self._record_period = record_period
        self._mono_mode = mono_mode
        # ______________________________________________________________Vectorized dbu calculator
        self._vdbu = np.vectorize(self._rdbu, signature='(m)->()')

        # ______________________________________________________________channels type and number processing
        if self.nchannels == 1:
            Master.get_channels = lambda x: x

        elif mono_mode:
            Master.get_channels = lambda x: np.mean(x.reshape((self._data_chunk, self.nchannels)),
                                                    axis=1)
        else:
            Master.get_channels = lambda x: np.append(*[[x[i::self.nchannels]] for i in range(self.nchannels)],
                                                      axis=0)

        if mono_mode:
            self._process_nchannel = 1
        else:
            self._process_nchannel = self.nchannels

        # ______________________________________________________________sample rate and pre filter processing
        self.prim_filter_cutoff = int(8e3)  # Hz
        self._frame_rate.append(self.prim_filter_cutoff * 2)
        downsample = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
        self._frame_rate[1] = int(self._frame_rate[0] / downsample)  # human voice high frequency
        self._filters = [scisig.firwin(10, self.prim_filter_cutoff, fs=self._frame_rate[0])]
        self._data_chunk = nperseg
        self.frame_rate = frame_rate
        # data type of buffer                       0
        # used in callback method(down sampling)    1
        # used in _win_nd                           2
        # rms constant value                        3
        #                                           4

        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sampwidth)]  # 1 2 4 8
        # f'<{self.data_chunk * self.nchannels}{sampwidth}'

        self._constants = ['<i{}'.format(self._sampwidth),
                           downsample,
                           range(self.nchannels)[1:],
                           ((2 ** (self._sampwidth * 8 - 1) - 1) / 1.225),
                           range(self.nchannels)]

        if data_format == SampleFormat.formatFloat32.value:
            self._constants[0] = '<f{}'.format(self._sampwidth)

        self._threads = []
        # _______________________________________________________user _database define
        try:
            self._database = pd.read_pickle(Master.USER_PATH, compression='xz')
        except:
            self._database = pd.DataFrame(columns=Master.DATABASE_COLS)

        self._local_database = pd.DataFrame(columns=Master.DATABASE_COLS)

        # _______________________________________________________pipeline _database and other definitions section

        try:
            self._semaphore = [threading.Semaphore()]

            dataq = queue.Queue(maxsize=Master.SOUND_BUFFER_SIZE)
            self._recordq = [threading.Event(), queue.Queue(maxsize=Master.SOUND_BUFFER_SIZE)]
            self._queue = queue.Queue()
            self._streamq = [threading.Event(), queue.Queue(maxsize=Master.SOUND_BUFFER_SIZE)]

            self._echo = threading.Event()
            self._primary_filter = threading.Event()
            # self._primary_filter.set()
            self.main_pipe_database = []
            self.branch_pipe_database = []
            # _______________________________________threading management
            self._functions = []
            # self._functions.append(self._process)
            self._functions.append(self.__post_process)

            for i in range(len(self._functions)):
                self._threads.append(threading.Thread(target=self._run, daemon=True, args=(len(self._threads),)))
                self._queue.put(i)

            # ____________________________________________________windowing object define and initialization
            # post, pre
            self._nhop = nperseg - noverlap
            self._noverlap = noverlap
            self._window = window
            # self._win_frame_rate = [self.main_fs, (self.primary_fc * 2)]
            # self._win_downsample_coef = int(np.round(self._win_frame_rate[0] / self._win_frame_rate[1]))
            # self._win_filter = scisig.firwin(30, self.primary_fc, fs=self._win_frame_rate[0])
            # self._win_window = get_window(self._window_type, self._nperseg)
            # self._win_nchannels_range = range(self.nchannels)

            if mono_mode:
                self._iwin_buffer = [np.zeros(nperseg)]
                self._win_buffer = [np.zeros(nperseg), np.zeros(nperseg)]
                self._win = self._win_mono
                self._iwin = self._iwin_mono
                self._upfirdn = lambda h, x, const: scisig.upfirdn(h, x, down=const)
            else:
                self._iwin_buffer = [np.zeros(nperseg) for i in range(self.nchannels)]
                self._win_buffer = [[np.zeros(nperseg), np.zeros(nperseg)] for i in range(self.nchannels)]

                # 2d mode
                self._upfirdn = np.vectorize(lambda h, x, const: scisig.upfirdn(h, x, down=const),
                                             signature='(n),(m),()->(c)')
                self._iwin = self._iwin_nd
                self._win = self._win_nd

            self._normal_stream = Master._Stream(self, dataq, process_type='queue')
            self._main_stream = Master._Stream(self, dataq, process_type='queue')
            # self.data_tst_buffer = [i for i in range(5)]
            # self.iwin_tst_buffer = [i for i in range(5)]
            self._refresh_ev = threading.Event()
            if self._ui_mode:
                print('Initialization completed!')

            self._master_mix_callback = master_mix_callback

            # master/ slave selection
            # _master_sync in slave mode must be deactivated
            # The _master_sync just used in the activated multi_stream pipelines
            self._master_sync = threading.Barrier(self._process_nchannel)

            self.slave_database = []
            # The _slave_sync just used in the activated slave's main pipelines
            self._slave_sync = None

        except:
            print('Initialization error!')
            raise

    def start(self):
        '''
        start audio streaming process
        :return: self object
        '''
        # stdInput2stdOutput mode
        stream_callback = self._stream_callback
        # user input mode
        if self.io_mode.startswith("usr"):
            stream_callback = self._ustream_callback

        try:
            self._pystream = self._paudio.open(format=self._sample_format,
                                               channels=self.nchannels,
                                               rate=self._frame_rate[0],
                                               frames_per_buffer=self._data_chunk,
                                               input_device_index=self._input_dev_id,
                                               input=True,
                                               stream_callback=stream_callback)
            for i in self._threads:
                i.start()

            # The server blocks all active threads
            # del self._threads[:]
            # self._queue.join()

        except:
            print('failed to start!')
            raise
        return self

    def _run(self, th_id):
        # self.a1=0
        self._functions[self._queue.get()](th_id)
        # Indicate that a formerly enqueued task is complete.
        # Used by queue consumer threads. For each get() used to fetch a task,
        # a subsequent call to task_done() tells the queue that the processing on the task is complete.

        # If a join() is currently blocking,
        # it will resume when all items have been processed
        # (meaning that a task_done() call was received for every item that had been put() into the queue).
        self._queue.task_done()


    def _exstream(self):
        while self._exstream_mode.is_set():
            in_data: np.ndarray = np.frombuffer(
                self._stream_file.read(self._stream_data_size),
                self._constants[0]).astype('f')

            try:
                in_data: np.ndarray = Master.get_channels(in_data)
            except ValueError:
                return

            if not in_data.shape[1] == self._data_chunk:
                if self._stream_loop_mode:
                    self._stream_file.seek(self._stream_data_pointer, 0)
                else:
                    # print('ok')
                    self._exstream_mode.clear()
                    self._stream_file.seek(self._stream_data_pointer, 0)
                    self._main_stream.clear()

                    self._stream_file = None
                    self._stream_loop_mode = None
                    self._stream_data_pointer = None
                    self._stream_data_size = None

                if self._stream_on_stop:
                    self._stream_on_stop()
                    self._stream_on_stop = None
                return
            self._main_stream.acquire()
            self._main_stream.put(in_data)  # .astype(self._constants[0])
            self._main_stream.release()
            if self._stream_type == StreamMode.normal:
                break

    # stdInput2stdOutput
    def _stream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        if self._exstream_mode.is_set():
            self._exstream()
            return None, 0

        elif self._master_mute_mode.is_set():
            if self._process_nchannel < 2:
                in_data = np.zeros((self._nperseg), 'f')
            else:
                in_data = np.zeros((self._process_nchannel, self._nperseg), 'f')
        else:
            in_data = np.frombuffer(in_data, self._constants[0]).astype('f')
            try:
                in_data = Master.get_channels(in_data)
            except ValueError:
                return None, 0

        self._main_stream.acquire()
        self._main_stream.put(in_data)  # .astype(self._constants[0])
        self._main_stream.release()

        return None, 0

    # user Input mode
    def _ustream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        if self._exstream_mode.is_set():
            if not self._exstream():
                return None, 0

        elif self._master_mute_mode.is_set():
            if self._process_nchannel < 2:
                in_data = np.zeros((self._nperseg), 'f')
            else:
                in_data = np.zeros((self._process_nchannel, self._nperseg), 'f')
        else:
            in_data = self.input_dev_callback(frame_count, time_info, status).astype('f')

        # in_data = np.fromstring(in_data, self._constants[0])
        self._main_stream.acquire()
        self._main_stream.put(in_data)  # .astype(self._constants[0])
        self._main_stream.release()

        # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
        # self.a1 = time.perf_counter()
        return None, 0

    def __post_process(self, thread_id):
        stream_out = self._paudio.open(
            format=self._sample_format,
            channels=self._process_nchannel,
            output_device_index=self._output_device_index,
            rate=self._frame_rate[0], input=False, output=True)
        stream_out.start_stream()
        # flag = self._primary_filter.is_set()
        # framerate = self._frame_rate[1]
        user_mode = "usr" in self.io_mode[4:]
        rec_ev, rec_queue = self._recordq
        while 1:

            # check primary filter state
            data = self._main_stream.get()
            # Tools.push(self.iwin_tst_buffer, data)

            if rec_ev.isSet():
                rec_queue.put_nowait(data)

            if not self._mono_mode:
                data = Master.shuffle2d_channels(data)
            # data = data.T.reshape(np.prod(data.shape))
            # print(data.shape)
            if self._echo.isSet():
                stream_out.write(data.tobytes())

            if user_mode:
                self.output_dev_callback(data)

            if self._refresh_ev.is_set():
                flag = self._primary_filter.is_set()
                self._refresh_ev.clear()
                # close current stream
                stream_out.close()
                # create new stream
                if flag:
                    rate = self._frame_rate[1]
                else:
                    rate = self._frame_rate[0]

                # print(rate)
                stream_out = self._paudio.open(
                    format=self._sample_format,
                    channels=self._process_nchannel,
                    rate=rate, input=False, output=True)
                stream_out.start_stream()
                self._main_stream.clear()

    def _recorder(self, enable_compressor=True,
                  noise_sampling_duration=10,
                  record_duration=10,
                  enable_ui=True,
                  play_recorded=True,
                  catching_precision=0.1,
                  echo_mode=False,
                  rec_start_callback=None):
        # clear data queue
        self._main_stream.clear()
        rec_ev, rec_queue = self._recordq
        progress = None
        if enable_ui:
            progress = tqdm.tqdm(range(100), f"Processing.. ")
            # Waiting for Process processing to be disabled
            progress.update(5)
        if enable_compressor:
            duration = int(np.round(noise_sampling_duration / self._record_period)) - 1
            if enable_ui:
                # start _main _process
                # record sample for  SAMPLE_CATCH_TIME duration
                step = 40 / duration
                progress.set_description('Processing environment')
        elif enable_ui:
            step = 40

        echo = self._echo.isSet()
        if enable_compressor:
            if echo and not echo_mode:
                self.echo(enable=False)
            rec_ev.set()
            sample = [rec_queue.get()]
            for i in range(duration):
                sample = np.vstack((sample, [rec_queue.get()]))
                if enable_ui: progress.update(step)
            rec_ev.clear()

            if echo and not echo_mode:
                self.echo()
            # if not self._mono_mode:
            #     sample = self.shuffle3d_channels(sample)
            vdbu = self._vdbu(sample)
            vdbu = vdbu[np.abs(vdbu) < np.inf]
            noise = np.mean(vdbu)
            # print(self._vdbu(sample))
            if enable_ui:
                self._safe_print(
                    f'\n^^^^^^^^^^Ready to record for  {record_duration} seconds^^^^^^^^^^^^^^')
                if not self._safe_input(
                        f'Are you ready? enter "yes"\nNoise Level: {noise} dbu').strip().lower() == 'yes':
                    # record sample for  SAMPLE_CATCH_TIME duration
                    # free input_data_stream before _process
                    return None

        while 1:
            if enable_ui:
                # reset progress
                progress.n = 50
                progress.last_print_n = 50
                progress.start_t = time.time()
                progress.last_print_t = time.time()
                progress.update()

            self._main_stream.clear()

            duration = int(np.round(record_duration / self._record_period)) - 1

            if enable_ui:
                step = 30 / duration
                progress.set_description('Recording the conversation..')

            if echo and not echo_mode:
                self.echo(enable=False)
            if callable(rec_start_callback):
                if enable_compressor:
                    tmp = rec_start_callback(self, {'noise': noise})
                else:
                    tmp = rec_start_callback(self)

                if not tmp:
                    return None
            rec_ev.set()
            sample = [rec_queue.get()]
            for i in range(duration):
                sample = np.vstack((sample, [rec_queue.get()]))
                if enable_ui:  progress.update(step)
            rec_ev.clear()

            # sample preprocessing
            if enable_compressor:
                sample = self._compressor(sample,
                                          progress,
                                          noise_base=noise + Master.SPEAK_LEVEL_GAP,
                                          final_period=catching_precision,
                                          base_period=self._record_period)

            if enable_ui:
                progress.update(10)
                # Waiting for Process processing to be enabled
                progress.close()

            if not self._mono_mode:
                sample = self.shuffle3d_channels(sample)
            sample = sample.astype(self._constants[0]).tobytes()
            if play_recorded:
                play(sample,
                     sample_format=self._sample_format,
                     nchannels=self._process_nchannel,
                     framerate=self.frame_rate)

            if echo and not echo_mode:
                self.echo()

            if enable_ui:
                if self._safe_input('\nTry again? "yes"').strip().lower() == 'yes':
                    continue
                self._safe_print('Recorded successfully!')
            ch = self._process_nchannel

            retval = {'size':len(sample),
                       'noise': None,
                      'frameRate': self.frame_rate,
                      'o': sample,
                      'nchannels': ch,
                      'sampleFormat': self._sample_format,
                      'nperseg': self._nperseg,
                      'duration': record_duration,
                      }

            if enable_compressor:
                retval['noise'] = noise
            return retval

    # sample must be a 2d array
    # final_period just worked in mono mode
    def _compressor(self, sample, progress, final_period=0.1, base_period=0.3, noise_base=-1e3):

        assert sample.shape[0] * base_period > 3
        if progress:
            progress.set_description('Processing..')
        # Break into different sections based on period time
        tmp = sample.shape[-1] * sample.shape[0]
        cols = Tools.near_divisor(tmp, final_period * sample.shape[-1] / base_period)
        if progress:
            progress.update(5)
        # Reshaping of arrays based on time period
        if self._mono_mode and final_period < base_period:
            sample = sample.reshape((int(tmp / cols), cols))
        if progress:
            progress.update(5)
        # time filter
        dbu = self._vdbu(sample)
        # tmp = sample.shape
        # if self._mono_mode or final_period < base_period or self.nchannels < 2:
        #     sample = sample[dbu > noise_base]
        # else:
        #     sample = (sample.reshape(np.prod(tmp[:2]), tmp[2])[dbu > noise_base])
        #     if sample.shape[0] % 2:
        #         sample = sample[:-1 * (sample.shape[0] % 2)]
        #     sample = sample.reshape(int(np.prod(sample.shape) / np.prod(tmp[1:])), *tmp[1:])
        if self._mono_mode or final_period < base_period:
            dbu = dbu[np.abs(dbu) < np.inf]
            sample = sample[dbu > noise_base]
        else:
            # dbu = dbu[np.absolute(dbu, axis=1) < np.inf]
            sample = sample[np.max(dbu, axis=1) > noise_base]

        # shape = sample.shape
        # sample = sample.reshape(shape[0] * shape[1])
        # LPF filter
        if progress:
            progress.update(5)
        # print(sample, sample.shape)
        return sample

    def _safe_input(self, *args, **kwargs):
        self._semaphore[0].acquire()
        res = input(*args, **kwargs)
        self._semaphore[0].release()
        return res

    def _safe_print(self, *args, **kwargs):
        self._semaphore[0].acquire(timeout=1)
        print(*args, **kwargs)
        self._semaphore[0].release()

    # RMS to dbu
    # signal maps to standard +4 dbu
    def _rdbu(self, arr):
        return Tools.dbu(np.max(arr) / self._constants[3])


    def add_file(self, filename: str, sample_format: SampleFormat = SampleFormat.formatUnknown,
                nchannels: int = None, sample_rate: int = None ,safe_load=True):
        '''
        Add an audio file to the local database. None value for the parameters means
        that they are automatically selected based on the properties of the audio file.

         Note:
          supported files format: WAV, FLAC, VORBIS, MP3

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        :param filename: path/name of the audio file
        :param sample_format: sample format
        :param nchannels: number of audio channels
        :param sample_rate: sample rate
        :param safe_load: load an audio file and modify it according to the 'Master' attributes.
        (sample rate, sample format, number of channels, etc).
        :return: Wrap object

        '''
        info = get_file_info(filename)
        if safe_load:
            sample_format = SampleMapValue[self._sample_format]
        elif sample_format is SampleFormat.formatUnknown:
            if info.sample_format is LibSampleFormat.UNKNOWN:
                sample_format = SampleMap[self._sample_format]
            else:
                sample_format = info.sample_format
        else:
            sample_format = SampleMap[sample_format]

        if nchannels is None:
            nchannels = info.nchannels
        if safe_load:
          sample_rate = self._sample_rate
        elif sample_rate is None:
            sample_rate = info.sample_rate


        record ={
            'size': None,
            'noise': None,
            'frameRate': sample_rate,
            'o': None,
            'sampleFormat': ISampleMap[sample_format].value,
            'nchannels': nchannels,
            'duration': info.duration,
            'nperseg': self._nperseg,
        }
        p0 = max(filename.rfind('\\'), filename.rfind('/'))
        p1 = filename.rfind('.')
        if p0 < 0:
            p0 = 0
        if p1 <= 0:
            p1 = None
        name = filename[p0: p1]
        if name in self._local_database.index or \
                name in self._database.index:
            raise KeyError('Record with the name of {}'
                           ' already registered in local or external database'.format(name))



        # print(list(record.keys()), list(self._local_database.columns))
        assert list(record.keys()) == list(self._local_database.columns)

        if safe_load and not self._mono_mode and record['nchannels'] > self.nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=name,
                                                                              ch0=record['nchannels'],
                                                                              ch1=self.nchannels))
        path = Master.DATA_PATH + name + Master.BUFFER_TYPE
        cache = self._cache()
        try:
            if path in cache:

                f = record['o']=  open(path, 'rb+')
                record['size'] = os.path.getsize(path)
                cache_info = f.read(Master.CACHE_INFO)

                cframe_rate, csample_format, cnchannels = np.frombuffer(cache_info, dtype='u8').tolist()
                csample_format = csample_format if csample_format else None

                if (cnchannels == self._process_nchannel and
                    csample_format == self._sample_format and
                    cframe_rate == self.frame_rate):
                    #print_en
                    # print('path in cache not load add file', cframe_rate, cnchannels, csample_format)
                    record['frameRate'] = cframe_rate
                    record['nchannels'] = cnchannels
                    record['sampleFormat'] = csample_format

                elif safe_load:
                    # print_en
                    # print('path in cache safe load add file')
                    record['o'] = f.read()
                    record = self._sync_record(record)
                    f.seek(0, 0)
                    # f = open(path, 'wb+')
                    f.truncate()
                    fsize = f.write(np.array([record['frameRate'],
                                      record['sampleFormat'] if record['sampleFormat'] else 0,
                                     record['nchannels']],
                                     dtype='u8').tobytes())
                    fsize += f.write(record['o'])
                    record['o'] = f
                    record['size'] = fsize
                else:
                    raise Error
            else:
                raise Error
        except Error:
            # print_en
            # print('new decode add file')
            record['o'] = data = decode_file(filename, sample_format, nchannels, sample_rate, DitherMode.NONE)
            if safe_load:
                # print(record['frameRate'])
                record = self._sync_record(record)
                # print(record['frameRate'])
            f = open(path, 'wb+')
            f.truncate()
            fsize = f.write(np.array([record['frameRate'],
                              record['sampleFormat'] if record['sampleFormat'] else 0,
                              record['nchannels']],
                              dtype='u8').tobytes())
            fsize += f.write(record['o'])
            record['o'] = f
            record['size'] = fsize

        f.flush()
        f.seek(Master.CACHE_INFO, 0)

        record['duration'] = record['size'] / (record['frameRate'] *
                                               record['nchannels'] *
                                               Audio.get_sample_size(record['sampleFormat']))
        self._local_database.loc[name] = record
        gc.collect()
        return self.load(name)

    def add(self, record, safe_load=True):
        '''
        Add new record to the local database.

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        :param record: can be an wrapped record, pandas series or an audio file in
                        mp3, WAV, FLAC or VORBIS format.
        :param safe_load: load an audio file and modify it according to the 'Master' attributes.
        :return: Wrapped object

        \nnote:
         The name of the new record can be changed with the following tuple that is passed to the function:
         (record object, new name of )


        '''
        name_type = type(record)
        if name_type is list or name_type is tuple:
            assert len(record) < 3

            rec_name = None
            if len(record) == 2:
                if type(rec_name) is str:
                    rec_name = record[1]

            if type(record[0]) is Wrap:
                record = record[0].get_data()

            elif type(record[0]) is pd.Series:
                pass
            else:
                raise TypeError

            if rec_name and type(rec_name) is str:
                record.name = rec_name

            elif rec_name is None:
                record.name = Tools.time_name()

            else:
                raise ValueError('second item in the list is the name of the new record')
            # print(record.name)
            return self.add(record, safe_load=safe_load)

        elif name_type is Wrap:
            series = record.get_data()
            return self.add(series, safe_load=safe_load)

        elif name_type is pd.Series:
            name = record.name
            assert list(record.keys()) == list(self._local_database.columns)

            if safe_load and not self._mono_mode and record['nchannels'] > self.nchannels:
                raise ImportError('number of channel for the {name}({ch0})'
                                  ' is not same as object channels({ch1})'.format(name=name,
                                                                                  ch0=record['nchannels'],
                                                                                         ch1=self.nchannels))
            data = record['o']
            if type(data) is not BufferedRandom:
                if safe_load:
                    record = self._sync_record(record)
                f = open(Master.DATA_PATH + name + Master.BUFFER_TYPE, 'wb+')
                f.truncate()

                newsize = f.write(np.array([record['frameRate'],
                                  record['sampleFormat'] if record['sampleFormat'] else 0,
                                 record['nchannels']],
                                 dtype='u8').tobytes())
                newsize += f.write(record['o'])

                f.flush()
                f.seek(Master.CACHE_INFO, 0)
                record['o'] = f
                record['size'] = newsize

            elif (not (Master.DATA_PATH + record.name + Master.BUFFER_TYPE) ==  data.name) or \
                    record.name in self._local_database.index or record.name in self._database.index:
                # new sliced Wrap data

                if  record.name in self._local_database.index or record.name in self._database.index:
                    record.name = Tools.time_name()
                prefile = record['o']
                prepos = prefile.tell(), 0
                newfile = record['o'] = open(Master.DATA_PATH + record.name + Master.BUFFER_TYPE, 'wb+')
                newsize = newfile.write(np.array([record['frameRate'],
                                       record['sampleFormat'] if record['sampleFormat'] else 0,
                                       record['nchannels']],
                                       dtype='u8').tobytes())
                newsize += newfile.write(prefile.read())

                prefile.seek(*prepos)
                newfile.seek(Master.CACHE_INFO, 0)
                newfile.flush()
                record['o'] = newfile
                record['size'] = newsize

            # if record.name  in self._local_database.index or \
            #     record.name in self._database.index:
            #     raise KeyError('Record with the name of {}'
            #                    ' already registered in local or external database'.format(name))
            self._local_database.loc[record.name] = record
            return self.wrap(record)

        elif name_type is str:
           return self.add_file(record, safe_load=safe_load)

        else:
            raise TypeError('The record must be an audio file, data frame or a Wrap object')

        gc.collect()

    def recorder(self,
                 name: str=None,
                 enable_compressor: bool=False,
                 noise_sampling_duration: Union[int, float]=1,
                 record_duration: Union[int, float]=10,
                 enable_ui: bool=True,
                 play_recorded: bool=True,
                 catching_precision: float=0.1,
                 echo_mode: bool=False,
                 rec_start_callback: callable=None):
        """
        record from main stream for a while.

        :param name: name of new record (optional)
        :param enable_compressor:   enable to compress the recorded data.(optional)
                                    The compressor deletes part of the signal that has
                                     lower energy than sampled noise.
        :param noise_sampling_duration: optional; noise sampling duration used in compressor.
        :param record_duration: determines the time of recording process
        :param enable_ui: user inteface mode
        :param play_recorded: Determines whether the recorded file is played after the recording process.
        :param catching_precision: Signal compression accuracy,
                                    more accuracy can be achieved with a smaller number
        :param echo_mode: It can disable the mainstream's echo mode, when the recorder is online.
        :param rec_start_callback: called after noise sampling.
        :return: wrapped object of record
        """
        gc.collect()
        if type(name) is str and name in self._local_database.index:
            raise KeyError('The entered name is already registered in the database.')
        if not name:
            name = Tools.time_name()

        data = self._recorder(enable_compressor=enable_compressor,
                              noise_sampling_duration=noise_sampling_duration,
                              record_duration=record_duration,
                              enable_ui=enable_ui,
                              play_recorded=play_recorded,
                              catching_precision=catching_precision,
                              echo_mode=echo_mode,
                              rec_start_callback=rec_start_callback)
        file = open(Master.DATA_PATH + name + Master.BUFFER_TYPE, 'wb+')
        file.truncate()
        file.write(np.array([data['frameRate'],
                          data['sampleFormat'] if data['sampleFormat'] else 0,
                          data['nchannels']],
                          dtype='u8').tobytes())

        file.write(data['o'])
        file.flush()
        file.seek(Master.CACHE_INFO, 0)
        data['o'] = file
        data['size'] += Master.CACHE_INFO
        self._local_database.loc[name] = data
        return self.wrap(self._local_database.loc[name].copy())

    def load_all(self, safe_load=True):
        '''
        load all of the saved records from the external database
                        to the local database.

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        :param safe_load: if safe load is enabled then load function tries to load a record
         in the local database based on the master  settings, like the frame rate and etc.

        :return:None
        '''
        self.load('', load_all=True, safe_load=safe_load)


    def load(self, name: str, load_all: bool=False, safe_load: bool=True,
             series: bool=False) -> Union[Wrap, pd.Series]:
        '''
        This method used to load a predefined recoed from the staable/external database to
         the local database. Trying to load a record that was previously loaded, outputs a wrapped version
         of the named record.

        :param name: predefined record name
        :param load_all: load all of the saved records from the external database
                        to the local database.
        :param safe_load: if safe load is enabled then load function tries to load a record
         in the local database based on the master  settings, like the frame rate and etc.
        :param series: if enabled then Trying to load a record that was previously loaded, outputs data series
        of the named record.
        :return: (optional) Wrapped object, pd.Series
        '''
        if load_all:
            for i in self._database.index:
                self.load(i, safe_load=safe_load)

            return
        # elif type(name) is pd.Series:
        #     ret
        else:
            if name in self._local_database.index:
                rec = self._local_database.loc[name].copy()
                if series:
                    return rec
                return self.wrap(rec)

            elif name in self._database.index:
                rec = self._database.loc[name].copy()
                file = open(rec['o'], 'rb+')
                file_size = os.path.getsize(rec['o'])
            else:
                raise ValueError('can not found the {name} in the local and the external databases'.format(name=name))

        if safe_load and not self._mono_mode and rec['nchannels'] > self.nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=name,
                                                                              ch0=rec['nchannels'],
                                                                              ch1=self.nchannels))
        if safe_load:
            rec['o'] = file.read()
            rec = self._sync_record(rec)
            rec['o'] = file

        if file_size > rec['size']:
            file.seek(rec['size'], 0)
        else:
            file.seek(Master.CACHE_INFO, 0)

        self._local_database.loc[name] = rec
        if series:
            return rec
        return self.wrap(rec)


    def get_record_info(self, name: str) -> dict:
        '''
        :param name: name of the registered record on the local or external database
        :return: information about saved record ['noiseLevel' 'frameRate'  'sizeInByte' 'duration'
            'nchannels' 'nperseg' 'name'].
        '''
        if name in self._local_database.index:
            rec = self._local_database.loc[name]

        elif name in self._database.index:
            rec = self._database.loc[name]
        else:
            raise ValueError('can not found the {name} in the local and the external databases'.format(name=name))
        return {
            'noiseLevel': str(rec['noise']) + ' <dbm>',
            'frameRate': rec['frameRate'],
            'sizeInByte': rec['size'],
            'duration': rec['duration'],
            'nchannels': rec['nchannels'],
            'nperseg': rec['nperseg'],
            'name': name,
        }

    def get_exrecord_info(self, name: str) -> dict:
        '''
        :param name: name of the registered record on the external database
        :return: information about saved record ['noiseLevel' 'frameRate'  'sizeInByte' 'duration'
            'nchannels' 'nperseg' 'name'].
        '''
        if name in self._database.index:
            rec = self._database.loc[name]
        else:
            raise ValueError('can not found the {name} in the external databases'.format(name=name))
        return {
            'noiseLevel': str(rec['noise']) + ' <dbm>',
            'frameRate': rec['frameRate'],
            'sizeInByte': rec['size'],
            'duration': rec['duration'],
            'nchannels': rec['nchannels'],
            'nperseg': rec['nperseg'],
            'name': name,
        }

    # save from local _database to stable memory
    def _sync_record(self, rec):
        axis = 0
        form = Audio.get_sample_size(rec['sampleFormat'])
        if rec['sampleFormat'] == SampleFormat.formatFloat32.value:
            form = '<f{}'.format(form)
        else:
            form = '<i{}'.format(form)
        data = np.frombuffer(rec['o'], form)
        if rec['nchannels'] == 1:
            if self.nchannels > rec['nchannels']:
                data = np.vstack([data for i in range(self.nchannels)])
                axis = 1
        # elif self.nchannels == 1 or self._mono_mode:
        #     data = np.mean(data.reshape(int(data.shape[-1:][0] / rec['nchannels']),
        #                                 rec['nchannels']),
        #                    axis=1)
        else:
            data = [[data[i::rec['nchannels']]] for i in range(self.nchannels)]
            data = np.append(*data, axis=0)
            axis = 1

        if not self.frame_rate == rec['frameRate']:
            data = scisig.resample(data,
                                   int((self.frame_rate * data.shape[-1:][0]) / rec['frameRate']),
                                   axis=axis)
        if rec['nchannels'] > 1:
            data = Master.shuffle2d_channels(data)

        rec['nchannels'] = self._process_nchannel
        rec['sampleFormat'] = self._sample_format
        rec['frameRate'] = self.frame_rate
        rec['o'] = data.astype(self._constants[0]).tobytes()
        rec['size'] = len(rec['o'])
        rec['duration'] = rec['size'] / (rec['frameRate'] *
                                         rec['nchannels'] *
                                         Audio.get_sample_size(rec['sampleFormat']))

        return rec

    def del_record(self, name: str, deep: bool=False):
        '''
        This function is used to delete a record from the internal database.
        :param name str: the name of preloaded record.
        :param deep bool: deep delete mode is used to remove the record  and its corresponding caches
         from the external database.
        :return: None
        '''
        local = name in self._local_database.index
        extern = name in self._database.index
        ex = ''
        if deep:
            ex = 'and the external '
        assert local or (extern and deep), ValueError('can not found the {name} in the '
                                             'local {ex}databases'.format(name=name, ex=ex))
        if local:
            file = self._local_database.loc[name, 'o']
            file.close()

            if not extern:
                tmp = list(file.name)
                tmp.insert(file.name.find(record.name), 'stream_')
                streamfile_name = ''.join(tmp)
                try:
                    os.remove(streamfile_name)
                except FileNotFoundError:
                    pass

                os.remove(file.name)
            self._local_database.drop(name, inplace=True)

        # print(self._local_database)
        if extern and deep:

            file = self._database.loc[name, 'o']
            tmp = list(file.name)
            tmp.insert(file.name.find(record.name), 'stream_')
            streamfile_name = ''.join(tmp)
            try:
                os.remove(streamfile_name)
            except FileNotFoundError:
                pass

            os.remove(file)
            self._database.drop(name, inplace=True)
            self._database.to_pickle(Master.USER_PATH, compression='xz')
        gc.collect()


    def save_as(self, record: Union[str, pd.Series, Wrap], file_path: str=SAVE_PATH):
        '''
        Convert the record to the wav audio format.
        :param record: record can be the name of registered record, a pandas series or an wrap object
        :param file_path: The name or path of the wav file(auto detection).
        A new name for the record to be converted can be placed at the end of the address.
        :return:None
        '''
        rec_type = type(record)
        if rec_type is str:
            record = self.load(record, series=True)
        elif rec_type is Wrap:
            record = record.get_data()
        elif rec_type is pd.Series:
            pass
        else:
            raise TypeError('please control the type of record')

        p0 = max(file_path.rfind('\\'), file_path.rfind('/'))
        p1 = file_path.rfind('.')
        if p0 < 0:
            p0 = 0

        if p1 <= 0:
            p1 = None

        if (p0 == len(file_path) - 1) or len(file_path) < 2:
            name = record.name
        else:
            # print(p0, p1, file_path[p0+1: p1])
            name = file_path[p0 : p1]

        name += '.wav'
        if p0:
            file_path = file_path[0: p0 + 1] + name
        else:
            file_path = Master.SAVE_PATH + name
        # print(file_path, p0, p1)
        file = record['o']
        file_pos = file.tell()
        data = file.read()
        file.seek(file_pos, 0)

        wav_write_file(file_path, data,
                       record['nchannels'], record['frameRate'],
                       Audio.get_sample_size(record['sampleFormat']))


    def save(self, name: str='None', save_all: bool=False):
        '''
        Save the preloaded record to the external database
        :param name: name of the preloaded record
        :param save_all: if true then it's tries to save all of the preloaded records
        :return:None
        '''
        if save_all:
            data = self._local_database.copy()
            for i in data.index:
                data.loc[i, 'o'] = Master.DATA_PATH + i + Master.BUFFER_TYPE
            self._database = self._database.append(data, verify_integrity=True)
        else:
            if name in self._database.index and not (name in self._local_database.index):
                raise ValueError(f'the same name already registered in the database file')
            try:
                data = self._local_database.loc[name].copy()
                data['o'] = Master.DATA_PATH + name + Master.BUFFER_TYPE
                self._database.loc[name] = data
            except KeyError:
                raise ValueError(f'can not found the {name} in the local database')

        self._database.to_pickle(Master.USER_PATH, compression='xz')

    def save_all(self):
        '''
        Save all of the preloaded records to the external database
        :return: None
        '''
        self.save(save_all=True)

    def get_exrecord_names(self) -> list:
        '''
        :return: list of the saved records in the external database
        '''
        return self.get_record_names(local_database=False)

    # 'local' or stable database
    def get_record_names(self, local_database: bool=True) -> list:
        '''
        :param local_database: if false then external database will be selected
        :return: list of the saved records in the external or internal database
        '''
        if local_database:
            return list(self._local_database.index)
        else:
            return list(self._database.index)

    # def save_process(self, name):
    #     assert type(name) is str
    #     self._database.loc['__process__' + name] = {'inp_id': self._input_dev_id,
    #                                                 'nchannels': self._process_nchannel,
    #                                                 'frame_rate': self._frame_rate,
    #                                                 'out_id': self._output_device_index,
    #                                                 'data_format': port.get_format_from_width(self._sampwidth),
    #                                                 'mono_mode': self._mono_mode,
    #                                                 'ui_mode': self._ui_mode,
    #                                                 'nperseg': self._nperseg,
    #                                                 'window': self._window,
    #                                                 'window_nhop': self._nhop,
    #                                                 'filter_state': self._primary_filter.is_set(),
    #                                                 }

    def get_nperseg(self):
        return self._nperseg

    def get_sample_rate(self):
        return self.frame_rate

    def stream(self, record: Union[str, Wrap, pd.Series],
               block_mode: bool=False,
               safe_load: bool=False,
               on_stop: callable=None,
               loop_mode: bool=False,
               use_cached_files=True,
               stream_mode:StreamMode = StreamMode.optimized) -> StreamControl:
        '''
        'Record' playback on the mainstream.

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        :param record: It could be a predefined record name, a wrapped record, or a pandas series.
        :param block_mode: This can be true, in which case the current thread will be
         blocked as long as the stream is busy.
        :param safe_load: load an audio file and modify it according to the 'Master' attributes(like the frame rate
        , number oof channels, etc).
        :param on_stop: An optional callback is called at the end of the streaming process.
        :param loop_mode: 'Record' playback on the continually.
        :param use_cached_files: enable additional cache maintaining process.
        :return: A StreamControl object
        '''
        #loop mode just worked in unblocking mode
        cache_head_check_size = 20

        rec_type = type(record)
        if rec_type is str:
            record = self.load(record, series=True)
        else:
            if rec_type is Wrap:
                assert record.is_packed(), BufferError('The {} is not packed!'.format(record))
                record = record.get_data()

            elif rec_type is pd.Series:
                record = record.copy()
            else:
                raise TypeError('please control the type of record')


            if not self._mono_mode and record['nchannels'] > self.nchannels:
                raise ImportError('number of channel for the {name}({ch0})'
                                  ' is not same as object channels({ch1})'.format(name=name,
                                                                                  ch0=record['nchannels'],
                                                                                  ch1=self.nchannels))
        assert  type(record['o']) is io.BufferedRandom, TypeError('The record object is not standard')
        file = record['o']
        file_pos = file.tell()
        tmp = list(file.name)
        tmp.insert(file.name.find(record.name), 'stream_')
        streamfile_name = ''.join(tmp)
        try:
            if use_cached_files:
                record_size = os.path.getsize(file.name)
                streamfile_size = os.path.getsize(streamfile_name)
                file.seek(0, 0)
                pre_head = file.read(Master.CACHE_INFO + cache_head_check_size)
                file.seek(file_pos, 0)

                if record_size == streamfile_size:
                    pass
                elif (record_size - record['size']) == streamfile_size:
                    pass
                    file_pos = Master.CACHE_INFO
                else:
                    raise FileNotFoundError

                streamfile = open(streamfile_name, 'rb+')

                # check compability of two files
                post_head = streamfile.read(Master.CACHE_INFO + cache_head_check_size)
                streamfile.seek(file_pos, 0)
                if not post_head == pre_head:
                    streamfile.close()
                    raise FileNotFoundError

                if safe_load:
                    record['o'] = streamfile.read()
                    record = self._sync_record(record)
                    file.seek(file_pos, 0)
                record['o'] = streamfile
            else:
                raise FileNotFoundError

        except FileNotFoundError:
            record['o'] = file.read()
            if safe_load:
                record = self._sync_record(record)
            file.seek(file_pos, 0)
            streamfile = open(streamfile_name, 'wb+')

            streamfile.write(np.array([record['frameRate'],
                              record['sampleFormat'] if record['sampleFormat'] else 0,
                             record['nchannels']],
                             dtype='u8').tobytes())

            streamfile.write(record['o'])
            streamfile.flush()
            streamfile.seek(Master.CACHE_INFO, 0)
            record['o'] = streamfile
            file_pos = Master.CACHE_INFO

        if block_mode:
            data_size = self._nperseg * self.nchannels * self._sampwidth
            self._main_stream.acquire()
            while 1:
                in_data = streamfile.read(data_size)
                if not in_data:
                    break
                in_data = np.frombuffer(in_data, self._constants[0]).astype('f')
                in_data = Master.get_channels(in_data)
                self._main_stream.put(in_data)  # .astype(self._constants[0])
            self._main_stream.release()
            self._main_stream.clear()
            streamfile.seek(file_pos, 0)
            if on_stop:
                on_stop()

        else:
            return self._stream_control(record, on_stop, loop_mode, stream_mode)

    def _stream_control(self, *args):
        return Master.StreamControl(self, *args)

    def mute(self):
        '''
        mute the mainstream
        :return: self
        '''
        self._master_mute_mode.set()
        return self


    def unmute(self):
        assert not self._exstream_mode.is_set()
        self._master_mute_mode.clear()
        return self


    def echo(self, record: Union[Wrap, str, pd.Series]=None,
             enable: bool=None, main_output_enable: bool=False) -> None:
        """
        Play "Record" on the operating system's default audio output.

         Note:
          If the 'record' argument takes the value None,
          the method controls the standard output activity of the master with the 'enable' argument.

        :param record: optional, default None;
         It could be a predefined record name, a wrapped record,
         or a pandas series.

        :param enable: optional, default None(trigger mode)
            determines that the standard output of the master is enable or not.

        :param main_output_enable:
            when the 'record' is not None, controls the standard output activity of the master
        :return: None
        """
        # Enable echo to system's default output or echo an pre recorded data
        # from local or external database
        if record is None:
            if enable is None:
                if self._echo.is_set():
                    self._echo.clear()
                else:
                    self._main_stream.clear()
                    self._echo.set()
            elif enable:
                self._main_stream.clear()
                self._echo.set()
            else:
                self._echo.clear()
        else:
            if type(record) is Wrap:
                assert record.is_packed()
                record = record.get_data()
            else:
                if type(record) is str:
                    record = self.load(record, series=True)
                elif type(record) is pd.Series:
                    pass
                else:
                    ValueError('unknown type')

            file = record['o']
            file_pos = file.tell()
            data = file.read()
            file.seek(file_pos, 0)

            flg = False
            # assert self.nchannels == record['nchannels'] and self._sampwidth == record['Sample Width']
            if not main_output_enable and self._echo.is_set():
                flg = True
                self._echo.clear()

            play(data,
                 sample_format=record['sampleFormat'],
                 nchannels=record['nchannels'],
                 framerate=record['frameRate'])

            if flg:
                self._echo.set()

    def disable_echo(self):
        '''
        disable the main stream echo mode
        :return: None
        '''
        self.echo(enable=False)

    def join(self, other):

        assert self.frame_rate == other.frame_rate and \
               self._nperseg == other.nperseg and \
               self._sampwidth == other._sampwidth

    def wrap(self, record: Union[str, pd.Series]):
        '''
        Create a Wrap object
        :param record: preloaded record or pandas series
        :return: Wrap object
        '''
        return Wrap(self, record)

    def _cache(self):
        path = []
        expath = []
        base_len = len(Master.DATA_PATH)
        for i in self._database.index:
            path.append(self._database.loc[i, 'o'][base_len:])
        for i in self._local_database.index:
            expath.append(self._local_database.loc[i, 'o'].name[base_len:])
        path += [i for i in expath if not i in path]

        listdir = os.listdir(Master.DATA_PATH)
        listdir = list([Master.DATA_PATH + item for item in listdir if item.endswith('.bin')])
        print(path, listdir)
        # listdir = [Master.DATA_PATH + item  for item in listdir if not item in path]
        for i in path:
            j = 0
            while j < len(listdir):

                if i in listdir[j]:
                    del(listdir[j])
                else:
                    j += 1

        return listdir

    def clean_cache(self):
        '''
         The audio data maintaining process has additional cached files to reduce dynamic
         memory usage and improve performance, meaning that, The audio data storage methods
         can have different execution times based on the cached files.
         This function used to clean additional cache files.
        :return: None
        '''
        cache = self._cache()
        for i in cache:
            os.remove(i)


    def _refresh(self, *args):
        win_len = self._nperseg

        if len(args):
            # primary filter disabled -> pfd
            if args[0] == 'p_f_d':
                self._nhop = win_len - self._noverlap
                self.frame_rate = self._frame_rate[0]
        # primary filter frequency change
        else:
            self._frame_rate[1] = (self.prim_filter_cutoff * 2)
            downsample = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
            self._frame_rate[1] = int(self._frame_rate[0] / downsample)  # human voice high frequency
            self._filters = [scisig.firwin(10, self.prim_filter_cutoff, fs=self._frame_rate[0])]
            # self._data_chunk = int(self._frame_rate[0] * self._record_period)
            # data type of buffer                       0
            # used in callback method(down sampling)    1
            self._constants[1] = downsample
            if self._primary_filter.is_set():
                self.frame_rate = self._frame_rate[1]

            if self._window_type:
                win_len = int(np.ceil(((self._data_chunk - 1) * 1 + len(self._filters[0])) / downsample))
                self._nhop = self._nperseg - self._noverlap
                self._nhop = int(self._nhop / self._data_chunk * win_len)

        if self._window is not None and (not self._nperseg == win_len):
            if type(self._window_type) is str or \
                    type(self._window_type) is tuple or \
                    type(self._window_type) == float:
                self._window = scisig.get_window(self._window_type, win_len)
            else:
                raise _Error("can't refresh static window")
            self._nperseg = win_len

        self._main_stream.acquire()

        if self._mono_mode:
            self._iwin_buffer = [np.zeros(win_len)]
            self._win_buffer = [np.zeros(win_len), np.zeros(win_len)]
        else:
            self._iwin_buffer = [np.zeros(win_len) for i in range(self.nchannels)]
            self._win_buffer = [[np.zeros(win_len), np.zeros(win_len)] for i in range(self.nchannels)]
        # format specifier used in struct.unpack    2
        # rms constant value                        3
        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sampwidth)]  # 1 2 4 8

        self._record_period = self._nperseg / self.frame_rate

        self._refresh_ev.set()

        self._main_stream.clear()
        self._main_stream.release()

class _Error(Exception):
    pass
