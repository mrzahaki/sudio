# the name of Allah

from ._register import Members as Mem
from ._register import static_vars
from ._tools import Tools
from ._audio import *
from ._pipeline import Pipeline
from ._process_common import *
import tqdm
from scipy.signal import upfirdn, firwin, lfilter, check_NOLA
from scipy.signal import get_window as scipy_window
import threading
import queue
import pandas as pd
import pyaudio
import numpy as np


@Mem.process.parent
@Mem.sudio.add
class Master:
    USER_PATH = './user.sufile'
    SOUND_BUFFER_SIZE = 1000
    SAMPLE_CATCH_TIME = 10  # s
    SAMPLE_CATCH_PRECISION = 0.1  # s
    SPEAK_LEVEL_GAP = 10  # dbu
    _shuffle3d_channels = np.vectorize(lambda x: x.T.reshape(np.prod(x.shape)),
                                       signature='(m,n)->(c)')

    def __init__(self,
                 std_input_dev_id: int = None,
                 std_output_dev_id: int = None,
                 frame_rate: int = 48000,
                 nchannels: int = 2,
                 data_format: int = 8,  # 16bit mode
                 mono_mode: bool = True,
                 ui_mode: bool = True,
                 nperseg: int = 256,
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
                 data_format: int = 8,  # 16bit mode
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

            data_format: int
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

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        np.warnings.filterwarnings('ignore', category=RuntimeWarning)

        self.branch_pipe_database_min_key = 0
        self._ui_mode = ui_mode
        self._window_type = window

        self.sample_rate = frame_rate
        self.nperseg = self._nperseg = nperseg
        # check _win arguments
        if noverlap is None:
            noverlap = nperseg // 2
        if type(self._window_type) is str or \
                type(self._window_type) is tuple or \
                type(self._window_type) == float:
            window = scipy_window(window, nperseg)

        if self._window_type:
            assert len(window) == nperseg, 'control size of window'
            if NOLA_check:
                assert check_NOLA(window, nperseg, noverlap)
        elif self._window_type is None:
            window = None
            self._window_type = None

        self.io_mode = IO_mode

        self.input_dev_callback = input_dev_callback
        self.output_dev_callback = output_dev_callback

        # _______________________________________________________UI mode std_input_dev_id calculator
        record_period = nperseg / frame_rate
        self._record_period = record_period
        self._mono_mode = mono_mode

        self._output_device_index = None
        p = self._paudio = pyaudio.PyAudio()  # Create an interface to PortAudio
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
                self.nchannels = dev[dev_id]['maxInputChannels']
                self._input_dev_id = dev_id

            else:
                dev = p.get_default_input_device_info()
                self._frame_rate = [int(dev['defaultSampleRate'])]
                self.nchannels = dev['maxInputChannels']
                self._input_dev_id = dev['index']

        else:
            self._frame_rate = [frame_rate]
            self.nchannels = nchannels
            self._input_dev_id = std_input_dev_id

        self._sampwidth = pyaudio.get_sample_size(data_format)
        # self._sampwidth = data_format
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

        # ______________________________________________________________Vectorized dbu calculator
        self._vdbu = np.vectorize(self.rdbu, signature='(m)->()')

        # ______________________________________________________________channels type and number processing

        if mono_mode:
            self.__class__.get_channels = lambda x: np.mean(x.reshape((self._data_chunk, self.nchannels)),
                                                     axis=1)
        else:
            self.__class__.get_channels = lambda x: np.append(*[[x[i::self.nchannels]] for i in range(self.nchannels)],
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
        self._filters = [firwin(5, self.prim_filter_cutoff, fs=self._frame_rate[0])]
        self._data_chunk = nperseg
        self.frame_rate = frame_rate
        # data type of buffer                       0
        # used in callback method(down sampling)    1
        # used in _win_nd                           2
        # rms constant value                        3
        #                                           4
        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sampwidth)]  # 1 2 4 8
        self._constants = [f'int{self._sampwidth * 8}',
                           downsample,
                           range(self.nchannels)[1:],
                           ((2 ** (self._sampwidth * 8 - 1) - 1) / 1.225),
                           range(self.nchannels)]
        self._threads = []
        # _______________________________________________________user _database define
        try:
            self._database = pd.read_pickle(self.__class__.USER_PATH, compression='xz')
        except:
            self._database = pd.DataFrame(
                columns=['Noise', 'Frame Rate', 'o', 'Sample Width', '_nchannels', 'Duraion', 'Base Period'])
        self._local_database = pd.DataFrame(
            columns=['Noise', 'Frame Rate', 'o', 'Sample Width', 'nchannels', 'Duraion', 'Base Period'])

        # _______________________________________________________pipeline _database and other definitions section

        try:
            self._semaphore = [threading.Semaphore()]

            dataq = queue.Queue(maxsize=self.__class__.SOUND_BUFFER_SIZE)
            self._recordq = [threading.Event(), queue.Queue(maxsize=self.__class__.SOUND_BUFFER_SIZE)]
            self._queue = queue.Queue()

            self._echo = threading.Event()
            self._primary_filter = threading.Event()
            # self._primary_filter.set()
            self.main_pipe_database = []
            self.branch_pipe_database = []
            self._pipeline_ev = threading.Event()
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
            # self._win_filter = firwin(30, self.primary_fc, fs=self._win_frame_rate[0])
            # self._win_window = get_window(self._window_type, self._nperseg)
            # self._win_nchannels_range = range(self.nchannels)

            if mono_mode:
                self._iwin_buffer = [np.zeros(nperseg)]
                self._win_buffer = [np.zeros(nperseg), np.zeros(nperseg)]
                self._win = self._win_mono
                self._iwin = self._iwin_mono
                self._upfirdn = lambda h, x, const: upfirdn(h, x, down=const)
                self._lfilter = lambda h, x: lfilter(h, [1.0], x).astype(self._constants[0])
            else:
                self._iwin_buffer = [np.zeros(nperseg) for i in range(self.nchannels)]
                self._win_buffer = [[np.zeros(nperseg), np.zeros(nperseg)] for i in range(self.nchannels)]

                # 2d mode
                self._upfirdn = np.vectorize(lambda h, x, const: upfirdn(h, x, down=const),
                                             signature='(n),(m),()->(c)')
                self._iwin = self._iwin_nd
                self._win = self._win_nd
                # 3d mode
                self._lfilter = np.vectorize(lambda h, x: lfilter(h, [1.0], x).astype(self._constants[0]),
                                             signature='(n),(m)->(c)')

            self._normal_stream = self.__class__._Stream(self, dataq, process_type='queue')
            self._main_stream = self.__class__._Stream(self, dataq, process_type='queue')

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
        # stdInput2stdOutput mode
        stream_callback = self._stream_callback
        # user input mode
        if self.io_mode.startswith("usr"):
            stream_callback = self._ustream_callback

        try:
            self._pystream = self._paudio.open(format=pyaudio.get_format_from_width(self._sampwidth),
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

    # stdInput2stdOutput
    def _stream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        in_data = np.fromstring(in_data, self._constants[0])
        in_data = self.__class__.get_channels(in_data)
        # print(in_data.shape)
        # in_data = np.fromstring(in_data, self._constants[0])
        # print(in_data.shape)
        try:
            self._main_stream.put(in_data.astype(self._constants[0]))
        except queue.Full:
            self._main_stream.sget()

        # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
        # self.a1 = time.perf_counter()
        return None, pyaudio.paContinue

    # user Input mode
    def _ustream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        in_data = self.input_dev_callback(frame_count, time_info, status)
        # in_data = np.fromstring(in_data, self._constants[0])
        try:
            self._main_stream.put(in_data.astype(self._constants[0]))
        except queue.Full:
            self._main_stream.sget()

        # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
        # self.a1 = time.perf_counter()
        return None, pyaudio.paContinue

    def __post_process(self, thread_id):
        stream_out = self._paudio.open(
            format=pyaudio.get_format_from_width(self._sampwidth),
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
                data = self.__class__.shuffle2d_channels(data)
            # data = data.T.reshape(np.prod(data.shape))
            # print(data.shape)
            if self._echo.isSet():
                stream_out.write(data.tostring())

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
                    format=pyaudio.get_format_from_width(self._sampwidth),
                    channels=self._process_nchannel,
                    rate=rate, input=False, output=True)
                stream_out.start_stream()
                self._main_stream.clear()

    def recorder(self, enable_compressor=True,
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
                                          noise_base=noise + self.__class__.SPEAK_LEVEL_GAP,
                                          final_period=catching_precision,
                                          base_period=self._record_period)

            if enable_ui:
                progress.update(10)
                # Waiting for Process processing to be enabled
                progress.close()

            if not self._mono_mode:
                sample = self.shuffle3d_channels(sample)

            if play_recorded:
                self._audio_play_wrapper(sample)

            if echo and not echo_mode:
                self.echo()

            if enable_ui:
                if self._safe_input('\nTry again? "yes"').strip().lower() == 'yes':
                    continue
                self._safe_print('Recorded successfully!')
            retval = {'o': sample,
                      'Frame Rate': self.frame_rate,
                      'channels': self.nchannels,
                      'Sample Width': self._sampwidth,
                      'Base Period': self._record_period,
                      'Duraion': record_duration}
            if enable_compressor:
                retval['Noise'] = noise
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
        # sample = self._lfilter(self._filters[1], sample)
        if progress:
            progress.update(5)
        # print(sample, sample.shape)
        return sample

    def _audio_play_wrapper(self, sample):
        # print(sample.shape)
        play(sample, inp_format='array', sampwidth=self._sampwidth, nchannels=self._process_nchannel,
             framerate=self.frame_rate)

    def _refresh(self, *args):
        win_len = self._nperseg

        if len(args):
            # primary filter disabled -> pfd
            if args[0] == 'p_f_d':
                self._nhop = win_len - self._noverlap

        # primary filter frequency change
        else:
            self._frame_rate[1] = (self.prim_filter_cutoff * 2)
            downsample = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
            self._frame_rate[1] = int(self._frame_rate[0] / downsample)  # human voice high frequency
            self._filters = [firwin(5, self.prim_filter_cutoff, fs=self._frame_rate[0])]
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


        if self._window and not self._nperseg == win_len:
            if type(self._window_type) is str or \
               type(self._window_type) is tuple or \
               type(self._window_type) == float:
                self._window = scipy_window(self._window_type, win_len)
            else:
                raise Error("can't refresh static window")
            self.nperseg = win_len

        if self._mono_mode:
            self._iwin_buffer = [np.zeros(win_len)]
            self._win_buffer = [np.zeros(win_len), np.zeros(win_len)]
        else:
            self._iwin_buffer = [np.zeros(win_len) for i in range(self.nchannels)]
            self._win_buffer = [[np.zeros(win_len), np.zeros(win_len)] for i in range(self.nchannels)]
        # format specifier used in struct.unpack    2
        # rms constant value                        3
        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sampwidth)]  # 1 2 4 8
        self._refresh_ev.set()
        self._main_stream.clear()

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
    def rdbu(self, arr):
        return Tools.dbu(np.max(arr) / self._constants[3])

    def add_record(self, name,
                   enable_compressor=True,
                   noise_sampling_duration=10,
                   record_duration=10,
                   enable_ui=True,
                   play_recorded=True,
                   catching_precision=0.1,
                   echo_mode=False,
                   rec_start_callback=None):

        if name in self._local_database.index:
            name = self._safe_input(
                'The entered name is already registered in the database.\nPlease enter another name:')
        if not name:
            raise ValueError

        self._local_database.loc[name] = self.recorder(enable_compressor=enable_compressor,
                                                       noise_sampling_duration=noise_sampling_duration,
                                                       record_duration=record_duration,
                                                       enable_ui=enable_ui,
                                                       play_recorded=play_recorded,
                                                       catching_precision=catching_precision,
                                                       echo_mode=echo_mode,
                                                       rec_start_callback=rec_start_callback)

    # load the name from stable database to local if defined
    def load_record(self, name, load_all=False):
        if load_all:
            self._local_database = self._local_database.append(self._database, verify_integrity=True)
        else:
            if name in self._local_database.index:
                return self._local_database.loc[name]
            elif not (name in self._database.index):
                raise ValueError('name not in self._database.index')
            rec = self._database.loc[name]
            self._local_database.loc[name] = rec
            return rec

            # save from local _database to stable memory

    def save_record(self, name, save_all=False):
        if save_all:
            self._database = self._database.append(self._local_database, verify_integrity=True)
        else:
            if name in self._database.index and not (name in self._local_database.index):
                raise ValueError
            self._database.loc[name] = self._local_database.loc[name]
        self._database.to_pickle(self.__class__.USER_PATH, compression='xz')

    # 'local' or stable database
    def get_record_names(self, database='local'):

        if database == 'local':
            return list(self._local_database.index)
        else:
            return list(self._database.index)

    # def save_process(self, name):
    #     assert type(name) is str
    #     self._database.loc['__process__' + name] = {'inp_id': self._input_dev_id,
    #                                                 'nchannels': self._process_nchannel,
    #                                                 'frame_rate': self._frame_rate,
    #                                                 'out_id': self._output_device_index,
    #                                                 'data_format': pyaudio.get_format_from_width(self._sampwidth),
    #                                                 'mono_mode': self._mono_mode,
    #                                                 'ui_mode': self._ui_mode,
    #                                                 'nperseg': self.nperseg,
    #                                                 'window': self._window,
    #                                                 'window_nhop': self._nhop,
    #                                                 'filter_state': self._primary_filter.is_set(),
    #                                                 }

    def echo(self, record=None, enable=True, main_output_enable=False):

        # Enable echo to system's default output or echo an pre recorded data from local or external database
        if record is None:
            if enable:
                self._main_stream.clear()
                self._echo.set()
            else:
                self._echo.clear()

        elif type(record) is str:
            record = self.load_record(record)
            flg = False
            # assert self.nchannels == record['nchannels'] and self._sampwidth == record['Sample Width']
            if not main_output_enable and self._echo.is_set():
                flg = True
                self._echo.clear()
            print(record['nchannels'])
            self._audio_play_wrapper(record['o'])

            if flg:
                self._echo.set()

    def join(self, other):

        assert self.frame_rate == other.frame_rate and \
               self.nperseg == other.nperseg and \
               self._sampwidth == other._sampwidth


class Error(Exception):
    pass

