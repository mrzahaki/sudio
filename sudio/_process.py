# the name of Allah

from ._register import Members as Mem
from ._register import static_vars
from ._tools import Tools
from ._audio import *
from ._pipeline import Pipeline
import tqdm
from scipy.signal import upfirdn, firwin, lfilter, get_window, check_NOLA
import threading
import queue
import pandas as pd
import pyaudio
import numpy as np


# @Mem._process.parent
@Mem.sudio.add
class Process():
    USER_PATH = './user.sufile'
    SOUND_BUFFER_SIZE = 1000
    SAMPLE_CATCH_TIME = 10  # s
    SAMPLE_CATCH_PRECISION = 0.1  # s
    SPEAK_LEVEL_GAP = 10  # dbu
    _shuffle3d_channels = np.vectorize(lambda x: x.T.reshape(np.prod(x.shape)),
                                       signature='(m,n)->(c)')



    def __init__(self,
                 std_input_dev_id=None,
                 std_output_dev_id=None,
                 frame_rate=48000,
                 nchannels=2,
                 data_format=pyaudio.paInt16,
                 mono_mode=True,
                 optimum_mono=False,
                 ui_mode=True,
                 nperseg=256,
                 noverlap=None,
                 window='hann',
                 NOLA_check=True,
                 IO_mode="stdInput2stdOutput",
                 input_dev_callback=None,
                 output_dev_callback=None):
        '''
        :param std_input_dev_id: Unsigned Integer(None to default)
        :param IO_mode:  Input/Output processing mode can be:
                        "stdInput2stdOutput":
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

        :param input_dev_callback:
                        callback function; default None;

                        :inputs:    None
                        :outputs:   Numpy N-channel data frame in  (Number of channels, Number of data per segment)
                                    dimensions.
                        :note: In 1 channel mode data frame have  (Number of data per segment, ) shape

                        can be used to define user input callback
                        this function used in "usrInput2usrOutput" and "usrInput2usrOutput" IO modes
                        and called by core every 1/frame_rate second and must returns frame data
                        in (Number of channels, Number of data per segment) dimensions numpy array.
                        each data must be in special format that defined in data_format section.

        :param output_dev_callback:
                        callback function; default None;

                        :inputs:    Numpy N-channel data frame in  (Number of channels, Number of data per segment)
                                    dimensions.
                        :outputs:
                        :note: In 1 channel mode data frame have  (Number of data per segment, ) shape

                        can be used to define user input callback
                        this function used in "usrInput2usrOutput" and "usrInput2usrOutput" IO modes
                        and called by core every 1/frame_rate second and must returns frame data
                        in (Number of channels, Number of data per segment) dimensions numpy array.
                        each data must be in special format that defined in data_format section.

        :param std_input_dev_id: Unsigned Integer
                        In standard input IO mode, the std_input_dev_id used to define that which windows standard
                        streams used by the core of processor.

                        :Note:  None argument can be used when ui mode is enabled to select input device id
                                in user interface mode simply.

        :param frame_rate:  Unsigned Integer; sample rate
        :param nchannels: Unsigned Integer; number of input channels
        :param data_format: Object: supported data format:
                        formatFloat32, formatInt32, formatInt24, formatInt16, formatInt8, formatUInt8
        :param mono_mode: Enabe mono mode
        :param optimum_mono:  Enabe optimized mono mode
        :param ui_mode: Enabe user interface mode(In recording)
        :param nperseg: number of samples per each data frame(in one channel)
        :param noverlap: number of overlap between defined windows
        :param window:  String;
                        boxcar, triang , blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris
                        nuttall, barthann, cosine, exponential, tukey, taylor, kaiser (needs beta),
                        gaussian (needs standard deviation), general_cosine (needs weighting coefficients),
                        general_gaussian (needs power, width), general_hamming (needs window coefficient),
                        dpss (needs normalized half-bandwidth), chebwin (needs attenuation)
        :param NOLA_check: NOLA check for window definition
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
        self.nperseg = nperseg
        # check _win arguments
        if noverlap is None:
            noverlap = nperseg // 2
        if type(window) == str:
            window = get_window(window, nperseg)

        if NOLA_check:
            assert check_NOLA(window, nperseg, noverlap)

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
                    msg = f'Index {idx}: '\
                        f' Name: {j["name"]},'\
                        f' Input Channels:{j["maxInputChannels"]},'\
                        f' Sample Rate:{j["defaultSampleRate"]},'\
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
                    msg = f'Index {idx}:'\
                        f'  Name: {j["name"]},'\
                        f' Channels:{j["maxOutputChannels"]},'\
                        f' Sample Rate:{j["defaultSampleRate"]},'\
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
            if optimum_mono:
                get_channels = np.vectorize(np.mean, signature='(m)->()')
                Process.get_channels = lambda x: get_channels(x.reshape((self._data_chunk, self.nchannels)))
            else:
                Process.get_channels = lambda x: x[::self.nchannels]
        else:
            Process.get_channels = lambda x: np.append(*[[x[i::self.nchannels]] for i in range(self.nchannels)],
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
            self._database = pd.read_pickle(Process.USER_PATH, compression='xz')
        except:
            self._database = pd.DataFrame(
                columns=['Noise', 'Frame Rate', 'o', 'Sample Width', '_nchannels', 'Duraion', 'Base Period'])
        self._local_database = pd.DataFrame(
            columns=['Noise', 'Frame Rate', 'o', 'Sample Width', 'nchannels', 'Duraion', 'Base Period'])

        # _______________________________________________________pipeline _database and other definitions section

        try:
            self._semaphore = [threading.Semaphore()]

            dataq = queue.Queue(maxsize=Process.SOUND_BUFFER_SIZE)
            self._recordq = [threading.Event(), queue.Queue(maxsize=Process.SOUND_BUFFER_SIZE)]
            self._queue = queue.Queue()

            self._echo = threading.Event()
            self._primary_filter = threading.Event()
            # self._primary_filter.set()
            self.main_pipe_database = []
            self.branch_pipe_database = []
            self._pipeline_ev = threading.Event()
            #_______________________________________threading management
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

            self._normal_stream = Process._Stream(self, dataq, process_type='queue')
            self._main_stream = Process._Stream(self, dataq, process_type='queue')

            # self.data_tst_buffer = [i for i in range(5)]
            # self.iwin_tst_buffer = [i for i in range(5)]
            self._refresh_ev = threading.Event()
            if self._ui_mode:
                print('Initialization completed!')

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
        in_data = Process.get_channels(in_data)
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

        in_data = self.input_dev_callback()
        # in_data = np.fromstring(in_data, self._constants[0])
        try:
            self._main_stream.put(in_data.astype(self._constants[0]))
        except queue.Full:
            self._main_stream.sget()

        # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
        # self.a1 = time.perf_counter()
        return None, pyaudio.paContinue

    def _win_mono(self, data):

        if self._primary_filter.is_set():
            data = upfirdn(self._filters[0], data, down=self._constants[1]).astype(self._constants[0])

        # Tools.push(self.data_tst_buffer, data)
        retval = np.vstack((self._win_buffer[1], np.hstack((self._win_buffer[1][self._nhop:],
                                                            self._win_buffer[0][:self._nhop])))) * self._window
        Tools.push(self._win_buffer, data)


        return retval

    def _win_nd(self, data):

        # brief: Windowing data
        # Param 'data' shape depends on number of input channels(e.g. for two channel stream, each chunk of
        # data must be with the shape of (2, chunk_size))
        if self._primary_filter.is_set():
            data = self._upfirdn(self._filters[0], data, self._constants[1]).astype(self._constants[0])

        # retval frame consists of two window
        # that each window have the shape same as 'data' param shape(e.g. for two channel stream:(2, 2, chunk_size))
        # note when primary_filter is enabled retval retval shape changes depend on upfirdn filter.
        # In general form retval shape is
        # (number of channels, number of windows(2), size of data chunk depend on primary_filter activity).

        final = []
        # Channel 0
        # final.append(np.vstack((self._win_buffer[0][1], np.hstack(
        #     (self._win_buffer[0][1][self._nhop:], self._win_buffer[0][0][:self._nhop])))) * self._window)
        # Tools.push(self._win_buffer[0], data[0])

        # range(self.nchannels)[1:]
        for i in self._constants[4]:
            final.append(np.vstack((self._win_buffer[i][1], np.hstack(
                (self._win_buffer[i][1][self._nhop:], self._win_buffer[i][0][:self._nhop])))) * self._window)
            Tools.push(self._win_buffer[i], data[i])
        # for 2 channel win must be an 2, 2, self._data_chunk(e.g. 256)
        # reshaping may create some errors
        # return data
        final = np.array(final)

        return final

    def _iwin_mono(self, win):
        retval = np.hstack((self._iwin_buffer[0][self._nhop:], win[1][:self._nhop])) + win[0]
        self._iwin_buffer[0] = win[1]
        # return data
        # plt.plot(retval, label='iwin')
        # plt.legend()
        # plt.show()
        return retval.astype(self._constants[0])

    # start from index1 data
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

        return retval.astype(self._constants[0])

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
                data = Process.shuffle2d_channels(data)
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
            if enable_ui :
                self._safe_print(
                    f'\n^^^^^^^^^^Ready to record for  {record_duration} seconds^^^^^^^^^^^^^^')
                if not self._safe_input(f'Are you ready? enter "yes"\nNoise Level: {noise} dbu').strip().lower() == 'yes':
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
            if callable(rec_start_callback) :
                if enable_compressor:
                    tmp = rec_start_callback(self, {'noise':noise})
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
                                         noise_base=noise + Process.SPEAK_LEVEL_GAP,
                                         final_period=catching_precision,
                                         base_period=self._record_period)

            if enable_ui:
                progress.update(10)
                # Waiting for Process processing to be enabled
                progress.close()

            if not self._mono_mode:
                sample = self.shuffle3d_channels(sample)

            if play_recorded:
                framerate = self._audio_play_wrapper(sample)

            if echo and not echo_mode:
                self.echo()

            if enable_ui:
                if self._safe_input('\nTry again? "yes"').strip().lower() == 'yes':
                    continue
                self._safe_print('Recorded successfully!')
            retval = {'o': sample,
                    'Frame Rate': framerate,
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
        if self._mono_mode and final_period < base_period:
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
        if self._primary_filter.is_set():
            framerate = self._frame_rate[1]
        else:
            framerate = self._frame_rate[0]
        # print(sample.shape)
        play(sample, inp_format='array', sampwidth=self._sampwidth, nchannels=self._process_nchannel,
             framerate=framerate)
        return framerate

    def _refresh(self, arg=None):

        if type(arg) is str:
            if arg == 'p_f_d':
                win_len = self.nperseg
                self._nhop = win_len - self._noverlap

        else:
            self._frame_rate[1] = (self.prim_filter_cutoff * 2)
            downsample = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
            self._frame_rate[1] = int(self._frame_rate[0] / downsample)  # human voice high frequency
            self._filters = [firwin(5, self.prim_filter_cutoff, fs=self._frame_rate[0])]
            # self._data_chunk = int(self._frame_rate[0] * self._record_period)
            # data type of buffer                       0
            # used in callback method(down sampling)    1
            self._constants[1] = downsample

            win_len = int(np.ceil(((self._data_chunk - 1) * 1 + len(self._filters[0])) / downsample))
            self._nhop = self.nperseg - self._noverlap
            self._nhop = int(self._nhop / self._data_chunk * win_len)

        self._window = get_window(self._window_type, win_len)
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

    # ____________________________________________________ pipe line
    def add_pipeline(self, name, pip, process_type='main', channel=None, stream_type='multithreading'):
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
            (self.nchannel, 2[2 windows in 1 frame], self.nperseg) dimension [None]
            or mono (2, self.nperseg) dimension.
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
            self.main_pipe_database.append(Process._Stream(self, pip, name, stream_type=stream_type, process_type=process_type))

        elif process_type == 'branch':
            self.branch_pipe_database.append(Process._Stream(self, pip, name, stream_type=stream_type, channel=channel, process_type=process_type))

        # reserved for the next versions
        else:
            pass


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
                self._main_stream.clear()
                self._main_stream.set(obj)


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
        self._database.to_pickle(Process.USER_PATH, compression='xz')

    # 'local' or stable database
    def get_record_names(self, database='local'):

        if database == 'local':
            return list(self._local_database.index)
        else:
            return list(self._database.index)

    def filter(self, enable=False, fc=None):
        # if fc == None dont change
        if enable:
            if self._primary_filter.is_set() and fc == self.prim_filter_cutoff:
                return
            elif fc:
                assert fc < (self._frame_rate[0] / 2)
                self.prim_filter_cutoff = fc
            self._primary_filter.set()
            self._refresh()
        elif self._primary_filter.is_set():
            self._primary_filter.clear()
            # self.prim_filter_cutoff = self._frame_rate[0] / 2 - 1e-6
            self._refresh('p_f_d')

        else:
            return


    # arr format: [[data0:[ch0] [ch1]]  [data1: [ch0] [ch1]], ...]
    @staticmethod
    def shuffle3d_channels(arr):
        arr = Process._shuffle3d_channels(arr)
        # res = np.array([])
        # for i in arr:
        #     res = np.append(res, Process.shuffle2d_channels(i))
        return arr.reshape(np.prod(arr.shape))
        # return res

    @staticmethod
    def shuffle2d_channels(arr):
        return arr.T.reshape(np.prod(arr.shape))

    class _Stream:

        def __init__(self, other, obj, name=None, stream_type='multithreading', channel=None,  process_type='main'):

            self.stream_type = stream_type
            self.process_obj = other
            self.process_type = process_type
            self.pip = obj
            self.channel = channel
            self.name = name
            self._data_indexer = lambda data: data[channel]

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
                if not channel is None and not(type(channel) is int and channel <= other.nchannels):
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

        def _main_put(self, data):
            data = self.process_obj._win(data)
            # print(data.shape)
            for branch in self.process_obj.branch_pipe_database:
                branch.put(branch._data_indexer(data))
            # print(self.process_obj.branch_pipe_database)
            self.pip.put(data)

        def _main_get(self, *args):
            data = self.process_obj._iwin(self.pip.get(*args))
            # print(data.shape)
            return data
        def _multi_main_put(self, data):

            if self.process_obj._primary_filter.is_set():
                data = self.process_obj._upfirdn(self.process_obj._filters[0], data, self.process_obj._constants[1]).astype(self.process_obj._constants[0])

            final = []
            # Channel 0
            win = np.vstack((self.process_obj._win_buffer[0][1], np.hstack(
                (self.process_obj._win_buffer[0][1][self.process_obj._nhop:], self.process_obj._win_buffer[0][0][:self.process_obj._nhop])))) * self.process_obj._window

            Tools.push(self.process_obj._win_buffer[0], data[0])
            self.pip[0].put(win)
            final.append(win)
            # range(self.process_obj.nchannels)[1:]
            for i in self.process_obj._constants[2]:
                win = np.vstack((self.process_obj._win_buffer[i][1], np.hstack(
                    (self.process_obj._win_buffer[i][1][self.process_obj._nhop:], self.process_obj._win_buffer[i][0][:self.process_obj._nhop])))) * self.process_obj._window

                Tools.push(self.process_obj._win_buffer[i], data[i])
                self.pip[i].put(win)
                final.append(win)
            # for 2 channel win must be an 2, 2, self.process_obj._data_chunk(e.g. 256)
            # reshaping may create some errors
            # return data
            for branch in self.process_obj.branch_pipe_database:
                branch.put(branch._data_indexer(final))
            

        def _multi_main_get(self, *args):
            # channel 0
            win = self.pip[0].get(*args)
            retval = np.hstack((self.process_obj._iwin_buffer[0][self.process_obj._nhop:], win[1][:self.process_obj._nhop])) + \
                     win[0]
            self.process_obj._iwin_buffer[0] = win[1]

            for i in self.process_obj._constants[2]:
                win = self.pip[i].get(*args)
                tmp = np.hstack((self.process_obj._iwin_buffer[i][self.process_obj._nhop:], win[1][:self.process_obj._nhop])) + \
                      win[0]
                retval = np.vstack((retval, tmp))
                self.process_obj._iwin_buffer[i] = win[1]

            return retval.astype(self.process_obj._constants[0])
        
        def _multi_main_clear(self):
            for i in self.pip:
                i.clear()

        def set(self, other):
            if isinstance(other, self.__class__):
                assert  ('main' in self.process_type or\
                         'multi_stream' in self.process_type or\
                         'queue' in self.process_type)\
                        and\
                        ('main' in other.process_type or\
                         'multi_stream' in other.process_type or\
                         'queue' in other.process_type),\
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
                 self.name) = ( other.put,
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


class Error(Exception):
    pass
