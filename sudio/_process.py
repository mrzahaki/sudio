from ._register import Members as Mem
from ._register import static_vars
from ._tools import Tools
from ._audio import *
from ._pipeline import Pipeline
import tqdm
from scipy.signal import upfirdn, firwin, lfilter, check_COLA, get_window, check_NOLA
import threading
import queue
import pandas as pd
import pyaudio
import numpy as np


# @Mem._process.parent
@Mem.sudio.add
class Process():
    USER_PATH = './user.sufile'
    HUMAN_VOICE_FREQ = int(8e3)  # Hz
    SOUND_BUFFER_SIZE = 500
    SAMPLE_CATCH_TIME = 10  # s
    SAMPLE_CATCH_PRECISION = 0.1  # s
    SPEAK_LEVEL_GAP = 10  # dbu
    _shuffle3d_channels = np.vectorize(lambda x: x.T.reshape(np.prod(x.shape)),
                                       signature='(m,n)->(c)')

    def __init__(self, input_dev_id=None, frame_rate=48000, nchannels=2,
                 data_format=pyaudio.paInt16,
                 mono_mode=True,
                 optimum_mono=False,
                 ui_mode=True,
                 nperseg=256,
                 noverlap=None,
                 window='hann',
                 NOLA_check=True):

        self._pipe_database = []
        self._functions = []
        self._functions = [self._main]
        self._functions.append(self._process)
        self._functions.append(self._play)
        self._ui_mode = ui_mode
        self._def_window = lambda length: get_window(window, length)

        # check _win arguments
        if noverlap is None:
            noverlap = nperseg // 2
        if type(window) == str:
            window = get_window(window, nperseg)
        if NOLA_check:
            # assert check_COLA(window, nperseg, noverlap)
            assert check_NOLA(window, nperseg, noverlap)
        # post, pre
        self._nhop = nperseg - noverlap
        self._noverlap = noverlap
        self._window = window

        record_period = nperseg / frame_rate
        self._record_period = record_period
        self._mono_mode = mono_mode
        if input_dev_id is None and self._ui_mode:

            rec = record(output='array', _ui_mode=False)
            self._nchannels = rec['nchannels']
            self._frame_rate = [rec['frame rate']]
            self._sampwidth = rec['sample width']
            self._input_dev_id = rec['dev_id']

        else:
            self._frame_rate = [frame_rate]
            self._sampwidth = pyaudio.get_sample_size(data_format)
            self._nchannels = nchannels
            self._input_dev_id = input_dev_id
        # Vectorized dbu calculator
        self._vdbu = np.vectorize(self.rdbu, signature='(m)->()')
        if mono_mode:
            if optimum_mono:
                get_channels = np.vectorize(np.mean, signature='(m)->()')
                Process.get_channels = lambda x: get_channels(x.reshape((self._data_chunk, self._nchannels)))
            else:
                Process.get_channels = lambda x: x[::self._nchannels]
        else:
            Process.get_channels = lambda x: np.append(*[[x[i::self._nchannels]] for i in range(self._nchannels)], axis=0)

        if mono_mode:
            self._process_nchannel = 1
        else:
            self._process_nchannel = self._nchannels

        self._frame_rate.append(Process.HUMAN_VOICE_FREQ * 2)
        downsample = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
        self._frame_rate[1] = int(self._frame_rate[0] / downsample)  # human voice high frequency
        self._filters = [firwin(30, Process.HUMAN_VOICE_FREQ, fs=self._frame_rate[0]),
                        firwin(30, int(self._frame_rate[1] / 2 - 1e-6), fs=self._frame_rate[1])]
        self._data_chunk = nperseg
        # data type of buffer                       0
        # used in callback method(down sampling)    1
        # used in _win_nd                           2
        # rms constant value                        3
        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sampwidth)]  # 1 2 4 8
        self._constants = [f'int{self._sampwidth * 8}',
                          downsample,
                          range(self._nchannels)[1:],
                          ((2 ** (self._sampwidth * 8 - 1) - 1) / 1.225)]
        self._threads = []
        # _______________________________________________________user _database section
        try:
            self._database = pd.read_pickle(Process.USER_PATH, compression='xz')
        except:
            self._database = pd.DataFrame(
                columns=['Noise', 'Frame Rate', 'o', 'Sample Width', '_nchannels', 'Duraion', 'Base Period'])
        self._local_database = pd.DataFrame(
            columns=['Noise', 'Frame Rate', 'o', 'Sample Width', 'nchannels', 'Duraion', 'Base Period'])

        # _______________________________________________________pipeline _database section

        try:
            self._semaphore = [threading.Semaphore()]

            self._paudio = pyaudio.PyAudio()  # Create an interface to PortAudio
            dataq = queue.Queue(maxsize=Process.SOUND_BUFFER_SIZE)
            self._queue = queue.Queue()

            self._echo = threading.Event()
            self._primary_filter = threading.Event()
            # self._primary_filter.set()

            self._pipeline_ev = threading.Event()

            # create threads and queues
            for i in range(len(self._functions)):
                self._threads.append(threading.Thread(target=self._run, daemon=True, args=(len(self._threads),)))
                self._queue.put(i)

            if mono_mode:
                self._iwin_buffer = [np.zeros(nperseg), np.zeros(nperseg)]
                self._win_buffer = [np.zeros(nperseg), np.zeros(nperseg)]

                self._win = self._win_mono
                self._iwin = self._iwin_mono
                self._upfirdn = lambda h, x, const: upfirdn(h, x, down=const)
                self._lfilter = lambda h, x: lfilter(h, [1.0], x).astype(self._constants[0])
            else:
                self._iwin_buffer = [[np.zeros(nperseg), np.zeros(nperseg)] for i in range(self._nchannels)]
                self._win_buffer = [[np.zeros(nperseg), np.zeros(nperseg)] for i in range(self._nchannels)]

                # 2d mode
                self._upfirdn = np.vectorize(lambda h, x, const: upfirdn(h, x, down=const),
                                            signature='(n),(m),()->(c)')
                self._iwin = self._iwin_nd
                self._win = self._win_nd
                # 3d mode
                self._lfilter = np.vectorize(lambda h, x: lfilter(h, [1.0], x).astype(self._constants[0]),
                                            signature='(n),(m)->(c)')

            self._normal_stream = Process._Stream(self, dataq, mode='other')
            self._main_stream = Process._Stream(self, dataq, mode='other')

            print('Initialization completed!')

        except:
            print('Initialization error!')
            raise

    def start(self):
        try:
            self._pystream = self._paudio.open(format=pyaudio.get_format_from_width(self._sampwidth),
                                           channels=self._nchannels,
                                           rate=self._frame_rate[0],
                                           frames_per_buffer=self._data_chunk,
                                           input_device_index=self._input_dev_id,
                                           input=True,
                                           stream_callback=self._stream_callback)
            for i in self._threads:
                i.start()

            # The server blocks all active threads
            # del self._threads[:]
            self._queue.join()

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

    def _stream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        in_data = Process.get_channels(np.fromstring(in_data, self._constants[0]))
        # in_data = np.fromstring(in_data, self._constants[0])
        try:
            self._main_stream.put(in_data.astype(self._constants[0]))
        except queue.Full:
            self._main_stream.get()
        # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
        # self.a1 = time.perf_counter()
        return None, pyaudio.paContinue

    def _win_mono(self, data):
        pass

    def _win_nd(self, data):
        # brief: Windowing data
        # Param 'data' shape depends on number of input channels(e.g. for two channel stream, each chunk of
        # data must be with the shape of (2, chunk_size))

        # retval frame consists of two window
        # that each window have the shape same as 'data' param shape(e.g. for two channel stream:(2, 2, chunk_size))
        # note when primary_filter is enabled retval retval shape changes depend on upfirdn filter.
        # In general form retval shape is
        # (number of windows(2), number of channels, size of data chunk depend on primary_filter activity).
        if self._primary_filter.isSet():
            data = self._upfirdn(self._filters[0], data, self._constants[1])

        final = []
        # Channel 0
        final.append(np.vstack((self._win_buffer[0][1], np.hstack((self._win_buffer[0][1][self._nhop:], self._win_buffer[0][0][:self._nhop])))) * self._window)
        Tools.push(self._win_buffer[0], data[0])

        # range(self._nchannels)[1:]
        for i in self._constants[2]:
            final.append(np.vstack((self._win_buffer[i][1], np.hstack((self._win_buffer[i][1][self._nhop:], self._win_buffer[i][0][:self._nhop])))) * self._window)
            Tools.push(self._win_buffer[i], data[i])
        # for 2 channel win must be an 2, 2, self._data_chunk(e.g. 256)
        # reshaping may create some errors
        # return data
        return np.array(final)

    def _iwin_mono(self, win):
        pass

    # start from index1 data
    def _iwin_nd(self, win):
        # for 2 channel win must be an 2[two ], 2, self._data_chunk(e.g. 256)
        # pre post, data,
        # 2 window per frame
        final = []
        for n_win in [0, 1]:
            retval = np.hstack((self._iwin_buffer[0][1][self._nhop:], win[n_win][0][:self._nhop])) + \
                     self._iwin_buffer[0][0]

            Tools.push(self._iwin_buffer[0], win[n_win][0])

            for i in self._constants[2]:
                tmp = np.hstack((self._iwin_buffer[i][1][self._nhop:], win[n_win][i][:self._nhop])) + \
                      self._iwin_buffer[i][0]

                retval = np.vstack((retval, tmp))
                Tools.push(self._iwin_buffer[i], win[n_win][i])
            final.append(retval)

        return np.array(final)

    def _play(self, thread_id):
        stream_out = self._paudio.open(
            format=pyaudio.get_format_from_width(self._sampwidth),
            channels=self._process_nchannel,
            rate=self._frame_rate[0], input=False, output=True)
        stream_out.start_stream()
        flag = self._primary_filter.isSet()
        framerate = self._frame_rate[1]
        while 1:
            self._echo.wait()
            # check primary filter state
            data = self._iwin(self._main_stream.get())
            if not self._mono_mode:
                data = Process.shuffle2d_channels(data)
            # data = data.T.reshape(np.prod(data.shape))

            stream_out.write(data.tostring())

            if not self._primary_filter.isSet() == flag or not self._frame_rate[1] == framerate:
                flag = self._primary_filter.isSet()

                # close current stream
                stream_out.close()
                # create new stream
                if flag:
                    framerate = rate = self._frame_rate[1]
                else:
                    rate = self._frame_rate[0]

                stream_out = self._paudio.open(
                    format=pyaudio.get_format_from_width(self._sampwidth),
                    channels=self._process_nchannel,
                    rate=rate, input=False, output=True)
                stream_out.start_stream()
                self._main_stream.clear()

    # working in two mode: 'recognition'->'r' mode and 'new-user'->'n' mode
    def _process(self, thread_id):
        while 1:
            time.sleep(1)
            # try:
            #     self.speaker_recognition_ev.wait()
            #     a = 20 * np.log10(np.max(
            #         correlate(self._main_stream.get(), self._local_database.loc['hussein']['o'][1:4, :].reshape(4830),
            #                   mode='valid')))
            #
            # except KeyError:
            #     while not len(self._local_database): pass

        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sampwidth)]  # 1 2 4 8
        # # data len int(self._data_chunk/self._constants[1]) + 10
        # format = f'<{int(self._data_chunk / self._constants[1]) + 10}{sampwidth}'
        # # print(format)
        # while 1:
        #     self._semaphore[1].acquire()
        #     if not len(self.recognized):
        #         self.dataq.get()
        #         continue
        #
        #     ding_data = struct.pack(format, *self.dataq.get().tolist())
        #     self._semaphore[1].release()

    def _main(self, thread_id):
        while 1:
            try:
                inp = input('\n_main>> ').strip().lower()

                # _________________________________________________________________add instruction
                if not inp.find('add '):
                    inp = inp.replace('add ', '', 1).strip()

                    if 'record ' in inp:
                        self._safe_print('^^^^^^^^^^Please Wait...^^^^^^^^^^^^^^')
                        enable_compressor = False
                        record_duration = Process.SAMPLE_CATCH_TIME
                        if '-t' in inp:
                            tmp = inp.split('-t')[1].strip().split(' ')
                            if not '=' == tmp[0]:
                                raise Error('Please enter valid statement')
                            record_duration = int(tmp[1].strip())
                        if '-c' in inp:
                            enable_compressor = True
                        self.add_record(inp.split('record ')[1],
                                        enable_compressor=enable_compressor,
                                        noise_sampling_duration=Process.SAMPLE_CATCH_TIME,
                                        record_duration=record_duration,
                                        catching_precision=Process.SAMPLE_CATCH_TIME)

                    else:
                        raise Error('Please enter valid statement')
                    continue

                # _________________________________________________________________echo mode
                if not inp.find('echo '):
                    inp = inp.replace('echo ', '', 1).strip()
                    if inp == 'on':
                        self.echo()
                        self._safe_print('^^^^^^^^^^^^^^^^Echo mode is active!^^^^^^^^^^^^^^^^')

                    elif inp == 'off':
                        self.echo(enable=False)

                    else:
                        raise Error('Please enter valid statement')
                    continue
                # _________________________________________________________________load instruction
                if not inp.find('load '):
                    inp = inp.replace('load ', '', 1).strip()
                    try:
                        if inp == '.':
                            load_all = True
                        elif inp:
                            load_all = False
                        else:
                            raise Error('Please enter valid statement')
                        self.load(inp, load_all=load_all)
                        print('^^^^^^^^^^^^^^^^Loaded successfully!^^^^^^^^^^^^^^^^')

                    except KeyError:
                        raise Error('Please enter valid statement.')
                    except ValueError:
                        raise Error('The entered name is already registered in the _database.')

                    continue
                # _________________________________________________________________save instruction
                # Set/Change current settings
                if not inp.find('save '):
                    inp = inp.replace('save ', '', 1).strip().lower()
                    try:
                        if inp == '.':
                            save_all = True
                        elif inp:
                            save_all = False
                        else:
                            raise Error('Please enter valid statement')
                        self.save(inp, save_all=save_all)

                        print('^^^^^^^^^^^^^^^^Saved successfully!^^^^^^^^^^^^^^^^')

                    except KeyError:
                        raise Error('Please enter valid statement.')
                    except ValueError:
                        raise Error('The entered name is already registered in the database.')

                    continue

                # _________________________________________________________________ls instruction
                # Set/Change current settings
                if not inp.find('ls'):

                    inp = inp.replace('ls', '', 1).strip()
                    if not inp:
                        if len(self._local_database):
                            for inx, i in enumerate(self._local_database.index):
                                print(f'{inx}: {i}')
                        else:
                            raise Error('^^^^^^^^^^^^^^^^^^^^^^Buffer is Empty ^^^^^^^^^^^^^^^^^^^^^^')

                    elif inp == 'database':
                        if len(self._database):
                            for inx, i in enumerate(self._database.index):
                                print(f'{inx}: {i}')
                        else:
                            raise Error('^^^^^^^^^^^^^^^^^^^^^^Buffer is Empty ^^^^^^^^^^^^^^^^^^^^^^')

                    elif not inp.find('pip'):
                        if len(self._pipe_database):
                            inp = inp.replace('pip', '', 1).strip()
                            if not inp:
                                for inx, i in enumerate(self._pipe_database):
                                    print(
                                        f'{inx}: {i.name}, State: {(i.pip.is_alive() and "Streamed") or (not i.pip.is_alive()) and "Stopped"}',
                                        f' Number of Stages: {len(i.pip)}')
                            else:
                                for i in self.pipe_databases:
                                    if inp == i[0]:
                                        print(
                                            f'{i.name}, State: {(i.pip.is_alive() and "Streamed") or (not i.pip.is_alive()) and "Stopped"}',
                                            f' Number of Stages: {len(i.pip)}')
                        else:
                            raise Error('^^^^^^^^^^^^^^^^^^^^^^Pipeline is Empty ^^^^^^^^^^^^^^^^^^^^^^')

                    else:
                        if len(self._local_database):
                            try:
                                self._safe_print(
                                    self._local_database.loc[inp][['Noise', 'Frame Rate', 'Sample Width', 'nchannels']])
                                if self._safe_input('Play sound? "yes"').strip().lower() == 'yes':
                                    play(self._local_database.loc[inp]['o'], inp_format='array',
                                         sampwidth=self._local_database.loc[inp]['Sample Width'],
                                         nchannels=self._local_database.loc[inp]['nchannels'],
                                         framerate=self._local_database.loc[inp]['Frame Rate'])
                            except KeyError:
                                raise Error('Please enter valid statement')
                        else:
                            raise Error('^^^^^^^^^^^^^^^^^^^^^^Buffer is Empty ^^^^^^^^^^^^^^^^^^^^^^')
                    continue
                # _________________________________________________________________set instruction
                # Set/Change current settings
                if not inp.find('set '):

                    inp = inp.replace('set ', '', 1).strip()
                    if not inp:
                        raise Error('Please enter valid statement')
                    # print(inp)
                    # Activate the primary filter
                    if 'primary filter' in inp:
                        inp = inp.replace('primary filter', '', 1).strip()
                        if not inp:
                            raise Error('Please enter valid statement')
                        # print(inp)
                        if 'on' in inp:
                            if self._primary_filter.isSet():
                                raise Error('Already activated!')
                            self._primary_filter.set()
                            inp = inp.replace('on', '', 1).strip()
                            # print(inp)
                            if '--fc' in inp:
                                # print(inp)
                                inp = inp.replace('--fc', '', 1).strip()
                                if inp.find('='):
                                    raise Error('Please enter valid statement')
                                inp = int(inp.split('=')[1].strip())
                                # print(inp)
                                assert inp < (self._frame_rate[0] / 2)
                                Process.HUMAN_VOICE_FREQ = inp
                            self._refresh()

                            self._safe_print('Ok')

                        elif 'off' in inp:
                            self._primary_filter.clear()
                            print('Ok')
                        else:
                            raise Error('Please enter valid statement')

                    elif 'pip' in inp:
                        inp = inp.replace('pip', '', 1).strip()
                        if not inp:
                            raise Error('Please enter valid statement')
                        # print(inp)
                        if 'on' in inp:

                            inp = inp.replace('on', '', 1).strip()
                            # print(inp)
                            if not inp:
                                raise Error('Please enter valid statement')
                            # name of pipeline
                            self.set_pipeline(inp)
                            self._safe_print('Ok')

                        elif 'off' in inp:
                            self.set_pipeline('', enable=False)
                            print('Ok')
                        else:
                            raise Error('Please enter valid statement')

                    # elif 'speaker recognition' in inp:
                    #     inp = inp.replace('speaker recognition', '', 1).strip()
                    #     if not inp:
                    #         raise Error('Please enter valid statement')
                    #
                    #     if 'on' in inp:
                    #         self.speaker_recognition_ev.set()
                    #         # inp = inp.replace('on', '', 1).strip()
                    #         self._safe_print('Ok')
                    #
                    #     elif 'off' in inp:
                    #         self.speaker_recognition_ev.clear()
                    #         print('Ok')
                    #     else:
                    #         raise Error('Please enter valid statement')
                    else:
                        raise Error('Please enter valid statement')
                    continue
                else:
                    raise Error('Please enter valid statement')

            except Error as msg:
                print('Error:! ', msg)

    def recorder(self, enable_compressor=True,
                 noise_sampling_duration=10,
                 record_duration=10,
                 enable_ui=True,
                 play_recorded=True,
                 catching_precision=0.1):
        # clear data queue
        self._main_stream.clear()

        if enable_ui: progress = tqdm.tqdm(range(100), f"Processing.. ")
        # Waiting for Process processing to be disabled

        if enable_ui: progress.update(5)


        # start _main _process
        # record sample for  SAMPLE_CATCH_TIME duration
        duration = int(np.round(noise_sampling_duration / self._record_period)) - 1

        if enable_ui:
            step = 40 / duration
            progress.set_description('Processing environment')

        echo = self._echo.isSet()
        if echo:
            # disable echo
            self.echo(enable=False)

        sample = [self._iwin(self._main_stream.get())]
        for i in range(duration):
            sample = np.vstack((sample, [self._iwin(self._main_stream.get())]))
            if enable_ui: progress.update(step)

        if echo:
            self.echo()
        noise = np.mean(self._vdbu(sample))

        flag = False
        if enable_ui:
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
            if echo:
                self.echo(enable=False)

            sample = [self._iwin(self._main_stream.get())]
            for i in range(duration):
                sample = np.vstack((sample, [self._iwin(self._main_stream.get())]))
                if enable_ui:  progress.update(step)

            # sample preprocessing
            if enable_compressor:
                sample = self.compressor(sample,
                                         progress,
                                         noise_base=noise + Process.SPEAK_LEVEL_GAP,
                                         final_period=catching_precision,
                                         base_period=self._record_period)
            if enable_ui: progress.update(10)
            # Waiting for Process processing to be enabled
            if play_recorded:
                if not self._mono_mode:
                    sample = self.shuffle3d_channels(sample)
                framerate = self._audio_play_wrapper(sample)

            if echo:
                self.echo()

            if enable_ui:
                if self._safe_input('\nTry again? "yes"').strip().lower() == 'yes':
                    continue
                progress.close()
                self._safe_print('Added successfully!')

            return {'o': sample,
                    'Noise': noise,
                    'Frame Rate': framerate,
                    'nchannels': self._process_nchannel,
                    'Sample Width': self._sampwidth,
                    'Base Period': self._record_period,
                    'Duraion': record_duration}

    # sample must be a 2d array
    # final_period just worked in mono mode
    def compressor(self, sample, progress, final_period=0.1, base_period=0.3, noise_base=-1e3):

        assert sample.shape[0] * base_period > 3
        progress.set_description('Processing..')
        # Break into different sections based on period time
        tmp = sample.shape[-1] * sample.shape[0]
        cols = Tools.near_divisor(tmp, final_period * sample.shape[-1] / base_period)
        progress.update(5)
        # Reshaping of arrays based on time period
        if self._mono_mode and final_period < base_period:
            sample = sample.reshape((int(tmp / cols), cols))
        progress.update(5)
        # time filter
        dbu = self._vdbu(sample)
        if self._mono_mode and final_period < base_period:
            sample = sample[dbu > noise_base]
        else:
            sample = sample[np.max(dbu, axis=1) > noise_base]

        # shape = sample.shape
        # sample = sample.reshape(shape[0] * shape[1])
        # LPF filter
        sample = self._lfilter(self._filters[1], sample)

        progress.update(5)
        # print(sample, sample.shape)
        return sample

    def _audio_play_wrapper(self, sample):
        if self._primary_filter.isSet():
            framerate = self._frame_rate[1]
        else:
            framerate = self._frame_rate[0]

        play(sample, inp_format='array', sampwidth=self._sampwidth, nchannels=self._process_nchannel, framerate=framerate)
        return framerate

    def _refresh(self):
        self._frame_rate[1] = (Process.HUMAN_VOICE_FREQ * 2)
        downsample = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
        self._frame_rate[1] = int(self._frame_rate[0] / downsample)  # human voice high frequency
        self._filters[1] = firwin(30, int(self._frame_rate[1] / 2 - 1e-6), fs=self._frame_rate[1])

        self._data_chunk = int(self._frame_rate[0] * self._record_period)
        # data type of buffer                       0
        # used in callback method(down sampling)    1
        self._constants[1] = downsample

        win_len = np.ceil(((self._data_chunk - 1) * 1 + len(self._filters[0])) / downsample)
        self._window = self._def_window(win_len)
        self._win_buffer = np.zeros(win_len)
        self._iwin_buffer = [np.zeros(win_len), np.zeros(win_len)]
        # format specifier used in struct.unpack    2
        # rms constant value                        3
        sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sampwidth)]  # 1 2 4 8


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

    def add_pipeline(self, name, pip):
        self._pipe_database.append(Process._Stream(self, pip, name, 'multiprocessing'))

    def set_pipeline(self, name, enable=True):
        if not enable:
            self._main_stream.clear()
            self._main_stream.set(self._normal_stream)
            return

        for obj in self._pipe_database:
            if name == obj.name:
                if obj.pip.is_alive():
                    self._main_stream.clear()
                    self._main_stream.set(obj)

                else:
                    raise Error(f'Error: {name} pipeline dont alive!')

    def add_record(self, name,
                   enable_compressor=True,
                   noise_sampling_duration=10,
                   record_duration=10,
                   enable_ui=True,
                   play_recorded=True,
                   catching_precision=0.1):

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
                                                      catching_precision=catching_precision)

    def echo(self, enable=True):
        if enable:
            self._main_stream.clear()
            self._echo.set()
        else:
            self._echo.clear()

    # load the name from database if defined
    def load(self, name, load_all=False):

        if load_all:
            self._local_database = self._local_database.append(self._database, verify_integrity=True)
        else:
            if name in self._local_database.index and not (name in self._database.index):
                raise ValueError
            self._local_database.loc[name] = self._database.loc[name]

    # save from local _database to stable memory
    def save(self, name, save_all=False):
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
        def __init__(self, other, obj, name=None, mode='multiprocessing'):
            self.mode = mode
            if mode == 'multiprocessing':
                obj.insert(0, other._win)
                self.put = obj.put
                self.clear = obj.clear
                self.get = obj.get
                self.pip = obj
                self.name = name

            else:
                self.put = obj.put_nowait
                self.clear = obj.queue.clear
                self.get = lambda: other._win(obj.get())

        def set(self, other):
            if isinstance(other, self.__class__):
                (self.put, self.clear, self.get) = (other.put, other.clear, other.get)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return (self.put, self.clear, self.get) == (other.put, other.clear, other.get)
            return False

        def __ne__(self, other):
            return not self.__eq__(other)

        def type(self):
            return self.mode


class Error(Exception):
    pass
