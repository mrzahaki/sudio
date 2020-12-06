from ._register import Members as Mem
from ._tools import Tools
from ._audio import *
from ._pipeline import Pipeline
import tqdm
from scipy.signal import upfirdn, firwin, lfilter, correlate
import threading
import queue
import pandas as pd
import pyaudio
import numpy as np


@Mem.sudio.add
class Process():
    DATABASE_PATH = './sudio.sufile'
    HUMAN_VOICE_FREQ = int(8e3)  # Hz
    SOUND_BUFFER_SIZE = 400
    SAMPLE_CATCH_TIME = 10  # s
    SAMPLE_CATCH_PRECISION = 0.1  # s
    SPEAK_LEVEL_GAP = 10  # dbu
    _shuffle3d_channels = np.vectorize(lambda x: x.T.reshape(np.prod(x.shape)),
                                     signature='(m,n)->(c)')

    def __init__(self, input_dev_id=None, frame_rate=48000, nchannels=2, record_period=0.3,
                 data_format=pyaudio.paInt16,
                 mono_mode=True,
                 optimum_mono=False):

        self.functions = []
        self.functions = [self.main]
        self.functions.append(self.process)
        self.functions.append(self.play)
        self.record_period = record_period
        self.mono_mode = mono_mode
        if input_dev_id is None:

            rec = record(output='array', ui_mode=False)
            self.nchannels = rec['nchannels']
            self.frame_rate = [rec['frame rate']]
            self.sampwidth = rec['sample width']
            self.input_dev_id = rec['dev_id']

        else:
            self.frame_rate = [frame_rate]
            self.sampwidth = pyaudio.get_sample_size(data_format)
            self.nchannels = nchannels
            self.input_dev_id = input_dev_id
        # Vectorized dbu calculator
        self.vdbu = np.vectorize(self.rdbu, signature='(m)->()')
        if mono_mode:
            if optimum_mono:
                get_channels = np.vectorize(np.mean, signature='(m)->()')
                Process.get_channels = lambda x: get_channels(x.reshape((self.data_chunk, self.nchannels)))
            else:
                Process.get_channels = lambda x: x[::self.nchannels]
        else:
            Process.get_channels = lambda x: np.append(*[[x[i::self.nchannels]] for i in range(self.nchannels)], axis=0)

        if mono_mode:
            self.process_nchannel = 1
        else:
            self.process_nchannel = self.nchannels

        self.frame_rate.append(Process.HUMAN_VOICE_FREQ * 2)
        downsample = int(np.round(self.frame_rate[0] / self.frame_rate[1]))
        self.frame_rate[1] = int(self.frame_rate[0] / downsample)  # human voice high frequency
        self.filters = [firwin(30, Process.HUMAN_VOICE_FREQ, fs=self.frame_rate[0]),
                        firwin(30, int(self.frame_rate[1] / 2 - 1e-6), fs=self.frame_rate[1])]
        self.data_chunk = int(self.frame_rate[0] * record_period)
        # data type of buffer                       0
        # used in callback method(down sampling)    1
        # None   2
        # rms constant value                        3
        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self.sampwidth)]  # 1 2 4 8
        self.constants = [f'int{self.sampwidth * 8}',
                          downsample,
                          0,
                          ((2 ** (self.sampwidth * 8 - 1) - 1) / 1.225)]
        self.threads = []
        try:
            self.database = pd.read_pickle(Process.DATABASE_PATH, compression='xz')
        except:
            self.database = pd.DataFrame(columns=['Noise', 'Frame Rate', 'o', 'Sample Width', 'nchannels'])

        self.local_database = pd.DataFrame(columns=['Noise', 'Frame Rate', 'o', 'Sample Width', 'nchannels'])

        try:
            self.semaphore = [threading.Semaphore()]

            self.paudio = pyaudio.PyAudio()  # Create an interface to PortAudio
            self.dataq = queue.Queue(maxsize=Process.SOUND_BUFFER_SIZE)
            self.queue = queue.Queue()

            self.echo = threading.Event()
            self.speaker_recognition_ev = threading.Event()
            self.primary_filter = threading.Event()
            self.primary_filter.set()

            # create threads and queues
            for i in range(len(self.functions)):
                self.threads.append(threading.Thread(target=self.run, daemon=True, args=(len(self.threads),)))
                self.queue.put(i)

            if mono_mode:
                self.upfirdn = lambda h, x, const: upfirdn(h, x, down=const)
                self.lfilter = lambda h, x: lfilter(h, [1.0], x).astype(self.constants[0])
            else:
                # 2d mode
                self.upfirdn = np.vectorize(lambda h, x, const: upfirdn(h, x, down=const),
                             signature='(n),(m),()->(c)')

                # 3d mode
                self.lfilter = np.vectorize(lambda h, x: lfilter(h, [1.0], x).astype(self.constants[0]),
                             signature='(n),(m)->(m)')



            print('Initialization completed!')

        except:
            print('Initialization error!')
            raise

    def start(self):
        try:
            self.stream = self.paudio.open(format=pyaudio.get_format_from_width(self.sampwidth),
                                           channels=self.nchannels,
                                           rate=self.frame_rate[0],
                                           frames_per_buffer=self.data_chunk,
                                           input_device_index=self.input_dev_id,
                                           input=True,
                                           stream_callback=self.stream_callback)
            for i in self.threads:
                i.start()

            # The server blocks all active threads
            # del self.threads[:]
            self.queue.join()

        except:
            print('failed to start!')
            raise

    def run(self, th_id):
        # self.a1=0
        self.functions[self.queue.get()](th_id)
        # Indicate that a formerly enqueued task is complete.
        # Used by queue consumer threads. For each get() used to fetch a task,
        # a subsequent call to task_done() tells the queue that the processing on the task is complete.

        # If a join() is currently blocking,
        # it will resume when all items have been processed
        # (meaning that a task_done() call was received for every item that had been put() into the queue).
        self.queue.task_done()

    def stream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        in_data = Process.get_channels(np.fromstring(in_data, self.constants[0]))
        # in_data = np.fromstring(in_data, self.constants[0])
        if self.primary_filter.isSet():
            in_data = self.upfirdn(self.filters[0], in_data, self.constants[1])
        try:
            self.dataq.put_nowait(in_data.astype(self.constants[0]))
        except queue.Full:
            pass
        # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
        # self.a1 = time.perf_counter()
        return None, pyaudio.paContinue

    def play(self, thread_id):
        stream_out = self.paudio.open(
            format=pyaudio.get_format_from_width(self.sampwidth),
            channels=self.process_nchannel,
            rate=self.frame_rate[1], input=False, output=True)
        stream_out.start_stream()
        flag = self.primary_filter.isSet()
        framerate = self.frame_rate[1]
        while 1:
            self.echo.wait()
            # check primary filter state
            data = self.dataq.get()
            if not self.mono_mode:
                data = Process.shuffle2d_channels(data)
            # data = data.T.reshape(np.prod(data.shape))

            stream_out.write(data.tostring())

            # stream_out.write(self.dataq.get().tostring())
            if not self.primary_filter.isSet() == flag or not self.frame_rate[1] == framerate:
                flag = self.primary_filter.isSet()

                # close current stream
                stream_out.close()
                # create new stream
                if flag:
                    framerate = rate = self.frame_rate[1]
                else:
                    rate = self.frame_rate[0]

                stream_out = self.paudio.open(
                    format=pyaudio.get_format_from_width(self.sampwidth),
                    channels=self.process_nchannel,
                    rate=rate, input=False, output=True)
                stream_out.start_stream()
                self.dataq.queue.clear()

    # working in two mode: 'recognition'->'r' mode and 'new-user'->'n' mode
    def process(self, thread_id):
        while 1:
            try:
                self.speaker_recognition_ev.wait()
                a = 20 * np.log10(np.max(
                    correlate(self.dataq.get(), self.local_database.loc['hussein']['o'][1:4, :].reshape(4830),
                              mode='valid')))
                print(a)
            except KeyError:
                while not len(self.local_database): pass

        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self.sampwidth)]  # 1 2 4 8
        # # data len int(self.data_chunk/self.constants[1]) + 10
        # format = f'<{int(self.data_chunk / self.constants[1]) + 10}{sampwidth}'
        # # print(format)
        # while 1:
        #     self.semaphore[1].acquire()
        #     if not len(self.recognized):
        #         self.dataq.get()
        #         continue
        #
        #     ding_data = struct.pack(format, *self.dataq.get().tolist())
        #     self.semaphore[1].release()

    def main(self, thread_id):
        while 1:
            try:
                inp = input('\nmain>> ').strip().lower()

                # _________________________________________________________________add instruction
                if not inp.find('add '):
                    inp = inp.replace('add ', '', 1).strip()

                    if not inp.find('user '):
                        self.safe_print('^^^^^^^^^^Please Wait...^^^^^^^^^^^^^^')
                        inp = inp.replace('user ', '', 1).strip()
                        if inp in self.local_database.index:
                            inp = self.safe_input(
                                'The entered name is already registered in the database.\nPlease enter another name:')
                        if not inp:
                            raise Error('Please enter valid statement')
                        self.local_database.loc[inp] = self.def_new_user()
                    else:
                        raise Error('Please enter valid statement')
                    continue

                # _________________________________________________________________echo mode
                if not inp.find('echo '):
                    inp = inp.replace('echo ', '', 1).strip()
                    if inp == 'on':
                        self.dataq.queue.clear()
                        self.echo.set()
                        self.safe_print('^^^^^^^^^^^^^^^^Echo mode is active!^^^^^^^^^^^^^^^^')

                    elif inp == 'off':
                        self.echo.clear()

                    else:
                        raise Error('Please enter valid statement')
                    continue
                # _________________________________________________________________load instruction
                if not inp.find('load '):
                    inp = inp.replace('load ', '', 1).strip()
                    try:
                        if inp == '.':
                            self.local_database = self.local_database.append(self.database, verify_integrity=True)
                        elif inp:
                            if inp in self.local_database.index:
                                raise ValueError
                            self.local_database.loc[inp] = self.database.loc[inp]
                        else:
                            raise Error('Please enter valid statement')

                        self.database.to_pickle(Process.DATABASE_PATH)
                        print('^^^^^^^^^^^^^^^^Loaded successfully!^^^^^^^^^^^^^^^^')

                    except KeyError:
                        raise Error('Please enter valid statement.')
                    except ValueError:
                        raise Error('The entered name is already registered in the database.')

                    continue
                # _________________________________________________________________save instruction
                # Set/Change current settings
                if not inp.find('save '):
                    inp = inp.replace('save ', '', 1).strip().lower()
                    try:
                        if inp == '.':
                            self.database = self.database.append(self.local_database, verify_integrity=True)
                        elif inp:
                            if inp in self.database.index:
                                raise ValueError
                            self.database.loc[inp] = self.local_database.loc[inp]
                        else:
                            raise Error('Please enter valid statement')

                        self.database.to_pickle(Process.DATABASE_PATH, compression='xz')
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
                        if len(self.local_database):
                            for inx, i in enumerate(self.local_database.index):
                                print(f'{inx}: {i}')
                        else:
                            raise Error('^^^^^^^^^^^^^^^^^^^^^^Buffer is Empty ^^^^^^^^^^^^^^^^^^^^^^')

                    elif inp == 'database':
                        if len(self.database):
                            for inx, i in enumerate(self.database.index):
                                print(f'{inx}: {i}')
                        else:
                            raise Error('^^^^^^^^^^^^^^^^^^^^^^Buffer is Empty ^^^^^^^^^^^^^^^^^^^^^^')
                    else:
                        if len(self.local_database):
                            try:
                                self.safe_print(
                                    self.local_database.loc[inp][['Noise', 'Frame Rate', 'Sample Width', 'nchannels']])
                                if self.safe_input('Play sound? "yes"').strip().lower() == 'yes':
                                    play(self.local_database.loc[inp]['o'], inp_format='array',
                                         sampwidth=self.local_database.loc[inp]['Sample Width'],
                                         nchannels=self.local_database.loc[inp]['nchannels'],
                                         framerate=self.local_database.loc[inp]['Frame Rate'])
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
                            self.primary_filter.set()
                            inp = inp.replace('on', '', 1).strip()
                            # print(inp)
                            if '--fc' in inp:
                                # print(inp)
                                inp = inp.replace('--fc', '', 1).strip()
                                if inp.find('='):
                                    raise Error('Please enter valid statement')
                                inp = int(inp.split('=')[1].strip())
                                # print(inp)
                                assert inp < (self.frame_rate[0] / 2)
                                Process.HUMAN_VOICE_FREQ = inp
                                self.refresh()

                            self.safe_print('Ok')

                        elif 'off' in inp:
                            self.primary_filter.clear()
                            print('Ok')
                        else:
                            raise Error('Please enter valid statement')

                    elif 'speaker recognition' in inp:
                        inp = inp.replace('speaker recognition', '', 1).strip()
                        if not inp:
                            raise Error('Please enter valid statement')

                        if 'on' in inp:
                            self.speaker_recognition_ev.set()
                            # inp = inp.replace('on', '', 1).strip()
                            self.safe_print('Ok')

                        elif 'off' in inp:
                            self.speaker_recognition_ev.clear()
                            print('Ok')
                        else:
                            raise Error('Please enter valid statement')
                    else:
                        raise Error('Please enter valid statement')
                    continue
                else:
                    raise Error('Please enter valid statement')

            except Error as msg:
                print('Error:! ', msg)

    def def_new_user(self):
        # clear data queue
        self.dataq.queue.clear()

        progress = tqdm.tqdm(range(100), f"Processing.. ")
        # Waiting for Process processing to be disabled
        speaker_recognition_ev = self.speaker_recognition_ev.isSet()
        if speaker_recognition_ev:
            self.speaker_recognition_ev.clear()
        progress.update(5)

        # start main process
        # record sample for  SAMPLE_CATCH_TIME duration
        duration = int(np.round(Process.SAMPLE_CATCH_TIME / self.record_period)) - 1
        step = 40 / duration

        progress.set_description('Processing environment')
        echo = self.echo.isSet()
        if echo:
            self.echo.clear()

        sample = [self.dataq.get()]
        for i in range(duration):
            sample = np.vstack((sample, [self.dataq.get()]))
            progress.update(step)

        if echo:
            self.echo.set()

        noise = np.mean(self.vdbu(sample))
        progress.update(5)

        while 1:
            # reset progress
            progress.n = 50
            progress.last_print_n = 50
            progress.start_t = time.time()
            progress.last_print_t = time.time()
            progress.update()

            self.safe_print(
                f'\n^^^^^^^^^^Please say your name in a quiet place for a limited time of {Process.SAMPLE_CATCH_TIME} seconds^^^^^^^^^^^^^^')
            if self.safe_input(f'Are you ready? enter "yes"\nNoise Level: {noise} dbu').strip().lower() == 'yes':
                # record sample for  SAMPLE_CATCH_TIME duration
                # free dataq before process
                self.dataq.queue.clear()

                step = 30 / duration
                progress.set_description('Recording the conversation..')

                if echo:
                    self.echo.clear()

                sample = [self.dataq.get()]
                for i in range(duration):
                    sample = np.vstack((sample, [self.dataq.get()]))
                    progress.update(step)

                # sample preprocessing
                sample = self.preprocess(sample,
                                         progress,
                                         noise_base=noise + Process.SPEAK_LEVEL_GAP,
                                         final_period=Process.SAMPLE_CATCH_PRECISION,
                                         base_period=self.record_period)
                progress.update(10)
                progress.refresh()
                # Waiting for Process processing to be enabled
                if not self.mono_mode:
                    sample = self.shuffle3d_channels(sample)
                framerate = self.audio_play_wrapper(sample)

                if echo:
                    self.dataq.queue.clear()
                    self.echo.set()

                if self.safe_input('\nTry again? "yes"').strip().lower() == 'yes':
                    continue

            progress.close()
            self.safe_print('Added successfully!')
            if speaker_recognition_ev:
                self.speaker_recognition_ev.set()

            return {'o': sample,
                    'Noise': noise,
                    'Frame Rate': framerate,
                    'nchannels': self.process_nchannel,
                    'Sample Width': self.sampwidth}

    # sample must be a 2d array
    # final_period just worked in mono mode
    def preprocess(self, sample, progress, final_period=0.1, base_period=0.3, noise_base=-1e3):

        assert sample.shape[0] * base_period > 3
        progress.set_description('Processing..')
        # Break into different sections based on period time
        tmp = sample.shape[-1] * sample.shape[0]
        cols = Tools.near_divisor(tmp, final_period * sample.shape[-1] / base_period)
        progress.update(5)
        # Reshaping of arrays based on time period
        if self.mono_mode:
            sample = sample.reshape((int(tmp / cols), cols))
        progress.update(5)
        # time filter
        dbu = self.vdbu(sample)
        if self.mono_mode:
            sample = sample[dbu > noise_base]
        else:
            sample = sample[np.max(dbu, axis=1) > noise_base]

        # shape = sample.shape
        # sample = sample.reshape(shape[0] * shape[1])
        # LPF filter
        sample = self.lfilter(self.filters[1], sample)

        progress.update(5)
        # print(sample, sample.shape)
        return sample

    def audio_play_wrapper(self, sample):
        if self.primary_filter.isSet():
            framerate = self.frame_rate[1]
        else:
            framerate = self.frame_rate[0]

        play(sample, inp_format='array', sampwidth=self.sampwidth, nchannels=self.process_nchannel, framerate=framerate)
        return framerate

    def refresh(self):
        self.frame_rate[1] = (Process.HUMAN_VOICE_FREQ * 2)
        downsample = int(np.round(self.frame_rate[0] / self.frame_rate[1]))
        self.frame_rate[1] = int(self.frame_rate[0] / downsample)  # human voice high frequency
        self.filters[1] = firwin(30, int(self.frame_rate[1] / 2 - 1e-6), fs=self.frame_rate[1])

        self.data_chunk = int(self.frame_rate[0] * self.record_period)
        # data type of buffer                       0
        # used in callback method(down sampling)    1
        self.constants[1] = downsample
        # format specifier used in struct.unpack    2
        # rms constant value                        3
        sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self.sampwidth)]  # 1 2 4 8

    def safe_input(self, *args, **kwargs):
        self.semaphore[0].acquire()
        res = input(*args, **kwargs)
        self.semaphore[0].release()
        return res

    def safe_print(self, *args, **kwargs):
        self.semaphore[0].acquire(timeout=1)
        print(*args, **kwargs)
        self.semaphore[0].release()

    # RMS to dbu
    # signal maps to standard +4 dbu
    def rdbu(self, arr):
        return Tools.dbu(np.max(arr) / self.constants[3])

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


class Error(Exception):
    pass
