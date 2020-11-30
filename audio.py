# the name of allah

import tqdm
import wave
import pyaudio
# from pyaudio import paInt16, paInt8, paInt32, paInt24
import time
import numpy as np
import struct
from scipy.signal import upfirdn, firwin, lfilter, correlate
from scipy import pi
import threading
import queue
from tools import Tools
import pandas as pd



class Audio():

    @staticmethod
    # output can be a 'array','wave' or standard wave 'frame'
    # note:params sampwidth, nchannels, framerate just work in frame and array type
    def play(inp: object, inp_format: object = 'wave', sampwidth: object = 2, nchannels: object = 1, framerate: object = 44100) -> object:
        if inp_format == 'wave':
            inp.rewind()
            ding_data = inp.readframes(inp.getnframes())
            sampwidth = inp.getsampwidth()
            nchannels = inp.getnchannels()
            framerate = inp.getframerate()

        elif inp_format == 'frame':
            ding_data = inp

        else:
            ding_data = inp.astype(f'int{sampwidth * 8}').tostring()


        audio = pyaudio.PyAudio()
        stream_out = audio.open(
            format=audio.get_format_from_width(sampwidth),
            channels=nchannels,
            rate=framerate, input=False, output=True)
        # print(audio.get_format_from_width(inp.getsampwidth()))
        stream_out.start_stream()
        stream_out.write(ding_data)

        # time.sleep(0.2)

        stream_out.stop_stream()
        stream_out.close()
        audio.terminate()

    @staticmethod
    # Format: h is short in C type
    # Format: l is long in C type
    # '?' -> _BOOL , 'h' -> short, 'i' -> int
    # c -> char,
    # s -> string example '3s'-> b'sbs'
    # f -> float, q-> long long
    # @  Byte order->native, Size->native, Alignment->native,
    # =  Byte order->native, Size->standard, Alignment->none,
    # <  Byte order->little-endian, Size->standard, Alignment->none,
    # >  Byte order->big-endian, Size->standard, Alignment->none,
    # !  Byte order->network (= big-endian), Size->standard, Alignment->none,
    # A format character may be preceded by an integral repeat count.
    #  For example, the format string '4h' means exactly the same as 'hhhh'.
    # https://stackoverflow.com/questions/23154400/read-the-data-of-a-single-channel-from-a-stereo-wave-file-in-python
    # output can be a 'array','wave' or standard wave 'frame'
    # output ncahnnels is equal to input ncahnnels
    # get all channels
    def get_channel(wav, ch_index=None, output='wave', out_wave_name='sound0'):

        nframes = wav.getnframes()
        ncahnnels = wav.getnchannels()

        frame_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(wav.getsampwidth())]  # 1 2 4 8
        wav.rewind()

        frame = wav.readframes(nframes)
        # convert the byte string into a list of ints, little-endian 16-bit samples
        signal = list(struct.unpack(f'<{nframes * ncahnnels}{frame_width}', frame))
        if not ch_index is None:
            ch_index %= ncahnnels
            # np.fromstring(signal, )
            signal = signal[ch_index::ncahnnels]

        if output == 'array':
            if ch_index is None:
                retval = []
                for i in range(ncahnnels):
                    retval += [signal[i::ncahnnels]]
                return np.array(retval)
            return np.array(signal)

        elif output == 'frame':
                return frame
        else:
            obj = wave.open(f'{out_wave_name}.wav', 'w')
            if ch_index is None:
                obj.setnchannels(ncahnnels)
            else:
                obj.setnchannels(1)  # mono
            obj.setsampwidth(wav.getsampwidth())
            obj.setframerate(wav.getframerate())
            obj.writeframes(frame)
            obj.close()

    @staticmethod
    # output/inp can be a 'array','wave' or standard wave 'frame' types
    # output ncahnnels is equal to wav ncahnnels
    # set inp as channel ch_index of wav
    # inp must be mono(nchannels == 1)
    def set_channel(wav, ch_index, inp, inp_format='wave', output='wave', out_wave_name='sound0'):
        wav_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(wav.getsampwidth())]  # 1 2 4 8

        if inp_format == 'wave':
            assert (inp.getsampwidth() == wav.getsampwidth() and
                    inp.getnframes() == wav.getnframes() and
                    inp.getnchannels() == 1 and
                    inp.getframerate == wav.getframerate())
            inp.rewind()
            signal = inp.readframes(inp.getnframes())
            frame_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(inp.getsampwidth())]  # 1 2 4 8
            assert wav_width == frame_width

            # convert the byte string into a list of ints, little-endian 16-bit samples
            signal = list(struct.unpack(f'<{inp.getnframes()}{frame_width}',
                                        signal))

        else:
            assert (len(inp) == wav.getnframes())

            if inp_format == 'frame':
                # convert the byte string into a list of ints, little-endian 16-bit samples
                signal = list(struct.unpack(f'<{wav.getnframes()}{wav_width}',
                                            inp))

            else:  # np array
                signal = inp.tolist()

        nchannels = wav.getnchannels()
        assert nchannels > 1
        ch_index %= nchannels

        wav.rewind()
        wav_signal = wav.readframes(wav.getnframes())
        # convert the byte string into a list of ints, little-endian 16-bit samples
        wav_signal = list(struct.unpack(f'<{wav.getnframes() * nchannels}{wav_width}',
                                        wav_signal))

        print(len(wav_signal), len(signal))
        for i, j in zip(range(ch_index, len(wav_signal), nchannels), range(ch_index, len(signal))):
            wav_signal[i] = signal[j]

        if output == 'array':
            return np.array(wav_signal)

        else:
            signal = struct.pack(f'<{len(wav_signal)}{wav_width}', *wav_signal)
            if output == 'frame':
                return signal
            else:
                obj = wave.open(f'./sounds/{out_wave_name}.wav', 'w')
                obj.setnchannels(nchannels)
                obj.setsampwidth(wav.getsampwidth())
                obj.setframerate(wav.getframerate())
                obj.writeframes(signal)
                obj.close()
                return wave.open(f'./sounds/{out_wave_name}.wav', 'r')

    @staticmethod
    # Record in chunks of 1024 samples
    # Record for record_time seconds(if record_time==0 then it just return one chunk of data)
    # output/inp can be a 'array','wave' or standard wave 'frame' types
    # data_format:
    # pyaudio.paInt16 = 8     16 bit int
    # pyaudio.paInt24 = 4     24 bit int
    # pyaudio.paInt32 = 2     32 bit int
    # pyaudio.paInt8 = 16     8 bit int
    # if output is equal to 'array' return type is a dictionary that contains signal properties
    # in fast mode(ui_mode=False) you must enter dev_id(input device id) and nchannels and rate params
    def record(dev_id=None,
               data_chunk=1024,
               output='array',
               record_time=0,
               filename="output.wav",
               data_format=pyaudio.paInt16,
               ui_mode=True,
               nchannels=2,
               rate=48000,
               fast_mode=False):

        p = pyaudio.PyAudio()  # Create an interface to PortAudio
        dev = []
        if dev_id is None:
            for i in range(p.get_device_count()):
                tmp = p.get_device_info_by_index(i)
                if tmp['maxInputChannels'] > 0:
                    dev.append(tmp)
            assert len(dev) > 0
            print('please choose input device from index :')
            for idx, j in enumerate(dev):
                print(f'Index {idx}: Name: {j["name"]}, Input Channels:{j["maxInputChannels"]}, Sample Rate:{j["defaultSampleRate"]}, Host Api:{j["hostApi"]}')

            while 1:
                try:
                    dev_id = int(input('index: '))
                    break
                except:
                    print('please enter valid index!')

            rate = int(dev[dev_id]['defaultSampleRate'])
            nchannels = dev[dev_id]['maxInputChannels']

        if ui_mode:
            print('Recording...')

        stream = p.open(format=data_format,
                        channels=nchannels,
                        rate=rate,
                        frames_per_buffer=data_chunk,
                        input_device_index=dev_id,
                        input=True)

        # Initialize array to store frames
        frames = stream.read(data_chunk)
        # Store data in chunks for 3 seconds
        for i in range(1, int(rate / data_chunk * record_time)):
            frames += stream.read(data_chunk)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        if ui_mode:
            print('Finished recording')

        frames = b''.join(frames)
        sample_width = p.get_sample_size(data_format)

        if output == 'frame':
            return frames

        elif output == 'wave':
            # Save the recorded data as a WAV file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(nchannels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(frames)
            wf.close()
            return wave.open(filename, 'r')

        else:  # output == 'array':

            signal = np.fromstring(frames, f'int{sample_width * 8}')
            if fast_mode:
                return signal
            return {'o': signal,
                    'sample width': sample_width,
                    'frame rate': rate,
                    'nchannels': nchannels,
                    'dev_id': dev_id}

    @staticmethod
    # output can be a 'array','wave' or standard wave 'frame' types
    # output ncahnnels is equal to input ncahnnels
    def del_channel(wav, ch_index, output='wave', out_wave_name='sound0'):
        return Audio.set_channel(wav, ch_index, np.zeros(wav.getnframes(), dtype='int'), inp_format='array',
                                 output=output, out_wave_name=out_wave_name)

    class Process():
        DATABASE_PATH = './sounds/data.pickle'
        HUMAN_VOICE_FREQ = int(8e3) #Hz
        SOUND_BUFFER_SIZE = 400
        SAMPLE_CATCH_TIME = 10 #s
        SAMPLE_CATCH_PRECISION = 0.1  # s
        SPEAK_LEVEL_GAP = 10 #dbu

        def __init__(self, input_dev_id=None, frame_rate=48000, nchannels=2, record_period=0.3,
                     data_format=pyaudio.paInt16,
                     optimum_mono=False):

            self.functions = []
            self.functions = [self.main]
            self.functions.append(self.speaker_recognition)
            self.functions.append(self.play)
            self.record_period = record_period

            if input_dev_id is None:
                rec = Audio.record(output='array', ui_mode=False)
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
            if optimum_mono:
                to_mono = np.vectorize(np.mean, signature='(m)->()')
                self.to_mono = lambda x: to_mono(x.reshape((self.data_chunk, self.nchannels)))
            else:
                self.to_mono = lambda x: x[::self.nchannels]

            self.frame_rate.append(Audio.Process.HUMAN_VOICE_FREQ * 2)
            downsample = int(np.round(self.frame_rate[0] / self.frame_rate[1]))
            self.frame_rate[1] = int(self.frame_rate[0] / downsample) # human voice high frequency
            self.filters = [firwin(30, Audio.Process.HUMAN_VOICE_FREQ, fs=self.frame_rate[0]), firwin(30, int(self.frame_rate[1]/2 - 1e-6), fs=self.frame_rate[1])]
            self.data_chunk = int(self.frame_rate[0] * record_period)
            # data type of buffer                       0
            # used in callback method(down sampling)    1
            # None   2
            # rms constant value                        3
            sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self.sampwidth)]  # 1 2 4 8
            self.constants = [f'int{self.sampwidth * 8}',
                              downsample,
                              0,
                              ((2 ** (self.sampwidth * 8 - 1) - 1) / 1.225)]
            self.threads = []
            try:
                self.database = pd.read_pickle(Audio.Process.DATABASE_PATH)
            except:
                self.database = pd.DataFrame(columns=['Noise', 'Frame Rate', 'o', 'Sample Width'])

            self.local_database = pd.DataFrame(columns=['Noise', 'Frame Rate', 'o', 'Sample Width'])

            try:
                self.semaphore = [threading.Semaphore()]

                self.paudio = pyaudio.PyAudio()  # Create an interface to PortAudio
                self.dataq = queue.Queue(maxsize=Audio.Process.SOUND_BUFFER_SIZE)
                self.queue = queue.Queue()

                self.echo = threading.Event()
                self.speaker_recognition_ev = threading.Event()
                self.primary_filter = threading.Event()
                self.primary_filter.set()

                # create threads and queues
                for i in range(len(self.functions)):
                    self.threads.append(threading.Thread(target=self.run, daemon=True, args=(len(self.threads),)))
                    self.queue.put(i)
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

        def stream_callback(self, in_data, frame_count,  time_info, status):  # PaCallbackFlags

            in_data = self.to_mono(np.fromstring(in_data, self.constants[0]))
            if self.primary_filter.isSet():
                in_data = upfirdn(self.filters[0], in_data, down=self.constants[1])
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
                channels=1,
                rate=self.frame_rate[1], input=False, output=True)
            stream_out.start_stream()
            flag = self.primary_filter.isSet()
            framerate = self.frame_rate[1]
            while 1:
                self.echo.wait()
                # check primary filter state
                stream_out.write(self.dataq.get().tostring())
                if not self.primary_filter.isSet() == flag or not self.frame_rate[1] == framerate:
                    flag = self.primary_filter.isSet()
                    print('yes')
                    # close current stream
                    stream_out.close()
                    # create new stream
                    if flag:
                        framerate = rate = self.frame_rate[1]
                    else:
                        rate = self.frame_rate[0]

                    stream_out = self.paudio.open(
                        format=pyaudio.get_format_from_width(self.sampwidth),
                        channels=1,
                        rate=rate, input=False, output=True)
                    stream_out.start_stream()
                    self.dataq.queue.clear()

        # working in two mode: 'recognition'->'r' mode and 'new-user'->'n' mode
        def speaker_recognition(self, thread_id):
            while 1:
                try:
                    self.speaker_recognition_ev.wait()
                    a = 20 * np.log10(np.max(correlate(self.dataq.get(), self.local_database.loc['hussein']['o'][1:4, :].reshape(4830), mode='valid')))
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
                    inp = input('\nProcess>> ').strip().lower()

                    # _________________________________________________________________add instruction
                    if not inp.find('add '):
                        inp = inp.replace('add ', '', 1).strip()

                        if not inp.find('user '):
                            self.safe_print('^^^^^^^^^^Please Wait...^^^^^^^^^^^^^^')
                            inp = inp.replace('user ', '', 1).strip()
                            if inp in self.local_database.index:
                                    inp = self.safe_input('The entered name is already registered in the database.\nPlease enter another name:')
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

                            self.database.to_pickle(Audio.Process.DATABASE_PATH)
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

                            self.database.to_pickle(Audio.Process.DATABASE_PATH)
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
                                    self.safe_print(self.local_database.loc[inp][['Noise', 'Frame Rate', 'Sample Width']])
                                    if self.safe_input('Play sound? "yes"').strip().lower() == 'yes':
                                        Audio.play(self.local_database.loc[inp]['o'], inp_format='array',
                                                   sampwidth=self.local_database.loc[inp]['Sample Width'], nchannels=1,
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
                                    Audio.Process.HUMAN_VOICE_FREQ = inp
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
            duration = int(np.round(Audio.Process.SAMPLE_CATCH_TIME / self.record_period)) - 1
            step = 40 / duration

            progress.set_description('Processing environment')
            echo = self.echo.isSet()
            if echo:
                self.echo.clear()

            sample = self.dataq.get()
            for i in range(duration):
                sample = np.vstack((sample, self.dataq.get()))
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
                    f'\n^^^^^^^^^^Please say your name in a quiet place for a limited time of {Audio.Process.SAMPLE_CATCH_TIME} seconds^^^^^^^^^^^^^^')
                if self.safe_input(f'Are you ready? enter "yes"\nNoise Level: {noise} dbu').strip().lower() == 'yes':
                    # record sample for  SAMPLE_CATCH_TIME duration
                    # free dataq before process
                    self.dataq.queue.clear()

                    step = 30 / duration
                    progress.set_description('Recording the conversation..')

                    if echo:
                        self.echo.clear()

                    sample = self.dataq.get()
                    for i in range(duration):
                        sample = np.vstack((sample, self.dataq.get()))
                        progress.update(step)

                    # sample preprocessing
                    sample = self.preprocess(sample,
                                             progress,
                                             noise_base=noise + Audio.Process.SPEAK_LEVEL_GAP,
                                             final_period=Audio.Process.SAMPLE_CATCH_PRECISION,
                                             base_period=self.record_period)
                    progress.update(10)
                    progress.refresh()
                    # Waiting for Process processing to be enabled
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
                        'Sample Width': self.sampwidth}

        # sample must be a 2d array
        def preprocess(self, sample, progress, final_period=0.1, base_period=0.3, noise_base=-1e3):

            progress.set_description('Processing..')
            # Break into different sections based on period time
            tmp = sample.shape[1] * sample.shape[0]
            cols = Tools.near_divisor(tmp, final_period * sample.shape[1] / base_period)
            progress.update(5)
            # Reshaping of arrays based on time period
            sample = sample.reshape((int(tmp / cols), cols))
            progress.update(5)

            # time filter
            dbu = self.vdbu(sample)
            sample = sample[dbu > noise_base]

            shape = sample.shape
            sample = sample.reshape(shape[0] * shape[1])
            # LPF filter
            sample = lfilter(self.filters[1], [1.0], sample).astype(self.constants[0]).reshape(shape)
            #sample = upfirdn(self.filters[1], sample).astype(self.constants[0])
            progress.update(5)

            # print(sample, sample.shape)
            return sample

        def audio_play_wrapper(self, sample):
            if self.primary_filter.isSet():
                framerate = self.frame_rate[1]
            else:
                framerate = self.frame_rate[0]
            Audio.play(sample, inp_format='array', sampwidth=self.sampwidth, nchannels=1, framerate=framerate)
            return framerate

        def refresh(self):
            self.frame_rate[1] = (Audio.Process.HUMAN_VOICE_FREQ * 2)
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

class Error(Exception):
    pass

class TEMP_ERROR(Exception):
    pass
