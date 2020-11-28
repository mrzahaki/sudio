# the name of allah

import wave
import pyaudio
# from pyaudio import paInt16, paInt8, paInt32, paInt24
import time
import numpy as np
import struct
from scipy.signal import upfirdn, firwin
from scipy import pi
import threading
import queue
from tools import Tools



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
            frame_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(sampwidth)]  # 1 2 4 8
            inp = inp.tolist()
            ding_data = struct.pack(f'<{len(inp)}{frame_width}', *inp)

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

        signal = wav.readframes(nframes)
        # convert the byte string into a list of ints, little-endian 16-bit samples
        signal = list(struct.unpack(f'<{nframes * ncahnnels}{frame_width}', signal))
        if not ch_index is None:
            ch_index %= ncahnnels
            signal = signal[ch_index::ncahnnels]

        if output == 'array':
            if ch_index is None:
                retval = []
                for i in range(ncahnnels):
                    retval += [signal[i::ncahnnels]]
                return np.array(retval)
            return np.array(signal)

        else:
            signal = struct.pack(f'<{len(signal)}{frame_width}', *signal)
            if output == 'frame':
                return signal
            else:
                obj = wave.open(f'{out_wave_name}.wav', 'w')
                if ch_index is None:
                    obj.setnchannels(ncahnnels)
                else:
                    obj.setnchannels(1)  # mono
                obj.setsampwidth(wav.getsampwidth())
                obj.setframerate(wav.getframerate())
                obj.writeframes(signal)
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
                obj = wave.open(f'{out_wave_name}.wav', 'w')
                obj.setnchannels(nchannels)
                obj.setsampwidth(wav.getsampwidth())
                obj.setframerate(wav.getframerate())
                obj.writeframes(signal)
                obj.close()
                return wave.open(f'{out_wave_name}.wav', 'r')

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
            for j in dev:
                print(f'index {j["index"]}: name: {j["name"]}')

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
        frames = [stream.read(data_chunk)]
        # Store data in chunks for 3 seconds
        for i in range(1, int(rate / data_chunk * record_time)):
            frames.append(stream.read(data_chunk))

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
            wf = wave.open(filename, 'r')

        else:  # output == 'array':
            wav_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(sample_width)]  # 1 2 4 8
            frame_size = int(len(frames) / nchannels)
            print(len(frames), ' ',frame_size, nchannels, wav_width, sample_width)
            signal = list(struct.unpack(f'<{frame_size}{wav_width}', frames))
            if fast_mode:
                return np.array(signal)
            return {'o': np.array(signal),
                    'sample width': sample_width,
                    'frame rate': rate,
                    'nchannels': nchannels,
                    'nframes': frame_size,
                    'dev_id': dev_id}


        @staticmethod
        # output can be a 'array','wave' or standard wave 'frame' types
        # output ncahnnels is equal to input ncahnnels
        def del_channel(wav, ch_index, output='wave', out_wave_name='sound0'):
            return Audio.set_channel(wav, ch_index, np.zeros(wav.getnframes(), dtype='int'), inp_format='array',
                                     output=output, out_wave_name=out_wave_name)

    class Recognition():
        HUMAN_VOICE_FREQ = int(8e3)
        SOUND_BUFFER_SIZE = 400
        SAMPLE_CATCH_TIME = 10
        def __init__(self, input_dev_id=None, frame_rate=48000, nchannels=2, record_period=0.3,
                     data_format=pyaudio.paInt16,
                     optimum_mono=False):

            self.recognized = []
            self.functions = []
            #self.functions = [self.main]
            self.functions.append(self.process)
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

            self.rms = np.vectorize(Tools.rms, signature='(m)->()')
            if optimum_mono:
                to_mono = np.vectorize(np.mean, signature='(m)->()')
                self.to_mono = lambda x: to_mono(x.reshape((self.data_chunk, self.nchannels)))
            else:
                self.to_mono = lambda x: x[::self.nchannels]

            self.frame_rate.append(Audio.Recognition.HUMAN_VOICE_FREQ * 2)
            downsample = int(np.round(self.frame_rate[0] / self.frame_rate[1]))
            self.frame_rate[1] = int(self.frame_rate[0] / downsample) # human voice high frequency
            self.filters = [firwin(30, Audio.Recognition.HUMAN_VOICE_FREQ, fs=self.frame_rate[0])]
            self.data_chunk = int(self.frame_rate[0] * record_period)
            # data type of buffer
            # used in callback method(down sampling)
            # format specifier used in struct.unpack
            sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self.sampwidth)]  # 1 2 4 8
            self.constants = [f'int{self.sampwidth * 8}',
                              downsample,
                              f'<{self.data_chunk * self.nchannels}{sampwidth}']
            print(self.frame_rate[0], self.frame_rate[1], self.constants[1])
            self.threads = []
            try:
                self.semaphore = [threading.Semaphore(), threading.Semaphore()]
                self.paudio = pyaudio.PyAudio()  # Create an interface to PortAudio
                self.dataq = queue.Queue(maxsize=Audio.Recognition.SOUND_BUFFER_SIZE)
                self.queue = queue.Queue()

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
            # a0 = time.perf_counter()
            in_data = np.array(struct.unpack(self.constants[2], in_data))
            in_data = self.to_mono(in_data)
            in_data = upfirdn(self.filters[0], in_data, down=self.constants[1]).astype(self.constants[0])
            self.dataq.put(in_data)

            # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
            # self.a1 = time.perf_counter()
            return None, pyaudio.paContinue

        def play(self, thread_id):
            stream_out = self.paudio.open(
                format=pyaudio.get_format_from_width(self.sampwidth),
                channels=1,
                rate=self.frame_rate[1], input=False, output=True)
            stream_out.start_stream()
            sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self.sampwidth)]  # 1 2 4 8
            # data len int(self.data_chunk/self.constants[1]) + 10
            format = f'<{int(self.data_chunk / self.constants[1]) + 10}{sampwidth}'

            while 1:
                a = self.dataq.get()
                #print(Tools.rms(a))
                print(Tools.dbu(Tools.rms(a) / ((2**(self.sampwidth*8) - 1) * 1.225)))
                #sample = self.dataq.get()
                #for i in range(int(np.round(Audio.Recognition.SAMPLE_CATCH_TIME / self.record_period)) - 1):
                #    sample = np.vstack((sample, self.dataq.get()))
                #self.preprocess(sample)
                ding_data = struct.pack(format, *a.tolist())
                stream_out.write(ding_data)

            stream_out.stop_stream()
            stream_out.close()
            pass

        # working in two mode: 'recognition'->'r' mode and 'new-user'->'n' mode
        def process(self, thread_id):
            self.play(0)
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
            try:
                while 1:
                    inp = input('>>').strip().lower()
                    if not inp.find('add '):
                        self.safe_print('^^^^^^^^^^Please Wait...^^^^^^^^^^^^^^')
                        inp = inp.replace('add ', '', 1).strip()

                        # Waiting for Recognition processing to be disabled
                        self.semaphore[1].acquire()

                        self.safe_print(f'^^^^^^^^^^Please say your name in a quiet place for a limited time\
                                         of {Audio.Recognition.SAMPLE_CATCH_TIME} seconds^^^^^^^^^^^^^^')

                        # record sample for  SAMPLE_CATCH_TIME duration
                        sample = self.dataq.get()
                        for i in range(int(np.round(Audio.Recognition.SAMPLE_CATCH_TIME / self.record_period)) - 1):
                            sample = np.vstack(sample, self.dataq.get())

                        # sample preprocessing
                        sample = self.preprocess(sample)

                        # Waiting for Recognition processing to be enabled

                        # self.recognized.append({
                        #     'name': inp
                        #     'o': sample
                        # })
                    else:
                        raise Error('Please enter valid statement')

            except Error as msg:
                print('Error:! ', msg)

        # sample must be a 2d array
        def preprocess(self, sample, final_period=0.1, base_period=0.3):
            # Break into different sections based on period time
            tmp = sample.shape[1] * sample.shape[0]
            cols = Tools.near_divisor(tmp, final_period * sample.shape[1] / base_period)
            # Reshaping of arrays based on time period
            sample = sample.reshape((int(tmp / cols), cols))
            rms = self.rms(sample)

        def safe_input(self, *args, **kwargs):
            self.semaphore[0].acquire()
            res = input(*args, **kwargs)
            self.semaphore[0].release()
            return res

        def safe_print(self, *args, **kwargs):
            self.semaphore[0].acquire(timeout=1)
            print(*args, **kwargs)
            self.semaphore[0].release()

class Error(Exception):
    pass

