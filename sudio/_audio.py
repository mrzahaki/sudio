# the name of allah
from ._register import Members as Mem
from ._tools import Tools
import wave
import pyaudio
import time
import numpy as np
import struct
from scipy.signal import upfirdn, firwin, get_window
import threading
import queue

# output can be a 'array','wave' or standard wave 'frame'
# note:params sampwidth, nchannels, framerate just work in frame and array type
@Mem.sudio.add
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
@Mem.sudio.add
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


# output/inp can be a 'array','wave' or standard wave 'frame' types
# output ncahnnels is equal to wav ncahnnels
# set inp as channel ch_index of wav
# inp must be mono(nchannels == 1)
@Mem.sudio.add
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
@Mem.sudio.add
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


# output can be a 'array','wave' or standard wave 'frame' types
# output ncahnnels is equal to input ncahnnels
@Mem.sudio.add
def del_channel(wav, ch_index, output='wave', out_wave_name='sound0'):
    return set_channel(wav, ch_index, np.zeros(wav.getnframes(), dtype='int'), inp_format='array',
                             output=output, out_wave_name=out_wave_name)


@Mem.sudio.add
def bark_scale(freq):
    return 13 * np.atan(freq * 76e-5) + 3.5 * np.atan((freq / 75e2) ** 2)


@Mem.sudio.add
def mel_scale(freq):
    return 2595 * np.log10(1 + freq / 700)

@Mem.sudio.add
def erb_scale(freq):
    return 24.7 * (1 + 4.37 * freq / 1000)

@Mem.sudio.add
def win_parser_mono(window, win_num):
    return window[win_num]


@Mem.sudio.add
def win_parser(window, win_num, nchannel):
    return window[nchannel][win_num]


@Mem.sudio.add
class Windowing:
    def __init__(self, mono_mode, nchannels, window_type, astype, filter_state, main_fs, primary_fc, nperseg, nhop ):
        self.main_fs = main_fs
        self.primary_fc = primary_fc
        self._astype = astype
        self._nchannels = nchannels
        self._nhop = nhop
        self._main_nperseg = self._nperseg = nperseg
        self._window_type = window_type
        self.def_filter_state = filter_state
        self._mono_mode = mono_mode

        self._frame_rate = [self.main_fs, (self.primary_fc * 2)]
        self._downsample_coef = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
        self._filter = firwin(30, self.primary_fc, fs=self._frame_rate[0])
        self._window = get_window(self._window_type, self._nperseg)
        self._nchannels_range = range(self._nchannels)

        self.refresh_ev = threading.Event()
        self.refresh_queue = queue.Queue(maxsize=2)
        self._filter_state = threading.Event()

        if self.def_filter_state:
            self._filter_state.set()

        if self._mono_mode:
            self._upfirdn = lambda h, x, const: upfirdn(h, x, down=const)
            self._win_buffer = [np.zeros(self._nperseg), np.zeros(self._nperseg)]
        else:
            self._upfirdn = np.vectorize(lambda h, x, const: upfirdn(h, x, down=const),
                                         signature='(n),(m),()->(c)')
            self._win_buffer = [[np.zeros(self._nperseg), np.zeros(self._nperseg)] for i in range(self._nchannels)]

    def init(self, other):
        pass

        # self.manager = other.manager
        # self.refresh_ev = self.manager.Event()
        # self.refresh_queue = self.manager.Queue(maxsize=2)
        # self._filter_state = self.manager.Event()




    def win_mono(self, data):
        # plt.plot(data, label='data')
        # plt.legend()
        # plt.show()
        if self.refresh_ev.is_set():
            self.refresh()
        if self._filter_state.is_set():
            data = self._upfirdn(self._filter, data, self._downsample_coef).astype(self._astype)

        # Tools.push(self.data_tst_buffer, data)
            # Tools.push(self.data_tst_buffer, data)
        retval = np.vstack((self._win_buffer[1], np.hstack((self._win_buffer[1][self._nhop:],
                                                            self._win_buffer[0][:self._nhop])))) * self._window
        Tools.push(self._win_buffer, data)
        return retval

    def win_nd(self, data):
        # print(data.shape)
        # brief: Windowing data
        # Param 'data' shape depends on number of input channels(e.g. for two channel stream, each chunk of
        # data must be with the shape of (2, chunk_size))

        # retval frame consists of two window
        # that each window have the shape same as 'data' param shape(e.g. for two channel stream:(2, 2, chunk_size))
        # note when primary_filter is enabled retval retval shape changes depend on upfirdn filter.
        # In general form retval shape is
        # (number of channels, number of windows(2), size of data chunk depend on primary_filter activity).
        if self.refresh_ev.is_set():
            self.refresh()
        if self._filter_state.is_set():
            data = self._upfirdn(self._filter, data, self._downsample_coef).astype(self._astype)

        final = []
        # Channel 0
        # final.append(np.vstack((self._win_buffer[0][1], np.hstack(
        #     (self._win_buffer[0][1][self._nhop:], self._win_buffer[0][0][:self._nhop])))) * self._window)
        # Tools.push(self._win_buffer[0], data[0])

        # range(self._nchannels)[1:]
        for i in self._nchannels_range:
            final.append(np.vstack((self._win_buffer[i][1], np.hstack(
                (self._win_buffer[i][1][self._nhop:], self._win_buffer[i][0][:self._nhop])))) * self._window)
            Tools.push(self._win_buffer[i], data[i])
        # for 2 channel win must be an 2, 2, self._data_chunk(e.g. 256)
        # reshaping may create some errors
        # return data
        return np.array(final)

    def refresh(self):
        primary_fc = self.refresh_queue.get()
        # filetr state
        if self.refresh_queue.get():
            self._filter_state.set()

            self._frame_rate[1] = (primary_fc * 2)
            self._downsample_coef = int(np.round(self._frame_rate[0] / self._frame_rate[1]))
            self._frame_rate[1] = int(self._frame_rate[0] / self._downsample_coef)  # human voice high frequency
            self._nperseg = int(np.ceil(((self._main_nperseg - 1) * 1 + len(self._filter)) / self._downsample_coef))
            self._nhop = int(self._nhop / self._main_nperseg * self._nperseg)

        elif self._filter_state.is_set():
            self._filter_state.clear()
            self._frame_rate[1] = self._frame_rate[0]
            self._nhop = int(self._nhop / self._nperseg * self._main_nperseg)
            self._nperseg = self._main_nperseg

        self._window = get_window(self._window_type, self._nperseg)
        if self._mono_mode:
            self._win_buffer = [np.zeros(self._nperseg), np.zeros(self._nperseg)]
        else:
            self._win_buffer = [[np.zeros(self._nperseg), np.zeros(self._nperseg)] for i in range(self._nchannels)]

        self.refresh_ev.clear()

    def update(self, primary_fc, filter_state):
        self.refresh_queue.put(primary_fc)
        self.refresh_queue.put(filter_state)
        self.refresh_ev.set()
