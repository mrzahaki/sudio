# the name of allah
from ._register import Members as Mem
import wave
import pyaudio
import time
import numpy as np
import struct


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

