import numpy as np
import pyaudio
import wave
# import pandas as pd
# from scipy.signal import upfirdn, firwin, lfilter, correlate
# from _pipeline import Pipeline


class Tools:
    @staticmethod
    # Find the number closest to num 1 that is divisible by num 2
    def near_divisible(divisible, num1):
        buf = divisible % num1, int(divisible/num1) + 1 * 6 % 20
        return divisible + [buf[0] * -1, buf[1]][buf.index(min(buf))]

    # Find the nearest integer divisor, so that the remainder of the division is zero.
    @staticmethod
    def near_divisor(num, divisor):
        div = int(np.round(num / divisor))
        res = np.array([0, div])
        while res[1] < num:
            if not num % res[1]:
                if res[0]:
                    res = (num / res)
                    div = np.abs(res - divisor)
                    return int(res[div == np.min(div)][0])
                res[0] = res[1]
                res[1] = div

            if res[0]:
                res[1] -= 1
            else:
                res[1] += 1
        raise ValueError

    @staticmethod
    def dbu(v):
        return 20 * np.log10(v / np.sqrt(.6))

    @staticmethod
    def adbu(dbu):
        return np.sqrt(.6) * 10 ** (dbu/20)

    @staticmethod
    def push(obj, data):
        obj.pop()
        obj.insert(0, data)

    @staticmethod
    def ipush(obj, data):
        obj.reverse()
        Tools.push(obj, data)
        obj.reverse()

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
    @staticmethod
    def record(dev_id=None,
               data_chunk=1024,
               output='array',
               record_time=0,
               filename="output.wav",
               data_format=8,# 16 bit
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
                print(
                    f'Index {idx}: Name: {j["name"]}, Input Channels:{j["maxInputChannels"]}, Sample Rate:{j["defaultSampleRate"]}, Host Api:{j["hostApi"]}')

            while 1:
                try:
                    dev_id = int(input('index for input dev: '))
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

        # frames = b''.join(frames)
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



