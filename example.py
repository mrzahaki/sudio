from sudio import Sudio, Pipeline
import  numpy as np
import time
import pyaudio
from scipy.signal import stft
# def hac(x):
#     a= x.T.reshape(np.prod(x.shape))
#     return np.array([a])
#
# _shuffle_channels = np.vectorize(hac, signature='(m,n)->(d)')

class Tmp:
    x =0

    def fun0(x):
        # print(x.shape)
        x[0] = x[0] / 2
        return x

    def fun1(x):
        x[1] *= 4
        return x

    def update(self, x):
        Tmp.x = x

def shuffle_channels(arr):
    print(arr.shape)
    # arr = _shuffle_channels(arr)
    print(arr.shape)
    return arr.reshape(np.prod(arr.shape))

if __name__=='__main__':
    I = 3
    if I == 1:
        # dev_id=None, data_chunk=1024, filename="output.wav", data_format=pyaudio.paInt16 ui_mode=True record_time=3
        # def play(inp, inp_format='wave', sampwidth=2, nchannels=1, framerate=44100):

        # 'o','sample width' 'frame rate': 'nchannels', 'nframes' 'dev_id'
        rec = Sudio.record(output='array')
        nchannels = rec['nchannels']
        framerate = rec['frame rate']
        sampwidth = rec['sample width']
        dev_id = rec['dev_id']
        #for i in range(20):
        rec = Sudio.record(output='array', dev_id=dev_id, ui_mode=False, rate=framerate, nchannels=nchannels, record_time=0.)
        Sudio.play(rec, inp_format='array', nchannels=nchannels, framerate=framerate, sampwidth=sampwidth)
    if I==2:
        # a = np.random.rand(4, 2, 5)
        # print(shuffle_channels(a).shape)
        a = Sudio.Process(record_period=0.2, std_input_dev_id=0, mono_mode=False)
        a.start()
        # samp = a.def_new_user()
        #arr = np.linspace(2.0, 500.0, num=300)
        #arr.reshape()
    if I == 3:
        a = Sudio.Process(std_input_dev_id=0, frame_rate=48000, nchannels=2,
                 data_format=pyaudio.paInt16,
                 mono_mode=False,
                 optimum_mono=False,
                 ui_mode=True,
                 nperseg=1440,
                 noverlap=None,
                 window='hann',
                 NOLA_check=True)

        pip = Pipeline(io_buffer_size=50, pipe_type='LiveProcessing')#DeadProcessing LiveProcessing
        pip.append(Tmp.fun0)
        pip.append(Tmp.fun1)
        pip.start()
        #
        # pip.put(10)
        # print(pip.get())
        #
        # t.update(10)
        # pip.put(10)
        # print(pip.get())

        a.add_pipeline('pip0', pip)
        a.set_pipeline('pip0')
        # a.primary_filter(enable=False, fc=45000)
        a.echo()
        a.start()