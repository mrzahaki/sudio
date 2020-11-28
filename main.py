from audio import Audio
import  numpy as np

if __name__=='__main__':
    I = 2
    if I == 1:
        # dev_id=None, data_chunk=1024, filename="output.wav", data_format=pyaudio.paInt16 ui_mode=True record_time=3
        # def play(inp, inp_format='wave', sampwidth=2, nchannels=1, framerate=44100):

        # 'o','sample width' 'frame rate': 'nchannels', 'nframes' 'dev_id'
        rec = Audio.record(output='array')
        nchannels = rec['nchannels']
        framerate = rec['frame rate']
        sampwidth = rec['sample width']
        dev_id = rec['dev_id']
        #for i in range(20):
        rec = Audio.record(output='array', dev_id=dev_id, ui_mode=False, rate=framerate, nchannels=nchannels, record_time=0.)
        Audio.play(rec, inp_format='array', nchannels=nchannels, framerate=framerate, sampwidth=sampwidth)
    if I==2:
        a = Audio.Recognition(record_period=0.1)
        a.start()
        #arr = np.linspace(2.0, 500.0, num=300)
        #arr.reshape()