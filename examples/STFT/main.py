from sudio import Master
from stft import init

if __name__ == '__main__':
    import os
    os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'

    # create new Master object
    master = Master(mono_mode=True, nperseg=4000)
    # start master kernel
    master.start()

    # initial STFT resources
    stft = init(master)
    # start STFT thread
    stft.start()

    # enable STDIN(As default system microphone)
    master.unmute()


