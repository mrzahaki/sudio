import os
# os.environ["KIVY_NO_CONSOLELOG"] = "1"

from win32api import GetSystemMetrics
from sudio import Sudio, Pipeline
import numpy as np
from scipy.fft import fft
from scipy.signal import lfilter, iirfilter
import queue
import threading
import time

from kivy.garden.knob import Knob
from kivy.graphics import Color, Ellipse, Line
from kivy.app import App
from kivy.properties import NumericProperty, ListProperty, ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.config import Config
# from kivy.core.window import Window
# from math import cos, sin

class Gfft(FloatLayout):

    LPF_CUTOFF_FREQUENCY = .9e3
    faxislay = ObjectProperty(None)
    points = ListProperty([])
    dt = NumericProperty(0)
    _update_points_animation_ev = None
    xpos_start = NumericProperty(0)
    xpos_end = NumericProperty(0)
    lpf_cutoff_frequency = NumericProperty(LPF_CUTOFF_FREQUENCY)
    y_offset = NumericProperty(0)
    y_pos_start = NumericProperty(0)
    y_pos_end = NumericProperty(0)
    outline_border = NumericProperty(0)

    freq_span_knob = ObjectProperty()
    freq_move_knob = ObjectProperty()



    def __init__(self, sudio, rwidth, rheight, max_buffer_size=100, **kwargs):
        super().__init__(**kwargs)

        # print(self.width, self.height)
        # index 1: used for knob update
        self.knob_update_buffer = {self.freq_span_knob:[], self.freq_move_knob:[]}
        self.knob_update_functions = {self.freq_span_knob: self.frequency_span_update, self.freq_move_knob: self.frequency_move_update}
        self.dataqueue = queue.Queue(maxsize=max_buffer_size)
        self.data_control_qeue = queue.Queue(maxsize=2)
        self.rwidth = rwidth
        self.rheight = rheight

        #____________________________constants
        self.frequency_axis_size = 25
        weight_axis_size = 10
        self.outline_border = 20
        self.xpos_start = 23
        self.xpos_end = self.rwidth - 3
        self.y_pos_start = 100
        self.y_pos_end = self.rheight-15
        self.y_offset = self.y_pos_start + 310
        self.wy_lable_start = self.y_pos_start
        # __________________freq_span
        self.knob_update_sample_number = 3# uint: 3 is max precision{3 to inf} must be odd number
        self.current_view_start = 0
        self.current_view_end = int(sudio.nperseg / 2)
        self.full_view_width = int(sudio.nperseg / 2)


        self.constant = [range(self.frequency_axis_size)]
        #________________________________________________________ calculate and update of weight axis
        wlabel = np.linspace(0, 100, num=weight_axis_size, endpoint=False)
        wlabel_x = np.full((weight_axis_size,), fill_value=(self.xpos_start - self.outline_border + 4))
        wpoint_x = np.full((weight_axis_size,), fill_value=(self.xpos_start))
        wlabel_y = np.linspace(self.wy_lable_start, self.y_pos_end, num=weight_axis_size, endpoint=False)
        weight_axis_points = np.vstack((wlabel_x, wlabel_y)).T.tolist()
        weight_line_points = np.vstack((wpoint_x, wlabel_y)).T.tolist()

        for i in range(weight_axis_size):
            label = Label(  text="{:.0f}".format(wlabel[i]),
                            font_size='9sp',
                            text_size=(self.rwidth, self.rheight),
                            pos=weight_axis_points[i])
            self.add_widget(label)

        self.waxis_color = Color(.5, .5, .5, .2)
        # self.xaxis_color.a=.8
        self.canvas.add(self.waxis_color)

        for i in range(weight_axis_size):
            line = Line(width=1,
                        points=(weight_line_points[i], (self.xpos_end, weight_line_points[i][1])))

            self.canvas.add(line)
        #________________________________________________________ calculate and update of frequency axis
        self.x_axis = np.linspace(self.xpos_start, self.xpos_end, num=self.full_view_width)[self.current_view_start: self.current_view_end]
        # self.frequency_axis = np.linspace(0, 1, num=self.frequency_axis_size, endpoint=False)
        # print(self.frequency_axis_update)

        self.flabel_y = np.full((self.frequency_axis_size,), fill_value=(self.y_pos_start-self.outline_border + 4))
        self.fpoint_y = np.full((self.frequency_axis_size,), fill_value=(self.y_pos_start))
        self.flabel_x = self.x_axis[::int(np.ceil(self.full_view_width / self.frequency_axis_size))]
        self.frequency_axis_points = np.vstack((self.flabel_x, self.flabel_y)).T.tolist()
        self.frequency_line_points = np.vstack((self.flabel_x, self.fpoint_y)).T.tolist()

        self.xlabels = []
        self.xlines = []

        for i in self.constant[0]:
            label = Label(  font_size='9sp',
                            text_size=(self.rwidth, self.rheight),
                            pos=self.frequency_axis_points[i])
            self.xlabels.append(label)
            self.add_widget(label)
            line = Line(width=1,
                        points=(self.frequency_line_points[i], (self.frequency_line_points[i][0], self.y_pos_end)))
            self.xlines.append(line)

        self.frequency_axis_update(self.current_view_start, self.current_view_end, fs=sudio.sample_rate)

        self.xaxis_color = Color(.5, .5, .5, .2)
        # self.xaxis_color.a=.8
        self.canvas.add(self.xaxis_color)

        for i in self.constant[0]:
            self.canvas.add(self.xlines[i])

        # map(self.add_widget, self.frequency_axis_label)
        # print(self.x_axis)
        # self.y_axis = np.zeros(sudio.nperseg)
        self.lpf_coef = iirfilter(5, Gfft.LPF_CUTOFF_FREQUENCY, btype='lowpass', fs=sudio.sample_rate)

        self.sudio = sudio
        pip = Pipeline(io_buffer_size=500, pipe_type='LiveProcessing')  # DeadProcessing LiveProcessing
        pip.start()
        pip.append(self.mono_fft_process)
        self.pip = pip
        self.start_flg = True

    def knob_update(self, *data):

        t = time.time()
        tmp = self.knob_update_buffer[data[0]]
        if len(tmp) == (self.knob_update_sample_number-1):
            buf = [data[1], t]
            velocity = np.array([])
            # process data, tmp is full
            # knob / sec
            for i in tmp[::-1]:
                velocity = np.append(velocity, (buf[0] - i[0]) / (buf[1] - i[1]))
                buf = i
            tmp.clear()
            if velocity.prod() < 0:
                return
            velocity = int(velocity.mean())
            # print(data)
            # velocity = velocity.__abs__().mean() * ((velocity.prod() > 0) * 2 -1) * ((velocity[0] > 0) * 2 -1)
            self.knob_update_functions[data[0]](velocity)

        else:
            tmp.append([data[1], t])


    def frequency_axis_update(self, freq_start, freq_width, fs=1):
        # freq_width = 1 :pi
        # freq_start = 0 :pi
        self.frequency_axis = np.linspace(freq_start / self.full_view_width, (freq_start + freq_width) / self.full_view_width,
                                          num=self.frequency_axis_size, endpoint=False) * fs / 2
        tmp = (self.frequency_axis >= 1e3)
        self.frequency_axis[tmp] /= 1e3
        tmp = np.where(tmp)[0][0]
        txt = "{:.2f}"
        # flabel_x = self.x_axis[::int(np.ceil(self.full_view_width / self.frequency_axis_size))]
        for i in self.constant[0]:
            if i == tmp:
                txt += 'k'
            self.xlabels[i].text = txt.format(self.frequency_axis[i])
            self.xlabels[i].x = float(self.flabel_x[i])
            # start point, end point

            self.frequency_line_points = np.vstack((self.flabel_x, self.fpoint_y)).T.tolist()
            self.xlines[i].points = (self.frequency_line_points[i], (self.frequency_line_points[i][0], self.y_pos_end))

    def frequency_span_update(self, velocity):
        vs, ve = self.current_view_start, self.current_view_end
        vs += velocity
        ve -= velocity

        if vs < 0 or ve > self.full_view_width:
            vs = 0
            ve = self.full_view_width
        elif (ve - vs) < (self.frequency_axis_size + 5):
            return

        width = ve - vs
        self.x_axis = np.linspace(self.xpos_start, self.xpos_end, num=width)
        self.flabel_x = np.arange(self.xpos_start, self.xpos_end, (self.xpos_end - self.xpos_start) / self.frequency_axis_size)

        self.frequency_line_points = np.vstack((self.flabel_x, self.fpoint_y)).T.tolist()
        self.current_view_start, self.current_view_end = vs, ve

        self.frequency_axis_update(self.current_view_start , width, fs=self.sudio.sample_rate)

    def frequency_move_update(self, velocity):
        vs, ve = self.current_view_start, self.current_view_end
        vs += velocity
        ve += velocity

        if vs < 0 or ve > self.full_view_width:
            return


        width = ve - vs

        self.current_view_start, self.current_view_end = vs, ve

        self.frequency_axis_update(self.current_view_start, width, fs=self.sudio.sample_rate)


    def fft_gui_start(self, do_animation):
        if do_animation:
            self._update_points_animation_ev = Clock.schedule_interval(self.update_points, 1/30)
            if self.start_flg:

                self.sudio.add_pipeline('pip0', self.pip, process_type='branch', channel=0)
                # self.sudio.set_pipeline('pip0')
                self.sudio.start()

                self.start_flg = False

        elif self._update_points_animation_ev is not None:
            self._update_points_animation_ev.cancel()

        return self

    def update_points(self, dt):

        self.dt += dt
        try:
            y_axis = self.dataqueue.get_nowait()
        except queue.Empty:
            return

        y_axis += self.y_offset
        # print('job 2 done')
        points = np.vstack((self.x_axis, y_axis[self.current_view_start: self.current_view_end]))
        self.points = list(Sudio.Process.shuffle2d_channels(points))

    # def normalizer_update(self, freq):
    #     self.lpf_cutoff_frequency = freq
    #     self.lpf_coef = iirfilter(5, self.lpf_cutoff_frequency, btype='lowpass', fs=sudio.sample_rate)

    def mono_fft_process(self, frame):
        try:
            data = 50 * np.log10(np.abs(fft(frame) / self.sudio.nperseg) ** 2)
            data = lfilter(self.lpf_coef[0],  self.lpf_coef[1], data)
            self.dataqueue.put_nowait(data[0])
            self.dataqueue.put_nowait(data[1])
        except:
            pass

        return frame


class GfftApp(App):
    def __init__(self, sudio, rwidth, rheight, max_buffer_size=100, **kwargs):
        super().__init__(**kwargs)
        self.sudio = sudio
        self.maxbf_size = max_buffer_size
        self.rwidth = rwidth
        self.rheight = rheight

    def build(self):
        return Gfft(self.sudio, self.rwidth, self.rheight, self.maxbf_size)

    def run(self):
        super().run()


def main_init(sudio, maxbuffer_size):
    rwidth = GetSystemMetrics(0)
    rheight = GetSystemMetrics(1) - 100

    Config.set('graphics', 'width', str(rwidth))
    Config.set('graphics', 'height', str(rheight))

    GfftApp(sudio, rwidth, rheight, maxbuffer_size).run()


if __name__ == '__main__':
    # Config.set('graphics', 'multisamples', '0')
    os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'
    # Config.set('graphics', 'fullscreen', 1)

    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    Builder.load_file('fft.kv')

    # sudio = Sudio.Process(std_input_dev_id=1, frame_rate=44000, nchannels=1,
    #                       data_format=Sudio.formatInt16,
    #                       mono_mode=True,
    #                       optimum_mono=False,
    #                       ui_mode=False,
    #                       nperseg=5000,
    #                       noverlap=None,
    #                       window='hann',
    #                       NOLA_check=True)
    #

    sudio = Sudio.Process(std_input_dev_id=None, frame_rate=44000, nchannels=2,
                          data_format=Sudio.formatInt16,
                          mono_mode=True,
                          optimum_mono=False,
                          ui_mode=False,
                          nperseg=4000,
                          noverlap=None,
                          window='hann',
                          NOLA_check=True)

    # sudio.primary_filter(enable=False, fc=45000)
    sudio.echo()
    # sudio.start()
    threading.Thread(target=main_init, args=(sudio, 50, )).start()

    input()
