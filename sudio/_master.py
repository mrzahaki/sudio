"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""

# Todo IO mode

import io
import scipy.signal as scisig
import threading
import queue
import pandas as pd
import numpy as np
from typing import Union, Callable
import warnings
import gc
import time
import os
from io import BufferedRandom


from sudio.types import StreamMode, RefreshError, SampleFormatToLib, LibToSampleFormat
from sudio.types import SampleFormat, SampleFormatEnumToLib, LibSampleFormat, DitherMode
from sudio.types import LibSampleFormatEnumToSample, PipelineProcessType
from sudio.wrap.wrapgenerator import WrapGenerator
from sudio.wrap.wrap import Wrap
from sudio.extras.exmath import find_nearest_divisor
from sudio.extras.exmath import voltage_to_dBu
from sudio.extras.strtool import generate_timestamp_name 
from sudio.extras.timed_indexed_string import TimedIndexedString 
from sudio.stream.stream import Stream
from sudio.stream.streamcontrol import StreamControl
from sudio.stream.utils import stdout_stream
from sudio.audioutils.window import multi_channel_overlap, single_channel_overlap
from sudio.audioutils.window import multi_channel_windowing, single_channel_windowing
from sudio.audioutils.codecinfo import get_file_info
from sudio.audioutils import interface as audiointerface
from sudio.audioutils.audio import Audio
from sudio.audioutils.io import write_wav_file
from sudio.audioutils.codec import decode_audio_file
from sudio.audioutils.channel import shuffle3d_channels, shuffle2d_channels, get_mute_mode_data, map_channels
from sudio.audioutils.sync import synchronize_audio
from sudio.audioutils.cacheutil import handle_cached_record, write_to_cached_file
from sudio.pipeline import Pipeline




class Master:

    DATA_PATH = './data/'
    USER_PATH = DATA_PATH + 'user.sufile'
    SAVE_PATH = DATA_PATH + 'export/'
    SAMPLE_CATCH_TIME = 10  # s
    SAMPLE_CATCH_PRECISION = 0.1  # s
    SPEAK_LEVEL_GAP = 10  # dbu
    CACHE_INFO = 4 * 8

    DATABASE_COLS = ['size', 'noise', 'frameRate', 'o', 'sampleFormat', 'nchannels', 'duration', 'nperseg']
    BUFFER_TYPE = '.bin'

    def __init__(self,
                 std_input_dev_id: int = None,
                 std_output_dev_id: int = None,
                 frame_rate: int = 48000,
                 nchannels: int = None,
                 data_format: SampleFormat = SampleFormat.formatInt16,  # 16bit mode
                 mono_mode: Union[bool, int] = False,
                 nperseg: int = 500,
                 noverlap: int = None,
                 window: object = 'hann',
                 NOLA_check: bool = True,
                 input_dev_callback: Callable = None,
                 output_dev_callback: Callable = None,
                 master_mix_callback: Callable = None,
                 buffer_size: int = 30):
        
        """
        Create a Master object to manage audio processing.

        Parameters
        ----------
        std_input_dev_id: int, optional
             If not provided, the input device id will be OS standard input device id.

        std_output_dev_id: int, optional
             If not provided, the output device id will OS standard output device id.

        frame_rate: int, optional
            Input channel sample rate. If std_input_dev_id is None, the value of
            frame_rate does not matter.

        nchannels: int, optional
            Number of input channels (inner channels). If std_input_dev_id is None, the value of
            nchannels does not matter.

        data_format: SampleFormat
            Specify the audio bit depths. Supported data formats (from audio):
            formatFloat32, formatInt32, formatInt24, formatInt16 (default),
            formatInt8, formatUInt8.

        mono_mode: bool, int optional
            If True, all input channels will be mixed to one channel. (integer type considered as specific input channel)

        nperseg: int, optional
            Number of samples per each data frame (window) in one channel.

        noverlap: int, default None
            Number of samples that overlap between defined windows. If not given, it will
            be selected automatically.

        window: string, float, tuple, ndarray, optional
            Type of window to create (string) or pre-designed window in numpy array format.
            Default ("hann"): Hanning function. None to disable windowing.

        NOLA_check: bool, optional
            Check whether the Nonzero Overlap Add (NOLA) constraint is met (if True).

        input_dev_callback: callable, default None
            Callback function for user-defined input stream, called by the core every 1/frame_rate second.
            Must return frame data in (Number of channels, Number of data per segment) dimensions numpy array.
            Each data must be in the format defined in data_format section.

        output_dev_callback: callable, default None
            Callback function for user-defined output stream, called by the core every 1/frame_rate second.
            Must return frame data in (Number of channels, Number of data per segment) dimensions numpy array.
            Each data must be in the format defined in the data_format section.

        master_mix_callback: callable, optional
            Callback used before the main processing stage in the master for controlling and mixing all
            slave channels to an ndarray of shape (master.nchannels, 2, master.nperseg).
            If not defined, the number of audio channels in all slaves must be the same as the master.

        Note
        -----
        - If the window requires no parameters, then `window` can be a string.
        - If the window requires parameters, then `window` must be a tuple with the first argument the
        string name of the window, and the next arguments the needed parameters.
        - If `window` is a floating-point number, it is interpreted as the beta parameter of the
        `scipy.signal.windows.kaiser` window.

        Window Types:
        boxcar, triang, blackman, hamming, hann (default), bartlett, flattop, parzen, bohman, blackmanharris,
        nuttall, barthann, cosine, exponential, tukey, taylor, kaiser (needs beta),
        gaussian (needs standard deviation), general_cosine (needs weighting coefficients),
        general_gaussian (needs power, width), general_hamming (needs window coefficient),
        dpss (needs normalized half-bandwidth), chebwin (needs attenuation)

        Note
        -----
        To enable inversion of an STFT via the inverse STFT in `istft`, the signal windowing must obey
        the constraint of "nonzero overlap add" (NOLA):
            \sum_{t}w^{2}[n-tH] \ne 0
        This ensures that the normalization factors in the denominator of the overlap-add inversion
        equation are not zero. Only very pathological windows will fail the NOLA constraint.

        Master/Slave Mode Details:
        - Every Process object can operate in master or slave modes.
        - Each input must be preprocessed before mixing different audio inputs, so the slave object is
        designed so that the post-processing stage is disabled and can be joined to master simply by
        using the 'join' method.
        """
        # ----------------------------------------------------- First Initialization Section -----------------------------------------------------

        # Create necessary directories if they don't exist
        try:
            os.mkdir(Master.DATA_PATH)
        except FileExistsError:
            pass
        try:
            os.mkdir(Master.SAVE_PATH)
        except:
            pass
        
        # Set initial values for various attributes
        self._sound_buffer_size = buffer_size
        self._stream_type = StreamMode.optimized
        data_format = data_format.value
        self.input_dev_callback = None
        self.output_dev_callback = None
        self._default_stream_callback = self._stream_callback

        # Initialize threading events and flags
        self._exstream_mode = threading.Event()
        self._master_mute_mode = threading.Event()
        self._stream_loop_mode = False
        self._stream_on_stop = False
        self._stream_data_pointer = 0
        self._stream_data_size = None
        self._stream_file = None
        self._nchannels = None

        # Suppress warnings
        warnings.filterwarnings("ignore")

        self._output_device_index = None
        # Create an interface to PortAudio
        self._audio_instance = Audio()

        # Set default number of input channels
        input_channels = int(1e6)

        # UI mode: Choose input device interactively
        if callable(input_dev_callback):
            self.input_dev_callback = input_dev_callback
            self._default_stream_callback = self._custom_stream_callback
            self._nchannels = nchannels

        elif std_input_dev_id is None:
            dev = self.get_default_input_device_info()
            frame_rate = self._frame_rate = int(dev['defaultSampleRate'])
            input_channels = dev['maxInputChannels']
            self._std_input_dev_id = dev['index']

        else:
            # Use specified input device ID
            self._std_input_dev_id = std_input_dev_id
            dev = self.get_device_info_by_index(std_input_dev_id)
            input_channels = dev['maxInputChannels']
            frame_rate = self._frame_rate = int(dev['defaultSampleRate'])


        # Set number of output channels to a large value initially
        self._sample_width = Audio.get_sample_size(data_format)
        self._sample_format = data_format
        output_channels = int(1e6)

        if callable(output_dev_callback):
            self.output_dev_callback = output_dev_callback
            self._default_stream_callback = self._custom_stream_callback
            self._nchannels = nchannels

        # UI mode: Choose output device interactively
        elif std_output_dev_id is None:
            # Use default output device
            dev = self.get_default_output_device_info()
            self._output_device_index = dev['index']
            output_channels = dev['maxOutputChannels']
        else:
            # Use specified input device ID
            self._std_input_dev_id = std_output_dev_id
            dev = self.get_device_info_by_index(std_output_dev_id)
            output_channels = dev['maxInputChannels']

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._nchannels = self._output_channels
  


        # Raise an error if no input or output device is found
        if self._nchannels == 0:
            raise ValueError('No input or output device found')

        # Set mono mode to True if there is only one channel
        if self._nchannels == 1:
            mono_mode = True

        # Mute the audio initially
        self.mute()

        # Set initial values for various attributes
        self.branch_pipe_database_min_key = 0
        self._window_type = window
        self._sample_rate = frame_rate
        self._nperseg = nperseg

        # Check _win arguments
        if noverlap is None:
            noverlap = nperseg // 2

        if type(self._window_type) is str or type(self._window_type) is tuple or type(self._window_type) == float:
            window = scisig.get_window(window, nperseg)

        # Check window size
        if self._window_type:
            assert len(window) == nperseg, 'Control size of window'
            if NOLA_check:
                assert scisig.check_NOLA(window, nperseg, noverlap)
        elif self._window_type is None:
            window = None
            self._window_type = None

        # ----------------------------------------------------- UI mode std_input_dev_id calculator -----------------------------------------------------

        # Calculate recording period based on window size and frame rate
        record_period = nperseg / frame_rate
        self._record_period = record_period

        # ----------------------------------------------------- Vectorized dbu calculator -----------------------------------------------------

        # Vectorize the dbu calculation function
        self._vdbu = np.vectorize(self._rdbu, signature='(m)->()')

        # Set the number of channels for processing
        if mono_mode:
            self._nchannels = 1
 

        # ----------------------------------------------------- Sample rate and pre-filter processing -----------------------------------------------------

        # Set data chunk size, frame rate, and other constants
        self._data_chunk = nperseg
        self._frame_rate = frame_rate
        self._sample_width_format_str = '<i{}'.format(self._sample_width)

        # Adjust constant for floating-point data format
        if data_format == SampleFormat.formatFloat32.value:
            self._sample_width_format_str = '<f{}'.format(self._sample_width)

        # Initialize thread list
        self._threads = []

        # ----------------------------------------------------- User _database define -----------------------------------------------------

        # Load user database from pickle file or create an empty DataFrame
        try:
            self._database = pd.read_pickle(Master.USER_PATH, compression='xz')
        except:
            self._database = pd.DataFrame(columns=Master.DATABASE_COLS)

        self._local_database = pd.DataFrame(columns=Master.DATABASE_COLS)

        # ----------------------------------------------------- Pipeline _database and other definitions section -----------------------------------------------------

        # Initialize semaphores, queues, and events
        self._semaphore = [threading.Semaphore()]
        self._recordq = [threading.Event(), queue.Queue(maxsize=0)]
        self._queue = queue.Queue()
        self._streamq = [threading.Event(), queue.Queue(maxsize=self._sound_buffer_size)]

        self._echo = threading.Event()
        self.main_pipe_database = []
        self.branch_pipe_database = []

        # ----------------------------------------------------- Threading management -----------------------------------------------------

        self._functions = []
        self._functions.append(self._stream_output_manager)

        # Create and start threads
        for i in range(len(self._functions)):
            self._threads.append(threading.Thread(target=self._run, daemon=True, args=(len(self._threads),)))
            self._queue.put(i)

        # ----------------------------------------------------- Windowing object define and initialization -----------------------------------------------------

        # Set parameters for windowing
        self._nhop = nperseg - noverlap
        self._noverlap = noverlap
        self._window = window

        # Initialize buffers and functions based on mono or multi-channel mode
        if mono_mode:
            self._iwin_buffer = [np.zeros(nperseg)]
            self._win_buffer = [np.zeros(nperseg), np.zeros(nperseg)]
            self._win = self._single_channel_windowing
            self._iwin = self._single_channel_overlap
            self._upfirdn = lambda h, x, const: scisig.upfirdn(h, x, down=const)
        else:
            self._iwin_buffer = [np.zeros(nperseg) for i in range(self._nchannels)]
            self._win_buffer = [[np.zeros(nperseg), np.zeros(nperseg)] for i in range(self._nchannels)]

            # 2D mode
            self._upfirdn = np.vectorize(lambda h, x, const: scisig.upfirdn(h, x, down=const), signature='(n),(m),()->(c)')
            self._iwin = self._multi_channel_overlap
            self._win = self._multi_channel_windowing

        # Create Stream objects for main and normal streams
        data_queue = queue.Queue(maxsize=self._sound_buffer_size)
        self._normal_stream = Stream(self, data_queue, process_type=PipelineProcessType.QUEUE)
        self._main_stream = Stream(self, data_queue, process_type=PipelineProcessType.QUEUE)
        self._main_stream.acquire()
        # ----------------------------------------------------- Miscellaneous initialization -----------------------------------------------------
        self._refresh_ev = threading.Event()

        self._master_mix_callback = master_mix_callback

        # Master/slave selection and synchronization
        self._master_sync = threading.Barrier(self._nchannels)
        self.slave_database = []
        self._slave_sync = None
        self._pystream = False
        self._mono_mode = mono_mode


    def start(self):
        '''
        start audio streaming process
        :return: self object
        '''
        assert not self._pystream, 'Master is Already Started'
        try:
            self._pystream = self._audio_instance.open_stream(format=self._sample_format,
                                                   channels=self._input_channels,
                                                   rate=self._frame_rate,
                                                   frames_per_buffer=self._data_chunk,
                                                   input_device_index=self._std_input_dev_id,
                                                   input=True,
                                                #    output=True,  # Enable output
                                                   stream_callback=self._default_stream_callback)
            for i in self._threads:
                i.start()

            # The server blocks all active threads
            # del self._threads[:]
            # self._queue.join()

        except:
            raise
        
        return self

    def _run(self, th_id):
        # self.a1=0
        self._functions[self._queue.get()](th_id)
        # Indicate that a formerly enqueued task is complete.
        # Used by queue consumer threads. For each get() used to fetch a task,
        # a subsequent call to task_done() tells the queue that the processing on the task is complete.

        # If a join() is currently blocking,
        # it will resume when all items have been processed
        # (meaning that a task_done() call was received for every item that had been put() into the queue).
        self._queue.task_done()


    def _exstream(self):
        while self._exstream_mode.is_set():
            in_data: np.ndarray = np.frombuffer(
                self._stream_file.read(self._stream_data_size),
                self._sample_width_format_str).astype('f')

            try:
                in_data: np.ndarray = map_channels(in_data, self._nchannels, self._nchannels, self._mono_mode)
            except ValueError:
                return

            if not in_data.shape[-1] == self._data_chunk:
                if self._stream_loop_mode:
                    self._stream_file.seek(self._stream_data_pointer, 0)
                else:
                    # print('ok')
                    self._exstream_mode.clear()
                    self._stream_file.seek(self._stream_data_pointer, 0)
                    self._main_stream.clear()

                    self._stream_file = None
                    self._stream_loop_mode = None
                    self._stream_data_pointer = None
                    self._stream_data_size = None

                if self._stream_on_stop:
                    self._stream_on_stop()
                    self._stream_on_stop = None
                return
            
            self._main_stream.acquire()
            self._main_stream.put(in_data)  
            self._main_stream.release()
            if self._stream_type == StreamMode.normal:
                break
    
    def _main_stream_safe_release(self):
        if self._main_stream.locked():
            self._main_stream.release()


    def _stream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags
        
        if self._exstream_mode.is_set():
            self._exstream()
            return None, 0

        elif self._master_mute_mode.is_set():
            in_data = get_mute_mode_data(self._nchannels, self._nperseg)
        else:
            in_data = np.frombuffer(in_data, self._sample_width_format_str).astype('f')
            try:
                in_data = map_channels(in_data, self._input_channels, self._nchannels, self._mono_mode)
            # except ValueError:
            except Exception as e:
                print(f"Error {e}")
                return None, 0

        try:
            self._main_stream.acquire()
            self._main_stream.put(in_data)  
            self._main_stream.release()

        except Exception as e:
            print(f"Error in stream callback: {e}")
            print(f"in_data shape: {in_data.shape}, type: {type(in_data)}")
            print(f"self._input_channels: {self._input_channels}, self._nchannels: {self._nchannels}")
            return None, 1  # Indicate that an error occurred

        return None, 0

    # user Input mode
    def _custom_stream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        if self._exstream_mode.is_set():
            if not self._exstream():
                return None, 0

        elif self._master_mute_mode.is_set():
            in_data = get_mute_mode_data(self._nchannels, self._nperseg)
        else:
            in_data = self.input_dev_callback(frame_count, time_info, status).astype('f')

        self._main_stream.acquire()
        self._main_stream.put(in_data)  
        self._main_stream.release()

        # print(f't={a0- time.perf_counter()}, tlo={self.a1 - a0}')
        # self.a1 = time.perf_counter()
        return None, 0

    def _stream_output_manager(self, thread_id):
        stream_out = self._audio_instance.open_stream(
                            format=self._sample_format,
                            channels=self._output_channels,
                            output_device_index=self._output_device_index,
                            rate=self._frame_rate, input=False, output=True)

        stream_out.start_stream()
        rec_ev, rec_queue = self._recordq
        while 1:
            try:
                data = self._main_stream.get(timeout=0.001)

                if rec_ev.is_set():
                    rec_queue.put_nowait(data)

                if not self._mono_mode:
                    data = shuffle2d_channels(data)

                if self._echo.is_set():
                    stream_out.write(data.tobytes())

                if self.output_dev_callback:
                    self.output_dev_callback(data)
            except Exception as e:
                pass

            if self._refresh_ev.is_set():
                self._refresh_ev.clear()
                # close current stream
                stream_out.close()
                # create new stream
                rate = self._frame_rate


                # print(rate)
                stream_out = self._audio_instance.open_stream(
                                    format=self._sample_format,
                                    channels=self._output_channels,
                                    rate=rate,
                                    input=False,
                                    output=True)
                                    

                stream_out.start_stream()
                self._main_stream.clear()


    def _safe_input(self, *args, **kwargs):
        self._semaphore[0].acquire()
        res = input(*args, **kwargs)
        self._semaphore[0].release()
        return res

    def _safe_print(self, *args, **kwargs):
        self._semaphore[0].acquire(timeout=1)
        print(*args, **kwargs)
        self._semaphore[0].release()

    # RMS to dbu
    # signal maps to standard +4 dbu
    def _rdbu(self, arr):
        return voltage_to_dBu(np.max(arr) / ((2 ** (self._sample_width * 8 - 1) - 1) / 1.225))


    def add_file(self, filename: str, sample_format: SampleFormat = SampleFormat.formatUnknown,
                nchannels: int = None, sample_rate: int = None ,safe_load=True):
        '''
        Add an audio file to the local database. None value for the parameters means
        that they are automatically selected based on the properties of the audio file.

         Note:
          supported files format: WAV, FLAC, VORBIS, MP3

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        :param filename: path/name of the audio file
        :param sample_format: sample format
        :param nchannels: number of audio channels
        :param sample_rate: sample rate
        :param safe_load: load an audio file and modify it according to the 'Master' attributes.
        (sample rate, sample format, number of channels, etc).
        :return: WrapGenerator object

        '''
        info = get_file_info(filename)
        if safe_load:
            sample_format = SampleFormatEnumToLib[self._sample_format]
        elif sample_format is SampleFormat.formatUnknown:
            if info.sample_format is LibSampleFormat.UNKNOWN:
                sample_format = SampleFormatToLib[self._sample_format]
            else:
                sample_format = info.sample_format
        else:
            sample_format = SampleFormatToLib[sample_format]

        if nchannels is None:
            nchannels = info.nchannels
        if safe_load:
          sample_rate = self._sample_rate
        elif sample_rate is None:
            sample_rate = info.sample_rate


        record ={
            'size': None,
            'noise': None,
            'frameRate': sample_rate,
            'o': None,
            'sampleFormat': LibToSampleFormat[sample_format].value,
            'nchannels': nchannels,
            'duration': info.duration,
            'nperseg': self._nperseg,
        }

        # print(record)

        p0 = max(filename.rfind('\\'), filename.rfind('/')) + 1
        p1 = filename.rfind('.')
        if p0 < 0:
            p0 = 0
        if p1 <= 0:
            p1 = None
        name = filename[p0: p1]
        if name in self._local_database.index or \
                name in self._database.index:
            raise KeyError('Record with the name of {}'
                           ' already registered in local or external database'.format(name))



        # print(list(record.keys()), list(self._local_database.columns))
        assert list(record.keys()) == list(self._local_database.columns)

        if safe_load and not self._mono_mode and record['nchannels'] > self._nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=name,
                                                                              ch0=record['nchannels'],
                                                                              ch1=self._nchannels))

        decoder = lambda: decode_audio_file(filename, sample_format, nchannels, sample_rate, DitherMode.NONE)

        record = (
        handle_cached_record(record,
                    TimedIndexedString(Master.DATA_PATH + name + Master.BUFFER_TYPE,
                                      start_before=Master.BUFFER_TYPE),
                    self,
                    safe_load = safe_load,
                    decoder=decoder)
        )

        self._local_database.loc[name] = record
        gc.collect()
        return self.load(name)


    def add(self, record, safe_load=True):
        '''
        Add new record to the local database.

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        :param record: can be an wrapped record, pandas series or an audio file in
                        mp3, WAV, FLAC or VORBIS format.
        :param safe_load: load an audio file and modify it according to the 'Master' attributes.
        :return: Wrapped object

        \nnote:
         The name of the new record can be changed with the following tuple that is passed to the function:
         (record object, new name of )


        '''
        name_type = type(record)
        if name_type is list or name_type is tuple:
            assert len(record) < 3

            rec_name = None
            if len(record) == 2:
                if type(rec_name) is str:
                    rec_name = record[1]

            if type(record[0]) is WrapGenerator:
                record = record[0].get_data()

            elif type(record[0]) is pd.Series:
                pass
            else:
                raise TypeError

            if rec_name and type(rec_name) is str:
                record.name = rec_name

            elif rec_name is None:
                record.name = generate_timestamp_name()

            else:
                raise ValueError('second item in the list is the name of the new record')
            # print(record.name)
            return self.add(record, safe_load=safe_load)

        elif name_type is Wrap or name_type is WrapGenerator:
            series = record.get_data()
            return self.add(series, safe_load=safe_load)

        elif name_type is pd.Series:
            name = record.name
            assert list(record.keys()) == list(self._local_database.columns)

            if safe_load and not self._mono_mode and record['nchannels'] > self._nchannels:
                raise ImportError('number of channel for the {name}({ch0})'
                                  ' is not same as object channels({ch1})'.format(name=name,
                                                                                  ch0=record['nchannels'],
                                                                                         ch1=self._nchannels))
            if type(record['o']) is not BufferedRandom:
                if  record.name in self._local_database.index or record.name in self._database.index:
                    record.name = generate_timestamp_name()
                    # print('yessssss')

                if safe_load:
                    record = self._sync_record(record)

                f, newsize = (
                    write_to_cached_file(record['size'],
                                record['frameRate'],
                                record['sampleFormat'] if record['sampleFormat'] else 0,
                                record['nchannels'],
                                file_name=Master.DATA_PATH + record.name + Master.BUFFER_TYPE,
                                data=record['o'],
                                pre_truncate=True,
                                after_seek=(Master.CACHE_INFO, 0),
                                after_flush=True,
                                sizeon_out=True)
                )
                record['o'] = f
                record['size'] = newsize

            elif (not (Master.DATA_PATH + record.name + Master.BUFFER_TYPE) ==  record.name) or \
                    record.name in self._local_database.index or record.name in self._database.index:
                # new sliced Wrap data

                if  record.name in self._local_database.index or record.name in self._database.index:
                    record.name = generate_timestamp_name()
                
                prefile = record['o']
                prepos = prefile.tell(), 0

                record['o'], newsize = (
                    write_to_cached_file(record['size'],
                                record['frameRate'],
                                record['sampleFormat'] if record['sampleFormat'] else 0,
                                record['nchannels'],
                                file_name=Master.DATA_PATH + record.name + Master.BUFFER_TYPE,
                                data=prefile.read(),
                                after_seek=prepos,
                                after_flush=True,
                                sizeon_out=True)
                )
                prefile.seek(*prepos)
                record['size'] = newsize

            # print(record)
            # if record.name  in self._local_database.index or \
            #     record.name in self._database.index:
            #     raise KeyError('Record with the name of {}'
            #                    ' already registered in local or external database'.format(name))
            self._local_database.loc[record.name] = record

            gc.collect()
            return self.wrap(record)

        elif name_type is str:
           gc.collect()
           return self.add_file(record, safe_load=safe_load)

        else:
            raise TypeError('The record must be an audio file, data frame or a Wrap object')


    def recorder(self, record_duration: float, name: str = None):
        """
        Record audio for a specified duration.
        
        Args:
            record_duration (float): Duration to record in seconds.
            name (str, optional): Name for the recording. If None, a timestamp-based name will be generated.
        
        Returns:
            WrapGenerator: A wrapped version of the recorded audio data.
        """
        if name is None:
            name = generate_timestamp_name('record')
        elif name in self._local_database.index:
            raise KeyError(f'The name "{name}" is already registered in the database.')

        rec_ev, rec_queue = self._recordq
        rec_ev.set()  # Start recording

        # Record for the specified duration
        start_time = time.time()
        recorded_data = []
        while time.time() - start_time < record_duration:
            data = rec_queue.get()
            recorded_data.append(data)

        rec_ev.clear()  # Stop recording

        # Process the recorded data
        sample = np.array(recorded_data)
        
        # Apply shuffle3d_channels if not in mono mode
        if not self._mono_mode:
            sample = shuffle3d_channels(sample)

        # Prepare the record data
        record_data = {
            'size': sample.nbytes,
            'noise': None,
            'frameRate': self._frame_rate,
            'nchannels': self._nchannels,
            'sampleFormat': self._sample_format,
            'nperseg': self._nperseg,
            'duration': record_duration,
        }

        # Write the recorded data to a file
        record_data['o'] = write_to_cached_file(
            record_data['size'],
            record_data['frameRate'],
            record_data['sampleFormat'],
            record_data['nchannels'],
            file_name=f"{Master.DATA_PATH}{name}{Master.BUFFER_TYPE}",
            data=sample.tobytes(),
            pre_truncate=True,
            after_seek=(Master.CACHE_INFO, 0),
            after_flush=True
        )

        # Update the size to include the cache info
        record_data['size'] += Master.CACHE_INFO

        # Add the record to the local database
        self._local_database.loc[name] = record_data

        return self.wrap(self._local_database.loc[name].copy())

    def load_all(self, safe_load=True):
        '''
        load all of the saved records from the external database
                        to the local database.

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        :param safe_load: if safe load is enabled then load function tries to load a record
         in the local database based on the master  settings, like the frame rate and etc.

        :return:None
        '''
        self.load('', load_all=True, safe_load=safe_load)


    def load(self, name: str, load_all: bool=False, safe_load: bool=True,
             series: bool=False) -> Union[WrapGenerator, pd.Series]:
        '''
        This method used to load a predefined recoed from the staable/external database to
         the local database. Trying to load a record that was previously loaded, outputs a wrapped version
         of the named record.

        :param name: predefined record name
        :param load_all: load all of the saved records from the external database
                        to the local database.
        :param safe_load: if safe load is enabled then load function tries to load a record
         in the local database based on the master  settings, like the frame rate and etc.
        :param series: if enabled then Trying to load a record that was previously loaded, outputs data series
        of the named record.
        :return: (optional) Wrapped object, pd.Series
        '''
        if load_all:
            for i in self._database.index:
                self.load(i, safe_load=safe_load)

            return
        # elif type(name) is pd.Series:
        #     ret
        else:
            if name in self._local_database.index:
                rec = self._local_database.loc[name].copy()
                if series:
                    return rec
                return self.wrap(rec)

            elif name in self._database.index:
                rec = self._database.loc[name].copy()
                file = open(rec['o'], 'rb+')
                file_size = os.path.getsize(rec['o'])
            else:
                raise ValueError('can not found the {name} in the local and the external databases'.format(name=name))

        if safe_load and not self._mono_mode and rec['nchannels'] > self._nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=name,
                                                                              ch0=rec['nchannels'],
                                                                              ch1=self._nchannels))
        if safe_load:
            rec['o'] = file.read()
            rec = self._sync_record(rec)
            rec['o'] = file

        if file_size > rec['size']:
            file.seek(rec['size'], 0)
        else:
            file.seek(Master.CACHE_INFO, 0)

        self._local_database.loc[name] = rec
        if series:
            return rec
        return self.wrap(rec)


    def get_record_info(self, name: str) -> dict:
        '''
        :param name: name of the registered record on the local or external database
        :return: information about saved record ['noiseLevel' 'frameRate'  'sizeInByte' 'duration'
            'nchannels' 'nperseg' 'name'].
        '''
        if name in self._local_database.index:
            rec = self._local_database.loc[name]

        elif name in self._database.index:
            rec = self._database.loc[name]
        else:
            raise ValueError('can not found the {name} in the local and the external databases'.format(name=name))
        return {
            'noiseLevel': str(rec['noise']) + ' <dbm>',
            'frameRate': rec['frameRate'],
            'sizeInByte': rec['size'],
            'duration': rec['duration'],
            'nchannels': rec['nchannels'],
            'nperseg': rec['nperseg'],
            'name': name,
            'sampleFormat': LibSampleFormatEnumToSample[rec['sampleFormat']].name
        }

    def get_exrecord_info(self, name: str) -> dict:
        '''
        :param name: name of the registered record on the external database
        :return: information about saved record ['noiseLevel' 'frameRate'  'sizeInByte' 'duration'
            'nchannels' 'nperseg' 'name'].
        '''
        if name in self._database.index:
            rec = self._database.loc[name]
        else:
            raise ValueError('can not found the {name} in the external databases'.format(name=name))
        return {
            'noiseLevel': str(rec['noise']) + ' <dbm>',
            'frameRate': rec['frameRate'],
            'sizeInByte': rec['size'],
            'duration': rec['duration'],
            'nchannels': rec['nchannels'],
            'nperseg': rec['nperseg'],
            'name': name,
        }


    def _syncable(self,
                 *target,
                 nchannels: int = None,
                 sample_rate: int = None,
                 sample_format_id: int = None):
        '''
         Determines whether the target can be synced with specified properties or not

        :param target: wrapped object\s
        :param nchannels: number of channels; if the value is None, the target will be compared to the 'self' properties.
        :param sample_rate: sample rate; if the value is None, the target will be compared to the 'self' properties.
        :param sample_format_id: if the value is None, the target will be compared to the 'self' properties.
        :return: returns only objects that need to be synchronized.
        '''
        nchannels = nchannels if nchannels else self._nchannels
        sample_format_id = self._sample_format if sample_format_id is None else sample_format_id
        sample_rate = sample_rate if sample_rate else self._sample_rate

        buffer = []
        for rec in target:
            assert type(rec) is Wrap or type(rec) is WrapGenerator
            tmp = rec.get_data()

            if (not tmp['nchannels'] == nchannels or
                not sample_rate == tmp['frameRate'] or
                not tmp['sampleFormat'] == sample_format_id):
                buffer.append(True)
            else:
                buffer.append(False)
        if len(buffer) == 1:
            return buffer[0]
        return buffer


    def syncable(self,
                 *target,
                 nchannels: int = None,
                 sample_rate: int = None,
                 sample_format: SampleFormat = SampleFormat.formatUnknown):
        '''
         Determines whether the target can be synced with specified properties or not

        :param target: wrapped object\s
        :param nchannels: number of channels; if the value is None, the target will be compared to the 'self' properties.
        :param sample_rate: sample rate; if the value is None, the target will be compared to the 'self' properties.
        :param sample_format: if the value is None, the target will be compared to the 'self' properties.
        :return: returns only objects that need to be synchronized.
        '''
        return self._syncable(*target, nchannels=nchannels, sample_rate=sample_rate, sample_format_id=sample_format.value)


    def sync(self,
             *targets,
             nchannels: int=None,
             sample_rate: int=None,
             sample_format: SampleFormat=SampleFormat.formatUnknown,
             output='wrapped'):
        '''
        Synchronizes targets in the Wrap object format with the specified properties
        :param targets: wrapped object\s
        :param nchannels: number of channels; if the value is None, the target will be synced to the 'self' properties.
        :param sample_rate: if the value is None, the target will be synced to the 'self' properties.
        :param sample_format: if the value is None, the target will be synced to the 'self' properties.
        :param output: can be 'wrapped', 'series' or 'ndarray_data'
        :return: returns synchronized objects.
        '''
        # syncable = list((self.syncable(*records,
        #                         nchannels=nchannels,
        #                         sample_rate=sample_rate,
        #                         sample_format=sample_format), ))


        nchannels = nchannels if nchannels else self._nchannels
        sample_format = self._sample_format if sample_format == SampleFormat.formatUnknown else sample_format.value
        sample_rate =  sample_rate if sample_rate else self._sample_rate

        out_type = 'ndarray' if output.startswith('n') else 'byte'

        # targets = list(val.get_data().copy() for idx, val in enumerate(records) if syncable[idx])
        # buffer = [*list(val.get_data().copy() for idx, val in enumerate(records) if not syncable[idx])]
        # for record in buffer:
        #     f = record['o']
        #     tmp = f.tell, 0
        #     record['o'] = f.read()
        #     f.seek(*tmp)
        #
        #     if out_type.startswith('n'):
        #         pass

        buffer = []
        for record in targets:
            record: pd.Series = record.get_data().copy()
            assert type(record) is pd.Series
            main_file: io.BufferedRandom = record['o']
            main_seek = main_file.tell(), 0
            record['o'] = main_file.read()
            main_file.seek(*main_seek)
            synchronize_audio(record, nchannels, sample_rate, sample_format, output_data=out_type)
            record.name = generate_timestamp_name(record.name)
            if output[0] == 'w':
                buffer.append(self.add(record))
            else:
                buffer.append(record)
        return tuple(buffer)


    # save from local _database to stable memory
    def _sync_record(self, rec):
        return synchronize_audio(rec, self._nchannels, self._frame_rate, self._sample_format)

    def del_record(self, name: str, deep: bool=False):
        '''
        This function is used to delete a record from the internal database.
        :param name str: the name of preloaded record.
        :param deep bool: deep delete mode is used to remove the record and its corresponding caches
        from the external database.
        :return: None
        '''
        local = name in self._local_database.index
        extern = name in self._database.index
        ex = ''
        if deep:
            ex = 'and the external '
        assert local or (extern and deep), ValueError(f'can not found the {name} in the '
                                            f'local {ex}databases'.format(name=name, ex=ex))
        if local:
            file = self._local_database.loc[name, 'o']
            if not file.closed:
                file.close()

            tmp = list(file.name)
            tmp.insert(file.name.find(name), 'stream_')
            streamfile_name = ''.join(tmp)
            try:
                os.remove(streamfile_name)
            except FileNotFoundError:
                pass
            except PermissionError:
                pass

            try:
                os.remove(file.name)
            except PermissionError:
                pass
            self._local_database.drop(name, inplace=True)

        if extern and deep:
            file = self._database.loc[name, 'o']
            if not file.closed:
                file.close()

            tmp = list(file.name)
            tmp.insert(file.name.find(name), 'stream_')
            streamfile_name = ''.join(tmp)
            try:
                os.remove(streamfile_name)
            except FileNotFoundError:
                pass
            except PermissionError:
                pass

            try:
                os.remove(file.name)
            except PermissionError:
                pass
            self._database.drop(name, inplace=True)
            self._database.to_pickle(Master.USER_PATH, compression='xz')
        gc.collect()


    def save_as(self, record: Union[str, pd.Series, Wrap, WrapGenerator], file_path: str=SAVE_PATH):
        '''
        Convert the record to the wav audio format.
        :param record: record can be the name of registered record, a pandas series or an wrap object
        :param file_path: The name or path of the wav file(auto detection).
        A new name for the record to be converted can be placed at the end of the address.
        :return:None
        '''
        rec_type = type(record)
        if rec_type is str:
            record = self.load(record, series=True)
        elif rec_type is Wrap or rec_type is WrapGenerator:
            record = record.get_data()
        elif rec_type is pd.Series:
            pass
        else:
            raise TypeError('please control the type of record')

        p0 = max(file_path.rfind('\\'), file_path.rfind('/'))
        p1 = file_path.rfind('.')
        if p0 < 0:
            p0 = 0

        if p1 <= 0:
            p1 = None

        if (p0 == len(file_path) - 1) or len(file_path) < 2:
            name = record.name
        else:
            # print(p0, p1, file_path[p0+1: p1])
            name = file_path[p0 : p1]

        name += '.wav'
        if p0:
            file_path = file_path[0: p0 + 1] + name
        else:
            file_path = Master.SAVE_PATH + name
        # print(file_path, p0, p1)
        file = record['o']
        file_pos = file.tell()
        data = file.read()
        file.seek(file_pos, 0)

        write_wav_file(file_path, data,
                       record['nchannels'], record['frameRate'],
                       Audio.get_sample_size(record['sampleFormat']))


    def save(self, name: str='None', save_all: bool=False):
        '''
        Save the preloaded record to the external database
        :param name: name of the preloaded record
        :param save_all: if true then it's tries to save all of the preloaded records
        :return:None
        '''
        if save_all:
            data = self._local_database.copy()
            for i in data.index:
                data.loc[i, 'o'] = Master.DATA_PATH + i + Master.BUFFER_TYPE
            self._database = self._database.append(data, verify_integrity=True)
        else:
            if name in self._database.index and not (name in self._local_database.index):
                raise ValueError(f'the same name already registered in the database file')
            try:
                data = self._local_database.loc[name].copy()
                data['o'] = Master.DATA_PATH + name + Master.BUFFER_TYPE
                self._database.loc[name] = data
            except KeyError:
                raise ValueError(f'can not found the {name} in the local database')

        self._database.to_pickle(Master.USER_PATH, compression='xz')

    def save_all(self):
        '''
        Save all of the preloaded records to the external database
        :return: None
        '''
        self.save(save_all=True)

    def get_exrecord_names(self) -> list:
        '''
        :return: list of the saved records in the external database
        '''
        return self.get_record_names(local_database=False)

    # 'local' or stable database
    def get_record_names(self, local_database: bool=True) -> list:
        '''
        :param local_database: if false then external database will be selected
        :return: list of the saved records in the external or internal database
        '''
        if local_database:
            return list(self._local_database.index)
        else:
            return list(self._database.index)


    def get_nperseg(self):
        return self._nperseg

    def get_nchannels(self):
        return self._nchannels

    def get_sample_rate(self):
        return self._frame_rate

    def stream(self, record: Union[str, Wrap, pd.Series, WrapGenerator],
               block_mode: bool=False,
               safe_load: bool=False,
               on_stop: callable=None,
               loop_mode: bool=False,
               use_cached_files=True,
               stream_mode:StreamMode = StreamMode.optimized) -> StreamControl:
        '''
        'Record' playback on the mainstream.

        Note:
         The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
         meaning that, The audio data storage methods can have different execution times based on the cached files.

        Note:
        The recorder can only capture normal streams(Non-optimized streams)

        :param record: It could be a predefined record name, a wrapped record, or a pandas series.
        :param block_mode: This can be true, in which case the current thread will be
         blocked as long as the stream is busy.
        :param safe_load: load an audio file and modify it according to the 'Master' attributes(like the frame rate
        , number oof channels, etc).
        :param on_stop: An optional callback is called at the end of the streaming process.
        :param loop_mode: 'Record' playback on the continually.
        :param use_cached_files: enable additional cache maintaining process.
        :return: A StreamControl object
        '''
        #loop mode just worked in unblocking mode
        cache_head_check_size = 20

        rec_type = type(record)
        if rec_type is str:
            record = self.load(record, series=True)
        else:
            if rec_type is Wrap or rec_type is WrapGenerator:
                assert rec_type is WrapGenerator or record.is_packed(), BufferError('The {} is not packed!'.format(record))
                record = record.get_data()

            elif rec_type is pd.Series:
                record = record.copy()
            else:
                raise TypeError('please control the type of record')


        if not self._mono_mode and record['nchannels'] > self._nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=record.name,
                                                                              ch0=record['nchannels'],
                                                                              ch1=self._nchannels))

        elif not self._frame_rate == record['frameRate']:
            warnings.warn('Warning, frame rate must be same')

        assert  type(record['o']) is io.BufferedRandom, TypeError('The record object is not standard')
        file = record['o']
        file_pos = file.tell()
        tmp = list(file.name)
        tmp.insert(file.name.find(record.name), 'stream_')
        streamfile_name = ''.join(tmp)
        try:
            if use_cached_files:
                record_size = os.path.getsize(file.name)
                streamfile_size = os.path.getsize(streamfile_name)
                file.seek(0, 0)
                pre_head = file.read(Master.CACHE_INFO + cache_head_check_size)
                file.seek(file_pos, 0)

                if record_size == streamfile_size:
                    pass
                elif (record_size - record['size']) == streamfile_size:
                    pass
                    file_pos = Master.CACHE_INFO
                else:
                    raise FileNotFoundError

                streamfile = open(streamfile_name, 'rb+')

                # check compability of two files
                post_head = streamfile.read(Master.CACHE_INFO + cache_head_check_size)
                streamfile.seek(file_pos, 0)
                if not post_head == pre_head:
                    streamfile.close()
                    raise FileNotFoundError

                if safe_load:
                    record['o'] = streamfile.read()
                    record = self._sync_record(record)
                    file.seek(file_pos, 0)
                record['o'] = streamfile
            else:
                raise FileNotFoundError

        except FileNotFoundError:
            record['o'] = file.read()
            if safe_load:
                record = self._sync_record(record)
            file.seek(file_pos, 0)

            record['o'] = streamfile = (
                write_to_cached_file(record['size'],
                            record['frameRate'],
                            record['sampleFormat'] if record['sampleFormat'] else 0,
                            record['nchannels'],
                            file_name=streamfile_name,
                            data=record['o'],
                            after_seek=(Master.CACHE_INFO, 0),
                            after_flush=True)
            )
            file_pos = Master.CACHE_INFO

        if block_mode:
            data_size = self._nperseg * self._nchannels * self._sample_width
            self._main_stream.acquire()
            while 1:
                in_data = streamfile.read(data_size)
                if not in_data:
                    break
                in_data = np.frombuffer(in_data, self._sample_width_format_str).astype('f')
                in_data = map_channels(in_data, self._nchannels, self._nchannels, self._mono_mode)
                self._main_stream.put(in_data)  
            self._main_stream.release()
            self._main_stream.clear()
            streamfile.seek(file_pos, 0)
            if on_stop:
                on_stop()

        else:
            return self._stream_control(record, on_stop, loop_mode, stream_mode)

    def _stream_control(self, *args):
        return StreamControl(self, *args)

    def mute(self):
        '''
        mute the mainstream
        '''
        self._master_mute_mode.set()


    def unmute(self):
        assert not self._exstream_mode.is_set(), "stream is busy"
        self._master_mute_mode.clear()
        self._main_stream.clear()  # Clear any stale data in the stream


    def is_muted(self):
        return self._master_mute_mode.is_set()


    def echo(self, record: Union[Wrap, str, pd.Series, WrapGenerator]=None,
             enable: bool=None, main_output_enable: bool=False):
        """
        Play "Record" on the operating system's default audio output.

         Note:
          If the 'record' argument takes the value None,
          the method controls the standard output activity of the master with the 'enable' argument.

        :param record: optional, default None;
         It could be a predefined record name, a wrapped record,
         or a pandas series.

        :param enable: optional, default None(trigger mode)
            determines that the standard output of the master is enable or not.

        :param main_output_enable:
            when the 'record' is not None, controls the standard output activity of the master
        :return: self
        """
        # Enable echo to system's default output or echo an pre recorded data
        # from local or external database
        if record is None:
            if enable is None:
                if self._echo.is_set():
                    self._echo.clear()
                else:
                    self._main_stream.clear()
                    self._echo.set()
            elif enable:
                self._main_stream.clear()
                self._echo.set()
            else:
                self._echo.clear()
        else:
            if type(record) is Wrap or type(record) is WrapGenerator:
                assert type(record) is WrapGenerator or record.is_packed()
                record = record.get_data()
            else:
                if type(record) is str:
                    record = self.load(record, series=True)
                elif type(record) is pd.Series:
                    pass
                else:
                    ValueError('unknown type')

            file = record['o']
            # file_pos = file.tell()
            file.seek(0, 0)
            data = file.read()
            file.seek(0, 0)
            # print(file)

            flg = False
            # assert self._nchannels == record['nchannels'] and self._sample_width == record['Sample Width']
            if not main_output_enable and self._echo.is_set():
                flg = True
                self._echo.clear()

            stdout_stream(data,
                 sample_format=record['sampleFormat'],
                 nchannels=record['nchannels'],
                 framerate=record['frameRate'])

            if flg:
                self._echo.set()

        # self.clean_cache()
        # return WrapGenerator(record)

    def disable_echo(self):
        '''
        disable the main stream echo mode
        :return: self
        '''
        return  self.echo(enable=False)


    def wrap(self, record: Union[str, pd.Series]):
        '''
        Create a Wrap object
        :param record: preloaded record or pandas series
        :return: Wrap object
        '''
        return WrapGenerator(self, record)

    def _cache(self):
        path = []
        expath = []
        base_len = len(Master.DATA_PATH)
        for i in self._database.index:
            path.append(self._database.loc[i, 'o'][base_len:])
        for i in self._local_database.index:
            expath.append(self._local_database.loc[i, 'o'].name[base_len:])
        path += [i for i in expath if not i in path]

        listdir = os.listdir(Master.DATA_PATH)
        listdir = list([Master.DATA_PATH + item for item in listdir if item.endswith(Master.BUFFER_TYPE)])
        # listdir = [Master.DATA_PATH + item  for item in listdir if not item in path]
        for i in path:
            j = 0
            while j < len(listdir):

                if i in listdir[j]:
                    del(listdir[j])
                else:
                    j += 1

        return listdir

    def clean_cache(self):
        '''
         The audio data maintaining process has additional cached files to reduce dynamic
         memory usage and improve performance, meaning that, The audio data storage methods
         can have different execution times based on the cached files.
         This function used to clean additional cache files.
        :return: self
        '''
        cache = self._cache()
        for i in cache:
            try:
                os.remove(i)
            except PermissionError:
                pass
        return self


    def _refresh(self, *args):
        win_len = self._nperseg

        if len(args):
            pass
        # primary filter frequency change
        else:

            if self._window_type:
                self._nhop = self._nperseg - self._noverlap
                self._nhop = int(self._nhop / self._data_chunk * win_len)

        if self._window is not None and (not self._nperseg == win_len):
            if type(self._window_type) is str or \
                    type(self._window_type) is tuple or \
                    type(self._window_type) == float:
                self._window = scisig.get_window(self._window_type, win_len)
            else:
                raise RefreshError("can't refresh static window")
            self._nperseg = win_len

        self._main_stream.acquire()

        if self._mono_mode:
            self._iwin_buffer = [np.zeros(win_len)]
            self._win_buffer = [np.zeros(win_len), np.zeros(win_len)]
        else:
            self._iwin_buffer = [np.zeros(win_len) for i in range(self._nchannels)]
            self._win_buffer = [[np.zeros(win_len), np.zeros(win_len)] for i in range(self._nchannels)]
        # format specifier used in struct.unpack    2
        # rms constant value                        3
        # sampwidth = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(self._sample_width)]  # 1 2 4 8

        self._record_period = self._nperseg / self._frame_rate

        self._refresh_ev.set()

        self._main_stream.clear()
        self._main_stream.release()

    def is_started(self):
        return self._pystream and True
    
    def get_window(self):
        """
        Retrieves information about the window used for processing.

        Returns:
            dict or None: A dictionary containing window information if available, or None if not set.
                - 'type': str, the type of window.
                - 'window': window data.
                - 'overlap': int, the overlap value.
                - 'size': int, the size of the window.
        """
        if self._window_type:
            return {'type': self._window_type,
                    'window': self._window,
                    'overlap': self._noverlap,
                    'size': len(self._window)}
        else:
            return None

    def disable_std_input(self):
        """
        Disables standard input stream by acquiring a lock.
        """
        # if not self._main_stream.locked():
        self._main_stream.acquire()

    def enable_std_input(self):
        """
        Enables standard input stream by clearing the lock.
        """
        # if self._main_stream.locked():
        self._main_stream.clear()
        self._main_stream.release()


    def add_pipeline(
            self, 
            pip, 
            name=None, 
            process_type: PipelineProcessType=PipelineProcessType.MAIN, 
            channel=None
            ):
        """
        Add a pipeline to the DSP core.

        Parameters:
        - name (str): Indicates the name of the pipeline.
        - pip (obj): Pipeline object or array of defined pipelines.
                    Note: In PipelineProcessType.MULTI_STREAM process type, pip must be an array of defined pipelines.
                    The size of the array must be the same as the number of input channels.
        - process_type (PipelineProcessType): Type of DSP core process.
                            Options: PipelineProcessType.MAIN, PipelineProcessType.BRANCH, PipelineProcessType.MULTI_STREAM
                            PipelineProcessType.MAIN: Processes input data and passes it to activated pipelines (if exist).
                            PipelineProcessType.BRANCH: Represents a branch pipeline with optional channel parameter.
                            PipelineProcessType.MULTI_STREAM: Represents a multi_stream pipeline mode. Requires an array of pipelines.
        - channel (obj): None or [0 to self.nchannel].
                        The input data passed to the pipeline can be a NumPy array in
                        (self.nchannel, 2[2 windows in 1 frame], self._nperseg) dimension [None]
                        or mono (2, self._nperseg) dimension.
                        In mono mode [self.nchannel = 1 or mono mode activated], channel must be None.

        Returns:
        - The pipeline must process data and return it to the core with the dimensions as same as the input.
        """
        stream_type = 'multithreading'

        stream = None

        if name is None:
            name = generate_timestamp_name()

        if process_type == PipelineProcessType.MAIN or process_type == PipelineProcessType.MULTI_STREAM:
            stream = Stream(self, pip, name, stream_type=stream_type, process_type=process_type)
            # n-dim main pipeline
            self.main_pipe_database.append(stream)
        elif process_type == PipelineProcessType.BRANCH:
            stream = Stream(self, pip, name, stream_type=stream_type, channel=channel, process_type=process_type)
            self.branch_pipe_database.append(stream)
                
        # Reserved for future versions
        else:
            pass

        return stream


    def set_pipeline(self, stream: Union[str, Stream]):
        if type(stream) is str:
            name = stream
        else:
            name = stream.name

        try:
            # Find the specified pipeline
            pipeline = next(obj for obj in self.main_pipe_database if obj.name == name)
            
            # Check if the pipeline is a main or multi_stream pipeline
            assert pipeline.process_type in [PipelineProcessType.MAIN, PipelineProcessType.MULTI_STREAM], \
                f"Pipeline {name} is not a MAIN or MULTI_STREAM type"

            try:
                # Disable the current pipeline if it exists
                self.disable_pipeline()
            except ValueError:
                pass
            

            # Acquire the lock before modifying the main stream
            self._main_stream.acquire()
            try:
                # Set the new pipeline
                self._main_stream.set(pipeline)

                if pipeline.process_type == PipelineProcessType.MULTI_STREAM:
                    # Ensure all sub-pipelines are alive in multi_stream mode
                    for i in pipeline.pip:
                        assert i.is_alive(), f'Error: Sub-pipeline in {name} is not Enabled!'
                        i.sync(self._master_sync)
                    
                    # Set each sub-pipeline as async
                    for i in pipeline.pip:
                        i.aasync()

                # Clear the main stream after setting the new pipeline
                # self._main_stream.clear()

            finally:
                # Always release the lock, even if an exception occurs
                self._main_stream.release()

        except StopIteration:
            raise ValueError(f"Pipeline {name} not found in main_pipe_database")
        except Exception as e:
            print(f"Error setting pipeline {name}: {str(e)}")
            # Attempt to restore the normal stream in case of error
            self.disable_pipeline()
            raise

    def disable_pipeline(self):
        if self._main_stream and hasattr(self._main_stream, 'pip'):
            # Stop processing new data
            self._main_stream.acquire()
            
            # Clear any remaining data in the pipeline
            if isinstance(self._main_stream.pip, Pipeline):
                self._main_stream.pip.clear()
            
            # Reset the main stream to the normal stream
            self._main_stream.set(self._normal_stream)
            
            # Clear the main stream
            self._main_stream.clear()
            
            # Release the lock
            self._main_stream.release()
        else:
            raise ValueError("No pipeline is currently set")             

    def clear_pipeline(self):
        if self._main_stream:
            self._main_stream.clear()
        for pipeline in self.main_pipe_database:
            pipeline.clear()
        for pipeline in self.branch_pipe_database:
            pipeline.clear()

    def _single_channel_windowing(self, data):
        """
        Performs windowing on a single channel of data.

        Parameters:
        - data (np.ndarray): The input data to be windowed.

        Returns:
        - np.ndarray: The windowed data.
        """
        # Check if the data is mono or multi-channel
        if self._window_type:
            retval = single_channel_windowing(
                data,
                self._win_buffer, 
                self._window,
                self._nhop
                )
        else:
            retval = data.astype(np.float64)
        return retval
    
    def _multi_channel_windowing(self, data):
        """
        Performs windowing on multiple channels of data.

        Parameters:
        - data (np.ndarray): The input data to be windowed.

        Returns:
        - np.ndarray: The windowed data.
        """
        # Check if the data is mono or multi-channel
        if self._window_type:
            retval = multi_channel_windowing(
                data,
                self._win_buffer,
                self._window,
                self._nhop,
                self._nchannels
                )
        else:
            retval = data.astype(np.float64)
        return retval
    

    def _single_channel_overlap(self, data):
        """
        Performs overlap-add on a single channel of data.

        Parameters:
        - data (np.ndarray): The input data to be processed.

        Returns:
        - np.ndarray: The processed data.
        """
        retval = single_channel_overlap(
            data,
            self._iwin_buffer,
            self._nhop
            )

        return retval
    

    def _multi_channel_overlap(self, data):
        """
        Performs overlap-add on multiple channels of data.

        Parameters:
        - data (np.ndarray): The input data to be processed.

        Returns:
        - np.ndarray: The processed data.
        """
        retval = multi_channel_overlap(
            data,
            self._iwin_buffer,
            self._nhop,
            self._nchannels,
            )

        return retval
    

    def set_window(self,
               window: object = 'hann',
               noverlap: int = None,
               NOLA_check: bool = True):
        
        if self._window_type == window and self._noverlap == noverlap:
            return
        else:
            self._window_type = window
            if noverlap is None:
                noverlap = self._nperseg // 2

            if type(window) is str or \
                    type(window) is tuple or \
                    type(window) == float:
                window = scisig.get_window(window, self._nperseg)

            elif window:
                assert len(window) == self._nperseg, 'control size of window'
                if NOLA_check:
                    assert scisig.check_NOLA(window, self._nperseg, noverlap)

            elif self._window_type is None:
                window = None
                self._window_type = None

        self._window = window
        # refresh
        self._noverlap = noverlap
        self._nhop = self._nperseg - self._noverlap
        if self._mono_mode:
            self._iwin_buffer = [np.zeros(self._nperseg)]
            self._win_buffer = [np.zeros(self._nperseg), np.zeros(self._nperseg)]
        else:
            self._iwin_buffer = [np.zeros(self._nperseg) for i in range(self._nchannels)]
            self._win_buffer = [[np.zeros(self._nperseg), np.zeros(self._nperseg)] for i in range(self._nchannels)]
        self._main_stream.clear()

    @staticmethod
    def get_default_input_device_info():
        """
        Retrieves information about the default input audio device.

        Returns:
            dict: A dictionary containing information about the default input device.
        """
        data = audiointerface.get_default_input_device_info()
        return data

    @staticmethod
    def get_default_output_device_info():
        """
        Retrieves information about the default output audio device.

        Returns:
            dict: A dictionary containing information about the default output device.
        """
        data = audiointerface.get_default_output_device_info()
        return data
    
    @staticmethod
    def get_device_count():
        """
        Gets the total number of available audio devices.

        Returns:
            int: The number of available audio devices.
        """
        data = audiointerface.get_device_count()
        return data
    
    @staticmethod
    def get_device_info_by_index(index: int):
        """
        Retrieves information about an audio device based on its index.

        Args:
            index (int): The index of the audio device.

        Returns:
            dict: A dictionary containing information about the specified audio device.
        """
        data = audiointerface.get_device_info_by_index(int(index))
        return data
    
    @staticmethod
    def get_input_devices():
        """
        Gets information about all available input audio devices.

        Returns:
            dict: A dictionary containing information about each available input device.
        """
        return audiointerface.get_input_devices()

    @staticmethod
    def get_output_devices():
        """
        Gets information about all available output audio devices.

        Returns:
            dict: A dictionary containing information about each available output device.
        """
        return audiointerface.get_output_devices()

    