"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""


import io
import scipy.signal as scisig
import threading
import queue
import numpy as np
from typing import Union, Callable
import warnings
import gc
import time
import os
from io import BufferedRandom
import traceback


from sudio.types import StreamMode, RefreshError, SampleFormatToLib, LibToSampleFormat
from sudio.types import SampleFormat, SampleFormatEnumToLib, LibSampleFormat, DitherMode
from sudio.types import LibSampleFormatEnumToSample, PipelineProcessType
from sudio.wrap.wrapgenerator import WrapGenerator
from sudio.wrap.wrap import Wrap
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
from sudio.audioutils.typeconversion import get_format_from_width
from sudio.pipeline import Pipeline
from sudio.metadata import AudioMetadataDatabase, AudioMetadata



class Master:

    CACHE_INFO = 4 * 8
    BUFFER_TYPE = '.bin'

    def __init__(self,
                 std_input_dev_id: int = None,
                 std_output_dev_id: int = None,
                 data_format: SampleFormat = SampleFormat.formatInt16,  # 16bit mode
                 nperseg: int = 500,
                 noverlap: int = None,
                 window: object = 'hann',
                 NOLA_check: bool = True,
                 input_dev_sample_rate: int = 48000,
                 input_dev_nchannels: int = None,
                 input_dev_callback: Callable = None,
                 output_dev_nchannels:int = None,
                 output_dev_callback: Callable = None,
                 buffer_size: int = 30,
                 audio_data_directory: str = './data/'
                 ):
        
        """
        Initialize the Master audio processing object.

        This class provides comprehensive audio processing capabilities, including
        device management, audio format handling, windowing, and pipeline processing.

        Parameters:
        -----------
        std_input_dev_id : int, optional
            Input device ID. If None, uses the system's default input device.
        
        std_output_dev_id : int, optional
            Output device ID. If None, uses the system's default output device.
        
        data_format : SampleFormat, optional
            Audio sample format. Default is 16-bit integer (formatInt16).
            Supported formats: formatFloat32, formatInt32, formatInt24, formatInt16, formatInt8, formatUInt8.
        
        nperseg : int, optional
            Number of samples per frame (window) in one channel. Default is 500.
        
        noverlap : int, optional
            Number of samples to overlap between frames. If None, defaults to nperseg // 2.
        
        window : str, float, tuple, or ndarray, optional
            Window type or array. Default is 'hann'.
            - If string: Specifies a scipy.signal window type (e.g., 'hamming', 'blackman').
            - If float: Interpreted as the beta parameter for scipy.signal.windows.kaiser.
            - If tuple: First element is the window name, followed by its parameters.
            - If ndarray: Direct specification of window values.
            - If None: Disables windowing.
        
        NOLA_check : bool, optional
            If True, checks if the window satisfies the Non-Zero Overlap Add (NOLA) constraint. Default is True.
        
        input_dev_sample_rate : int, optional
            Sample rate for the input device. Default is 48000 Hz. Only used if input_dev_callback is provided.
        
        input_dev_nchannels : int, optional
            Number of input channels. Required if input_dev_callback is provided.
        
        input_dev_callback : Callable, optional
            Custom callback function for input stream processing.
            Should return a numpy array of shape (nchannels, nperseg) with appropriate data format.
        
        output_dev_nchannels : int, optional
            Number of output channels. Required if output_dev_callback is provided.
        
        output_dev_callback : Callable, optional
            Custom callback function for output stream processing.
            Receives a numpy array of shape (nchannels, nperseg) with processed audio data.
        
        buffer_size : int, optional
            Size of the internal audio buffer in frames. Default is 30.

        Attributes:
        -----------
        Numerous internal attributes are set for managing audio streams, processing pipelines,
        and maintaining state. These are not typically accessed directly by users.

        Methods:
        --------
        Various methods are available for audio processing, pipeline management, and device control.
        Refer to individual method docstrings for details.

        Notes:
        ------
        - The class uses threading for efficient audio stream management.
        - Window functions are crucial for spectral analysis and should be chosen carefully.
        - NOLA constraint ensures proper reconstruction in overlap-add methods.
        - Custom callbacks allow for flexible input/output handling but require careful implementation.

        Examples:
        ---------
        # Basic initialization with default settings
        master = Master()

        # Custom configuration
        master = Master(std_input_dev_id=1, data_format=SampleFormat.formatFloat32,
                        nperseg=1024, window='hamming')

        # Using custom callbacks
        def input_callback(frame_count, time_info, status):
            # Custom input processing
            return processed_data

        master = Master(input_dev_callback=input_callback, input_dev_nchannels=2,
                        input_dev_sample_rate=44100)

        # Start audio processing
        master.start()
        """
        # ----------------------------------------------------- First Initialization Section -----------------------------------------------------

        # Set initial values for various attributes
        self._sound_buffer_size = buffer_size
        self._stream_type = StreamMode.optimized
        self._sample_format_type = data_format
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
        self._sample_rate = None
        self._audio_data_directory = audio_data_directory

        # Create necessary directories if they don't exist
        try:
            os.mkdir(self._audio_data_directory)
        except FileExistsError:
            pass
        
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
            input_channels = input_dev_nchannels
            self._sample_rate = input_dev_sample_rate

        elif std_input_dev_id is None:
            dev = self.get_default_input_device_info()
            self._sample_rate = int(dev['defaultSampleRate'])
            input_channels = dev['maxInputChannels']
            self._std_input_dev_id = dev['index']

        else:
            # Use specified input device ID
            self._std_input_dev_id = std_input_dev_id
            dev = self.get_device_info_by_index(std_input_dev_id)
            input_channels = dev['maxInputChannels']
            self._sample_rate = int(dev['defaultSampleRate'])


        data_format = data_format.value
        # Set number of output channels to a large value initially
        self._sample_width = Audio.get_sample_size(data_format)
        self._sample_format = data_format
        output_channels = int(1e6)

        if callable(output_dev_callback):
            self.output_dev_callback = output_dev_callback
            output_channels = output_dev_nchannels
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
            output_channels = dev['maxOutputChannels']

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._nchannels = self._output_channels
  


        # Raise an error if no input or output device is found
        if self._nchannels == 0:
            raise ValueError('No input or output device found')

        # Mute the audio initially
        self.mute()

        # Set initial values for various attributes
        self.branch_pipe_database_min_key = 0
        self._window_type = window
        self._nperseg = nperseg

        # Check window arguments
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
        # ----------------------------------------------------- Sample rate and pre-filter processing -----------------------------------------------------

        # Set data chunk size, frame rate, and other constants
        self._data_chunk = nperseg
        self._sample_width_format_str = '<i{}'.format(self._sample_width)

        # Adjust constant for floating-point data format
        if data_format == SampleFormat.formatFloat32.value:
            self._sample_width_format_str = '<f{}'.format(self._sample_width)

        # Initialize thread list
        self._threads = []

        # ----------------------------------------------------- User _database define -----------------------------------------------------

        self._local_database = AudioMetadataDatabase()

        # ----------------------------------------------------- Pipeline _database and other definitions section -----------------------------------------------------

        self._recordq = [threading.Event(), queue.Queue(maxsize=0)]
        self._queue = queue.Queue()

        self._echo_flag = threading.Event()
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


        self._overlap_buffer = [np.zeros(nperseg) for i in range(self._nchannels)]
        self._windowing_buffer = [[np.zeros(nperseg), np.zeros(nperseg)] for i in range(self._nchannels)]

        # Create Stream objects for main and normal streams
        data_queue = queue.Queue(maxsize=self._sound_buffer_size)
        self._normal_stream = Stream(self, data_queue, process_type=PipelineProcessType.QUEUE)
        self._main_stream = Stream(self, data_queue, process_type=PipelineProcessType.QUEUE)
        # ----------------------------------------------------- Miscellaneous initialization -----------------------------------------------------
        self._refresh_ev = threading.Event()

        # Master selection and synchronization
        self._master_sync = threading.Barrier(self._nchannels)
        self._pystream = False


    def start(self):
        """
        Start the audio processing stream.

        This method initializes and starts the audio stream, activating all configured
        audio processing pipelines and enabling real-time audio input/output.

        Returns:
        --------
        self : Master
            Returns the instance of the Master class, allowing for method chaining.

        Raises:
        -------
        AssertionError
            If the audio stream is already started.
        
        Other exceptions may be raised depending on the underlying audio library's
        behavior when opening the stream.

        Notes:
        ------
        - This method must be called after all desired configurations (like setting
        input/output devices, adding pipelines, etc.) have been made.
        - Once started, the audio stream runs in a separate thread, allowing for
        concurrent processing.
        - The method blocks until all active threads are initialized, ensuring
        the system is fully operational before returning.

        Examples:
        ---------
        # Basic usage
        master = Master()
        master.start()

        # Method chaining
        master = Master().start()

        # Start after configuration
        master = Master()
        master.add_pipeline(some_pipeline)
        master.set_window('hamming')
        master.start()
        """
        assert not self._pystream, 'Master is Already Started'
        try:
            self._pystream = self._audio_instance.open_stream(format=self._sample_format,
                                                   channels=self._input_channels,
                                                   rate=self._sample_rate,
                                                   frames_per_buffer=self._data_chunk,
                                                   input_device_index=self._std_input_dev_id,
                                                   input=True,
                                                   stream_callback=self._default_stream_callback)
            for i in self._threads:
                i.start()


        except:
            raise
        
        return self

    def _run(self, th_id):
        self._functions[self._queue.get()](th_id)
        self._queue.task_done()


    def _exstream(self):
        while self._exstream_mode.is_set():
            in_data: np.ndarray = np.frombuffer(
                self._stream_file.read(self._stream_data_size),
                self._sample_width_format_str).astype('f')
            try:
                in_data: np.ndarray = map_channels(in_data, self._nchannels, self._nchannels)
            except ValueError:
                return

            if not in_data.shape[-1] == self._data_chunk:
                if self._stream_loop_mode:
                    self._stream_file.seek(self._stream_data_pointer, 0)
                else:
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
                in_data = map_channels(in_data, self._input_channels, self._nchannels)
            except Exception as e:
                error_msg = f"Error in audio channel mapping: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())  # This will print the full stack trace

                return None, 0
        try:
            self._main_stream.acquire()
            self._main_stream.put(in_data)  
            self._main_stream.release()

        except Exception as e:
            print(f"Error in stream callback: {e}")
            print(f"in_data shape: {in_data.shape}, type: {type(in_data)}")
            print(f"self._input_channels: {self._input_channels}, self._nchannels: {self._nchannels}")
            print(traceback.format_exc())  # This will print the full stack trace
            return None, 1  # Indicate that an error occurred

        return None, 0


    def _custom_stream_callback(self, in_data, frame_count, time_info, status):  # PaCallbackFlags

        if self._exstream_mode.is_set():
            if not self._exstream():
                return None, 0

        elif self._master_mute_mode.is_set():
            in_data = get_mute_mode_data(self._nchannels, self._nperseg)
        else:
            in_data = self.input_dev_callback(frame_count, time_info, status).astype('f')
            try:
                in_data = map_channels(in_data, self._input_channels, self._nchannels)
            except Exception as e:
                error_msg = f"Error in audio channel mapping: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())  # This will print the full stack trace

        try:
            self._main_stream.acquire()
            self._main_stream.put(in_data)  
            self._main_stream.release()

        except Exception as e:
            print(f"Error in stream callback: {e}")
            print(f"in_data shape: {in_data.shape}, type: {type(in_data)}")
            print(f"self._input_channels: {self._input_channels}, self._nchannels: {self._nchannels}")
            print(traceback.format_exc())  # This will print the full stack trace
            return None, 1  # Indicate that an error occurred

        # self.a1 = time.perf_counter()
        return None, 0

    def _stream_output_manager(self, thread_id):
        stream_out = self._audio_instance.open_stream(
                            format=self._sample_format,
                            channels=self._output_channels,
                            output_device_index=self._output_device_index,
                            rate=self._sample_rate, input=False, output=True)

        stream_out.start_stream()
        rec_ev, rec_queue = self._recordq
        while 1:
            try:
                data = self._main_stream.get(timeout=0.0001)

                if rec_ev.is_set():
                    rec_queue.put_nowait(data)

                data = shuffle2d_channels(data)

                if self._echo_flag.is_set():
                    stream_out.write(data.tobytes())

                if self.output_dev_callback:
                    self.output_dev_callback(data)
            except Exception as e:
                pass
                # error_msg = f"Error in _stream_output_manager: {type(e).__name__}: {str(e)}"
                # print(error_msg)
                # print(traceback.format_exc())  

            if self._refresh_ev.is_set():
                self._refresh_ev.clear()
                # close current stream
                stream_out.close()
                # create new stream
                rate = self._sample_rate


                stream_out = self._audio_instance.open_stream(
                                    format=self._sample_format,
                                    channels=self._output_channels,
                                    rate=rate,
                                    input=False,
                                    output=True)
                                    

                stream_out.start_stream()
                self._main_stream.clear()

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


        p0 = max(filename.rfind('\\'), filename.rfind('/')) + 1
        p1 = filename.rfind('.')
        if p0 < 0:
            p0 = 0
        if p1 <= 0:
            p1 = None
        name = filename[p0: p1]
        if name in self._local_database.index():
            name = name + generate_timestamp_name()


        if safe_load and  record['nchannels'] > self._nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=name,
                                                                              ch0=record['nchannels'],
                                                                              ch1=self._nchannels))

        decoder = lambda: decode_audio_file(filename, sample_format, nchannels, sample_rate, DitherMode.NONE)

        record = (
        handle_cached_record(record,
                    TimedIndexedString(self._audio_data_directory + name + Master.BUFFER_TYPE,
                                      start_before=Master.BUFFER_TYPE),
                    self,
                    safe_load = safe_load,
                    decoder=decoder)
        )

        metadata = AudioMetadata(name, **record)
        self._local_database.add_record(metadata)
        gc.collect()
        return self.load(name)


    def add(self, record, safe_load=True):
        """
        Add a new record to the database.

        This method can handle various types of input including wrapped records,
        Record objects, or audio files in mp3, WAV, FLAC, or VORBIS format.

        Parameters:
        -----------
        record : Union[Wrap, str, AudioMetadata, WrapGenerator, Tuple[Union[Wrap, str, AudioMetadata, WrapGenerator], str]]
            The record to be added. Can be:
            - A Wrap or WrapGenerator object
            - A string representing the path to an audio file
            - AudioMetadata containing record data
            - A tuple containing the record and a custom name for the record

        safe_load : bool, optional
            If True (default), the audio file is loaded and modified according to
            the Master object's attributes. This ensures compatibility with the
            current audio processing settings.

        Returns:
        --------
        WrapGenerator
            A wrapped version of the added record.

        Notes:
        ------
        - If a tuple is passed as the record parameter, the second element will be
        used as the name for the new record. Otherwise, a name is automatically generated.
        - The method uses cached files to optimize memory usage and improve performance.
        As a result, execution times may vary depending on the state of these cached files.
        - Supported audio file formats: mp3, WAV, FLAC, VORBIS

        Raises:
        -------
        ValueError
            If the record type is not recognized or if there's an issue with file format.
        ImportError
            If the number of channels in the record is incompatible with the Master object's settings.

        Examples:
        ---------
        # Adding an audio file
        master.add("path/to/audio.wav")

        # Adding a wrapped record with a custom name
        wrapped_record = some_wrap_generator()
        master.add((wrapped_record, "my_custom_name"))

        # Adding AudioMetadata record
        series_record = AudioMetadata({...})  # Record data
        master.add(series_record)
        """
        name_type = type(record)
        if name_type is list or name_type is tuple:
            assert len(record) < 3

            rec_name = None
            if len(record) == 2:
                if type(rec_name) is str:
                    rec_name = record[1]

            if type(record[0]) is WrapGenerator:
                record = record[0].get_data()

            elif type(record[0]) is AudioMetadata:
                pass
            else:
                raise TypeError('record data should be an instance of AudioMetadata, or WrapGenerator')

            if rec_name and type(rec_name) is str:
                record.name = rec_name

            elif rec_name is None:
                record.name = generate_timestamp_name()

            else:
                raise ValueError('second item in the list is the name of the new record')
            
            return self.add(record, safe_load=safe_load)

        elif name_type is Wrap or name_type is WrapGenerator:
            record = record.get_data()
            return self.add(record, safe_load=safe_load)

        elif name_type is AudioMetadata:
            name = record.name
            if safe_load and record['nchannels'] > self._nchannels:
                raise ImportError('number of channel for the {name}({ch0})'
                                ' is not same as object channels({ch1})'.format(name=name,
                                                                                ch0=record['nchannels'],
                                                                                ch1=self._nchannels))
            if type(record['o']) is not BufferedRandom:
                if record.name in self._local_database.index():
                    record.name = generate_timestamp_name()

                if safe_load:
                    record = self._sync_record(record)

                f, newsize = (
                    write_to_cached_file(record['size'],
                                record['frameRate'],
                                record['sampleFormat'] if record['sampleFormat'] else 0,
                                record['nchannels'],
                                file_name=self._audio_data_directory + record.name + Master.BUFFER_TYPE,
                                data=record['o'],
                                pre_truncate=True,
                                after_seek=(Master.CACHE_INFO, 0),
                                after_flush=True,
                                sizeon_out=True)
                )
                record['o'] = f
                record['size'] = newsize

            elif (not (self._audio_data_directory + record.name + Master.BUFFER_TYPE) == record.name) or \
                    record.name in self._local_database.index():
                # new sliced Wrap data

                if record.name in self._local_database.index():
                    record.name = generate_timestamp_name()
                
                prefile = record['o']
                prepos = prefile.tell(), 0

                record['o'], newsize = (
                    write_to_cached_file(record['size'],
                                record['frameRate'],
                                record['sampleFormat'] if record['sampleFormat'] else 0,
                                record['nchannels'],
                                file_name=self._audio_data_directory + record.name + Master.BUFFER_TYPE,
                                data=prefile.read(),
                                after_seek=prepos,
                                after_flush=True,
                                sizeon_out=True)
                )
                prefile.seek(*prepos)
                record['size'] = newsize
            self._local_database.add_record(record)

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

        This method captures audio input for a given duration and stores it as a new record
        in the Master object's database.

        Parameters:
        -----------
        record_duration : float
            The duration of the recording in seconds.

        name : str, optional
            A custom name for the recorded audio. If None, a timestamp-based name is generated.

        Returns:
        --------
        Wrap
            A wrapped version of the recorded audio data.

        Raises:
        -------
        KeyError
            If the provided name already exists in the database.

        Notes:
        ------
        - The recording uses the current audio input settings of the Master object
        (sample rate, number of channels, etc.).
        - The recorded audio is automatically added to the Master's database and can be
        accessed later using the provided or generated name.
        - This method temporarily modifies the internal state of the Master object to
        facilitate recording. It restores the previous state after recording is complete.

        Examples:
        ---------
        # Record for 5 seconds with an auto-generated name
        recorded_audio = master.recorder(5)

        # Record for 10 seconds with a custom name
        recorded_audio = master.recorder(10, name="my_recording")

        # Use the recorded audio
        master.play(recorded_audio)
        """
        if name is None:
            name = generate_timestamp_name('record')
        elif name in self._local_database.index():
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
        sample = shuffle3d_channels(sample)

        # Prepare the record data
        record_data = {
            'size': sample.nbytes,
            'noise': None,
            'frameRate': self._sample_rate,
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
            file_name=f"{self._audio_data_directory}{name}{Master.BUFFER_TYPE}",
            data=sample.tobytes(),
            pre_truncate=True,
            after_seek=(Master.CACHE_INFO, 0),
            after_flush=True
        )

        # Update the size to include the cache info
        record_data['size'] += Master.CACHE_INFO

        # Add the record to the local database
        metadata = AudioMetadata(name, **record_data)
        self._local_database.add_record(metadata)

        return self.wrap(metadata.copy())


    def load(self, name: str, safe_load: bool=True,
         series: bool=False) -> Union[WrapGenerator, AudioMetadata]:
        '''
        This method used to load a predefined recoed from database. 
         Trying to load a record that was previously loaded, outputs a wrapped version
         of the named record.

        :param name: record name
        :param safe_load: if safe load is enabled then load function tries to load a record
         in the local database based on the master  settings, like the frame rate and etc.
        :param series: if enabled then Trying to load a record that was previously loaded, outputs data series
        of the named record.
        :return: (optional) Wrapped object, AudioMetadata
        '''

        if name in self._local_database.index():
            rec = self._local_database.get_record(name).copy()
            if series:
                return rec
            return self.wrap(rec)
        else:
            raise ValueError('can not found the {name} in database'.format(name=name))

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

        self._local_database.add_record(rec)
        if series:
            return rec
        return self.wrap(rec)
    

    def get_record_info(self, record: Union[str, WrapGenerator, Wrap]) -> dict:
        '''
        :param name: name of the registered record on database
        :return: information about saved record ['frameRate'  'sizeInByte' 'duration'
            'nchannels' 'nperseg' 'name'].
        '''
        if type(record) is WrapGenerator or Wrap:
            name = record.name
        elif type(record) is str:
            name = record
        else:
            raise TypeError('record must be an instance of WrapGenerator, Wrap or str')
        
        if name in self._local_database.index():
            rec = self._local_database.get_record(name)

        else:
            raise ValueError('can not found the {name} in database'.format(name=name))
        return {
            'frameRate': rec['frameRate'],
            'sizeInByte': rec['size'],
            'duration': rec['duration'],
            'nchannels': rec['nchannels'],
            'nperseg': rec['nperseg'],
            'name': name,
            'sampleFormat': LibSampleFormatEnumToSample[rec['sampleFormat']].name
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
        nchannels = nchannels if nchannels else self._nchannels
        sample_format = self._sample_format if sample_format == SampleFormat.formatUnknown else sample_format.value
        sample_rate = sample_rate if sample_rate else self._sample_rate

        out_type = 'ndarray' if output.startswith('n') else 'byte'

        buffer = []
        for record in targets:
            record: AudioMetadata = record.get_data().copy()
            assert isinstance(record, AudioMetadata)
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


    def _sync_record(self, rec):
        return synchronize_audio(rec, self._nchannels, self._sample_rate, self._sample_format)

    def del_record(self, record: Union[str, AudioMetadata, Wrap, WrapGenerator]):
        '''
        This function is used to delete a record from the internal database.
        :param name Union[str, AudioMetadata]: preloaded record.
        :return: None
        '''

        if type(record) is AudioMetadata or type(record) is Wrap or type(record) is WrapGenerator:
            name = record.name
        elif type(record) is str:
            name = record
        else:
            raise TypeError('please control the type of record')

        local = name in self._local_database.index()

        ex = ''
        assert local, ValueError(f'can not found the {name} in the '
                                            f'local {ex}databases'.format(name=name, ex=ex))
        if local:
            file = self._local_database.get_record(name)['o']
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
            self._local_database.remove_record(name)

        gc.collect()


    def export(self, record: Union[str, AudioMetadata, Wrap, WrapGenerator], file_path: str='./'):
        '''
        Convert the record to the wav audio format.
        :param record: record can be the name of registered record, an AudioMetadata or an wrap object
        :param file_path: The name or path of the wav file(auto detection).
        A new name for the record to be converted can be placed at the end of the address.
        :return:None
        '''
        rec_type = type(record)
        if rec_type is str:
            record = self.load(record, series=True)
        elif rec_type is Wrap or rec_type is WrapGenerator:
            record = record.get_data()
        elif rec_type is AudioMetadata:
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
            name = file_path[p0 : p1]

        name += '.wav'
        if p0:
            file_path = file_path[0: p0 + 1] + name
        else:
            file_path =  name
            
        file = record['o']
        file_pos = file.tell()
        data = file.read()
        file.seek(file_pos, 0)

        write_wav_file(file_path, data,
                       record['nchannels'], record['frameRate'],
                       Audio.get_sample_size(record['sampleFormat']))


    def get_record_names(self) -> list:
        '''
        :return: list of the saved records in database
        '''
        return list(self._local_database.index())



    def get_nperseg(self):
        return self._nperseg

    def get_nchannels(self):
        return self._nchannels

    def get_sample_rate(self):
        return self._sample_rate

    def stream(self, record: Union[str, Wrap, AudioMetadata, WrapGenerator],
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

        :param record: It could be a predefined record name, a wrapped record, or an AudioMetadata type object.
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

            elif rec_type is AudioMetadata:
                record = record.copy()
            else:
                raise TypeError('please control the type of record')


        if record['nchannels'] > self._nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=record.name,
                                                                              ch0=record['nchannels'],
                                                                              ch1=self._nchannels))

        elif not self._sample_rate == record['frameRate']:
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
                in_data = map_channels(in_data, self._nchannels, self._nchannels)
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


    def echo(self, record: Union[Wrap, str, AudioMetadata, WrapGenerator]=None,
             enable: bool=None, main_output_enable: bool=False):
        """
        Play "Record" on the operating system's default audio output.

         Note:
          If the 'record' argument takes the value None,
          the method controls the standard output activity of the master with the 'enable' argument.

        :param record: optional, default None;
         It could be a predefined record name, a wrapped record,
         or AudioMetadata.

        :param enable: optional, default None(trigger mode)
            determines that the standard output of the master is enable or not.

        :param main_output_enable:
            when the 'record' is not None, controls the standard output activity of the master
        :return: self
        """
        # Enable echo to system's default output or echo an pre recorded data
        if record is None:
            if enable is None:
                if self._echo_flag.is_set():
                    self._echo_flag.clear()
                else:
                    self._main_stream.clear()
                    self._echo_flag.set()
            elif enable:
                self._main_stream.clear()
                self._echo_flag.set()
            else:
                self._echo_flag.clear()
        else:
            if type(record) is Wrap or type(record) is WrapGenerator:
                assert type(record) is WrapGenerator or record.is_packed()
                record = record.get_data()
            else:
                if type(record) is str:
                    record = self.load(record, series=True)
                elif type(record) is AudioMetadata:
                    pass
                else:
                    ValueError('unknown type')

            file = record['o']
            # file_pos = file.tell()
            file.seek(0, 0)
            data = file.read()
            file.seek(0, 0)

            flg = False
            # assert self._nchannels == record['nchannels'] and self._sample_width == record['Sample Width']
            if not main_output_enable and self._echo_flag.is_set():
                flg = True
                self._echo_flag.clear()

            stdout_stream(data,
                 sample_format=record['sampleFormat'],
                 nchannels=record['nchannels'],
                 framerate=record['frameRate'])

            if flg:
                self._echo_flag.set()

        # self.clean_cache()
        # return WrapGenerator(record)

    def disable_echo(self):
        '''
        disable the main stream echo mode
        :return: self
        '''
        return  self.echo(enable=False)


    def wrap(self, record: Union[str, AudioMetadata]):
        '''
        Create a Wrap object
        :param record: preloaded record or AudioMetadata
        :return: Wrap object
        '''
        return WrapGenerator(self, record)

    def _cache(self):
        path = []
        expath = []
        base_len = len(self._audio_data_directory)
        for i in self._local_database.index():
            record = self._local_database.get_record(i)
            expath.append(record['o'].name[base_len:])
        path += [i for i in expath if i not in path]

        listdir = os.listdir(self._audio_data_directory)
        listdir = list([self._audio_data_directory + item for item in listdir if item.endswith(Master.BUFFER_TYPE)])
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


        self._overlap_buffer = [np.zeros(win_len) for i in range(self._nchannels)]
        self._windowing_buffer = [[np.zeros(win_len), np.zeros(win_len)] for i in range(self._nchannels)]

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
        if self._main_stream.locked():
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
                self._windowing_buffer, 
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
                self._windowing_buffer,
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
            self._overlap_buffer,
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
            self._overlap_buffer,
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

        self._overlap_buffer = [np.zeros(self._nperseg) for i in range(self._nchannels)]
        self._windowing_buffer = [[np.zeros(self._nperseg), np.zeros(self._nperseg)] for i in range(self._nchannels)]
        self._main_stream.clear()

    def get_sample_format(self)->SampleFormat:
        return self._sample_format_type

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

    