
# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


import scipy.signal as scisig
import threading
import queue
import numpy as np
from typing import Union, Callable
import warnings
import gc
import time
import os
from io import BufferedRandom, IOBase
import traceback
from pathlib import Path


from .types import StreamMode, RefreshError
from .types import PipelineProcessType
from .process.audio_wrap import AudioWrap
from .stream.stream import Stream
from .stream.streamcontrol import StreamControl
from .utils.strtool import generate_timestamp_name 
from .utils.timed_indexed_string import TimedIndexedString 
from .utils.window import multi_channel_overlap, single_channel_overlap
from .utils.window import multi_channel_windowing, single_channel_windowing
from .utils.channel import shuffle3d_channels, shuffle2d_channels, get_mute_mode_data, map_channels
from .utils.functional import type_check, filter_kwargs as fw, exclude_kwargs as ekw
from .utils.typeconversion import dtype_descriptor
from .audiosys import synchronize_audio, handle_cached_record, write_to_cached_file
from .pipeline import Pipeline
from .metadata import AudioRecordDatabase, AudioMetadata
from .io import SampleFormat, codec, write_to_default_output, AudioStream
from .io import AudioDeviceInfo, get_sample_size, FileFormat
from .io._webio import WebAudioIO
from .generator import Generator


class Master:

    CACHE_INFO_SIZE = 4 * 8
    BUFFER_TYPE = '.su'

    def __init__(self,
                 input_device_index: int = None,
                 output_device_index: int = None,
                 data_format: SampleFormat = SampleFormat.SIGNED16,
                 nperseg: int = 500,
                 noverlap: int = None,
                 window: object = 'hann',
                 NOLA_check: bool = True,
                 input_dev_sample_rate: int = 44100,
                 input_dev_nchannels: int = 2,
                 input_dev_callback: Callable = None,
                 output_dev_nchannels:int = 2,
                 output_dev_callback: Callable = None,
                 buffer_size: int = 30,
                 audio_data_directory: str = './sudio/',
                 ):
        
        """
        The `Master` class is responsible for managing audio data streams, applying windowing, 
        and handling input/output devices. This class provides various methods for processing, 
        recording, and playing audio data, with functionality to handle different audio formats, 
        sample rates, and device configurations.


		Parameters
        ----------
        - input_device_index : int, optional
            Index of the input audio device. If None, uses the system's default input device.
        - output_device_index : int, optional
            Index of the output audio device. If None, uses the system's default output device.
        - data_format : SampleFormat, default=SampleFormat.SIGNED16
            Audio sample format (e.g., FLOAT32, SIGNED16, UNSIGNED8)
        - nperseg : int, default=500
            Number of samples per segment for windowing and processing
        - noverlap : int, optional
            Number of overlapping samples between segments. 
            If None, defaults to half of nperseg.
        - window : str, float, tuple, or ndarray, default='hann'
            Window function type for signal processing:

            - String: scipy.signal window type (e.g., 'hamming', 'blackman')
            - Float: Beta parameter for Kaiser window
            - Tuple: Window name with parameters
            - ndarray: Custom window values
            - None: Disables windowing

        - NOLA_check : bool, default=True
            Perform Non-Overlap-Add (NOLA) constraint verification
        - input_dev_sample_rate : int, default=44100
            Input device sample rate in Hz. Behavior depends on input:

            - If a specific value is provided: Uses the given sample rate
            - If None: Automatically selects the default sample rate of the input device(if it exist)
            - If the selected rate is unsupported, raises an error
            - Recommended range typically between 8000 Hz and 96000 Hz

        - input_dev_nchannels : int, default=2
            Number of input device channels
        - input_dev_callback : callable, optional
            Custom callback for input device processing
        - output_dev_nchannels : int, default=2
            Number of output device channels
        - output_dev_callback : callable, optional
            Custom callback for output device processing
        - buffer_size : int, default=30
            Size of the audio stream buffer
        - audio_data_directory : str, default='./sudio/'
            Directory for storing audio data files

        Notes
        -----
        
        - Various methods are available for audio processing, pipeline management, and device control.
          Refer to individual method docstrings for details.
        - The class uses multi-threading for efficient audio stream management.
        - Window functions are crucial for spectral analysis and should be chosen carefully.
        - NOLA constraint ensures proper reconstruction in overlap-add methods.
        - Custom callbacks allow for flexible input/output handling but require careful implementation.
        """

        self._sound_buffer_size = buffer_size
        self._stream_type = StreamMode.optimized
        self._sample_format_type = data_format
        self.input_dev_callback = None
        self.output_dev_callback = None
        self._default_stream_callback = self._input_stream_callback
        self._sample_format = data_format
        self._nperseg = nperseg
        self._data_chunk = nperseg
        self._sample_width = get_sample_size(data_format)
        self._sample_width_format_str = dtype_descriptor(data_format)

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
        try:
            os.mkdir(self._audio_data_directory)
        except FileExistsError:
            pass

        self._output_device_index = None
        input_channels = int(1e6)
        output_channels = int(1e6)

        if input_dev_sample_rate is not None:
            assert isinstance(input_dev_sample_rate, int), TypeError('control input_dev_sample_rate')

        try:
            if callable(input_dev_callback):
                self.input_dev_callback = input_dev_callback
                self._default_stream_callback = self._custom_stream_callback

            try:
                count = self.get_device_count()
            except Exception as e:
                raise EOFError(f'{e}')
            if count < 1: raise EOFError('No audio input devices were found')
        
            if input_device_index is None:
                try:
                    dev = self.get_default_input_device_info()
                except:
                    raise EOFError(f'No default input info or device')
                
            else:
                try:
                    int(input_device_index)
                except ValueError:
                    raise ValueError('invalid literal for input_device_index.')
                
                try:
                    dev = self.get_device_info_by_index(input_device_index)
                except:
                    raise EOFError(f'No input info or device for index {input_device_index}')

            try:
                self._sample_rate = int(dev.default_sample_rate) if input_dev_sample_rate is None else input_dev_sample_rate
                assert self._sample_rate > 0.0
            except:
                raise EOFError(f'Input device samplerate unsupported')
            
            try:
                input_channels = int(dev.max_input_channels)
                assert input_channels > 0
            except:
                raise EOFError(f'Input device channels unsupported')
            
            try:
                self._input_device_index = int(dev.index)

                audio_input_stream = AudioStream()
                try:
                    audio_input_stream.open(
                        input_dev_index = self._input_device_index,
                        sample_rate = self._sample_rate,
                        format=self._sample_format,
                        input_channels=input_channels,
                        frames_per_buffer=self._data_chunk,
                        enable_input=True,
                        enable_output=False,
                        ) 
                finally:
                    audio_input_stream.close()
                
            except Exception as e:
                raise EOFError(f'Input device of {self._input_device_index}  {e}')


        except EOFError as e:
            warnings.warn(f"{e}. some features will be unavailable.")
            self._input_device_index = None

            try:
                input_channels = int(input_dev_nchannels)  
                assert input_channels > 0
            except:
                raise ValueError('Unsupported input dev channels.')
            
            try:
                self._sample_rate = int(input_dev_sample_rate)  # Default sample rate
                assert self._sample_rate > 0
            except:
                raise ValueError('Unsupported samplerate.')


        try:
            if callable(output_dev_callback):
                self.output_dev_callback = output_dev_callback
            try:
                count = self.get_device_count()
            except Exception as e:
                raise EOFError(f'{e}')
            if count < 1: raise EOFError('No audio output devices were found')
            
            if output_device_index is None:
                try:
                    dev = self.get_default_output_device_info()
                except:
                    raise EOFError(f'No default output info or device')
                
            else:
                try:
                    int(output_device_index)
                except ValueError:
                    raise ValueError('invalid literal for output_device_index.')
                    
                try:
                    dev = self.get_device_info_by_index(output_device_index)
                except:
                    raise EOFError(f'No output info or device for index {output_device_index}')

            try:
                output_channels = dev.max_output_channels
                assert output_channels > 0
            except:
                raise EOFError(f'Output device channels unsupported')
            try:
                self._output_device_index = dev.index

                audio_output_stream = AudioStream()
                try:
                    audio_output_stream.open(
                        output_dev_index = self._output_device_index,
                        sample_rate = self._sample_rate,
                        format=self._sample_format,
                        output_channels=output_channels,
                        frames_per_buffer=self._data_chunk,
                        enable_input=False,
                        enable_output=True,
                        ) 
                finally:
                    audio_output_stream.close()
                    
            except Exception as e:
                raise EOFError(f'Output device of {self._output_device_index} {e}')

        except EOFError as e:
            warnings.warn(f"{e}. some features will be unavailable.")
            self._output_device_index = None

            try:
                output_channels = int(output_dev_nchannels)  
                assert output_channels > 0
            except:
                raise ValueError('Unsupported output dev channels.')

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._nchannels = self._output_channels

        if self._nchannels == 0:
            raise ValueError('No input or output device found')
        self.mute()
        self.branch_pipe_database_min_key = 0
        self._window_type = window

        if noverlap is None:
            noverlap = nperseg // 2
        if type(self._window_type) is str or type(self._window_type) is tuple or type(self._window_type) == float:
            window = scisig.get_window(window, nperseg)
        if self._window_type:
            assert len(window) == nperseg, 'Control size of window'
            if NOLA_check:
                assert scisig.check_NOLA(window, nperseg, noverlap)
        elif self._window_type is None:
            window = None
            self._window_type = None

        self._threads = []
        self._local_database = AudioRecordDatabase()
        self._recordq = [threading.Event(), queue.Queue(maxsize=0)]
        self._queue = queue.Queue()
        self._echo_flag = threading.Event()
        self.main_pipe_database = []
        self.branch_pipe_database = []
        self._functions = []
        for i in range(len(self._functions)):
            self._threads.append(threading.Thread(target=self._run, daemon=True, args=(len(self._threads),)))
            self._queue.put(i)
        # Set parameters for windowing
        self._nhop = nperseg - noverlap
        self._noverlap = noverlap
        self._window = window
        self._overlap_buffer = [np.zeros(nperseg) for i in range(self._nchannels)]
        self._windowing_buffer = [[np.zeros(nperseg), np.zeros(nperseg)] for i in range(self._nchannels)]
        data_queue = queue.Queue(maxsize=self._sound_buffer_size)
        self._normal_stream = Stream(self, data_queue, process_type=PipelineProcessType.QUEUE)
        self._main_stream = Stream(self, data_queue, process_type=PipelineProcessType.QUEUE)
        self._refresh_ev = threading.Event()
        # Master selection and synchronization
        self._master_sync = threading.Barrier(self._nchannels)
        self._audio_input_stream = False
        self._audio_output_stream = False
        self.clean_cache()


    def start(
            self,
            ):
        """
        Starts the audio input and output streams and launches any registered threads.

        Returns
        -------

        self : Master
            Returns the instance of the Master class for method chaining.
        """
        assert self._output_device_index, "No output device, streaming unavailable"
        assert self._input_device_index, "No input device, streaming unavailable"
        assert not self._audio_input_stream, 'Master is Already Started'


        try:
            self._audio_input_stream = AudioStream()
            self._audio_output_stream = AudioStream()

            self._audio_input_stream.open(
                input_dev_index = self._input_device_index,
                sample_rate = self._sample_rate,
                format=self._sample_format,
                input_channels=self._input_channels,
                frames_per_buffer=self._data_chunk,
                enable_input=True,
                enable_output=False,
                input_callback=self._default_stream_callback
                )
            
            self._audio_output_stream.open(
                    output_dev_index=self._output_device_index,
                    frames_per_buffer=self._data_chunk,
                    sample_rate=self._sample_rate, 
                    format=self._sample_format,
                    output_channels=self._output_channels,
                    enable_input=False, 
                    enable_output=True,
                    output_callback=self._output_stream_callback
                )

            self._audio_output_stream.start()
            self._audio_input_stream.start()
            
            for i in self._threads:
                i.start()


        except:
            raise

    def _run(self, th_id):
        """
        Grabs a function from the queue and runs it with the given thread ID, 
        then marks the task as complete. Essentially a simple worker method 
        to keep things moving smoothly.
        """
        self._functions[self._queue.get()](th_id)
        self._queue.task_done()


    def _exstream(self):
        """
        Handles external stream file reading and processing.

        Reads audio data chunk by chunk from a file, manages looping or 
        stopping behavior, and puts the data into the main stream. Handles 
        edge cases like reaching end of file or needing to loop back.
        """
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
        """
        Safely releases the main stream lock if it's currently locked.

        Prevents potential deadlocks by checking and releasing the stream 
        lock only if it's actually held. A little insurance policy for 
        thread safety.
        """
        if self._main_stream.locked():
            self._main_stream.release()


    def _input_stream_callback(self, in_data, frame_count, format):  
        """
        Processes incoming audio input stream data.

        Handles different modes like external streaming, mute, and normal 
        input. Manages channel mapping, puts data into the main stream, and 
        handles potential errors gracefully. The Swiss Army knife of 
        audio input processing.
        """

        if self._exstream_mode.is_set():
            self._exstream()
            return None, True
        
        elif self._master_mute_mode.is_set():
            in_data = get_mute_mode_data(self._nchannels, self._nperseg)
        else:
            in_data = np.frombuffer(in_data, self._sample_width_format_str).astype('f')
            try:
                in_data = map_channels(in_data, self._input_channels, self._nchannels)
            except Exception as e:
                error_msg = f"Error in audio channel mapping: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())

                return None, True
        try:
            self._main_stream.acquire()
            self._main_stream.put(in_data)  
            self._main_stream.release()

        except Exception as e:
            # print(f"Error in stream callback: {e}")
            # print(f"in_data shape: {in_data.shape}, type: {type(in_data)}")
            # print(f"self._input_channels: {self._input_channels}, self._nchannels: {self._nchannels}")
            # print(traceback.format_exc())  # This will print the full stack trace
            return None, False  # Indicate that an error occurred

        return None, True


    def _custom_stream_callback(self, in_data, frame_count, format):  

        if self._exstream_mode.is_set():
            if not self._exstream():
                return None, 0

        elif self._master_mute_mode.is_set():
            in_data = get_mute_mode_data(self._nchannels, self._nperseg)
        else:
            in_data = self.input_dev_callback(frame_count, format).astype('f')
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

    def _output_stream_callback(self, frame_count, format):
        rec_ev, rec_queue = self._recordq
        while 1:
            data = None

            try:
                data = self._main_stream.get(timeout=0.0001)

                if rec_ev.is_set():
                    rec_queue.put_nowait(data)
            except:
                pass

            try:
                if data is not None:
                    data = shuffle2d_channels(data)

                    if self.output_dev_callback:
                        self.output_dev_callback(data)
                    
                    if self._echo_flag.is_set():
                        return (data.tobytes(), True)

            except Exception as e:
                error_msg = f"Error in _output_stream_callback: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())  

            if self._refresh_ev.is_set():
                self._refresh_ev.clear()
                # close current stream
                try:
                    self._audio_output_stream.stop()
                    self._audio_output_stream.close()
                except:
                    pass
                    # print(f"Error closing stream: {type(e).__name__}: {str(e)}")
                    # print(traceback.format_exc())    
                            
                # create new stream
                try:
                    self._audio_output_stream.open(
                        output_dev_index=self._output_device_index,
                        frames_per_buffer=self._data_chunk,
                        sample_rate=self._sample_rate, 
                        format=self._sample_format,
                        output_channels=self._output_channels,
                        enable_input=False, 
                        enable_output=True,
                        output_callback=self._output_stream_callback
                    )
                    self._audio_output_stream.start()

                except Exception as e:
                    print(f"Error opening new stream: {type(e).__name__}: {str(e)}")  
                    print(traceback.format_exc())                     

                self._main_stream.clear()


    def add_file(self, filename: str, sample_format: SampleFormat = SampleFormat.UNKNOWN,
                nchannels: int = None, sample_rate: int = None ,safe_load=True):
        '''
        Adds an audio file to the database with optional format and parameter adjustments.

        Supports WAV, FLAC, VORBIS, and MP3 file formats.

        Parameters:
        -----------

        filename : str
            Path/name of the audio file to add.
        sample_format : SampleFormat, optional
            Desired sample format. Defaults to automatically detecting or using master's format.
        nchannels : int, optional
            Number of audio channels. Defaults to file's original channel count.
        sample_rate : int, optional
            Desired sample rate. Defaults to file's original rate.
        safe_load : bool, default True
            If True, modifies file to match master object's audio attributes.

        Returns:
        --------

        AudioWrap
            Wrapped audio file with processed metadata and data.

        Raises:
        -------

        ImportError
            If safe_load is True and file's channel count exceeds master's channels.

        '''
        info = codec.get_file_info(filename)
        if safe_load:
            sample_format = self._sample_format
        elif sample_format is SampleFormat.UNKNOWN:
            if info.sample_format is SampleFormat.UNKNOWN:
                sample_format = self._sample_format
            else:
                sample_format = info.sample_format
        else:
            sample_format = sample_format

        if nchannels is None:
            nchannels = info.nchannels
        if safe_load:
          sample_rate = self._sample_rate
        elif sample_rate is None:
            sample_rate = info.sample_rate

        p0 = max(filename.rfind('\\'), filename.rfind('/')) + 1
        p1 = filename.rfind('.')
        if p0 < 0:
            p0 = 0
        if p1 <= 0:
            p1 = None
        name = filename[p0: p1]
        if name in self._local_database.index():
            name = name + generate_timestamp_name()

        record = AudioMetadata(name, **{
                'size': None,
                'frameRate': sample_rate,
                'o': None,
                'sampleFormat': sample_format,
                'nchannels': nchannels,
                'duration': info.duration,
            }
        )

        if safe_load and  record.nchannels > self._nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=name,
                                                                              ch0=record.nchannels,
                                                                              ch1=self._nchannels))
        decoder = lambda: codec.decode_audio_file(filename, sample_format, nchannels, sample_rate)
        path_server = TimedIndexedString(
                self._audio_data_directory + name + Master.BUFFER_TYPE,
                start_before=Master.BUFFER_TYPE
                )
        orphaned_cache = self.prune_cache()

        record = handle_cached_record(
            record=record,
            path_server=path_server,
            orphaned_cache=orphaned_cache,
            cache_info_size=self.__class__.CACHE_INFO_SIZE,
            decoder=decoder,
            sync_sample_format_id=self._sample_format,
            sync_nchannels=self._nchannels,
            sync_sample_rate=self._sample_rate,
            safe_load = safe_load,
            )
        self._local_database.add_record(record)
        gc.collect()
        return self.load(name)


    def add(self, record, safe_load:bool=True):
        """
        Adds audio data to the local database from various input types.

        Supports adding:
        - AudioWrap objects
        - Audio file paths (mp3, WAV, FLAC, VORBIS)
        - AudioMetadata records

        Parameters:
        -----------

        record : Union[AudioWrap, str, AudioMetadata]
            The audio record to add to the database.
        safe_load : bool, default True
            If True, ensures record matches master object's audio specifications.

        Returns:
        --------

        AudioWrap
            Processed and wrapped audio record.

        Raises:
        -------

        ImportError
            If safe_load is True and record's channel count exceeds master's.
        TypeError
            If record is not a supported type.

        Notes:
        ------

        - Uses cached files for optimized memory management
        - Execution time may vary based on cached file state

        Examples:
        ---------

        .. code-block:: python
            
            master = sudio.Master()
            audio = master.add('./alan kujay.mp3')

            master1 = sudio.Master()
            audio1 = master1.add(audio)

            master.echo(audio[:10])
            master1.echo(audio1[:10])
        """
        if isinstance(record, AudioWrap):
            record = record.get_data()
            return self.add(record, safe_load=safe_load)

        elif isinstance(record, AudioMetadata):
            name = record.name
            if safe_load and record.nchannels > self._nchannels:
                raise ImportError('number of channel for the {name}({ch0})'
                                ' is not same as object channels({ch1})'.format(name=name,
                                                                                ch0=record.nchannels,
                                                                                ch1=self._nchannels))
            if type(record.o) is not BufferedRandom:
                if (not record.name.strip()) or (record.name in self._local_database.index()):
                    record.name = generate_timestamp_name()

                if safe_load:
                    record = self._sync_record(record)

                f, newsize = (
                    write_to_cached_file(record.size,
                                record.frameRate,
                                record.sampleFormat if record.sampleFormat else 0,
                                record.nchannels,
                                file_name=self._audio_data_directory + record.name + Master.BUFFER_TYPE,
                                data=record.o,
                                pre_truncate=True,
                                after_seek=(Master.CACHE_INFO_SIZE, 0),
                                after_flush=True,
                                size_on_output=True)
                )
                record.o = f
                record.size = newsize

            elif (not (self._audio_data_directory + record.name + Master.BUFFER_TYPE) == record.name) or \
                    record.name in self._local_database.index():
                if record.name in self._local_database.index():
                    record.name = generate_timestamp_name()
                
                prefile = record.o
                prepos = prefile.tell(), 0

                record.o, newsize = (
                    write_to_cached_file(record.size,
                                record.frameRate,
                                record.sampleFormat if record.sampleFormat else 0,
                                record.nchannels,
                                file_name=self._audio_data_directory + record.name + Master.BUFFER_TYPE,
                                data=prefile.read(),
                                after_seek=prepos,
                                after_flush=True,
                                size_on_output=True)
                )
                prefile.seek(*prepos)
                record.size = newsize
            self._local_database.add_record(record)

            gc.collect()
            return self.wrap(record)

        elif isinstance(record, str):
            gc.collect()
            return self.add_file(record, safe_load=safe_load)

        else:
            raise TypeError('The record must be an audio file, data frame or a AudioWrap object')


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

        :return: AudioWrap instance


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

        **Record for 5 seconds with an auto-generated name**

         >>> recorded_audio = master.recorder(5)

        **Record for 10 seconds with a custom name**

         >>> recorded_audio = master.recorder(10, name="my_recording")

        **Use the recorded audio**

         >>> master.echo(recorded_audio)
        """

        assert self.is_started(), "instance not started, use start()"

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

        sample = np.array(recorded_data)
        sample = shuffle3d_channels(sample)

        metadata = AudioMetadata(
            name, 
            **{
                'size': sample.nbytes,
                'frameRate': self._sample_rate,
                'nchannels': self._nchannels,
                'sampleFormat': self._sample_format,
                'duration': record_duration,
            }
        )

        metadata.o = write_to_cached_file(
            metadata.size,
            metadata.frameRate,
            metadata.sampleFormat,
            metadata.nchannels,
            file_name=f"{self._audio_data_directory}{name}{Master.BUFFER_TYPE}",
            data=sample.tobytes(),
            pre_truncate=True,
            after_seek=(Master.CACHE_INFO_SIZE, 0),
            after_flush=True
        )

        metadata.size += Master.CACHE_INFO_SIZE
        self._local_database.add_record(metadata)
        return self.wrap(metadata.copy())


    def load(self, name: str, safe_load: bool=True,
         series: bool=False) -> Union[AudioWrap, AudioMetadata]:
        '''
        Loads a record from the local database. Trying to load a record that was previously loaded, 
        outputs a wrapped version of the named record.

        :param name: record name
        :param safe_load: Flag to safely load the record. if safe load is enabled then load function tries to load a record
         in the local database based on the master settings, like the frame rate and etc (default: `True`).
        :param series:  Return the record as a series (default: `False`).
        :return: (optional) AudioWrap object, AudioMetadata
        
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
            rec.o = file.read()
            rec = self._sync_record(rec)
            rec.o = file

        if file_size > rec.size:
            file.seek(rec.size, 0)
        else:
            file.seek(Master.CACHE_INFO_SIZE, 0)

        self._local_database.add_record(rec)
        if series:
            return rec
        return self.wrap(rec)
    

    def get_record_info(self, record: Union[str, AudioWrap]) -> dict:
        '''
        Retrieves metadata for a given record.

        :param record: The record (str, or AudioWrap) whose info is requested.

        :return: information about saved record in a dict format ['frameRate'  'sizeInByte' 'duration'
            'nchannels' 'name'].
        '''
        if isinstance(record, AudioWrap):
            name = record.name
        elif isinstance(record, str):
            name = record
        else:
            raise TypeError('record must be an instance of AudioWrap or str')
        
        if name in self._local_database.index():
            rec = self._local_database.get_record(name)

        else:
            raise ValueError('can not found the {name} in database'.format(name=name))
        return {
            'frameRate': rec['frameRate'],
            'sizeInByte': rec['size'],
            'duration': rec['duration'],
            'nchannels': rec['nchannels'],
            'name': name,
            'sampleFormat': rec['sampleFormat'].name
        }

    def _syncable(self,
                 *target,
                 nchannels: int = None,
                 sample_rate: int = None,
                 sample_format_id: int = None):
        '''
         Determines whether the target can be synced with specified properties or not

        :param target: wrapped objects
        :param nchannels: number of channels; if the value is None, the target will be compared to the 'self' properties.
        :param sample_rate: sample rate; if the value is None, the target will be compared to the 'self' properties.
        :param sample_format_id: if the value is None, the target will be compared to the 'self' properties.


        :return: only objects that need to be synchronized.
        '''
        nchannels = nchannels if nchannels else self._nchannels
        sample_format_id = self._sample_format if sample_format_id is None else sample_format_id
        sample_rate = sample_rate if sample_rate else self._sample_rate

        buffer = []
        for rec in target:
            assert isinstance(rec, AudioWrap)
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
                 sample_format: SampleFormat = SampleFormat.UNKNOWN):
        '''
         Prepares a list of targets to be synchronized. Determines whether the target can be synced with specified properties or not

        :param target: Targets to sync. wrapped objects
        :param nchannels: Number of channels (default: `None`); if the value is None, the target will be compared to the 'self' properties.
        :param sample_rate: Sample rate (default: `None`); if the value is None, the target will be compared to the 'self' properties.
        :param sample_format: Sample format (default: `SampleFormat.UNKNOWN`); if the value is None, the target will be compared to the 'self' properties.

        :return: only objects that need to be synchronized.
        '''
        return self._syncable(*target, nchannels=nchannels, sample_rate=sample_rate, sample_format_id=sample_format)


    def sync(self,
            *targets,
            nchannels: int=None,
            sample_rate: int=None,
            sample_format: SampleFormat=SampleFormat.UNKNOWN,
            output='wrapped'):
        '''
        Synchronizes audio across multiple records. Synchronizes targets in the AudioWrap object format with the specified properties.

        :param targets: Records to sync. wrapped objects.
        :param nchannels: Number of channels (default: `None`); if the value is None, the target will be synced to the 'self' properties.
        :param sample_rate: Sample rate (default: `None`); if the value is None, the target will be synced to the 'self' properties.
        :param sample_format: if the value is None, the target will be synced to the 'self' properties.
        :param output: can be 'wrapped', 'series' or 'ndarray_data'

        :return: synchronized objects.
        '''
        nchannels = nchannels if nchannels else self._nchannels
        sample_format = self._sample_format if sample_format == SampleFormat.UNKNOWN else sample_format
        sample_rate = sample_rate if sample_rate else self._sample_rate

        out_type = 'ndarray' if output.startswith('n') else 'byte'

        buffer = []
        for record in targets:
            record: AudioMetadata = record.get_data().copy()
            assert isinstance(record, AudioMetadata)
            main_file: BufferedRandom = record.o
            main_seek = main_file.tell(), 0
            record.o = main_file.read()
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

    def del_record(self, record: Union[str, AudioMetadata, AudioWrap]):
        '''
        Deletes a record from the local database.

        :param record: Record to delete (str, AudioMetadata, or AudioWrap).
        '''

        if isinstance(record, (AudioMetadata, AudioWrap)):
            name = record.name
        elif isinstance(record, str):
            name = record
        else:
            raise TypeError('please control the type of record')

        local = name in self._local_database.index()

        ex = ''
        assert local, ValueError(f'can not found the {name} in the '
                                            f'local {ex}databases'.format(name=name, ex=ex))
        if local:
            file = self._local_database.get_record(name).o
            if not file.closed:
                file.close()

            tmp = list(file.name)
            tmp.insert(file.name.find(name), 'stream_')
            streamfile_name = ''.join(tmp)
            try:
                os.remove(streamfile_name)
            except (FileNotFoundError, PermissionError):
                pass
            try:
                os.remove(file.name)
            except (PermissionError, FileNotFoundError):
                pass
            self._local_database.remove_record(name)

        gc.collect()

    def export(self, obj: Union[str, AudioMetadata, AudioWrap], file_path: str = './', format: FileFormat = FileFormat.UNKNOWN, quality: float = 0.5, bitrate: int = 128):
        '''
        Exports a record to a file in WAV, MP3, FLAC, or VORBIS format. The output format can be specified either through the `format` 
        argument or derived from the file extension in the `file_path`. If a file extension ('.wav', '.mp3', '.flac', or '.ogg') is 
        included in `file_path`, it takes precedence over the `format` argument. If no extension is provided, the 
        `format` argument is used, defaulting to WAV if set to FileFormat.UNKNOWN. The exported file is saved at the 
        specified `file_path`.

        :param obj: Record to export (str, AudioMetadata, or AudioWrap).
                    - str: Path to a file to be loaded and exported.
                    - AudioMetadata: A metadata object containing audio data.
                    - AudioWrap: Objects that wrap or generate the audio data.
        :param file_path: Path to save the exported file (default: './').
                        - A new filename can be specified at the end of the path.
                        - If a valid file extension ('.wav', '.mp3', '.flac', or '.ogg') is provided, it determines the output format, overriding the `format` argument.
                        - If no extension is included and the path is set to './', the name of the record is used.
        :param format: Output format (FileFormat.WAV, FileFormat.MP3, FileFormat.FLAC, or FileFormat.VORBIS). Defaults to FileFormat.UNKNOWN, 
                    which results in WAV being chosen unless a valid extension is provided in `file_path`.
        :param quality: Quality setting for encoding (default: 0.5).
                        - For WAV: Ignored
                        - For MP3: Converted to scale 0-9 (0 highest, 9 lowest)
                        - For FLAC: Converted to scale 0-8 (0 fastest/lowest, 8 slowest/highest)
                        - For VORBIS: Used directly (0.0 lowest, 1.0 highest)
        :param bitrate: Bitrate for MP3 encoding in kbps (default: 128). Only used if the format is MP3.
        :return: None

        Raises:

        - TypeError: Raised if `obj` is not one of the expected types (str, AudioMetadata, or AudioWrap).
        - ValueError: Raised if an unsupported format is provided.
        '''

        file_path = file_path.strip()
        if isinstance(obj, str): 
            record = self.load(obj, series=True)
        elif isinstance(obj, AudioWrap):
            record = obj.get_data()
        elif isinstance(obj, AudioMetadata):
            pass
        else:
            raise TypeError('please control the type of record')
        assert 0 <= quality <= 1, ValueError('Invalid quality factor')

        p0 = max(file_path.rfind('\\'), file_path.rfind('/'))
        p1 = file_path.rfind('.')
        if p0 < 0:
            p0 = 0
        if p1 <= 0:
            p1 = None
        
        if (not file_path) or (file_path == "./") or (file_path == ".") or (file_path == "/"):
            name = record.name
        else:
            name = file_path[p0 : p1]
        
        name_format = None
        if p1 is not None:
            name_format = file_path[p1 + 1:].lower()
            name_format = FileFormat.WAV if name_format == 'wav' else \
                        FileFormat.MP3 if name_format == 'mp3' else \
                        FileFormat.FLAC if name_format == 'flac' else \
                        FileFormat.VORBIS if name_format == 'ogg' else None

        format = name_format if name_format is not None else FileFormat.WAV if format == FileFormat.UNKNOWN else format
        supported = {FileFormat.WAV: 'wav', FileFormat.MP3: 'mp3', FileFormat.FLAC: 'flac', FileFormat.VORBIS: 'ogg'}
        if format not in supported:
            raise ValueError("Format must be either 'wav', 'mp3', 'flac', or 'ogg'")
        
        name += f'.{supported[format]}'
        if p0:
            file_path = file_path[0: p0 + 1] + name
        else:
            file_path = name
        
        file = record.o
        file_pos = file.tell()
        data = file.read()
        file.seek(file_pos, 0)
        
        if format == FileFormat.WAV:
            codec.encode_wav_file(
                file_path,
                data,
                record.sampleFormat,
                record.nchannels,
                record.frameRate,
            )
        elif format == FileFormat.MP3:
            mp3_quality = int(quality * 9)  # Convert 0-1 to 0-9 scale
            codec.encode_mp3_file(
                file_path,
                data,
                record.sampleFormat,
                record.nchannels,
                record.frameRate,
                bitrate,
                mp3_quality
            )
        elif format == FileFormat.FLAC:
            flac_compression = int(quality * 8)  # Convert 0-1 to 0-8 scale
            codec.encode_flac_file(
                file_path,
                data,
                record.sampleFormat,
                record.nchannels,
                record.frameRate,
                flac_compression
            )
        elif format == FileFormat.VORBIS:
            codec.encode_vorbis_file(
                file_path,
                data,
                record.sampleFormat,
                record.nchannels,
                record.frameRate,
                quality
            )


    def get_record_names(self) -> list:
        '''
        Returns a list of record names in the local database.
        '''
        return list(self._local_database.index())



    def get_nperseg(self):
        '''
        Returns the number of segments per window.
        '''
        return self._nperseg

    def get_nchannels(self):
        '''
        Returns the number of audio channels.
        '''
        return self._nchannels

    def get_sample_rate(self):
        '''
        Returns the sample rate of the master instanse core processor.
        '''
        return self._sample_rate

    def stream(self, record: Union[str, AudioMetadata, AudioWrap],
               block_mode: bool=False,
               safe_load: bool=False,
               on_stop: callable=None,
               loop_mode: bool=False,
               use_cached_files=True,
               stream_mode:StreamMode = StreamMode.optimized) -> StreamControl:
        '''
        Streams a record with optional loop and safe load modes.

        Note:
        
            The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance,
            meaning that, The audio data storage methods can have different execution times based on the cached files.

        Note:

            The recorder can only capture normal streams(Non-optimized streams)

        :param record: Record to stream (str, AudioMetadata, or AudioWrap).
        :param block_mode: Whether to block the stream (default: `False`).
        :param safe_load: Whether to safely load the record (default: `False`). 
         load an audio file and modify it according to the 'Master' attributes(like the frame rate, number oof channels, etc).
        :param on_stop: Callback for when the stream stops (default: `None`).
        :param loop_mode: Whether to enable loop mode (default: `False`).
        :param use_cached_files: Whether to use cached files (default: `True`).
        :param stream_mode: Streaming mode (default: `StreamMode.optimized`).

        :return: A StreamControl object
        '''

        assert self.is_started(), "instance not started, use start()"

        # loop mode dont workes in blocking mode
        cache_head_check_size = 20

        if isinstance(record, str):
            record = self.load(record, series=True)
        else:
            if isinstance(record, AudioWrap): 
                assert record.is_packed(), BufferError('The {} is not packed!'.format(record))
                record = record.get_data()

            elif isinstance(record, AudioMetadata):
                record = record.copy()
            else:
                raise TypeError('please control the type of record')


        if record.nchannels > self._nchannels:
            raise ImportError('number of channel for the {name}({ch0})'
                              ' is not same as object channels({ch1})'.format(name=record.name,
                                                                              ch0=record.nchannels,
                                                                              ch1=self._nchannels))

        elif not self._sample_rate == record.frameRate:
            warnings.warn(f'Warning, frame rate must be same, use {self.__class__}')

        assert  type(record.o) is BufferedRandom, TypeError('The record object is not standard')
        file = record.o
        file_pos = file.tell()
        tmp = list(file.name)
        tmp.insert(file.name.find(record.name), 'stream_')
        streamfile_name = ''.join(tmp)
        try:
            if use_cached_files:
                record_size = os.path.getsize(file.name)
                streamfile_size = os.path.getsize(streamfile_name)
                file.seek(0, 0)
                pre_head = file.read(Master.CACHE_INFO_SIZE + cache_head_check_size)
                file.seek(file_pos, 0)

                if record_size == streamfile_size:
                    pass
                elif (record_size - record.size) == streamfile_size:
                    pass
                    file_pos = Master.CACHE_INFO_SIZE
                else:
                    raise FileNotFoundError

                streamfile = open(streamfile_name, 'rb+')

                post_head = streamfile.read(Master.CACHE_INFO_SIZE + cache_head_check_size)
                streamfile.seek(file_pos, 0)
                if not post_head == pre_head:
                    streamfile.close()
                    raise FileNotFoundError

                if safe_load:
                    record.o = streamfile.read()
                    record = self._sync_record(record)
                    file.seek(file_pos, 0)
                record.o = streamfile
            else:
                raise FileNotFoundError

        except FileNotFoundError:
            record.o = file.read()
            if safe_load:
                record = self._sync_record(record)
            file.seek(file_pos, 0)

            record.o = streamfile = (
                write_to_cached_file(record.size,
                            record.frameRate,
                            record.sampleFormat if record.sampleFormat else 0,
                            record.nchannels,
                            file_name=streamfile_name,
                            data=record.o,
                            after_seek=(Master.CACHE_INFO_SIZE, 0),
                            after_flush=True)
            )
            file_pos = Master.CACHE_INFO_SIZE

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
        Mutes the master main stream.
        '''
        self._master_mute_mode.set()


    def unmute(self):
        '''
        Unmutes the master main stream.
        '''
        assert self.is_started(), "instance not started, use start()"
        assert not self._exstream_mode.is_set(), "stream is busy"
        self._master_mute_mode.clear()
        self._main_stream.clear()  # Clear any stale data in the stream


    def is_muted(self):
        '''
        Checks if the audio stream is muted.
        '''
        assert self.is_started(), "instance not started, use start()"
        return self._master_mute_mode.is_set()

    def echo(self, *args: Union[str, AudioMetadata, AudioWrap],
             enable: bool=None, main_output_enable: bool=False):
        
        """
        Play audio or manage audio streaming output functionality.

        Provides flexible audio playback and echo control with multiple input types 
        and configuration options.

        Parameters:
        -----------
        
        *args : str or AudioMetadata or AudioWrap
            Audio sources to play:
            - File path (str)
            - AudioMetadata object
            - AudioWrap object
            Multiple sources can be passed simultaneously

        enable : bool, optional
            Controls real-time echo behavior when no specific audio is provided:
            - None (default): Toggle echo on/off
            - True: Enable echo
            - False: Disable echo

        main_output_enable : bool, default False
            Determine whether to maintain the main audio stream's output during playback.
            Helps prevent potential audio feedback.

        Behavior:
        ---------
        
        - With no arguments: Manages real-time echo state
        - With audio sources: Plays specified audio through default output
        - Fallback to web audio if system audio is unavailable
        - Supports multiple audio playbacks

        Note: 
        -----

        If system audio fails, it'll try playing through your web browser 
        using HTML5 audio (if you're in a supported environment like Jupyter).

        Examples:
        ---------

            >>> master = sudio.Master()
            
            >>> master.add('audio1.ogg')
            >>> master.add('audio2.ogg')

            >>> # Play multiple audio sources
            >>> master.echo('audio1', 'audio2.wav')
            
            >>> # Toggle real-time echo
            >>> master.echo(enable=True)
        
        """
        if not len(args):
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
            for rec in args:
                self._echo(rec, main_output_enable)

    def _echo(
        self, 
        record:Union[str, AudioMetadata, AudioWrap],
        main_output_enable: bool=False
        ):
        """
        Internal method for audio playback with flexible input handling.

        Processes different audio input types and manages playback through 
        system or web audio output mechanisms.

        Parameters:
        -----------

        record : str or AudioMetadata or AudioWrap
            Audio source to play:
            - File path (str)
            - AudioMetadata object
            - AudioWrap object with packed data

        main_output_enable : bool, default False
            Control main output stream during playback.
            Helps manage potential audio feedback scenarios.

        Raises:
        -------
        
        ValueError
            If an unsupported input type is provided.
        RuntimeError
            When no compatible audio output method is available.

        Notes:
        ------

        - Automatically handles different audio input formats
        - Attempts system audio output with fallback to web audio
        - Manages echo flag state during playback
        - Validates output device availability
        """
        if isinstance(record, AudioWrap): 
                assert  record.is_packed(), 'record should be packed!'
                record_data = record.get_data()
        else:
            if isinstance(record, str):
                record_data = self.load(record, series=True)
            elif isinstance(record, AudioMetadata):
                record_data = record
            else:
                ValueError('unknown type')

        db = record_data.o
        if isinstance(db, IOBase):
            assert not db.closed, "cache file is inaccessible."
            # file_pos = file.tell()
            db.seek(0, 0)
            data = db.read()
            db.seek(0, 0)
        elif isinstance(record_data.o, bytes):
            data = db
            warnings.warn('use .add to sync before usage.')
        else:
            raise TypeError('unknown type')

        flg = False
        if not main_output_enable and self._echo_flag.is_set():
            flg = True
            self._echo_flag.clear()

        try:
            assert self._output_device_index, "No output devices were found."
            write_to_default_output(
                data,
                record_data.sampleFormat,
                record_data.nchannels,
                record_data.frameRate,
                self._output_device_index
            )
        except Exception as e:
            try:
                if WebAudioIO.is_web_audio_supported():
                    result = WebAudioIO.play_audio_data(
                        data, 
                        record_data.sampleFormat, 
                        record_data.nchannels, 
                        record_data.frameRate,
                        )
                    assert result
                    warnings.warn(f'{e}')
                else:
                    raise
            except:
                raise RuntimeError(f"{e}")
                
        if flg:
            self._echo_flag.set()

    def disable_echo(self):
        '''
        Disables the echo functionality.
        '''
        assert self.is_started(), "instance not started, use start()"
        self.echo(enable=False)


    def wrap(self, record: Union[str, AudioMetadata]):
        '''
        wraps a record as a `AudioWrap`.
        
        :param record: Record to wrap (str or AudioMetadata).
        '''
        return AudioWrap(self, record)

    def prune_cache(self):
        """
        Retrieve a list of unused or orphaned cache files in the audio data directory.

        Scans the audio data directory and identifies cache files that are no longer
        referenced in the local database, helping manage file system resources.

        Returns:
        --------
        list
            Absolute file paths of cache files not associated with any current audio record.

        Notes:
        ------
        - Compares existing files against local database records
        - Filters out currently used cache files
        - Useful for identifying potential cleanup candidates
        """

        path = []
        expath = []
        base_len = len(self._audio_data_directory)
        for i in self._local_database.index():
            record = self._local_database.get_record(i)
            expath.append(record.o.name[base_len:])
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

    def clean_cache(self, max_retries = 3, retry_delay = .1):
        '''
        The audio data maintaining process has additional cached files to reduce dynamic
        memory usage and improve performance, meaning that, The audio data storage methods
        can have different execution times based on the cached files.
        This function used to clean the audio cache by removing cached files.
        
        The function implements retry logic for handling permission errors and ensures
        files are properly deleted across different operating systems.
        '''
        cache = self.prune_cache()
        
        for file_path in cache:
            path = Path(file_path)
            if not path.exists():
                continue
                
            for attempt in range(max_retries):
                try:
                    try:
                        path.chmod(0o666)
                    except:
                        pass
                        
                    path.unlink(missing_ok=True)
                    break  # Success - exit retry loop
                    
                except PermissionError:
                    if attempt < max_retries - 1:
                        # Wait before retry to allow potential file locks to clear
                        time.sleep(retry_delay)
                        continue
                        
                except OSError as e:
                    # Handle other OS-specific errors
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
        time.sleep(retry_delay)


    def _refresh(self, *args):
        """
        Resets and reconfigures the signal processing buffers when things change.

        Handles window length updates, regenerates window functions, and clears 
        internal buffers. Useful when you need to adjust processing parameters 
        on the fly without completely rebuilding the entire stream setup.

        Optionally takes arguments for specific refresh scenarios, but can also 
        do a standard refresh based on current config.
        """

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
        '''
        Checks if the audio input and output streams are started.
        '''
        return self._audio_input_stream and self._audio_output_stream
    
    def get_window(self):
        """
        Retrieves the current window configuration.

        :return: dict or None: A dictionary containing window information if available, or None if not set.

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
        Disables standard input stream by acquiring the main stream's lock object.
        """
        # if not self._main_stream.locked():
        assert self.is_started(), "instance not started, use start()"
        self._main_stream.acquire()

    def enable_std_input(self):
        """
        Enables standard input stream by clearing the main stream's lock.
        """
        assert self.is_started(), "instance not started, use start()"
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
        Adds a new processing pipeline.

        Parameters:
        -----------

        - pip (obj): Pipeline object or array of defined pipelines.
         
        Note:
         In **PipelineProcessType.MULTI_STREAM** process type, pip must be an array of defined pipelines.
         The size of the array must be the same as the number of input channels.
         
        - name (str): Indicates the name of the pipeline.
        - process_type (PipelineProcessType): Type of processing pipeline (default: `PipelineProcessType.MAIN`). it can be:
         - **PipelineProcessType.MAIN**: Processes input data and passes it to activated pipelines (if exist).
         - **PipelineProcessType.BRANCH**: Represents a branch pipeline with optional channel parameter.
         - **PipelineProcessType.MULTI_STREAM**: Represents a multi_stream pipeline mode. Requires an array of pipelines.
        - channel (obj): None or [0 to self.nchannel].
         The input data passed to the pipeline can be a NumPy array in
         (self.nchannel, 2[2 windows in 1 frame], self._nperseg) dimension [None]
         or mono (2, self._nperseg) dimension.
         In mono mode [self.nchannel = 1 or mono mode activated], channel must be None.
        
        Note:
        
         The pipeline must process data and return it to the core with the dimensions as same as the input.

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
                
        else:
            pass

        return stream


    def set_pipeline(self, stream: Union[str, Stream]):
        '''
        sets the main processing pipeline.
        '''

        assert self.is_started(), "instance not started, use start()"

        if type(stream) is str:
            name = stream
        else:
            name = stream.name

        try:
            # find the specified pipeline
            pipeline = next(obj for obj in self.main_pipe_database if obj.name == name)
            
            assert pipeline.process_type in [PipelineProcessType.MAIN, PipelineProcessType.MULTI_STREAM], \
                f"Pipeline {name} is not a MAIN or MULTI_STREAM type"

            try:
                self.disable_pipeline()
            except ValueError:
                pass
            
            self._main_stream.acquire()
            try:
                self._main_stream.set(pipeline)
                
                if pipeline.process_type == PipelineProcessType.MULTI_STREAM:
                    for i in pipeline.pip:
                        assert i.is_alive(), f'Error: Sub-pipeline in {name} is not Enabled!'
                        i.sync(self._master_sync)
                    
                    for i in pipeline.pip:
                        i.aasync()

                # self._main_stream.clear()

            finally:
                self._main_stream.release()

        except StopIteration:
            raise ValueError(f"Pipeline {name} not found in main_pipe_database")
        except Exception as e:
            print(f"Error setting pipeline {name}: {str(e)}")
            # Attempt to restore the normal stream in case of error
            self.disable_pipeline()
            raise

    def disable_pipeline(self):
        '''
        Disables the current processing pipeline.
        '''

        assert self.is_started(), "instance not started, use start()"
        if self._main_stream and hasattr(self._main_stream, 'pip'):
            self._main_stream.acquire()
            if isinstance(self._main_stream.pip, Pipeline):
                self._main_stream.pip.clear()
            self._main_stream.set(self._normal_stream)
            
            self._main_stream.clear()
            
            self._main_stream.release()
        else:
            raise ValueError("No pipeline is currently set")             

    def clear_pipeline(self):
        '''
        Clears all pipeline's data.
        '''

        assert self.is_started(), "instance not started, use start()"
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
        '''
        Configures the window function for audio processing.

        :param window: The window function (default: `'hann'`).
        :param noverlap: Number of overlapping segments (default: `None`).
        :param NOLA_check: Perform the NOLA check (default: `True`).
        '''
        assert self.is_started(), "instance not started, use start()"

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
        '''
        Returns the sample format of the master instance.
        '''
        return self._sample_format_type

    def _get_sample_format(self):
        '''
        Returns the sample format of the master instance.
        '''
        return self._sample_format

    def generate(
            self, 
            Gen: Generator, 
            *args, 
            **kwargs
            ):
        """
        Generates audio using a specified Generator subclass and adds it to the database.

        Creates an instance of the provided Generator class, configures it with the Master's 
        audio parameters, and processes the generated audio through the system pipeline.

        Parameters:
        -----------
        Gen : Generator subclass
            Generator class to use for audio synthesis. Must inherit from `Generator`.
        *args : tuple
            Positional arguments forwarded to both:
            - Generator.__init__() for configuration
            - Generator.generate() for synthesis parameters
        **kwargs : dict
            Keyword arguments filtered to:
            - Generator.__init__() for object initialization
            - Generator.generate() for synthesis control
            See specific Generator subclass documentation for valid parameters.

        Returns:
        --------
        AudioWrap
            Wrapped audio record containing metadata and access to audio data.

        Raises:
        -------
        TypeError
            If `Gen` is not a subclass of Generator.
        ValueError
            If invalid keyword arguments are provided for the Generator's methods.

        Examples:
        ---------
        **Basic Sine Wave Generation**
        ```python
        master = Master()
        audio = master.generate(SineWave, 
                            duration=2, 
                            base_freq=440, 
                            amplitude=-6)
        ```

        **FM Synthesis with Harmonics**
        ```python
        master.generate(SineWave,
                    [220, 880],  # Frequency components
                    duration=5,
                    base_freq=110,
                    type=7,       # 7 harmonics
                    modulation_depth=50,
                    modulation_freq=2.5)
        ```

        **Multi-channel Phase Effects**
        ```python
        master.generate(SineWave,
                    [1000, 3000],
                    duration=3,
                    phase_different_map=[0, [0, 20]],  # Stereo phase variation
                    nchannels=2)
        ```

        **Dynamic Parameter Sweeps**
        ```python
        # Frequency sweep from 20Hz to 20kHz over 10 seconds
        master.generate(SineWave,
                    duration=10,
                    base_freq=[20, 20000],
                    amplitude_spline=True)  # Smooth amplitude transitions
        ```

        Notes:
        ------
        - The Master's audio parameters (sample rate, channels, etc.) are automatically
        forwarded to the Generator
        - Uses kwarg filtering to ensure parameters reach the correct generator methods
        - Generated audio is immediately added to the Master's database for processing
        """
        if not issubclass(Gen, Generator):
            raise TypeError('Gen must be a subclass of Generator')
        
        if ekw(
            Gen.__call__, 
            Gen.generate, 
            cls=Gen, 
            property='__init__', 
            **kwargs
            ):
            raise ValueError('Invalid keyword arguments')
        
        gen = Gen(
            *args,
            sample_rate=self._sample_rate,
            nchannels=self._nchannels,
            data_nperseg=self._nperseg,
            target_sample_format=self._sample_format,
            **fw(
                cls=Gen,
                property='__init__', 
                **kwargs
                ),
        )         

        metadata = gen(
            *args,
            **fw(Gen.generate, gen.__call__, **kwargs),
        )   

        record = self.add(metadata)
        return record

    @staticmethod
    def get_default_input_device_info()-> AudioDeviceInfo:
        """
        Returns information about the default input audio device.

        :return AudioDeviceInfo
        """
        data = AudioStream.get_default_input_device()
        return data

    @staticmethod
    def get_default_output_device_info()-> AudioDeviceInfo:
        """
        Returns information about the default output audio device.

        :return AudioDeviceInfo
        """
        data = AudioStream.get_default_output_device()
        return data
    
    @staticmethod
    def get_device_count()-> int:
        """
        Returns the number of available audio devices.
        """
        data = AudioStream.get_device_count()
        return data
    
    @staticmethod
    def get_device_info_by_index(index: int)-> AudioDeviceInfo:
        """
        Returns information about a specific audio device by index.

        :param index: The index of the audio device (int).

        :return AudioDeviceInfo
        """
        data = AudioStream.get_device_info_by_index(int(index))
        return data
    
    @staticmethod
    def get_input_devices():
        """
        Returns a list of available input devices.
        """
        return AudioStream.get_input_devices()

    @staticmethod
    def get_output_devices():
        """
        Returns a list of available output devices.
        """
        return AudioStream.get_output_devices()
