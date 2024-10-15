
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/

from sudio.stream.streamingtimecontroller import StreamingTimeController


class StreamControl:
    """
    The StreamControl class is defined to control the mainstream audio playback for a special 'record'.
    """

    time = StreamingTimeController()

    def __init__(self, master, record, on_stop, loop_mode, stream_mode):
        '''
        Initialize the StreamControl class to manage the mainstream audio playback for a specific 'record'.

        Args:
        - master: An instance of the main audio processing class.
        - record: A dictionary containing information about the audio file to be streamed.
        - on_stop: Callback function to execute when audio playback stops.
        - loop_mode: Boolean indicating whether to enable loop mode.
        - stream_mode: Type of streaming operation (e.g., 'read', 'write').

        Attributes:
        - self._master: Reference to the main audio processing instance.
        - self._stream_type: Type of streaming operation.
        - self._stream_file: File object representing the audio file.
        - self._stream_file_size: Size of the audio file.
        - self._size_calculator: Lambda function to calculate the size of a data chunk.
        - self._stream_data_size: Size of the audio data to be streamed.
        - self._stream_data_pointer: Current position in the audio file.
        - self._stream_on_stop: Callback function for when audio playback stops.
        - self._stream_loop_mode: Boolean indicating whether loop mode is enabled.
        - self.duration: Duration of the audio file.
        - self._time_calculator: Lambda function to convert time to byte offset.
        - self._itime_calculator: Lambda function to convert byte offset to time.

        Raises:
        - PermissionError: If attempting to initialize while another file is streaming.

        Note:
        - This class facilitates the management of audio playback for a specific audio file, ensuring proper synchronization
        and handling of playback-related operations.
        '''

        self._master = master
        self._stream_type = stream_mode
        self._stream_file = record['o']
        self._stream_file_size = record['size']
        self._size_calculator = lambda: (self._master._data_chunk *
                                         self._master._nchannels *
                                         self._master._sample_width)
        self._stream_data_size = self._size_calculator()
        self._stream_data_pointer = self._stream_file.tell()
        self._stream_on_stop = on_stop
        self._stream_loop_mode = loop_mode

        self.duration = record['duration']
        self._time_calculator = lambda t: int(self._master._sample_rate *
                                              self._master._nchannels *
                                              self._master._sample_width *
                                              t)

        self._itime_calculator = lambda byte: byte / (self._master._sample_rate *
                                                      self._master._nchannels *
                                                      self._master._sample_width)
        
        self._stream_busy_error = "Another file is currently streaming"
        self._stream_empty_error = "The stream is currently empty"


    def _ready(self):
        """
        Ensure the stream is ready for playback.
        """
        assert not self._is_streaming(), PermissionError(self._stream_busy_error)

        if not self.isready():
            self._master._exstream_mode.clear()
            if self._master._stream_file is not None:
                self._master._stream_file.seek(self._stream_data_pointer, 0)

            self._master._stream_type = self._stream_type
            self._master._stream_file = self._stream_file
            self._master._stream_data_size = self._stream_data_size
            self._master._stream_data_pointer = self._stream_data_pointer
            self._master._stream_on_stop = self._stream_on_stop
            self._master._stream_loop_mode = self._stream_loop_mode

    def isready(self):
        """
        Check if the stream is ready for playback.

        Returns:
        - bool: True if ready, False otherwise
        """
        return (self._master._stream_type == self._stream_type and
                self._stream_file == self._master._stream_file and
                self._stream_data_size == self._master._stream_data_size and
                self._stream_on_stop == self._master._stream_on_stop and
                self._stream_loop_mode == self._master._stream_loop_mode)

    def is_streaming(self):
        """
        Check if the stream is currently in the streaming state.

        Returns:
        - bool: True if streaming, False otherwise
        """
        if self.isready():
            return self._master._exstream_mode.is_set()
        else:
            return False

    def _is_streaming(self):
        """
        Check if the stream is currently in the streaming state (internal use).

        Returns:
        - bool: True if streaming, False otherwise
        """
        return self._master._exstream_mode.is_set()

    def start(self):
        """
        Start the audio playback stream.
        """
        self._ready()
        self._master._exstream_mode.set()
        self._master._main_stream_safe_release()
        self._master._main_stream.clear()


    def resume(self):
        """
        Resume the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        self._master._exstream_mode.set()
        self._master._main_stream.clear()

    def stop(self):
        """
        Stop the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self._master._stream_file.seek(self._stream_data_pointer, 0)
        self._master._exstream_mode.clear()
        # self._master._main_stream.clear()

    def pause(self):
        """
        Pause the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self._master._exstream_mode.clear()
        # self._master._main_stream.clear()

    def enable_loop(self):
        """
        Enable looping for the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self._master._stream_loop_mode = True
        self._stream_loop_mode = True

    def disable_loop(self):
        """
        Disable looping for the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self._master._stream_loop_mode = False
        self._stream_loop_mode = False
