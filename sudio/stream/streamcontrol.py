

from sudio._register import Members as Mem
from sudio.stream.streamingtimecontroller import StreamingTimeController

@Mem.master.add
class StreamControl:
    """
    The StreamControl class is defined to control the mainstream audio playback for a special 'record'.
    """

    time = StreamingTimeController()

    def __init__(self, other, record, on_stop, loop_mode, stream_mode):
        '''
        Initialize the StreamControl class to manage the mainstream audio playback for a specific 'record'.

        Args:
        - other: An instance of the main audio processing class.
        - record: A dictionary containing information about the audio file to be streamed.
        - on_stop: Callback function to execute when audio playback stops.
        - loop_mode: Boolean indicating whether to enable loop mode.
        - stream_mode: Type of streaming operation (e.g., 'read', 'write').

        Attributes:
        - self.other: Reference to the main audio processing instance.
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

        self.other = other
        self._stream_type = stream_mode
        self._stream_file = record['o']
        self._stream_file_size = record['size']
        self._size_calculator = lambda: (self.other._data_chunk *
                                         self.other.nchannels *
                                         self.other._sampwidth)
        self._stream_data_size = self._size_calculator()
        self._stream_data_pointer = self._stream_file.tell()
        self._stream_on_stop = on_stop
        self._stream_loop_mode = loop_mode

        self.duration = record['duration']
        self._time_calculator = lambda t: int(self.other._frame_rate *
                                              self.other.nchannels *
                                              self.other._sampwidth *
                                              t)

        self._itime_calculator = lambda byte: byte / (self.other._frame_rate *
                                                      self.other.nchannels *
                                                      self.other._sampwidth)
        
        self._stream_busy_error = "Another file is currently streaming"
        self._stream_empty_error = "The stream is currently empty"


    def _ready(self):
        """
        Ensure the stream is ready for playback.
        """
        assert not self._is_streaming(), PermissionError(self._stream_busy_error)

        if not self.isready():
            self.other._exstream_mode.clear()
            if self.other._stream_file is not None:
                self.other._stream_file.seek(self._stream_data_pointer, 0)

            self.other._stream_type = self._stream_type
            self.other._stream_file = self._stream_file
            self.other._stream_data_size = self._stream_data_size
            self.other._stream_data_pointer = self._stream_data_pointer
            self.other._stream_on_stop = self._stream_on_stop
            self.other._stream_loop_mode = self._stream_loop_mode

    def isready(self):
        """
        Check if the stream is ready for playback.

        Returns:
        - bool: True if ready, False otherwise
        """
        return (self.other._stream_type == self._stream_type and
                self._stream_file == self.other._stream_file and
                self._stream_data_size == self.other._stream_data_size and
                self._stream_on_stop == self.other._stream_on_stop and
                self._stream_loop_mode == self.other._stream_loop_mode)

    def is_streaming(self):
        """
        Check if the stream is currently in the streaming state.

        Returns:
        - bool: True if streaming, False otherwise
        """
        if self.isready():
            return self.other._exstream_mode.is_set()
        else:
            return False

    def _is_streaming(self):
        """
        Check if the stream is currently in the streaming state (internal use).

        Returns:
        - bool: True if streaming, False otherwise
        """
        return self.other._exstream_mode.is_set()

    def start(self):
        """
        Start the audio playback stream.
        """
        self._ready()
        self.other._exstream_mode.set()
        self.other._main_stream.clear()

    def resume(self):
        """
        Resume the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        self.other._exstream_mode.set()
        self.other._main_stream.clear()

    def stop(self):
        """
        Stop the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self.other._exstream_mode.clear()
        self.other._stream_file.seek(self._stream_data_pointer, 0)
        # self.other._main_stream.clear()

    def pause(self):
        """
        Pause the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self.other._exstream_mode.clear()
        # self.other._main_stream.clear()

    def enable_loop(self):
        """
        Enable looping for the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self.other._stream_loop_mode = True
        self._stream_loop_mode = True

    def disable_loop(self):
        """
        Disable looping for the audio playback stream.
        """
        assert self.isready(), PermissionError(self._stream_busy_error)
        assert self._is_streaming(), PermissionError(self._stream_empty_error)
        self.other._stream_loop_mode = False
        self._stream_loop_mode = False