import pyaudio


class Audio(pyaudio.PyAudio):
    """
    Customized class extending PyAudio functionality for audio stream handling.

    Attributes:
        None

    Methods:
        open_stream(*args, **kwargs):
            Opens an audio stream with the specified parameters.
        
        get_sample_size(format: int):
            Static method to retrieve the sample size for a given audio format.

        to_int(obj, key):
            Static method to convert the value associated with a key in a dictionary to an integer.

    Usage:
        audio_instance = Audio()
        stream = audio_instance.open_stream(format=pyaudio.paInt16, channels=2, rate=44100)
        sample_size = Audio.get_sample_size(pyaudio.paInt16)
    """
    def open_stream(self,
                    *args,
                    **kwargs
                    ):
        to_int = Audio.to_int
        to_int(kwargs, 'format')
        to_int(kwargs, 'channels')
        to_int(kwargs, 'rate')
        to_int(kwargs, 'frames_per_buffer')
        to_int(kwargs, 'input_device_index')
        to_int(kwargs, 'output_device_index')

        return self.open(*args, **kwargs)

    @staticmethod
    def get_sample_size(format: int):
        """
        Static method to retrieve the sample size for a given audio format.

        Args:
            format (int): Audio format identifier.

        Returns:
            int: Sample size in bytes.

        Usage:
            sample_size = Audio.get_sample_size(pyaudio.paInt16)
        """
        return pyaudio.get_sample_size(int(format))

    @staticmethod
    def to_int(obj, key):
        """
        Static method to convert the value associated with a key in a dictionary to an integer.

        Args:
            obj (dict): Dictionary containing key-value pairs.
            key (str): Key for which the value needs to be converted to an integer.

        Returns:
            None

        Usage:
            to_int(kwargs, 'format')
        """
        try:
            if not obj[key] is None:
                obj[key] = int(obj[key])
        except KeyError:
            pass
