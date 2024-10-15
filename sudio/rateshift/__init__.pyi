
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


from typing import Callable
import numpy as np

class ConverterType:
    sinc_best: 'ConverterType'
    sinc_medium: 'ConverterType'
    sinc_fastest: 'ConverterType'
    zero_order_hold: 'ConverterType'
    linear: 'ConverterType'


class Resampler:
    def __init__(self, converter_type: ConverterType, channels: int = 1) -> None:
        """
        Initialize a Resampler object.

        Args:
            converter_type (ConverterType): The type of converter to use.
            channels (int, optional): Number of channels in the audio. Defaults to 1.
        """
        pass

    def process(self, input: np.ndarray, sr_ratio: float, end_of_input: bool = False) -> np.ndarray:
        """
        Process audio data through the resampler.

        Args:
            input (numpy.ndarray): Input audio data as a NumPy array.
            sr_ratio (float): The ratio of output sample rate to input sample rate.
            end_of_input (bool, optional): Flag indicating if this is the last batch. Defaults to False.

        Returns:
            numpy.ndarray: Resampled audio data.
        """
        pass

    def set_ratio(self, ratio: float) -> None:
        """
        Set a new resampling ratio.

        Args:
            ratio (float): The new resampling ratio.
        """
        pass

    def reset(self) -> None:
        """
        Reset the internal state of the resampler.
        """
        pass


class CallbackResampler:
    def __init__(self, callback: Callable[[int], np.ndarray], ratio: float, converter_type: ConverterType, channels: int) -> None:
        """
        Initialize a CallbackResampler object.

        Args:
            callback (Callable): A function that provides input audio data.
            ratio (float): The initial resampling ratio.
            converter_type (ConverterType): The type of converter to use.
            channels (int): Number of channels in the audio.
        """
        pass

    def read(self, frames: int) -> np.ndarray:
        """
        Read resampled audio data.

        Args:
            frames (int): Number of frames to read.

        Returns:
            numpy.ndarray: Resampled audio data.
        """
        pass

    def set_starting_ratio(self, ratio: float) -> None:
        """
        Set a new starting ratio for the resampler.

        Args:
            ratio (float): The new starting ratio.
        """
        pass

    def reset(self) -> None:
        """
        Reset the internal state of the resampler.
        """
        pass


def resample(input: np.ndarray, sr_ratio: float, converter_type: ConverterType, channels: int = 1) -> np.ndarray:
    """
    Resample audio data.

    This is a convenience function for simple resampling tasks.

    Args:
        input (numpy.ndarray): Input audio data as a NumPy array.
        sr_ratio (float): The ratio of output sample rate to input sample rate.
        converter_type (ConverterType): The type of converter to use.
        channels (int, optional): Number of channels in the audio. Defaults to 1.

    Returns:
        numpy.ndarray: Resampled audio data.
    """
    pass
