#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


from sudio.io import SampleFormat
import numpy as np



def convert_array_type(arr:np.ndarray, sample_format:SampleFormat):
    """
    Convert the data type of a NumPy array based on the given SampleFormat.
    
    Args:
    arr (np.ndarray): Input NumPy array
    sample_format (SampleFormat): Desired output format
    
    Returns:
    np.ndarray: Converted NumPy array
    """
    if sample_format == SampleFormat.FLOAT32:
        return arr.astype(np.float32)
    elif sample_format == SampleFormat.SIGNED32:
        return arr.astype(np.int32)
    elif sample_format == SampleFormat.SIGNED16:
        return arr.astype(np.int16)
    elif sample_format == SampleFormat.UNSIGNED8:
        return arr.astype(np.int8)
    elif sample_format == SampleFormat.UNSIGNED8:
        return arr.astype(np.uint8)
    elif sample_format == SampleFormat.UNKNOWN:
        return arr  # Return the original array without conversion
    else:
        raise ValueError("Unsupported sample format")