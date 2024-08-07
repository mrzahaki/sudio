from sudio.types import LibSampleFormat, MiniaudioError, SampleFormat
import numpy as np


def get_sample_width_from_format(sample_format: LibSampleFormat) -> int:
    """
    Get the sample width (in bytes) based on the specified LibSampleFormat.

    Args:
        sample_format (LibSampleFormat): The sample format.

    Returns:
        int: The sample width in bytes.

    Raises:
        MiniaudioError: If the sample format is unsupported.
    """
    widths = {
        LibSampleFormat.UNSIGNED8: 1,
        LibSampleFormat.SIGNED16: 2,
        LibSampleFormat.SIGNED24: 3,
        LibSampleFormat.SIGNED32: 4,
        LibSampleFormat.FLOAT32: 4
    }
    if sample_format in widths:
        return widths[sample_format]
    raise MiniaudioError("Unsupported sample format", sample_format)





def get_format_from_width(sample_width: int, is_float: bool = False) -> LibSampleFormat:
    """
    Get the LibSampleFormat based on the specified sample width and float flag.

    Args:
        sample_width (int): The sample width in bytes.
        is_float (bool, optional): Whether the format is float. Defaults to False.

    Returns:
        LibSampleFormat: The corresponding LibSampleFormat.

    Raises:
        MiniaudioError: If the sample width is unsupported.
    """
    if is_float:
        return LibSampleFormat.FLOAT32
    elif sample_width == 1:
        return LibSampleFormat.UNSIGNED8
    elif sample_width == 2:
        return LibSampleFormat.SIGNED16
    elif sample_width == 3:
        return LibSampleFormat.SIGNED24
    elif sample_width == 4:
        return LibSampleFormat.SIGNED32
    else:
        raise MiniaudioError("Unsupported sample width", sample_width)
    



def convert_array_type(arr:np.ndarray, sample_format:SampleFormat):
    """
    Convert the data type of a NumPy array based on the given SampleFormat.
    
    Args:
    arr (np.ndarray): Input NumPy array
    sample_format (SampleFormat): Desired output format
    
    Returns:
    np.ndarray: Converted NumPy array
    """
    if sample_format == SampleFormat.formatFloat32:
        return arr.astype(np.float32)
    elif sample_format == SampleFormat.formatInt32:
        return arr.astype(np.int32)
    elif sample_format == SampleFormat.formatInt16:
        return arr.astype(np.int16)
    elif sample_format == SampleFormat.formatInt8:
        return arr.astype(np.int8)
    elif sample_format == SampleFormat.formatUInt8:
        return arr.astype(np.uint8)
    elif sample_format == SampleFormat.formatUnknown:
        return arr  # Return the original array without conversion
    else:
        raise ValueError("Unsupported sample format")