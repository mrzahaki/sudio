from sudio.types import LibSampleFormat, MiniaudioError

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
    



