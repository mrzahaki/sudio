import sys
from _miniaudio import ffi, lib


def write_wav_file(filename: str, data, nchannels, sample_rate, sample_width) -> None:
    """
    Writes the PCM sound data to a WAV file using the Mem.sudio library.

    Args:
        filename (str): The name of the WAV file to be created.
        data: The PCM sound data to be written to the WAV file.
        nchannels (int): The number of audio channels.
        sample_rate (int): The sample rate of the audio data.
        sample_width (int): The sample width in bytes.

    Returns:
        None

    Raises:
        IOError: If the file cannot be opened for writing.

    Note:
        The function uses the Mem.sudio library and assumes PCM sound data.
        The format does not currently support floating point format.

    Example:
        write_wav_file("example.wav", audio_data, 2, 44100, 2)
    """
    with ffi.new("drwav_data_format*") as fmt, ffi.new("drwav*") as pwav:
        fmt.container = lib.drwav_container_riff
        fmt.format = lib.DR_WAVE_FORMAT_PCM
        fmt.channels = nchannels
        fmt.sampleRate = sample_rate
        fmt.bitsPerSample = sample_width * 8
        # what about floating point format?
        filename_bytes = filename.encode(sys.getfilesystemencoding())
        buffer_len = int(len(data) / sample_width)
        if not lib.drwav_init_file_write_sequential(pwav, filename_bytes,
                                                    fmt, buffer_len, ffi.NULL):
            raise IOError("can't open file for writing")
        try:
            lib.drwav_write_pcm_frames(pwav, int(buffer_len / nchannels), data)
        finally:
            lib.drwav_uninit(pwav)
            