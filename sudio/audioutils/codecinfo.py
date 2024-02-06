import os
from _miniaudio import ffi, lib

from sudio.audioutils.fileinfo import AudioFileInfo
from sudio.types import FileFormat, LibSampleFormat, DecodeError
from sudio.extras.io import get_encoded_filename_bytes
from sudio.audioutils.typeconversion import get_format_from_width


def get_file_info(filename: str) -> AudioFileInfo:
    """
    Fetches information about the audio file.

    Parameters:
    - filename (str): The path or name of the audio file.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio file.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".ogg", ".vorbis"):
        return vorbis_get_file_info(filename)
    elif ext == ".mp3":
        return mp3_get_file_info(filename)
    elif ext == ".flac":
        return flac_get_file_info(filename)
    elif ext == ".wav":
        return wav_get_file_info(filename)
    raise DecodeError("unsupported file format")


def vorbis_get_file_info(filename: str) -> AudioFileInfo:
    """
    Fetches information about the audio file in vorbis format.

    Parameters:
    - filename (str): The path or name of the audio file.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio file.
    """
    filenamebytes = get_encoded_filename_bytes(filename)
    with ffi.new("int *") as error:
        vorbis = lib.stb_vorbis_open_filename(filenamebytes, error, ffi.NULL)
        if not vorbis:
            raise DecodeError("could not open/decode file")
        try:
            info = lib.stb_vorbis_get_info(vorbis)
            duration = lib.stb_vorbis_stream_length_in_seconds(vorbis)
            num_frames = lib.stb_vorbis_stream_length_in_samples(vorbis)
            return AudioFileInfo(filename, FileFormat.VORBIS, info.channels, info.sample_rate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.stb_vorbis_close(vorbis)


def vorbis_get_info(data: bytes) -> AudioFileInfo:
    """
    Fetches information about the audio data in vorbis format.

    Parameters:
    - data (bytes): Audio data in bytes.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio data.
    """
    with ffi.new("int *") as error:
        vorbis = lib.stb_vorbis_open_memory(data, len(data), error, ffi.NULL)
        if not vorbis:
            raise DecodeError("could not open/decode data")
        try:
            info = lib.stb_vorbis_get_info(vorbis)
            duration = lib.stb_vorbis_stream_length_in_seconds(vorbis)
            num_frames = lib.stb_vorbis_stream_length_in_samples(vorbis)
            return AudioFileInfo("<memory>", FileFormat.VORBIS, info.channels, info.sample_rate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.stb_vorbis_close(vorbis)


def flac_get_file_info(filename: str) -> AudioFileInfo:
    """
    Fetches information about the audio file in FLAC format.

    Parameters:
    - filename (str): The path or name of the audio file.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio file.
    """
    filenamebytes = get_encoded_filename_bytes(filename)
    flac = lib.drflac_open_file(filenamebytes, ffi.NULL)
    if not flac:
        raise DecodeError("could not open/decode file")
    try:
        duration = flac.totalPCMFrameCount / flac.sampleRate
        sample_width = flac.bitsPerSample // 8
        return AudioFileInfo(filename, FileFormat.FLAC, flac.channels, flac.sampleRate,
                             get_format_from_width(sample_width), duration, flac.totalPCMFrameCount)
    finally:
        lib.drflac_close(flac)


def flac_get_info(data: bytes) -> AudioFileInfo:
    """
    Fetches information about the audio data in FLAC format.

    Parameters:
    - data (bytes): Audio data in bytes.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio data.
    """
    flac = lib.drflac_open_memory(data, len(data), ffi.NULL)
    if not flac:
        raise DecodeError("could not open/decode data")
    try:
        duration = flac.totalPCMFrameCount / flac.sampleRate
        sample_width = flac.bitsPerSample // 8
        return AudioFileInfo("<memory>", FileFormat.FLAC, flac.channels, flac.sampleRate,
                             get_format_from_width(sample_width), duration, flac.totalPCMFrameCount)
    finally:
        lib.drflac_close(flac)


def mp3_get_file_info(filename: str) -> AudioFileInfo:
    """
    Fetches information about the audio file in MP3 format.

    Parameters:
    - filename (str): The path or name of the audio file.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio file.
    """
    filenamebytes = get_encoded_filename_bytes(filename)
    with ffi.new("drmp3 *") as mp3:
        if not lib.drmp3_init_file(mp3, filenamebytes, ffi.NULL):
            raise DecodeError("could not open/decode file")
        try:
            num_frames = lib.drmp3_get_pcm_frame_count(mp3)
            duration = num_frames / mp3.sampleRate
            return AudioFileInfo(filename, FileFormat.MP3, mp3.channels, mp3.sampleRate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.drmp3_uninit(mp3)


def mp3_get_info(data: bytes) -> AudioFileInfo:
    """
    Fetches information about the audio data in MP3 format.

    Parameters:
    - data (bytes): Audio data in bytes.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio data.
    """
    with ffi.new("drmp3 *") as mp3:
        if not lib.drmp3_init_memory(mp3, data, len(data), ffi.NULL):
            raise DecodeError("could not open/decode data")
        try:
            num_frames = lib.drmp3_get_pcm_frame_count(mp3)
            duration = num_frames / mp3.sampleRate
            return AudioFileInfo("<memory>", FileFormat.MP3, mp3.channels, mp3.sampleRate,
                                 LibSampleFormat.SIGNED16, duration, num_frames)
        finally:
            lib.drmp3_uninit(mp3)


def wav_get_file_info(filename: str) -> AudioFileInfo:
    """
    Fetches information about the audio file in WAV format.

    Parameters:
    - filename (str): The path or name of the audio file.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio file.
    """
    filenamebytes = get_encoded_filename_bytes(filename)
    with ffi.new("drwav*") as wav:
        if not lib.drwav_init_file(wav, filenamebytes, ffi.NULL):
            raise DecodeError("could not open/decode file")
        try:
            duration = wav.totalPCMFrameCount / wav.sampleRate
            sample_width = wav.bitsPerSample // 8
            is_float = wav.translatedFormatTag == lib.DR_WAVE_FORMAT_IEEE_FLOAT
            return AudioFileInfo(filename, FileFormat.WAV, wav.channels, wav.sampleRate,
                                 get_format_from_width(sample_width, is_float), duration, wav.totalPCMFrameCount)
        finally:
            lib.drwav_uninit(wav)


def wav_get_info(data: bytes) -> AudioFileInfo:
    """
    Fetches information about the audio data in WAV format.

    Parameters:
    - data (bytes): Audio data in bytes.

    Returns:
    - AudioFileInfo: Object containing various properties of the audio data.
    """
    with ffi.new("drwav*") as wav:
        if not lib.drwav_init_memory(wav, data, len(data), ffi.NULL):
            raise DecodeError("could not open/decode data")
        try:
            duration = wav.totalPCMFrameCount / wav.sampleRate
            sample_width = wav.bitsPerSample // 8
            is_float = wav.translatedFormatTag == lib.DR_WAVE_FORMAT_IEEE_FLOAT
            return AudioFileInfo("<memory>", FileFormat.WAV, wav.channels, wav.sampleRate,
                                 get_format_from_width(sample_width, is_float), duration, wav.totalPCMFrameCount)
        finally:
            lib.drwav_uninit(wav)
