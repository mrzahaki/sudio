
from typing import Generator, Optional, Any, Callable
import array
from _miniaudio import ffi, lib

from sudio.types import LibSampleFormat, DitherMode, DecodeError
from sudio.audioutils.typeconversion import get_sample_width_from_format
from sudio.extras.arraytool import get_array_proto_from_format
from sudio.extras.io import get_encoded_filename_bytes


def generate_samples_stream(frames_to_read: int, 
                            nchannels: int, 
                            output_format: LibSampleFormat,
                            decoder: ffi.CData, 
                            data: Any,
                            on_close: Optional[Callable] = None) -> Generator[array.array, int, None]:
    """
    Generates a stream of audio samples from an audio decoder.

    Args:
        frames_to_read (int): Number of frames to read in each iteration.
        nchannels (int): Number of audio channels.
        output_format (LibSampleFormat): The desired sample format for the output.
        decoder (ffi.CData): The audio decoder object.
        data (Any): Additional data associated with the generator.
        on_close (Optional[Callable]): A callback function to be executed when the generator is closed.

    Yields:
        array.array: A stream of audio samples represented as an array.

    Raises:
        None

    Notes:
        The generator is designed to yield chunks of audio samples on each iteration.
        It leverages the provided audio decoder to read PCM frames and convert them to the desired format.
        The generator continues to yield samples until there are no more frames to read from the decoder.

    Example:
        ```
        frames_to_read = 1024
        nchannels = 2
        output_format = LibSampleFormat.SIGNED16
        decoder = initialize_audio_decoder("example.wav")
        data = {'metadata': {'title': 'Example Song'}}
        sample_stream = generate_samples_stream(frames_to_read, nchannels, output_format, decoder, data)
        
        for samples_chunk in sample_stream:
            # Process the audio samples in the desired format.
            process_audio(samples_chunk)
        ```
    """
    _reference = data    # Ensure that any data passed in is not garbage collected.
    sample_width = get_sample_width_from_format(output_format)
    samples_proto = get_array_proto_from_format(output_format)
    allocated_buffer_frames = max(frames_to_read, 16384)

    try:
        with ffi.new("int8_t[]", allocated_buffer_frames * nchannels * sample_width) as decodebuffer:
            buf_ptr = ffi.cast("void *", decodebuffer)
            want_frames = (yield samples_proto) or frames_to_read

            while True:
                num_frames = lib.ma_decoder_read_pcm_frames(decoder, buf_ptr, want_frames)
                if num_frames <= 0:
                    break

                buffer = ffi.buffer(decodebuffer, num_frames * sample_width * nchannels)
                samples = array.array(samples_proto.typecode)
                samples.frombytes(buffer)

                want_frames = (yield samples) or frames_to_read

    finally:
        if on_close:
            on_close()

        # Clean up the audio decoder resources.
        lib.ma_decoder_uninit(decoder)


def stream_audio_file(filename: str, 
                    output_format: LibSampleFormat = LibSampleFormat.SIGNED16, 
                    nchannels: int = 2,
                    sample_rate: int = 44100, 
                    frames_to_read: int = 1024,
                    dither: DitherMode = DitherMode.NONE, 
                    seek_frame: int = 0) -> Generator[array.array, int, None]:
    """
    Streams chunks of raw PCM samples from an audio file.

    Args:
        filename (str): The name of the audio file to be streamed.
        output_format (LibSampleFormat): The desired sample format for the output.
        nchannels (int): The number of audio channels.
        sample_rate (int): The sample rate of the audio data.
        frames_to_read (int): Number of frames to read in each iteration.
        dither (DitherMode): The dithering mode to be applied during decoding.
        seek_frame (int): The frame to seek to before streaming starts.

    Yields:
        array.array: A stream of audio samples represented as an array.

    Raises:
        DecodeError: If the decoding process fails.

    Notes:
        This generator function decodes and streams audio data from the specified file in chunks.
        It supports seeking to a specific frame before streaming begins.
        Use the `send()` method to specify the number of frames to read in each iteration.

    Example:
        ```
        filename = "example.wav"
        output_format = LibSampleFormat.SIGNED16
        nchannels = 2
        sample_rate = 44100
        frames_to_read = 1024
        dither = DitherMode.NONE
        seek_frame = 0

        audio_stream = stream_audio_file(filename, output_format, nchannels, sample_rate, frames_to_read, dither, seek_frame)

        for samples_chunk in audio_stream:
            # Process the audio samples in the desired format.
            process_audio(samples_chunk)
        ```
    """
    filename_bytes = get_encoded_filename_bytes(filename)
    decoder = ffi.new("ma_decoder *")
    decoder_config = lib.ma_decoder_config_init(output_format.value, nchannels, sample_rate)
    decoder_config.ditherMode = dither.value

    result = lib.ma_decoder_init_file(filename_bytes, ffi.addressof(decoder_config), decoder)
    if result != lib.MA_SUCCESS:
        raise DecodeError("Failed to initialize decoder", result)

    if seek_frame > 0:
        result = lib.ma_decoder_seek_to_pcm_frame(decoder, seek_frame)
        if result != lib.MA_SUCCESS:
            raise DecodeError("Failed to seek to frame", result)

    generator = generate_samples_stream(frames_to_read, nchannels, output_format, decoder, None)
    dummy = next(generator)
    assert len(dummy) == 0

    return generator

