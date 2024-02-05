from _miniaudio import ffi, lib

from sudio.types import LibSampleFormat, DitherMode, DecodeError
from sudio.audioutils.typeconversion import get_sample_width_from_format
from sudio.extras.io import get_encoded_filename_bytes



def decode_audio_file(filename: str, 
                      output_format: LibSampleFormat = LibSampleFormat.SIGNED16,
                      nchannels: int = 2, 
                      sample_rate: int = 44100, 
                      dither: DitherMode = DitherMode.NONE) -> bytes:
    """
    Convenience function to decode any supported audio file to raw PCM samples in the chosen format.

    Args:
        filename (str): The name of the audio file to be decoded.
        output_format (LibSampleFormat): The desired sample format for the output.
        nchannels (int): The number of audio channels.
        sample_rate (int): The sample rate of the audio data.
        dither (DitherMode): The dithering mode to be applied during decoding.

    Returns:
        bytes: Raw PCM samples in the specified format.

    Raises:
        DecodeError: If the decoding process fails.
    """
    sample_width = get_sample_width_from_format(output_format)
    filename_bytes = get_encoded_filename_bytes(filename)
    
    with ffi.new("ma_uint64 *") as frames, ffi.new("void **") as memory:
        decoder_config = lib.ma_decoder_config_init(output_format.value, nchannels, sample_rate)
        decoder_config.ditherMode = dither.value
        
        result = lib.ma_decode_file(filename_bytes, ffi.addressof(decoder_config), frames, memory)
        if result != lib.MA_SUCCESS:
            raise DecodeError("Failed to decode file", result)
        
        buflen = frames[0] * nchannels * sample_width
        buffer = ffi.buffer(memory[0], buflen)
        decoded_bytes = bytes(buffer)
        
        lib.ma_free(memory[0], ffi.NULL)
        
    return decoded_bytes
