import numpy as np
import struct
import wave


def shuffle3d_channels(arr):
    """
    Shuffles the channels of a 3D array and returns a flattened result.

    Parameters:
    - arr (numpy.ndarray): Input 3D array of shape (m, n, c), where m and n are dimensions,
                          and c is the number of channels.

    Returns:
    - numpy.ndarray: Flattened array with shuffled channels.
    """
    reshaped_arr = arr.transpose(2, 0, 1).reshape(arr.shape[-1], -1)
    # arr format: [[data0:[ch0] [ch1]]  [data1: [ch0] [ch1]], ...]
    return reshaped_arr.reshape(-1)


# @Mem.master.add
def shuffle2d_channels(arr):
    """
    Shuffles the channels of a 2D array and returns a flattened result.

    Parameters:
    - arr (numpy.ndarray): Input 2D array of shape (m, n), where m and n are dimensions.

    Returns:
    - numpy.ndarray: Flattened array with shuffled channels.
    """
    return arr.T.reshape(-1)


def get_channel_data(wav, ch_index=None, output='wave', out_wave_name='sound0'):
    """
    Extracts a specific channel or all channels from a WAV file.

    Parameters:
    - wav (wave.Wave_read): Input WAV file opened in read mode.
    - ch_index (int or None): Index of the channel to extract. If None, extracts all channels.
    - output (str): Output format, can be 'array', 'wave', or 'frame'.
    - out_wave_name (str): Output WAV file name (used when output is 'wave').

    Returns:
    - numpy.ndarray or bytes: Depending on the 'output' parameter.
        - If 'array', returns a NumPy array containing the extracted channel(s).
        - If 'wave', writes a new WAV file and returns None.
        - If 'frame', returns the raw frames as bytes.

    Note:
    - The 'wave' output option writes a new WAV file if 'ch_index' is None (all channels), otherwise, it creates a mono WAV file.
    """
    nframes = wav.getnframes()
    ncahnnels = wav.getnchannels()

    frame_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(wav.getsampwidth())]  # 1 2 4 8
    wav.rewind()

    frame = wav.readframes(nframes)
    # convert the byte string into a list of ints, little-endian samples
    signal = list(struct.unpack(f'<{nframes * ncahnnels}{frame_width}', frame))

    if not ch_index is None:
        ch_index %= ncahnnels
        signal = signal[ch_index::ncahnnels]

    if output == 'array':
        if ch_index is None:
            retval = []
            for i in range(ncahnnels):
                retval += [signal[i::ncahnnels]]
            return np.array(retval)
        return np.array(signal)

    elif output == 'frame':
        return frame
    else:
        obj = wave.open(f'{out_wave_name}.wav', 'w')
        if ch_index is None:
            obj.setnchannels(ncahnnels)
        else:
            obj.setnchannels(1)  # mono
        obj.setsampwidth(wav.getsampwidth())
        obj.setframerate(wav.getframerate())
        obj.writeframes(frame)
        obj.close()


def set_channel_data(wav, ch_index, inp, inp_format='wave', output='wave', out_wave_name='sound0'):
    """
    Sets the specified channel of a WAV file to a new signal.

    Parameters:
    - wav (wave.Wave_write): Input WAV file opened in write mode.
    - ch_index (int): Index of the channel to set.
    - inp (numpy.ndarray, wave.Wave_read, or bytes): Input signal to set the channel. 
        - If 'wave.Wave_read', it extracts the frames from the input wave file.
        - If 'bytes', it directly uses the input frames as bytes.
        - If 'numpy.ndarray', it assumes a mono signal.
    - inp_format (str): Format of the input signal ('wave', 'frame', or 'array').
    - output (str): Output format, can be 'array', 'wave', or 'frame'.
    - out_wave_name (str): Output WAV file name (used when output is 'wave').

    Returns:
    - numpy.ndarray, wave.Wave_read, or bytes: Depending on the 'output' parameter.
        - If 'array', returns a NumPy array containing the modified signal.
        - If 'wave', writes a new WAV file and returns the opened file.
        - If 'frame', returns the modified frames as bytes.
    """
    wav_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(wav.getsampwidth())]  # 1 2 4 8

    if inp_format == 'wave':
        assert (inp.getsampwidth() == wav.getsampwidth() and
                inp.getnframes() == wav.getnframes() and
                inp.getnchannels() == 1 and
                inp.getframerate == wav.getframerate())
        inp.rewind()
        signal = inp.readframes(inp.getnframes())
        frame_width = ['c', 'h', 'i', 'q'][[1, 2, 4, 8].index(inp.getsampwidth())]  # 1 2 4 8
        assert wav_width == frame_width

        # convert the byte string into a list of ints, little-endian 16-bit samples
        signal = list(struct.unpack(f'<{inp.getnframes()}{frame_width}',
                                    signal))

    else:
        assert (len(inp) == wav.getnframes())

        if inp_format == 'frame':
            # convert the byte string into a list of ints, little-endian 16-bit samples
            signal = list(struct.unpack(f'<{wav.getnframes()}{wav_width}',
                                        inp))

        else:  # np array
            signal = inp.tolist()

    nchannels = wav.getnchannels()
    assert nchannels > 1
    ch_index %= nchannels

    wav.rewind()
    wav_signal = wav.readframes(wav.getnframes())
    # convert the byte string into a list of ints, little-endian 16-bit samples
    wav_signal = list(struct.unpack(f'<{wav.getnframes() * nchannels}{wav_width}',
                                    wav_signal))

    # print(len(wav_signal), len(signal))
    for i, j in zip(range(ch_index, len(wav_signal), nchannels), range(ch_index, len(signal))):
        wav_signal[i] = signal[j]

    if output == 'array':
        return np.array(wav_signal)

    else:
        signal = struct.pack(f'<{len(wav_signal)}{wav_width}', *wav_signal)
        if output == 'frame':
            return signal
        else:
            obj = wave.open(f'./sounds/{out_wave_name}.wav', 'w')
            obj.setnchannels(nchannels)
            obj.setsampwidth(wav.getsampwidth())
            obj.setframerate(wav.getframerate())
            obj.writeframes(signal)
            obj.close()
            return wave.open(f'./sounds/{out_wave_name}.wav', 'r')


def delete_channel(wav, ch_index, output='wave', out_wave_name='sound0'):
    """
    Deletes a specified channel from a WAV file.

    Parameters:
    - wav (wave.Wave_write): Input WAV file opened in write mode.
    - ch_index (int): Index of the channel to delete.
    - output (str): Output format, can be 'array', 'wave', or 'frame'.
    - out_wave_name (str): Output WAV file name (used when output is 'wave').

    Returns:
    - numpy.ndarray, wave.Wave_read, or bytes: Depending on the 'output' parameter.
        - If 'array', returns a NumPy array containing the modified signal.
        - If 'wave', writes a new WAV file and returns the opened file.
        - If 'frame', returns the modified frames as bytes.

    Note:
    - The deleted channel is replaced with zeros.
    """
    return set_channel_data(wav, ch_index, np.zeros(wav.getnframes(), dtype='int'), inp_format='array',
                       output=output, out_wave_name=out_wave_name)


