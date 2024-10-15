
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


import numpy as np
from sudio.rateshift import rateshift
from sudio.io import SampleFormat
from sudio.io import get_sample_size
from sudio.audiosys.channel import shuffle2d_channels


def synchronize_audio(rec,
                      nchannels: int,
                      sample_rate: int,
                      sample_format_id: int,
                      output_data='byte') -> dict:
    """
    Synchronizes audio data with the specified parameters.

    Parameters:
    - rec (dict): The input audio recording data.
    - nchannels (int): The desired number of channels.
    - sample_rate (int): The desired sample rate.
    - sample_format_id (int): The desired sample format ID.
    - output_data (str): Output data format, can be 'byte' or 'ndarray'.

    Returns:
    - dict: The synchronized audio recording data.

    Notes:
    - This function performs channel adjustment, resampling, and sample format conversion.
    - The input `rec` dictionary is modified in-place.

    Usage:
    ```python
    new_rec = synchronize_audio(rec, nchannels=2, sample_rate=44100, sample_format_id=SampleFormat.SIGNED16.value)
    ```

    :param rec: The input audio recording data.
    :param nchannels: The desired number of channels.
    :param sample_rate: The desired sample rate.
    :param sample_format_id: The desired sample format ID.
    :param output_data: Output data format, can be 'byte' or 'ndarray'.
    :return: The synchronized audio recording data.
    """

    form = get_sample_size(rec['sampleFormat'])
    if rec['sampleFormat'] == SampleFormat.FLOAT32:
        form = '<f{}'.format(form)
    else:
        form = '<i{}'.format(form)
    data = np.frombuffer(rec['o'], form)
    if rec['nchannels'] == 1:
        if nchannels > rec['nchannels']:
            data = np.vstack([data for i in range(nchannels)])
            rec['nchannels'] = nchannels

    else:
        # Safety update: Ensure all arrays have the same size
        channel_data = [data[i::rec['nchannels']] for i in range(nchannels)]
        min_length = min(len(channel) for channel in channel_data)
        channel_data = [channel[:min_length] for channel in channel_data]
        data = np.array(channel_data)

    if not sample_rate == rec['frameRate']:
        scale = sample_rate / rec['frameRate']

        if len(data.shape) == 1:
            # mono
            data = rateshift.resample(data, scale, rateshift.ConverterType.sinc_fastest, 1)
        else:
            # multi channel
            res = rateshift.resample(data[0], scale, rateshift.ConverterType.sinc_fastest, 1)
            for i in data[1:]:
                res = np.vstack((res,
                                 rateshift.resample(i, scale, rateshift.ConverterType.sinc_fastest, 1)))
            data = res

    if output_data.startswith('b') and rec['nchannels'] > 1:
        data = shuffle2d_channels(data)

    rec['nchannels'] = nchannels
    rec['sampleFormat'] = sample_format_id

    form = get_sample_size(sample_format_id)
    if sample_format_id == SampleFormat.FLOAT32:
        form = '<f{}'.format(form)
    else:
        form = '<i{}'.format(form)

    if output_data.startswith('b'):
        rec['o'] = data.astype(form).tobytes()
    else:
        rec['o'] = data.astype(form)

    rec['size'] = len(rec['o'])
    rec['frameRate'] = sample_rate

    rec['duration'] = rec['size'] / (rec['frameRate'] *
                                     rec['nchannels'] *
                                     get_sample_size(rec['sampleFormat']))

    return rec