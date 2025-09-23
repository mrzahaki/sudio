
# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio



import numpy as np
from sudio.rateshift import ConverterType, resample
from sudio.io import SampleFormat
from sudio.io import get_sample_size
from sudio.utils.channel import shuffle2d_channels
from sudio.metadata import AudioMetadata


def synchronize_audio(rec: AudioMetadata,
                      nchannels: int,
                      sample_rate: int,
                      sample_format_id: int,
                      output_data='byte') -> dict:
    """
    Synchronizes and transforms audio recording parameters.

    Modifies audio recording by adjusting channels, resampling, 
    and converting sample format.

    Args:
    -----

        - rec: Audio recording metadata
        - nchannels: Desired number of audio channels
        - sample_rate: Target sample rate
        - sample_format_id: Desired audio sample format
        - output_data: Output data format ('byte' or 'ndarray')

    Returns:
    --------

        - Modified audio recording metadata
    """

    form = get_sample_size(rec.sampleFormat)
    if rec.sampleFormat == SampleFormat.FLOAT32:
        form = '<f{}'.format(form)
    else:
        form = '<i{}'.format(form)
    data = np.frombuffer(rec.o, form)
    if rec.nchannels == 1:
        if nchannels > rec.nchannels:
            data = np.vstack([data for i in range(nchannels)])
            rec.nchannels = nchannels

    else:
        # Safety update: Ensure all arrays have the same size
        channel_data = [data[i::rec.nchannels] for i in range(nchannels)]
        min_length = min(len(channel) for channel in channel_data)
        channel_data = [channel[:min_length] for channel in channel_data]
        data = np.array(channel_data)

    if not sample_rate == rec.frameRate:
        scale = sample_rate / rec.frameRate
        dtype = data.dtype
        data = data.astype(np.float32)
        data = resample(data, scale, ConverterType.sinc_fastest)
        data.astype(dtype)

    if output_data.startswith('b') and rec.nchannels > 1:
        data = shuffle2d_channels(data)

    rec.nchannels = nchannels
    rec.sampleFormat = sample_format_id

    form = get_sample_size(sample_format_id)
    if sample_format_id == SampleFormat.FLOAT32:
        form = '<f{}'.format(form)
    else:
        form = '<i{}'.format(form)

    if output_data.startswith('b'):
        rec.o = data.astype(form).tobytes()
    else:
        rec.o = data.astype(form)

    rec.size = len(rec.o)
    rec.frameRate = sample_rate

    rec.duration = rec.size / (rec.frameRate *
                                     rec.nchannels *
                                     get_sample_size(rec.sampleFormat))

    return rec