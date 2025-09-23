

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


def shuffle3d_channels(arr):
    """
    Shuffles the channels of a 3D array and returns a flattened result.
    
    Parameters:
    - arr (numpy.ndarray): Input 3D array of shape (frames, channels, samples_per_frame)
    
    Returns:
    - numpy.ndarray: Flattened array with interleaved channels.
    """
    frames, channels, samples_per_frame = arr.shape
    # Reshape to (frames * samples_per_frame, channels)
    reshaped = arr.transpose(0, 2, 1).reshape(-1, channels)
    # Interleave channels and flatten
    return reshaped.flatten()


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


def get_mute_mode_data(nchannel, nperseg):
        if nchannel < 2:
            return np.zeros((nperseg), 'f')
        else:
            return np.zeros((nchannel, nperseg), 'f')
        

def map_channels(in_data:np.ndarray, in_channels, out_channels):
    """
    Map input audio channels to desired output channels.

    Args:
    in_data (np.ndarray): Input audio data.
    in_channels (int): Number of input channels.
    out_channels (int): Number of desired output channels.
    data_chunk (int): Size of data chunk for processing.

    Returns:
    np.ndarray: Processed audio data with desired number of channels.
    """

    if in_channels == 1:
        output = np.expand_dims(in_data, 0)
    else:
        # Reshape multi-channel data
        output = in_data.reshape(-1, in_channels).T

    # Upmixing
    if in_channels < out_channels:
        # Duplicate last channel for additional output channels
        output = np.vstack((output, np.tile(output[-1], (out_channels - in_channels, 1))))

    # Downmixing
    elif in_channels > out_channels:
        output = np.mean(output, axis=0, keepdims=True)
        output = np.tile(output[-1], (in_channels - out_channels, 1))

        # if in_channels == 2 and out_channels == 1:
        #     # Stereo to mono
        #     output = np.mean(output, axis=0, keepdims=True)
        # else:
        #     # General downmixing (average channels)
        #     output = output[:out_channels]
    
    return output