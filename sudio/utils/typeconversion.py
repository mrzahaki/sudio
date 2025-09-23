
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



from sudio.io import SampleFormat
import numpy as np



def convert_array_type(arr:np.ndarray, target_format:SampleFormat, source_format:SampleFormat=SampleFormat.UNKNOWN):
    """
    Convert the data type of a NumPy array based on the given SampleFormat.
    
    Args:
    arr (np.ndarray): Input NumPy array
    target_format (SampleFormat): Desired output format
    
    Returns:
    np.ndarray: Converted NumPy array
    """
    if not (source_format == SampleFormat.UNKNOWN) and not(source_format == target_format):
        if not source_format == SampleFormat.FLOAT32:
            arr = arr.astype(np.float32)

        if source_format == SampleFormat.SIGNED32:
            arr = arr / 2**31
        elif source_format == SampleFormat.SIGNED24:
            arr = arr / 2**23
        elif source_format == SampleFormat.SIGNED16:
            arr = arr / 2**15
        elif source_format == SampleFormat.UNSIGNED8:
            arr = arr / 2**8
    
        if target_format == SampleFormat.FLOAT32:
            ...
        elif target_format == SampleFormat.SIGNED32:
            arr = np.clip(arr * (2 ** 31), -(2 ** 31), (2 ** 31) - 1)
        elif target_format == SampleFormat.SIGNED24:
            arr = np.clip(arr * (2 ** 23), -(2 ** 23), (2 ** 23) - 1)
        elif target_format == SampleFormat.SIGNED16:
            arr = np.clip(arr * (2 ** 15), -(2 ** 15), (2 ** 15) - 1)
        elif target_format == SampleFormat.UNSIGNED8:
            arr = np.clip(arr * 128 + 128, 0, 255)


    if target_format == SampleFormat.FLOAT32:
        return arr.astype(np.float32)
    elif target_format == SampleFormat.SIGNED32 or target_format == SampleFormat.SIGNED24:
        return arr.astype(np.int32)
    elif target_format == SampleFormat.SIGNED16:
        return arr.astype(np.int16)
    elif target_format == SampleFormat.UNSIGNED8:
        return arr.astype(np.uint8)
    elif target_format == SampleFormat.UNKNOWN:
        return arr  
    else:
        raise ValueError("Unsupported sample format")
