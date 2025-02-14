# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True



# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


import numpy as _np
cimport numpy as _np
cimport cython


cpdef int find_nearest_divisible(int reference_number, int divisor):
    """
    Finds the number closest to 'reference_number' that is divisible by 'divisor'.

    Args:
        reference_number (int): The reference number.
        divisor (int): The divisor.
    """
    cdef int modulo = reference_number % divisor
    cdef int quotient = reference_number // divisor
    cdef int option1 = reference_number - modulo
    cdef int option2 = reference_number + (divisor - modulo)

    return option1 if abs(option1 - reference_number) <= abs(option2 - reference_number) else option2

cpdef int find_nearest_divisor(int num, int divisor) except? -1:
    """
    Finds the nearest divisor with zero remainder for 'num'.

    Args:
        num (int): The dividend.
        divisor (int): The candidate divisor.

    Returns:
        int: The nearest divisor.
    """
    cdef int div = int(round(num / divisor))
    cdef int lower = div
    cdef int upper = div
    
    while upper < num:
        if num % lower == 0:
            return lower
        if num % upper == 0:
            return upper
        
        lower -= 1
        upper += 1
    
    raise ValueError("No divisor with a zero remainder found.")



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef db2amp(db):
    """
    Convert decibels to amplitude.

    Args:
        db (int, float, ndarray): Decibel value(s)

    """
    return _np.power(10.0, (db / 20.0), dtype=_np.float64)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef amp2db(amp):
    """
    Convert amplitude to decibels.

    Args:
        amp (int, float, ndarray): Amplitude value(s)

    """
    return 20.0 * _np.log10(amp, dtype=_np.float64)


cpdef _np.ndarray normalize(_np.ndarray data, 
              float peak_level=0.0, 
              bint equalize_channels=False, 
              float dc_offset=0.0
              ):
    """
    adjusts audio signal levels to optimize dynamic range and channel balance.

    This function provides audio normalization with control over peak levels 
    and channel processing. It allows you to set a target peak amplitude, optionally equalize 
    across multiple channels, and apply a configurable DC offset.

    Parameters:
    -----------
    data : ndarray
        The raw audio data to be normalized. Supports both single-channel and multi-channel inputs[interleaved].
    
    peak_level : float, optional
        Desired peak amplitude in decibels. When set to 0.0, uses the maximum existing amplitude 
        as the reference point. Allows precise control over output signal strength.
    
    equalize_channels : bool, optional
        When True, normalizes all channels to the same maximum absolute amplitude. 
        Useful for maintaining consistent levels across stereo or multi-channel recordings.
    
    dc_offset : float, optional
        Applies an offset relative to the maximum amplitude. Range is typically between -1 and 1. 
        Helps address minor signal imbalances or subtle audio artifacts.
    
    Returns:
    --------
    ndarray
        The normalized audio data, preserving the original input dimensionality(based on sudio standard).
    """

    cdef:
        Py_ssize_t nchannels = 1
        Py_ssize_t nsamples
        _np.ndarray[float, ndim=2] data_2d
        _np.ndarray[float, ndim=1] max_abs_val
        float peak, dc, norm_peak
        Py_ssize_t i, j
    
    
    if data.ndim > 1:
        data_2d = data
        nchannels = data.shape[0]
    else:
        data_2d = _np.expand_dims(data, 0)

    
    # mae
    if equalize_channels:
        max_abs_val = _np.full(nchannels, _np.max(_np.abs(data_2d), axis=None))
    else:
        max_abs_val = _np.max(_np.abs(data_2d), axis=1)
    
    nsamples = data.shape[1]
    
    # dc and norm
    peak = db2amp(peak_level) if peak_level != 0.0 else 1.0
    for i in range(nchannels):
        dc = max_abs_val[i] * dc_offset 
        norm_peak = peak / (max_abs_val[i] + dc)
        if max_abs_val[i] > 0:
            for j in range(nsamples):
                data_2d[i, j] += dc
                data_2d[i, j] *= norm_peak
    
    if nchannels == 1:
        data = data_2d.squeeze(0)

    return data


__all__ = [
    'find_nearest_divisible', 
    'find_nearest_divisor', 
    'db2amp', 
    'amp2db', 
    'normalize' 
    ]

    