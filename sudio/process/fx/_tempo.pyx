# distutils: language=c++
# cython: language_level=3


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

# distutils: language=c++
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
cimport sudio.process.fx._fade_envelope as fade_envelope

from libc.math cimport sqrt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sudio.utils.math cimport db2amp

DEF DEFAULT_SEQUENCE_MS = 82
DEF DEFAULT_SEEKWINDOW_MS = 28  
DEF DEFAULT_OVERLAP_MS = 12


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray tempo_cy(
    np.ndarray input_audio,
    np.ndarray[double, ndim=1] envelope,
    int sample_rate=44100,
    int sequence_ms=DEFAULT_SEQUENCE_MS,
    int seekwindow_ms=DEFAULT_SEEKWINDOW_MS, 
    int overlap_ms=DEFAULT_OVERLAP_MS,
    bint enable_spline = False,
    double spline_sigma = 0.1,
    double fade_max_db = 0.0,
    double fade_min_db = -60.0,
    int envbuffer = 400,
    int envlen = 600,
    double default_tempo = 1.0
):

    
    input_audio = np.asarray(input_audio, dtype=np.float32)
    if input_audio.ndim == 1:
        input_audio = input_audio[np.newaxis, :]
    
    cdef:
        np.ndarray[double, ndim=1] tempo_values
        
    
    if len(envelope) > 1:
       tempo_values =  fade_envelope.prepare_envelope(
            envlen,
            envelope,
            enable_spline,
            spline_sigma,
            fade_max_db,
            fade_min_db,
            envbuffer
        )
    else:
        tempo_values = np.full(envlen, default_tempo, dtype=np.float64)

    intp = interp1d(
            np.linspace(0, 1, len(tempo_values)), 
            tempo_values
        )

    cdef np.ndarray result =  _tempo_cy(
            input_audio,
            intp,
            sample_rate,
            sequence_ms,
            seekwindow_ms,
            overlap_ms
        )
    
    return result




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray _tempo_cy(
    np.ndarray input_audio,
    object intp,
    int sample_rate=44100,
    int sequence_ms=DEFAULT_SEQUENCE_MS,
    int seekwindow_ms=DEFAULT_SEEKWINDOW_MS, 
    int overlap_ms=DEFAULT_OVERLAP_MS,
):

    
    input_audio = np.asarray(input_audio, dtype=np.float32)
    if input_audio.ndim == 1:
        input_audio = input_audio[np.newaxis, :]
    
    cdef:
        int channels = input_audio.shape[0]
        int frames = input_audio.shape[1]
        int overlap_length = (sample_rate * overlap_ms) // 1000
        int sequence_length = (sample_rate * sequence_ms) // 1000
        int seekwindow_length = (sample_rate * seekwindow_ms) // 1000
        int output_frames = frames * 2  # Generous initial allocation
        int input_pos = 0
        int output_pos = 0
        int best_offset
        double corr, best_corr
        int i, ch
        float scale1, scale2
        float[:] signal1_view
        float[:] signal2_view
        np.ndarray[np.float32_t, ndim=2] output_buffer = np.zeros((channels, output_frames), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] mid_buffer = np.zeros((channels, overlap_length), dtype=np.float32)
        
        double current_tempo
        double skip_fract = 0.0
        int skip
        double nominal_skip
        double input_progress_ratio
    

    while input_pos + seekwindow_length < frames:
        input_progress_ratio = float(input_pos) / frames
        current_tempo = float(intp(input_progress_ratio))
        
        nominal_skip = current_tempo * (sequence_length - overlap_length)
        skip = int(skip_fract + nominal_skip + 0.5)
        skip_fract += nominal_skip - skip
        
        best_offset = 0
        best_corr = -1.0
        for i in range(seekwindow_length - overlap_length):
            signal1_view = input_audio[0, input_pos + i:input_pos + i + overlap_length]
            signal2_view = mid_buffer[0, :overlap_length]
            corr = calc_correlation(signal1_view, signal2_view)
            if corr > best_corr:
                best_corr = corr
                best_offset = i
        
        for ch in range(channels):
            for i in range(overlap_length):
                scale1 = float(i) / overlap_length
                scale2 = 1.0 - scale1
                output_buffer[ch, output_pos + i] = (
                    input_audio[ch, input_pos + best_offset + i] * scale1 + 
                    mid_buffer[ch, i] * scale2
                )
        
        sequence_offset = input_pos + best_offset + overlap_length
        sequence_length_current = min(sequence_length - overlap_length, 
                                   frames - sequence_offset)
        
        if sequence_length_current > 0:
            for ch in range(channels):
                output_buffer[ch, output_pos + overlap_length:
                            output_pos + overlap_length + sequence_length_current] = \
                    input_audio[ch, sequence_offset:
                              sequence_offset + sequence_length_current]
        
        if sequence_offset + sequence_length_current - overlap_length < frames:
            for ch in range(channels):
                mid_buffer[ch, :] = input_audio[ch,
                    sequence_offset + sequence_length_current - overlap_length:
                    sequence_offset + sequence_length_current]
        
        input_pos += skip
        output_pos += sequence_length_current
    
    #trim output buffer  mono/multi-channel
    result = output_buffer[:, :output_pos]
    if input_audio.shape[0] == 1:
        result = result[0]
        
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double calc_correlation(float[:] signal1, float[:] signal2) nogil:
    """Calculate normalized cross-correlation between two signals"""
    cdef:
        int length = signal1.shape[0]
        int i
        double corr = 0.0
        double norm1 = 0.0
        double norm2 = 0.0
        
    for i in range(length):
        corr += signal1[i] * signal2[i]
        norm1 += signal1[i] * signal1[i]
        norm2 += signal2[i] * signal2[i]
        
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
        
    return corr / sqrt(norm1 * norm2)
