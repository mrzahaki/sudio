# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

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
cimport numpy as np
cimport cython
from sudio.process.fx._tempo cimport _tempo_cy
from sudio.process.fx._fade_envelope cimport prepare_envelope
from scipy.interpolate import interp1d
from sudio.rateshift import ConverterType, resample, Resampler

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray pitch_shifter_cy(
        np.ndarray input_audio, 
        np.ndarray[double, ndim=1] envelope,
        float ratio=1.0, 
        int sample_rate=44100,
        bint enable_spline=True,
        float spline_sigma = 0.1,
        float fade_max_db = 0.0,
        float fade_min_db = -60.0,
        int envbuffer = 400,
        int envlen = 600,
        int frame_length = 40, 
        int sequence_ms = 82,
        int seekwindow_ms = 28, 
        int overlap_ms = 12,
        object converter_type=ConverterType.sinc_fastest
):

    if len(envelope) > 1:
        envelope = 1.0 / envelope
        envelope = prepare_envelope(
            envlen,
            envelope,
            enable_spline,
            spline_sigma,
            fade_max_db,
            fade_min_db,
            envbuffer
        )
    else:
        ratio = 1.0 / ratio
        envelope = np.full(envlen, ratio, dtype=np.float64)


    intp = interp1d(
            np.linspace(0, 1, len(envelope)), 
            envelope
        )
    
    cdef np.ndarray tempo_res = _tempo_cy(
        input_audio,
        intp,
        sample_rate=sample_rate,
        sequence_ms=sequence_ms,
        seekwindow_ms=seekwindow_ms,
        overlap_ms=overlap_ms,
    )
    
    if tempo_res.ndim == 1:
        tempo_res = tempo_res[np.newaxis, :]


    cdef np.ndarray[np.float32_t, ndim=2] result
    cdef int nchannels = tempo_res.shape[0]
    cdef int samples = tempo_res.shape[1]
    cdef int data_chunk = (sample_rate * frame_length) // 1000
    cdef int total_steps = samples // data_chunk
    cdef int current_pos
    cdef float current_ratio
    cdef np.ndarray[np.float32_t, ndim=1] frame
    cdef np.ndarray[np.float32_t, ndim=2] resampled


    if len(envelope) > 1:
        result = np.zeros((nchannels, 0), dtype=np.float32)
        resampler = Resampler(converter_type, nchannels)
        
        for i in range(total_steps):
            current_pos = i * data_chunk
            is_last_chunk = current_pos + data_chunk >= samples
            current_ratio = float(intp(float(current_pos) / samples))
            resampler.set_ratio(current_ratio)
            frame = tempo_res[:, current_pos: current_pos + data_chunk].T.flatten()
            frame = resampler.process(frame, current_ratio, is_last_chunk)
            resampled = frame.reshape(len(frame) // nchannels, nchannels).T
            result = np.concatenate((result, resampled), axis=1)
    
    else:
        result = resample(
            tempo_res,
            ratio,
            converter_type   
        )

    if nchannels == 1:
        result = result[0]
    
    return result



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray _pitch_shifter_cy(
        np.ndarray input_audio, 
        object intp,
        int sample_rate=44100,
        int frame_length = 85, 
        int sequence_ms = 82,
        int seekwindow_ms = 28, 
        int overlap_ms = 12,
        object converter_type=ConverterType.sinc_fastest
):

    cdef np.ndarray tempo_res = _tempo_cy(
        input_audio,
        intp,
        sample_rate=sample_rate,
        sequence_ms=sequence_ms,
        seekwindow_ms=seekwindow_ms,
        overlap_ms=overlap_ms,
    )
    
    if tempo_res.ndim == 1:
        tempo_res = tempo_res[np.newaxis, :]


    cdef np.ndarray[np.float32_t, ndim=2] result
    cdef int nchannels = tempo_res.shape[0]
    cdef int samples = tempo_res.shape[1]
    cdef int data_chunk = (sample_rate * frame_length) // 1000
    cdef int total_steps = samples // data_chunk
    cdef int current_pos
    cdef float current_ratio
    cdef np.ndarray[np.float32_t, ndim=1] frame
    cdef np.ndarray[np.float32_t, ndim=2] resampled


    result = np.zeros((nchannels, 0), dtype=np.float32)
    resampler = Resampler(converter_type, nchannels)
    
    for i in range(total_steps):
        current_pos = i * data_chunk
        is_last_chunk = current_pos + data_chunk >= samples
        current_ratio = float(intp(float(current_pos) / samples))
        frame = tempo_res[:, current_pos: current_pos + data_chunk].T.flatten()
        frame = resampler.process(frame, current_ratio, is_last_chunk)
        resampled = frame.reshape(len(frame) // nchannels, nchannels).T
        result = np.concatenate((result, resampled), axis=1)
    

    if nchannels == 1:
        result = result[0]
    
    return result

