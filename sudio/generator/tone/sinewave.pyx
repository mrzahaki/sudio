# distutils: language = c++

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

import numpy as np
cimport numpy as np
from libc.math cimport sin, M_PI
from cython.parallel import prange
import cython
from libcpp.vector cimport vector
from libc.stdint cimport int32_t


cdef extern from "numpy/npy_math.h":
    float NPY_PI


ctypedef np.float32_t DTYPE_t
ctypedef float TYPE
GTYPE = np.float32

# Declare the C++ template function
cdef extern from "sinewave.h" namespace "":
    void integrate_harmonics[AmplitudeT,BaseFreqT,ModDepthT,ModFreqT,PhaseT,T](
        vector[T]& output,
        int32_t size,
        const AmplitudeT& amplitude,
        const BaseFreqT& base_freq,
        const ModDepthT& mod_depth,
        const ModFreqT& mod_freq,
        const vector[T]& time_vector,
        const PhaseT& phase,
        int32_t tonetype
    ) nogil
    
    void calculate_harmonics[AmplitudeT,BaseFreqT,ModDepthT,ModFreqT,PhaseT,T](
        vector[T]& output,
        int32_t size,
        const AmplitudeT& amplitude,
        const BaseFreqT& base_freq,
        const ModDepthT& mod_depth,
        const ModFreqT& mod_freq,
        const vector[T]& time_vector,
        const PhaseT& phase,
        int32_t tonetype
    ) nogil
    
    vector[T] add2vector[T](
        const vector[T]& v1,
        const vector[T]& v2,
    ) nogil





    
def generate_wave(
    int tone_type,
    np.ndarray[DTYPE_t, ndim=1] phase,
    np.ndarray[DTYPE_t, ndim=1] mod_freq,
    np.ndarray[DTYPE_t, ndim=1] mod_depth,
    np.ndarray[DTYPE_t, ndim=1] time_vector,
    np.ndarray[DTYPE_t, ndim=1] base_freq,
    np.ndarray[DTYPE_t, ndim=1] amplitude,
    list freq_comp_amp,
    list frequency_components,
    list phase_diff_map,
    int nchannels,
):
    cdef:
        int i, idx, ipx
        bint single_fca = len(freq_comp_amp) == 1
        np.ndarray[DTYPE_t, ndim=2] output_buffer
        vector[DTYPE_t] time_vector_cpp
        vector[DTYPE_t] buffervect0
        vector[DTYPE_t] fc_vector
        vector[DTYPE_t] fca_vector
        vector[DTYPE_t] amplitude_vector
        vector[DTYPE_t] base_freq_vector
        vector[DTYPE_t] phase_vector
        vector[DTYPE_t] combined_phase
        vector[DTYPE_t] mod_depth_vector
        vector[DTYPE_t] mod_freq_vector
        vector[DTYPE_t] pd_vector
        np.ndarray[DTYPE_t, ndim=1] current_freq_array
        DTYPE_t* freq_ptr
        int32_t size = len(time_vector)

    time_vector_cpp.assign(&time_vector[0], &time_vector[0] + size)
    amplitude_vector.assign(&amplitude[0], &amplitude[0] + amplitude.size)
    base_freq_vector.assign(&base_freq[0], &base_freq[0] + base_freq.size)
    phase_vector.assign(&phase[0], &phase[0] + phase.size)
    mod_depth_vector.assign(&mod_depth[0], &mod_depth[0] + mod_depth.size)
    mod_freq_vector.assign(&mod_freq[0], &mod_freq[0] + mod_freq.size)
    output_buffer = np.zeros((nchannels, size), dtype=GTYPE)
    buffervect0.resize(size)


    for ipx in range(nchannels):
        if len(phase_diff_map) == nchannels:
            current_freq_array = <np.ndarray[DTYPE_t, ndim=1]>phase_diff_map[ipx]
            freq_ptr = <DTYPE_t*>current_freq_array.data
            pd_vector.assign(freq_ptr, freq_ptr + <np.uint32_t>current_freq_array.shape[0])
        else:
            pd_vector.resize(1)
            pd_vector[0] = 0.0

        with nogil:

            combined_phase = add2vector(phase_vector, pd_vector)

            calculate_harmonics(
                buffervect0,
                size,
                amplitude_vector,
                base_freq_vector,
                mod_depth_vector,
                mod_freq_vector,
                time_vector_cpp,
                combined_phase,
                tone_type
            )
        
        for idx in range(len(frequency_components)):
            current_freq_array = <np.ndarray[DTYPE_t, ndim=1]>frequency_components[idx]
            freq_ptr = <DTYPE_t*>current_freq_array.data
            fc_vector.assign(freq_ptr, freq_ptr + <np.uint32_t>current_freq_array.shape[0])
            
            if single_fca:
                current_freq_array = <np.ndarray[DTYPE_t, ndim=1]>freq_comp_amp[0]
            else:
                current_freq_array = <np.ndarray[DTYPE_t, ndim=1]>freq_comp_amp[idx]
            freq_ptr = <DTYPE_t*>current_freq_array.data
            fca_vector.assign(freq_ptr, freq_ptr + <np.uint32_t>current_freq_array.shape[0])


            with nogil:

                integrate_harmonics(
                    buffervect0,
                    size,
                    fca_vector,
                    fc_vector,
                    mod_depth_vector,
                    mod_freq_vector,
                    time_vector_cpp,
                    combined_phase,
                    tone_type
                )

        with nogil:
            for i in range(size):
                output_buffer[ipx, i] = buffervect0[i]

    return output_buffer
