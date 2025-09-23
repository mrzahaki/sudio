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


cimport numpy as np

cpdef np.ndarray pitch_shifter_cy(
        np.ndarray input_audio, 
        np.ndarray[double, ndim=1] envelope,
        float ratio=*,
        int sample_rate=*,
        bint enable_spline=*,
        float spline_sigma =*,
        float fade_max_db =*,
        float fade_min_db =*,
        int envbuffer =*,
        int envlen =*,
        int frame_length =*,
        int sequence_ms =*,
        int seekwindow_ms =*,
        int overlap_ms =*,
        object converter_type=*,
)


cpdef np.ndarray _pitch_shifter_cy(
    np.ndarray input_audio, 
    object intp,
    int sample_rate=*,
    int frame_length =*,
    int sequence_ms =*,
    int seekwindow_ms =*,
    int overlap_ms =*,
    object converter_type=*,
)

