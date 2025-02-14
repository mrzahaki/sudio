# cython: language_level=3

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

cimport numpy as _np


# cdef int DEFAULT_SEQUENCE_MS = 82
# cdef int DEFAULT_SEEKWINDOW_MS = 28
# cdef int DEFAULT_OVERLAP_MS = 12


cpdef _np.ndarray tempo_cy(
    _np.ndarray input_audio,
    _np.ndarray[double, ndim=1] envelope,
    int sample_rate=*,
    int sequence_ms=*,
    int seekwindow_ms=*,
    int overlap_ms=*,
    bint enable_spline=*,
    double spline_sigma=*,
    double fade_max_db=*,
    double fade_min_db=*,
    int envbuffer=*,
    int envlen=*,
    double default_tempo=*
)


cpdef _np.ndarray _tempo_cy(
    _np.ndarray input_audio,
    object intp,
    int sample_rate=*,
    int sequence_ms=*,
    int seekwindow_ms=*, 
    int overlap_ms=*,
)

