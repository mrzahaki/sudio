# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True


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

DTYPE = _np.float32
ctypedef _np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def channel_mixer(
    _np.ndarray[DTYPE_t, ndim=2] data, 
    _np.ndarray[DTYPE_t, ndim=2] correlation,
):

    cdef int nchannels = data.shape[0]
    cdef int nsamples = data.shape[1]
    
    cdef _np.ndarray[DTYPE_t, ndim=2] result = _np.zeros((nchannels, nsamples), dtype=DTYPE)
    
    cdef int i, j, k
    cdef DTYPE_t temp
    
    for i in range(nchannels):
        for j in range(nsamples):
            temp = 0.0
            for k in range(nchannels):
                temp += correlation[i, k] * data[k, j]
            result[i, j] = temp
    
    return result

