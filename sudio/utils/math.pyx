# cython: language_level=3
# distutils: extra_compile_args = -O3


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
    return np.power(10.0, (db / 20.0), dtype=np.float64)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef amp2db(amp):
    """
    Convert amplitude to decibels.

    Args:
        amp (int, float, ndarray): Amplitude value(s)

    """
    return 20.0 * np.log10(amp, dtype=np.float64)

    