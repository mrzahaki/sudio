cimport numpy as cnp


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




cpdef int find_nearest_divisible(int reference_number, int divisor)
cpdef int find_nearest_divisor(int num, int divisor) except? -1
cpdef db2amp(db)
cpdef amp2db(amp)
