

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



import os
import sys

def get_encoded_filename_bytes(filepath: str) -> bytes:
    """
    Encode the given file path string into bytes using the system's filesystem encoding.

    Parameters:
    - filepath (str): The input file path.

    Returns:
    - bytes: The encoded file path as bytes.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    """
    expanded_filepath = os.path.expanduser(filepath)
    
    if not os.path.isfile(expanded_filepath):
        raise FileNotFoundError(filepath)

    return expanded_filepath.encode(sys.getfilesystemencoding())
