
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


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
