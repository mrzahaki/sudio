

try:
    import os
    import sys
    import ctypes

    # Redirect stderr to /dev/null
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    # Define our error handler type
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)

    def py_error_handler(filename, line, function, err, fmt):
        pass  # Ignore the error messages

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

    # Load the ALSA library
    asound = ctypes.CDLL('libasound.so')

    # Set the error handler
    asound.snd_lib_error_set_handler(c_error_handler)
    sys.stderr = stderr
except:
    pass

from sudio.audioutils.miniaudio import _lib

# Restore stderr