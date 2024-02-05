from sudio._register import Members as Mem
from sudio.audioutils.audio import Audio

def get_default_input_device_info():
    """
    Retrieves information about the default input audio device.

    Returns:
        dict: A dictionary containing information about the default input device.
    """
    au = Audio()
    data = au.get_default_input_device_info()
    au.terminate()
    return data


def get_device_count():
    """
    Gets the total number of available audio devices.

    Returns:
        int: The number of available audio devices.
    """
    au = Audio()
    data = au.get_device_count()
    au.terminate()
    return data


def get_device_info_by_index(index: int):
    """
    Retrieves information about an audio device based on its index.

    Args:
        index (int): The index of the audio device.

    Returns:
        dict: A dictionary containing information about the specified audio device.
    """
    au = Audio()
    data = au.get_device_info_by_index(int(index))
    au.terminate()
    return data


def get_input_devices():
    """
    Gets information about all available input audio devices.

    Returns:
        dict: A dictionary containing information about each available input device.
    """
    p = Audio()
    dev = {}
    default_dev = p.get_default_input_device_info()['index']
    for i in range(p.get_device_count()):
        tmp = p.get_device_info_by_index(i)
        if tmp['maxInputChannels'] > 0:
            i = tmp['index']
            del tmp['index']
            dev[i] = tmp
            if i == default_dev:
                dev['defaultDevice'] = True
            else:
                dev['defaultDevice'] = False
    p.terminate()
    assert len(dev) > 0
    return dev


def get_output_devices():
    """
    Gets information about all available output audio devices.

    Returns:
        dict: A dictionary containing information about each available output device.
    """
    p = Audio()
    dev = {}
    default_dev = p.get_default_output_device_info()['index']
    for i in range(p.get_device_count()):
        tmp = p.get_device_info_by_index(i)
        if tmp['maxOutputChannels'] > 0:
            i = tmp['index']
            del tmp['index']
            dev[i] = tmp
            if i == default_dev:
                dev['defaultDevice'] = True
            else:
                dev['defaultDevice'] = False
    p.terminate()
    assert len(dev) > 0
    return dev
