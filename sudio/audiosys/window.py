#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


import numpy as np
from sudio.utils.arraytool import push


def win_parser_mono(window, win_num):
    return window[win_num]


def win_parser(window, win_num, nchannel):
    return window[nchannel][win_num]


def single_channel_windowing(
        data:np.ndarray, 
        windowing_buffer:list, 
        window:np.ndarray, 
        nhop:int, 
        ):

    retval = np.vstack((windowing_buffer[1], np.hstack((windowing_buffer[1][nhop:],
                                                        windowing_buffer[0][:nhop])))) * window
    push(windowing_buffer, data)

    return retval


def multi_channel_windowing(
        data:np.ndarray,
        windowing_buffer:list,
        window:np.ndarray,
        nhop:int,
        nchannels:int,
        ):

    retval = []
    for i in range(nchannels):
        retval.append(np.vstack((windowing_buffer[i][1], np.hstack(
            (windowing_buffer[i][1][nhop:], windowing_buffer[i][0][:nhop])))) * window)
        push(windowing_buffer[i], data[i])
    return np.array(retval)


def single_channel_overlap(
        data:np.ndarray,
        overlap_buffer:list,
        nhop:int,
        ):
    
    retval = np.hstack((overlap_buffer[0][nhop:], data[1][:nhop])) + data[0]
    overlap_buffer[0] = data[1]
    return retval


def multi_channel_overlap(
        data:np.ndarray,
        overlap_buffer:list,
        nhop:int,
        nchannels:int,
        ):
    # data.shape =>(number of channels, number of windows(2), size of data chunk depend on primary_filter activity).
    # _overlap_buffer => [buffer 0, buffer1, buffer(number of channels)]
    # for 2 channel data must be an 2[two ], 2, self._data_chunk(e.g. 256)
    # pre post, data,
    # 2 window per frame

    # retval = np.hstack((data[n-1][nhop:], current_win[n+1][:nhop])) + current_win[n]
    # data[n-1] =  current_win[n+1]
    retval = np.hstack((overlap_buffer[0][nhop:], win_parser(data, 1, 0)[:nhop])) + \
             win_parser(data, 0, 0)
    overlap_buffer[0] = win_parser(data, 1, 0)

    for i in range(1, nchannels):
        tmp = np.hstack((overlap_buffer[i][nhop:], win_parser(data, 1, i)[:nhop])) + \
              win_parser(data, 0, i)
        retval = np.vstack((retval, tmp))
        overlap_buffer[i] = win_parser(data, 1, i)

    return retval




