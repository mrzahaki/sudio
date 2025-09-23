
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
    windowing_buffer.pop()
    windowing_buffer.insert(0, data)

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
        windowing_buffer[i].pop()
        windowing_buffer[i].insert(0, data[i])
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




