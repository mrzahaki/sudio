"""
with the name of ALLAH:
 SUDIO (https://github.com/MrZahaki/sudio)

 audio processing platform

 Author: hussein zahaki (hossein.zahaki.mansoor@gmail.com)

 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""

import time
import numpy as np
# import pyaudio
# import wave
# import pandas as pd
# from scipy.signal import upfirdn, firwin, lfilter, correlate
# from _pipeline import Pipeline


class Tools:
    @staticmethod
    def _str_conv(st):
        if st.isnumeric():
            st = int(st)
        elif st.replace('.', '', 1).isnumeric():
            st = float(st)
        elif '{' in st and '}' in st:
            st = st.strip('{').strip('}')
            st = Tools.str2dict(st)
        elif '[' in st and ']' in st:
            st = st.strip('[').strip(']')
            st = Tools.str2list(st)
        return st

    @staticmethod
    def str2dict(st, dict_eq=':', item_sep=','):
        if not st: return None
        buff = {}
        i = [i.strip().split(dict_eq) for i in st.split(item_sep)]
        for j in i:
            if j:
                value = j[1] if len(j) > 1 else None
                if value:
                    value = Tools._str_conv(value)
                buff[j[0]] = value
        return buff

    @staticmethod
    def str2list(st):
        if not st: return None
        buff = []
        i = [i.strip() for i in st.split(',')]
        for value in i:
            if value:
                value = Tools._str_conv(value)
            buff.append(value)
        return buff

    @staticmethod
    # Find the number closest to num 1 that is divisible by num 2
    def near_divisible(divisible, num1):
        buf = divisible % num1, int(divisible/num1) + 1 * 6 % 20
        return divisible + [buf[0] * -1, buf[1]][buf.index(min(buf))]

    # Find the nearest integer divisor, so that the remainder of the division is zero.
    @staticmethod
    def near_divisor(num, divisor):
        div = int(np.round(num / divisor))
        res = np.array([0, div])
        while res[1] < num:
            if not num % res[1]:
                if res[0]:
                    res = (num / res)
                    div = np.abs(res - divisor)
                    return int(res[div == np.min(div)][0])
                res[0] = res[1]
                res[1] = div

            if res[0]:
                res[1] -= 1
            else:
                res[1] += 1
        raise ValueError

    @staticmethod
    def dbu(v):
        return 20 * np.log10(v / np.sqrt(.6))

    @staticmethod
    def adbu(dbu):
        return np.sqrt(.6) * 10 ** (dbu/20)

    @staticmethod
    def push(obj, data):
        obj.pop()
        obj.insert(0, data)

    @staticmethod
    def ipush(obj, data):
        obj.reverse()
        Tools.push(obj, data)
        obj.reverse()

    @staticmethod
    def time_name():
        name = time.gmtime()
        name = str(name.tm_year) + str(name.tm_mon) + str(name.tm_mday) + '_' + str(name.tm_sec) + str(
            time.time() % 1)[2:6]
        return name
