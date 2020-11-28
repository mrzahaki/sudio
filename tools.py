import numpy as np


class Tools:
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
    def rms(arr):
        return np.max(arr) / np.sqrt(2)
        #return np.sqrt(np.abs(np.mean(arr[arr >= 0] ** 2)))



