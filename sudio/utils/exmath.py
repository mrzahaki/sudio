
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


import numpy as np

def find_nearest_divisible(reference_number, divisor):
    """
    Finds the number closest to 'reference_number' that is divisible by 'divisor'.

    Args:
        reference_number (int): The reference number.
        divisor (int): The divisor.

    Returns:
        int: The number closest to 'reference_number' that is divisible by 'divisor'.

    Example:
        >>> find_nearest_divisible(17, 5)
        15
        >>> find_nearest_divisible(30, 7)
        28
        >>> find_nearest_divisible(42, 8)
        40
    """
    buf = reference_number % divisor, int(reference_number / divisor) + 1 * 6 % 20
    return reference_number + [buf[0] * -1, buf[1]][buf.index(min(buf))]



def find_nearest_divisor(num, divisor):
    """
    Finds the nearest divisor with zero remainder for 'num'.

    Args:
        num (int): The dividend.
        divisor (int): The candidate divisor.

    Returns:
        int: The nearest divisor.

    Raises:
        ValueError: If no divisor with a zero remainder is found.

    Example:
        >>> find_nearest_divisor(15, 4)
        3
        >>> find_nearest_divisor(25, 6)
        5
        >>> find_nearest_divisor(18, 7)
        6

    Note:
        This function uses NumPy for mathematical operations.
    """
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
        
    raise ValueError("No divisor with a zero remainder found.")



def voltage_to_dBu(v):
    """
    Converts a voltage 'v' to decibels relative to 1 microvolt (uV).

    Args:
        v (float): The input voltage.

    Returns:
        float: The corresponding value in decibels (dBu).

    Example:
        >>> voltage_to_dBu(1e-6)
        0.0
        >>> voltage_to_dBu(1e-5)
        20.0
        >>> voltage_to_dBu(1e-4)
        40.0
    """
    return 20 * np.log10(v / np.sqrt(0.6))

def dBu_to_voltage(dbu):
    """
    Converts a value in decibels relative to 1 microvolt (dBu) to its corresponding voltage.

    Args:
        dbu (float): The input value in decibels.

    Returns:
        float: The corresponding voltage.

    Example:
        >>> dBu_to_voltage(0.0)
        1e-06
        >>> dBu_to_voltage(20.0)
        1e-05
        >>> dBu_to_voltage(40.0)
        0.0001
    """
    return np.sqrt(0.6) * 10 ** (dbu / 20)

