
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


import array
from sudio.types import MiniaudioError
from sudio.io import SampleFormat


def push(obj, data):
    """
    Inserts 'data' at the beginning of the list 'obj' and removes the last element.

    Args:
        obj (list): The list to be modified.
        data: The data to be inserted.

    Example:
        >>> my_list = [1, 2, 3, 4, 5]
        >>> push(my_list, 6)
        >>> print(my_list)
        [6, 1, 2, 3, 4]

    Note:
        This function modifies the input list in place.
    """
    obj.pop()
    obj.insert(0, data)


def ipush(obj, data):
    """
    Inserts 'data' at the beginning of the reversed list 'obj' and reverses it back.

    Args:
        obj (list): The list to be modified.
        data: The data to be inserted.

    Example:
        >>> my_list = [1, 2, 3, 4, 5]
        >>> ipush(my_list, 6)
        >>> print(my_list)
        [6, 5, 4, 3, 2]

    Note:
        This function modifies the input list in place.
    """
    obj.reverse()
    push(obj, data)
    obj.reverse()


def create_integer_array_of_size(itemsize: int) -> array.array:
    """
    Creates an integer array with the specified item size.

    Args:
        itemsize (int): The desired item size for the array.

    Returns:
        array.array: An integer array with the specified item size.

    Raises:
        ValueError: If an array with the specified item size cannot be created.
    """
    for typecode in "Bhilq":
        a = array.array(typecode)
        if a.itemsize == itemsize:
            return a
    raise ValueError("Cannot create array with the specified item size.")


def get_array_proto_from_format(sample_format: SampleFormat) -> array.array:
    """
    Get the array prototype based on the specified SampleFormat.

    Args:
        sample_format (SampleFormat): The sample format.

    Returns:
        array.array: The array prototype for the specified sample format.

    Raises:
        MiniaudioError: If the sample format cannot be used directly and needs conversion.
    """
    arrays = {
        SampleFormat.UNSIGNED8: create_integer_array_of_size(1),
        SampleFormat.SIGNED16: create_integer_array_of_size(2),
        SampleFormat.SIGNED32: create_integer_array_of_size(4),
        SampleFormat.FLOAT32: array.array('f')
    }
    if sample_format in arrays:
        return arrays[sample_format]
    raise MiniaudioError("The requested sample format cannot be used directly: "
                         + sample_format.name + " (convert it first)")