
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
