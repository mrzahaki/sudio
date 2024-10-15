#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/

import time

def convert_string_to_python_type(st):
    """
    Converts the input string 'st' to a more appropriate Python data type based on its content.

    Args:
        st (str): The input string to be converted.

    Returns:
        Union[int, float, dict, list, str]: The converted value. If 'st' represents a numeric value, it
        is converted to an integer or float. If 'st' contains curly braces, it is converted to a dictionary.
        If 'st' contains square brackets, it is converted to a list. Otherwise, the original string is returned.

    Example:
        >>> convert_string_to_python_type("42")
        42
        >>> convert_string_to_python_type("3.14")
        3.14
        >>> convert_string_to_python_type("{ 'key': 'value' }")
        {'key': 'value'}
        >>> convert_string_to_python_type("[1, 2, 3]")
        [1, 2, 3]
        >>> convert_string_to_python_type("Hello World")
        'Hello World'
    """
    if st.isnumeric():
        st = int(st)
    elif st.replace('.', '', 1).isnumeric():
        st = float(st)
    elif '{' in st and '}' in st:
        st = st.strip('{').strip('}')
        st = parse_dictionary_string(st)
    elif '[' in st and ']' in st:
        st = st.strip('[').strip(']')
        st = parse_list_string(st)
    return st


def parse_dictionary_string(st, dict_eq=':', item_sep=','):
    """
    Converts a string 'st' representing a dictionary-like structure to a Python dictionary.

    Args:
        st (str): The input string representing the dictionary.
        dict_eq (str, optional): The string used as the key-value separator. Defaults to ':'.
        item_sep (str, optional): The string used as the item separator. Defaults to ','.

    Returns:
        dict: The converted dictionary.

    Example:
        >>> parse_dictionary_string("{key1:value1, key2:value2}")
        {'key1': 'value1', 'key2': 'value2'}
        >>> parse_dictionary_string("name:John, age:30, city:New York", dict_eq=':', item_sep=', ')
        {'name': 'John', 'age': 30, 'city': 'New York'}
        >>> parse_dictionary_string("a=1; b=2; c=3", dict_eq='=', item_sep='; ')
        {'a': 1, 'b': 2, 'c': 3}
    """
    if not st:
        return None
    buff = {}
    i = [i.strip().split(dict_eq) for i in st.split(item_sep)]
    for j in i:
        if j:
            value = j[1] if len(j) > 1 else None
            if value:
                value = convert_string_to_python_type(value)
            buff[j[0]] = value
    return buff


def parse_list_string(st):
    """
    Converts a string 'st' representing a list-like structure to a Python list.

    Args:
        st (str): The input string representing the list.

    Returns:
        list: The converted list.

    Example:
        >>> parse_list_string("[1, 2, 3, 4, 5]")
        [1, 2, 3, 4, 5]
        >>> parse_list_string("apple, orange, banana, mango")
        ['apple', 'orange', 'banana', 'mango']
        >>> parse_list_string("3.14, 2.71, 1.618", convert_string_to_python_type=float)
        [3.14, 2.71, 1.618]
    """
    if not st:
        return None
    buff = []
    i = [i.strip() for i in st.split(',')]
    for value in i:
        if value:
            value = convert_string_to_python_type(value)
        buff.append(value)
    return buff



def generate_timestamp_name(seed=None):
    """
    Generates a timestamp-based name.

    Args:
        seed (str, optional): Seed value for additional uniqueness. Defaults to None.

    Returns:
        str: The generated timestamp-based name.

    Example:
        >>> generate_timestamp_name()
        'YYYYMMDD_SSMMM'
        >>> generate_timestamp_name("custom_seed")
        'custom_seedDD_SSMMM'

    Note:
        - 'YYYY' represents the year.
        - 'MM' represents the month.
        - 'DD' represents the day.
        - 'SS' represents the second.
        - 'MMM' represents the millisecond.
        - The seed, if provided, is appended to the beginning of the name.
    """
    current_time = time.gmtime()
    if seed:
        name = str(seed) + str(current_time.tm_mday) + str(current_time.tm_sec) + str(time.time() % 1)[2:5]
    else:
        name = (str(current_time.tm_year) + str(current_time.tm_mon) +
                str(current_time.tm_mday) + '_' + str(current_time.tm_sec) + str(time.time() % 1)[2:5])
    return name


