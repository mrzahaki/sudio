#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/

from sudio.utils.strtool import generate_timestamp_name


class TimedIndexedString:
    def __init__(self,
                 string: str,
                 start_from: int = None,
                 first_index: int = 0,
                 start_before: str = None,
                 max_index: int = None,
                 seed: str = ''):
        """
        A class for generating indexed strings with optional time-based suffixes.

        Args:
            base_string (str): The initial string.
            start_from (int, optional): Starting index. Defaults to None.
            first_index (int, optional): Initial index. Defaults to 0.
            start_before (str, optional): Character before which the index should start. Defaults to None.
            max_index (int, optional): Maximum index. Defaults to None.
            seed (str, optional): Seed for additional uniqueness. Defaults to ''.
        """
        if start_from is not None:
            index = start_from
        elif start_before is not None:
            index = string.index(start_before)
        else:
            raise ValueError("Either 'start_from' or 'start_before' must be provided.")

        self._count = first_index
        self._first_index = first_index
        self._max_index = max_index
        self._str: list = [(string[:index] + '_' + seed + '_') if seed and (seed not in string) else string[:index] + '_',
                           '',
                           string[index:]]
        self._current_str = ''

    def __call__(self, *args, timed_random: bool = False, timed_regular: bool = False, seed: str = ''):
        """
        Generates an indexed string.

        Args:
            *args: Additional string elements to append to the index.
            timed_random (bool, optional): Whether to add a timed random suffix. Defaults to False.
            timed_regular (bool, optional): Whether to add a timed regular suffix. Defaults to False.
            seed (str, optional): Seed for additional uniqueness. Defaults to ''.

        Returns:
            str: The generated indexed string.

        Raises:
            OverflowError: If the maximum index is reached.

        Example:
            >>> indexed_name = TimedIndexedString("example_", start_before="_", max_index=3)
            >>> indexed_name("suffix")  # 'example_0_suffix'
            >>> indexed_name("extra", timed_regular=True)  # 'example_1_extra_YYYYMMDD_SSMMM'
        """
        if self._max_index and self._count > self._max_index:
            raise OverflowError("Maximum index reached.")

        self._str[1] = str(self._count) + ''.join(args)
        self._count += 1

        if seed:
            self._str[0] += '_' + seed

        if timed_random:
            self._str[1] += generate_timestamp_name('_')
        elif timed_regular:
            self._str[1] += '_' + generate_timestamp_name()

        self._current_str = ''.join(self._str)
        return self._current_str

    def reset(self):
        """
        Resets the index counter to its initial value.
        """
        self._count = self._first_index

    def get(self):
        """
        Gets the current indexed string.

        Returns:
            str: The current indexed string.
        """
        return self._current_str
