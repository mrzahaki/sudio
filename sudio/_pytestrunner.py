import sys
import os

class PytestRunner:
    
    def __init__(self, module_name):
        """
        Initialize the pytest runner for a specific module.

        Parameters
        ----------
        module_name : str
            Name of the module to be tested
        """
        self.module_name = module_name
        self.__module__ = module_name

    def __call__(self, label='default', verbose=1, extra_argv=None,
                 doctests=False, coverage=False, durations=-1, tests=None):
        """
        Run tests for module using pytest.

        Parameters
        ----------
        label : str, optional
            Test label or marker to run. Defaults to 'default'.
        verbose : int, optional
            Verbosity level for test outputs. 
            1: Quiet, 2: More verbose, 3: Most verbose. Default is 1.
        extra_argv : list, optional
            Additional arguments to pass to pytest.
        doctests : bool, optional
            Enable running doctests. Default is False.
        coverage : bool, optional
            Generate code coverage report. Default is False.
            Requires pytest-cov to be installed.
        durations : int, optional
            Control test duration reporting:
            - Negative: No duration reporting
            - 0: Report duration of all tests
            - Positive: Report duration of slowest tests. Default is -1.
        tests : str or list, optional
            Specific test(s) to run. If None, runs all tests for the module.

        Returns
        -------
        bool
            True if tests pass, False otherwise.
        """
        import pytest

        # get the module path
        try:
            module = sys.modules[self.module_name]
            module_path = os.path.abspath(module.__path__[0])
        except (KeyError, AttributeError):
            module_path = self.module_name

        pytest_args = ["-l"]  # local variables on failure
        if verbose == 1:
            pytest_args += ["-q"]  # Quiet mode
        elif verbose > 1:
            pytest_args += ["-" + "v"*(verbose - 1)]  # Increase verbosity

        # optional arguments
        if extra_argv:
            pytest_args += list(extra_argv)
        if doctests:
            pytest_args += ["--doctest-modules"]
        if coverage:
            pytest_args += ["--cov=" + module_path]
        if label != 'default':
            pytest_args += ["-m", label]
        if durations >= 0:
            pytest_args += [f"--durations={durations}"]
        if tests is None:
            tests = [self.module_name]
        elif isinstance(tests, str):
            tests = [tests]

        pytest_args += ["--pyargs"] + list(tests)
        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return code == 0

def get_test_runner(module_name):
    """
    Create a test runner for a given module.

    Parameters
    ----------
    module_name : str
        Name of the module to create a test runner for.

    Returns
    -------
    PytestRunner
        A pytest runner configured for the specified module.
    """
    return PytestRunner(module_name)


__all__ = ['PytestRunner', 'get_test_runner']