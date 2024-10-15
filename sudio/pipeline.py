#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/

import threading
import queue
import time
from typing import Union
import traceback
from sudio.types.pipelineonbusytype import PipelineOnBusyType

class Pipeline(threading.Thread):
    """
    Pipeline class for audio processing.

    Attributes:
        max_size (int): Maximum size of the pipeline.
        io_buffer_size (int): Size of the I/O buffer.
        on_busy (Union[float, str]): Action to take when the pipeline is busy.
        list_dispatch (bool): Flag indicating whether to use list dispatch mode.
        _timeout (float): Timeout value for operations on queues.
        _pipeline (list): List to store the processing pipeline.
        _sync (threading.Event): Synchronization event.
        input_line (queue.Queue): Input data queue.
        _pipinput_queue (queue.Queue): Internal queue for pipeline input.
        _pipoutput_queue (queue.Queue): Internal queue for pipeline output.
        output_line (queue.Queue): Output data queue.
        refresh_ev (threading.Event): Refresh event for updating the pipeline.
        init_queue (queue.Queue): Queue for initialization functions.
        _list_dispatch (bool): Flag indicating whether to use list dispatch mode.

    Methods:
        __call__(self, data): Callable method for putting data into the input queue.
        clear(self): Clears both input and output queues.
        run(self): Main thread method for running the pipeline.
        _run_dispatched(self): Runs the pipeline in list dispatch mode.
        _run_norm(self): Runs the pipeline in normal mode.
        _post_process(self, retval): Performs post-processing and puts the result into the output queue.
        refresh(self): Refreshes the pipeline by processing initialization and pipeline input queues.
        insert(self, index, *func, args=(), init=()): Inserts functions into the pipeline at a specified index.
        append(self, *func, args=(), init=()): Appends functions to the end of the pipeline.
        sync(self, barrier): Synchronizes the pipeline using a threading barrier.
        aasync(self): Resets synchronization, making the pipeline asynchronous.
        delay(self): Delays the pipeline and returns the time taken for processing.
        set_timeout(self, t): Sets the timeout value for queue operations.
        get_timeout(self): Gets the current timeout value.
        __delitem__(self, key): Deletes an item from the pipeline.
        __len__(self): Gets the current length of the pipeline.
        __getitem__(self, key): Gets an item or slice from the pipeline.

    Raises:
        ValueError: If an unsupported value is provided for 'on_busy'.
        BufferError: If the pipeline is empty during a delay operation.
        ConnectionError: If the pipeline is not started during a delay operation.
    """
    def __init__(self, max_size: int = 0,
                 io_buffer_size: int = 10,
                 on_busy: Union[PipelineOnBusyType, str] = PipelineOnBusyType.BLOCK,
                 list_dispatch: bool = False):
        """
        Initializes the Pipeline instance with the specified parameters.

        Args:
            max_size (int): Maximum size of the pipeline.
            io_buffer_size (int): Size of the I/O buffer.
            on_busy (Union[float, PipelineOnBusyType]): Action to take when the pipeline is busy. DROP or BLOCK, float for timeout
            list_dispatch (bool): Flag indicating whether to use list dispatch mode.
        """
        super(Pipeline, self).__init__(daemon=True)
        self._timeout = 100e-3
        self._pipeline = []
        self._sync = None
        self.io_buffer_size = io_buffer_size
        self.input_line = queue.Queue(maxsize=io_buffer_size)
        self._pipinput_queue = queue.Queue(maxsize=max_size)
        self._pipoutput_queue = queue.Queue(maxsize=max_size)
        self.output_line = queue.Queue(maxsize=io_buffer_size)
        self.max_size = max_size
        self.on_busy = on_busy
        self.refresh_ev = threading.Event()
        self.init_queue = queue.Queue()
        self._list_dispatch = list_dispatch

        if on_busy == PipelineOnBusyType.DROP:
            self._process_type = (False, None)
        elif on_busy == PipelineOnBusyType.BLOCK:
            self._process_type = (True, None)
        elif type(on_busy) == float:
            self._process_type = (True, on_busy)
        else:
            raise ValueError

        self.put = self.__call__
        self.get = self.output_line.get
        self.put_nowait = self.input_line.put_nowait
        self.get_nowait = self.output_line.get_nowait

    def __call__(self, data):
        """
        Callable method for putting data into the input queue.

        Args:
            data: Data to be put into the input queue.
        """
        self.input_line.put(data, block=self._process_type[0], timeout=self._process_type[1])

    def clear(self):
        """
        Clears the input and output queues of the pipeline.

        Args:
            None

        Returns:
            None
        """
        while not self.input_line.empty():
            self.input_line.get()
        while not self.output_line.empty():
            self.output_line.get()

    def run(self):
        """
        The main execution loop of the pipeline. Handles dispatching to either _run_dispatched or _run_norm based on
        the pipeline mode (list dispatch or normal).

        Args:
            None

        Returns:
            None
        """
        while 1:
            try:
                if self._list_dispatch:
                    while 1:
                        self.refresh()
                        self._run_dispatched()

                else:
                    # Normal mode
                    while 1:
                        # print('Run normal ', time.time())
                        self.refresh()
                        self._run_norm()

            except IndexError:
                while not len(self._pipeline):
                    try:
                        if self._sync:
                            self._sync.wait()
                        self.output_line.put(self.input_line.get(timeout=self._timeout), timeout=self._timeout)
                    except queue.Empty:
                        pass
                    self.refresh()
            except (queue.Full, queue.Empty):
                pass

    def _run_dispatched(self):
        """
        Executes the pipeline in list dispatch mode.

        Args:
            None

        Returns:
            None
        """
        try:
            data = self.input_line.get(timeout=self._timeout)
            ret_val = self._pipeline[0][0](data, *self._pipeline[0][1:])
            for func in self._pipeline[1:]:
                ret_val = func[0](ret_val, *func[1:])
            self._post_process(ret_val)
        except queue.Empty:
            time.sleep(0.001)
        except queue.Full:
            pass
        except Exception as e:
            error_msg = f"Error in pipeline processing: {type(e).__name__}: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())  # This will print the full stack trace

    def _run_norm(self):
        """
        Executes the pipeline in normal mode.

        Args:
            None

        Returns:
            None
        """
        try:
            data = self.input_line.get(timeout=self._timeout)
            ret_val = self._pipeline[0][0](data, *self._pipeline[0][1:])
            for func in self._pipeline[1:]:
                ret_val = func[0](ret_val, *func[1:])
            self._post_process(ret_val)
        except queue.Empty:
            time.sleep(0.001)
        except queue.Full:
            pass
        except Exception as e:
            error_msg = f"Error in pipeline processing: {type(e).__name__}: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())

    def _post_process(self, retval):
        """
        Handles the post-processing steps after executing the pipeline.

        Args:
            retval: The return value from the pipeline execution.

        Returns:
            None
        """
        if self._sync:
            self._sync.wait()
        self.output_line.put(retval,
                             block=self._process_type[0],
                             timeout=self._timeout)

    def refresh(self):
        """
        Refreshes the pipeline, handling initialization, appending, inserting, and other operations.

        Args:
            None

        Returns:
            None
        """
        if self.refresh_ev.is_set():
            while not self.init_queue.empty():
                # init
                self.init_queue.get()(self)

            while not self._pipinput_queue.empty():
                # init
                args = self._pipinput_queue.get()
                # append mode

                if args[0] is None:
                    # print('o1')
                    self._pipeline.append(args[1])

                # insert mode
                elif type(args[0]) == int:
                    self._pipeline.insert(*args)

                elif type(args[0]) == str:
                    if args[0] == 'len':
                        self._pipoutput_queue.put(('len', len(self._pipeline)))
                    elif args[0] == 'key':
                        self._pipoutput_queue.put(('key', self._pipeline[args[1]]))
                    elif args[0] == 'del':
                        self.input_line.empty()
                        del self._pipeline[args[1]]
                    elif args[0] == 'time':
                        self.input_line.empty()
                        if self._list_dispatch:
                            t0 = time.thread_thread_time_ns()
                            self._run_dispatched()
                            tdiff = time.thread_time_ns() - t0
                        else:
                            t0 = time.thread_time_ns()
                            self._run_norm()
                            tdiff = time.thread_time_ns() - t0
                        self._pipoutput_queue.put(('time', tdiff * 1e-3))
                    elif args[0] == 'index':
                        index = -1
                        for i, item in enumerate(self._pipeline):
                            if item[0] == args[1]:
                                index = i
                                break
                        self._pipoutput_queue.put(('index', index))
                    elif args[0] == 'update_args':
                        func_or_index, new_args = args[1], args[2]
                        if isinstance(func_or_index, int):
                            if 0 <= func_or_index < len(self._pipeline):
                                self._pipeline[func_or_index] = [self._pipeline[func_or_index][0], *new_args]
                                self._pipoutput_queue.put(('update_args', 'success'))
                            else:
                                self._pipoutput_queue.put(('update_args', 'error'))
                        else:
                            updated = False
                            for i, item in enumerate(self._pipeline):
                                if item[0] == func_or_index:
                                    self._pipeline[i] = [item[0], *new_args]
                                    updated = True
                                    break
                            if updated:
                                self._pipoutput_queue.put(('update_args', 'success'))
                            else:
                                self._pipoutput_queue.put(('update_args', 'error'))
            self.refresh_ev.clear()

    def insert(self,
               index: int,
               *func: callable,
               args: Union[list, tuple, object] = (),
               init: Union[list, tuple, callable] = ()):
        """
        Inserts functions into the pipeline at the specified index.

        Args:
            index (int): The index at which to insert the functions.
            *func (callable): The functions to insert.
            args (Union[list, tuple, object]): Arguments to be passed to the functions.
            init (Union[list, tuple, callable]): Initialization functions or arguments.

        Returns:
            Pipeline: The modified pipeline instance.
        """
        if self.max_size:
            assert self.__len__() + len(func) < self.max_size
        elif init:
            if type(init) is callable:
                self.init_queue.put(init)
            elif type(init) is tuple or type(init) is list:
                for i in init:
                    if i:
                        self.init_queue.put(i)

        for num, i in enumerate(func):
            assert callable(i), 'TypeError'
            if args:
                # self._pipeline.append([i, *args[num]])
                try:
                    if type(args[num]) is tuple or type(args[num]) is list:
                        self._pipinput_queue.put([index, [i, *args[num]]])
                except KeyError:

                    if type(args) is tuple or type(args) is list:
                        if not args[num] is None:
                            self._pipinput_queue.put([index, [i, args[num]]])
                    else:
                        self._pipinput_queue.put([index, [i, args]])

            else:
                # self._pipeline.append([i])
                self._pipinput_queue.put([index, [i]])

        self.refresh_ev.set()
        while self.refresh_ev.is_set() and self.is_alive():
            pass

        return self

    # def join(self, *pip, args=(), init=()):
    #
    #     if len(init):
    #         self.init_queue.put(init[0])
    #         self.refresh_ev.set()
    #         while self.refresh_ev.is_set(): pass
    #
    #     self._pipeline.insert(index, [func, *args])
    #     return self

    def append(self,
               *func: callable,
               args: Union[list, tuple, object] = (),
               init: Union[list, tuple, callable] = ()):
        """
        Appends functions to the end of the pipeline.

        Args:
            *func (callable): The functions to append.
            args (Union[list, tuple, object]): Arguments to be passed to the functions.
            init (Union[list, tuple, callable]): Initialization functions or arguments.

        Returns:
            Pipeline: The modified pipeline instance.
        """
        return self.insert(self.__len__(),
                           *func,
                           args=args,
                           init=init)

    def sync(self, barrier: threading.Barrier):
        """
        Sets a synchronization barrier for the pipeline.

        Args:
            barrier (threading.Barrier): The synchronization barrier.

        Returns:
            None
        """
        self._sync = barrier

    def aasync(self):
        """
        Disables synchronization for the pipeline.

        Args:
            None

        Returns:
            None
        """
        self._sync = None

    def delay(self):
        """
        Delays the pipeline and returns the execution time.

        Args:
            None

        Returns:
            float: The execution time in seconds.
        """
        if not self.__len__():
            raise BufferError("Pipeline is empty ")

        elif self.is_alive():
            self._pipinput_queue.put(('time', 0))
            self.refresh_ev.set()
            while self.refresh_ev.is_set():
                pass

            while not self._pipoutput_queue.empty():
                data = self._pipoutput_queue.get()
                if data[0] == 'time':
                    return data[1]
                self._pipoutput_queue.put(data)
        else:
            raise ConnectionError("Pipeline is not started")

    def set_timeout(self, t: Union[float, int]):
        """
        Sets the timeout value for blocking operations.

        Args:
            t (Union[float, int]): The timeout value in seconds.

        Returns:
            None
        """
        self._timeout = t

    def get_timeout(self):
        """
        Retrieves the current timeout value.

        Args:
            None

        Returns:
            Union[float, int]: The current timeout value in seconds.
        """
        return self._timeout


    def __delitem__(self, key):
        """
        Deletes an item from the pipeline at the specified index.

        Args:
            key: The index of the item to delete.

        Returns:
            None
        """
        if self.is_alive():
            self._pipinput_queue.put(('del', key))
            self.refresh_ev.set()
            while self.refresh_ev.is_set(): pass
        else:
            self.input_line.empty()
            del self._pipeline[key]
        # self.input_line.empty()
        # del self._pipeline[key]

    def __len__(self):
        """
        Retrieves the current length of the pipeline.

        Args:
            None

        Returns:
            int: The current length of the pipeline.
        """
        if self.is_alive():
            self._pipinput_queue.put(('len', 0))
            self.refresh_ev.set()
            while self.refresh_ev.is_set(): pass

            while not self._pipoutput_queue.empty():
                data = self._pipoutput_queue.get()
                if data[0] == 'len':
                    return data[1]
                self._pipoutput_queue.put(data)
        else:
            return len(self._pipeline)
        # return len(self._pipeline)

    def __getitem__(self, key):
        """
        Retrieves an item from the pipeline at the specified index.

        Args:
            key: The index of the item to retrieve.

        Returns:
            Any: The item at the specified index.
        """
        if self.is_alive():
            self._pipinput_queue.put(('key', key))
            self.refresh_ev.set()
            while self.refresh_ev.is_set(): pass

            data = []
            while not self._pipoutput_queue.empty():
                data = self._pipoutput_queue.get()
                if data[0] == 'key':
                    data = data[1]
                else:
                    self._pipoutput_queue.put(data)
        else:
            data = self._pipeline[key]
        try:
            assert type(key) == slice
            # indices = key.indices(len(self._pipeline))
            return Pipeline(max_size=self.max_size, io_buffer_size=self.io_buffer_size,
                            on_busy=self.on_busy).append(*data)
        except AssertionError:
            return self._pipeline[key]
        

    def index(self, func):
        """
        Returns the index of the first occurrence of the specified function in the pipeline.

        Args:
            func (callable): The function to search for in the pipeline.

        Returns:
            int: The index of the function if found, -1 otherwise.

        Raises:
            ValueError: If the function is not found in the pipeline.
        """
        if self.is_alive():
            self._pipinput_queue.put(('index', func))
            self.refresh_ev.set()
            while self.refresh_ev.is_set():
                pass

            while not self._pipoutput_queue.empty():
                data = self._pipoutput_queue.get()
                if data[0] == 'index':
                    if data[1] == -1:
                        raise ValueError(f"{func} is not in the pipeline")
                    return data[1]
                self._pipoutput_queue.put(data)
        else:
            for i, item in enumerate(self._pipeline):
                if item[0] == func:
                    return i
            raise ValueError(f"{func} is not in the pipeline")
        
    
    def update_args(self, func_or_index, *new_args):
        """
        Updates the arguments of a function in the pipeline.

        Args:
            func_or_index (Union[callable, int]): The function or its index in the pipeline.
            *new_args: The new arguments to be used for the function.

        Returns:
            None

        Raises:
            ValueError: If the function is not found in the pipeline.
            IndexError: If the provided index is out of range.
        """
        if self.is_alive():
            self._pipinput_queue.put(('update_args', func_or_index, new_args))
            self.refresh_ev.set()
            while self.refresh_ev.is_set():
                pass

            while not self._pipoutput_queue.empty():
                data = self._pipoutput_queue.get()
                if data[0] == 'update_args':
                    if data[1] == 'error':
                        raise ValueError(f"Function or index {func_or_index} not found in the pipeline")
                    break
                self._pipoutput_queue.put(data)
        else:
            if isinstance(func_or_index, int):
                if 0 <= func_or_index < len(self._pipeline):
                    self._pipeline[func_or_index] = [self._pipeline[func_or_index][0], *new_args]
                else:
                    raise IndexError("Index out of range")
            else:
                for i, item in enumerate(self._pipeline):
                    if item[0] == func_or_index:
                        self._pipeline[i] = [item[0], *new_args]
                        return
                raise ValueError(f"Function {func_or_index} not found in the pipeline")