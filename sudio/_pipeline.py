# the name of allah
# import multiprocessing
import threading
import queue
import time


class Pipeline(threading.Thread):
    # type LiveProcessing, DeadProcessing or timeout value
    def __init__(self, max_size=0, io_buffer_size=10, pipe_type='LiveProcessing'):
        super(Pipeline, self).__init__(daemon=True)
        # self.manager = threading.Manager()
        # self.namespace = manager.Namespace()
        self._timeout = 30
        self._pipeline = []
        self.io_buffer_size = io_buffer_size
        self.input_line = queue.Queue(maxsize=io_buffer_size)
        self.pipinput_queue = queue.Queue(maxsize=max_size)
        self.pipoutput_queue = queue.Queue(maxsize=max_size)
        self.output_line = queue.Queue(maxsize=io_buffer_size)
        self.max_size = max_size
        self.pipe_type = pipe_type
        self.refresh_ev = threading.Event()
        self.init_queue = queue.Queue()
        # block value, timeout
        if pipe_type == 'LiveProcessing':
            self._process_type = (False, None)
        elif pipe_type == 'DeadProcessing':
            self._process_type = (True, None)
        elif type(pipe_type) == float:
            self._process_type = (True, pipe_type)
        else:
            raise ValueError

        self.put = self.__call__
        self.get = self.output_line.get
        self.put_nowait = self.input_line.put_nowait
        self.get_nowait = self.output_line.get_nowait

    def __call__(self, data):
        self.input_line.put(data, block=self._process_type[0], timeout=self._process_type[1])

    def clear(self):
        while not self.input_line.empty():
            self.input_line.get()
        while not self.output_line.empty():
            self.output_line.get()

    def run(self):
        while 1:
            try:
                while 1:
                    self.refresh()
                    # call from pipeline
                    ret_val = self._pipeline[0][0](*self._pipeline[0][1:], self.input_line.get(timeout=self._timeout))
                    for i in self._pipeline[1:]:
                        ret_val = i[0](*i[1:], ret_val)
                    self.output_line.put(ret_val, block=self._process_type[0], timeout=self._timeout)


            except IndexError:
                while not len(self._pipeline):
                    try:
                        self.output_line.put(self.input_line.get_nowait(), timeout=self._timeout)
                    except queue.Empty:
                        pass
                    self.refresh()

            except (queue.Full, queue.Empty):
                pass

    def refresh(self):
        if self.refresh_ev.is_set():
            while not self.init_queue.empty():
                # init
                self.init_queue.get()(self)

            while not self.pipinput_queue.empty():
                # init
                args = self.pipinput_queue.get()
                # append mode

                if args[0] is None:
                    # print('o1')
                    self._pipeline.append(args[1])

                # insert mode
                elif type(args[0]) == int:
                    self._pipeline.insert(*args)

                elif type(args[0]) == str:
                    if args[0] == 'len':
                        self.pipoutput_queue.put(('len', len(self._pipeline)))
                    elif args[0] == 'key':
                        self.pipoutput_queue.put(('key', self._pipeline[args[1]]))
                    elif args[0] == 'del':
                        self.input_line.empty()
                        del self._pipeline[args[1]]

            self.refresh_ev.clear()

    def insert(self, index:int, *func, args=(), init=()):

        if self.max_size:
            assert self.__len__() + len(func) < self.max_size

        for i in init:
            if i:
                self.init_queue.put(i)

        for num, i in enumerate(func):
            assert type(i) == type(self.insert), 'TypeError'
            if args:
                # self._pipeline.append([i, *args[num]])
                self.pipinput_queue.put([index, [i, *args[num]]])
            else:
                # self._pipeline.append([i])
                self.pipinput_queue.put([index, [i]])
        self.refresh_ev.set()
        while self.refresh_ev.is_set() and self.is_alive(): pass
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

    def append(self, *func, args=(), init=()):
        """append new functions to pipeline

        Parameters
        ----------
        func : list or object
            Functions to be added to the pipe
        args: tuple
            Input Arguments of Functions
        init: tuple
            Initial functions(access to pipeline shared memory)

        Returns
        -------
        y : current object


        Examples
        --------
        Simple operations:

        >>> from scipy.signal import upfirdn
        >>> h = firwin(30, 8000, fs=72000)
        >>>
        >>> def simple_sum(x, y):
        >>>     return x + y
        >>> pip = Pipeline().append(upfirdn, simple_sum, args=((h), (2)))
        """
        if self.max_size:
            assert self.__len__() + len(func) < self.max_size

        for i in init:
            if i:
                self.init_queue.put(i)

        for num, i in enumerate(func):
            assert type(i) == type(self.insert), 'TypeError'
            if args:
                # self._pipeline.append([i, *args[num]])
                self.pipinput_queue.put([None, [i, *args[num]]])
            else:
                # self._pipeline.append([i])
                self.pipinput_queue.put([None, [i]])
        self.refresh_ev.set()
        # print('o1')
        while self.refresh_ev.is_set() and self.is_alive(): pass

        return self

    def __delitem__(self, key):
        if self.is_alive():
            self.pipinput_queue.put(('del', key))
            self.refresh_ev.set()
            while self.refresh_ev.is_set(): pass
        else:
            self.input_line.empty()
            del self._pipeline[key]
        # self.input_line.empty()
        # del self._pipeline[key]

    def __len__(self):
        if self.is_alive():
            self.pipinput_queue.put(('len', 0))
            self.refresh_ev.set()
            while self.refresh_ev.is_set(): pass

            while not self.pipoutput_queue.empty():
                data = self.pipoutput_queue.get()
                if data[0] == 'len':
                    return data[1]
                self.pipoutput_queue.put(data)
        else:
            return len(self._pipeline)
        # return len(self._pipeline)

    def __getitem__(self, key):

        if self.is_alive():
            self.pipinput_queue.put(('key', key))
            self.refresh_ev.set()
            while self.refresh_ev.is_set(): pass

            data = []
            while not self.pipoutput_queue.empty():
                data = self.pipoutput_queue.get()
                if data[0] == 'key':
                    data = data[1]
                else:
                    self.pipoutput_queue.put(data)
        else:
            data = self._pipeline[key]
        try:
            assert type(key) == slice
            # indices = key.indices(len(self._pipeline))
            return Pipeline(max_size=self.max_size, io_buffer_size=self.io_buffer_size,
                            pipe_type=self.pipe_type).append(*data)
        except AssertionError:
            return self._pipeline[key]
