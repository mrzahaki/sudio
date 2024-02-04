"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""
import threading
import queue
import time
from typing import Union


class Pipeline(threading.Thread):
    # type drop, block or {timeout value}
    def __init__(self, max_size: int = 0,
                 io_buffer_size: int = 10,
                 on_busy: Union[float, str] = 'drop',
                 list_dispatch: bool = False):

        super(Pipeline, self).__init__(daemon=True)
        # self.manager = threading.Manager()
        # self.namespace = manager.Namespace()
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

        # block value, timeout
        if on_busy == 'drop':
            self._process_type = (False, None)
        elif on_busy == 'block':
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
        self.input_line.put(data, block=self._process_type[0], timeout=self._process_type[1])

    def clear(self):
        while not self.input_line.empty():
            self.input_line.get()
        while not self.output_line.empty():
            self.output_line.get()

    def run(self):
        data = None
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
                # print('IndexError ', time.time())
                if data:
                    if self._sync:
                        self._sync.wait()
                    self.output_line.put(data, timeout=self._timeout)
                while not len(self._pipeline):
                    try:
                        if self._sync:
                            self._sync.wait()
                        self.output_line.put(self.input_line.get(timeout=self._timeout), timeout=self._timeout)
                    except queue.Empty:
                        pass
                    self.refresh()
                # raise
            except (queue.Full, queue.Empty):
                pass

    def _run_dispatched(self):
        # print('data')
        # call from pipeline
        # print(data)
        ret_val = self._pipeline[0][0](*self._pipeline[0][1:],
                                       *self.input_line.get(timeout=self._timeout))
        for i in self._pipeline[1:]:
            ret_val = i[0](*i[1:], *ret_val)

        self._post_process(ret_val)

    def _run_norm(self):
        # call from pipeline
        ret_val = self._pipeline[0][0](*self._pipeline[0][1:],
                                       self.input_line.get(timeout=self._timeout))
        for i in self._pipeline[1:]:
            ret_val = i[0](*i[1:], ret_val)

        self._post_process(ret_val)

    def _post_process(self, retval):

        if self._sync:
            self._sync.wait()
        self.output_line.put(retval,
                             block=self._process_type[0],
                             timeout=self._timeout)

    def refresh(self):
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

            self.refresh_ev.clear()

    def insert(self,
               index: int,
               *func: callable,
               args: Union[list, tuple, object] = (),
               init: Union[list, tuple, callable] = ()):

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
                if type(args[num]) is tuple or type(args[num]) is list:
                    self._pipinput_queue.put([index, [i, *args[num]]])
                elif type(args) is tuple or type(args) is list:
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

        return self.insert(self.__len__(),
                           *func,
                           args=args,
                           init=init)

    def sync(self, barrier: threading.Barrier):
        self._sync = barrier

    def aasync(self):
        self._sync = None

    def delay(self):

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
        self._timeout = t

    def get_timeout(self):
        return self._timeout

        a = []
        a.__delitem__()

    def __delitem__(self, key):
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