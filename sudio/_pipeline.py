# the name of allah
import multiprocessing
import queue
import time


class Pipeline(multiprocessing.Process):
    # type LiveProcessing, DeadProcessing or timeout value
    def __init__(self, max_size=None, io_buffer_size=10, pipe_type='LiveProcessing'):
        super(Pipeline, self).__init__(daemon=True)
        manager = multiprocessing.Manager()
        # self.namespace = manager.Namespace()
        self.pipeline = manager.list([])
        self.io_buffer_size = io_buffer_size
        self.input_line = multiprocessing.Queue(maxsize=io_buffer_size)
        self.output_line = multiprocessing.Queue(maxsize=io_buffer_size)
        self.max_size = max_size
        self.pipe_type = pipe_type

        self.put = self.__call__
        self.get = self.output_line.get

        # block value, timeout
        if pipe_type == 'LiveProcessing':
            self.process_type = (False, None)
        elif pipe_type == 'DeadProcessing':
            self.process_type = (True, None)
        elif type(pipe_type) == float:
            self.process_type = (True, pipe_type)
        else:
            raise ValueError

    def __call__(self, *args):
        try:
            args = list(args)
            while args:
                self.input_line.put(args.pop(), block=self.process_type[0], timeout=self.process_type[1])
        except (queue.Empty, queue.Full):
            raise

    def run(self):
        while 1:
            try:
                while 1:
                    ret_val = self.pipeline[0](self.input_line.get_nowait())
                    for i in self.pipeline[1:]:
                        ret_val = i(ret_val)
                    self.output_line.put(ret_val, block=self.process_type[0], timeout=self.process_type[1])

            except IndexError:
                while not len(self.pipeline):
                    self.output_line.put(self.input_line.get(), block=self.process_type[0], timeout=self.process_type[1])
            except (queue.Full, queue.Empty):
                pass

    def append(self, *args):
        if self.max_size:
            assert len(self.pipeline) + len(args) < self.max_size

        for i in args:
            self.pipeline.append(i)

        return self

    def __delitem__(self, key):
        self.input_line.empty()
        del self.pipeline[key]

    def __len__(self):
        return len(self.pipeline)

    def __getitem__(self, key):
        try:
            assert type(key) == slice
            # indices = key.indices(len(self.pipeline))
            return Pipeline(max_size=self.max_size, io_buffer_size=self.io_buffer_size, pipe_type=self.pipe_type).append(*self.pipeline[key])
        except AssertionError:
            return  self.pipeline[key]


