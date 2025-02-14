from .enum import StreamMode
from .error import (
    RefreshError,
    DecodeError,
    StreamError,
)
from .pipelineprocesstype import PipelineProcessType
from .pipelineonbusytype import PipelineOnBusyType
from sudio.io import FileFormat, SampleFormat, DitherMode
from .name import Name
import numpy as _np
from typing import Union


EnvelopeType = Union[float, list, tuple, _np.ndarray, int]


__all__ = [
    'FileFormat',
    'SampleFormat',
    'DitherMode',
    'StreamMode',
    'RefreshError',
    'DecodeError',
    'StreamError',
    'PipelineProcessType',
    'PipelineOnBusyType',
    'Name',
    'EnvelopeType',

]
