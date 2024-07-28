from enum import Enum

class PipelineProcessType(Enum):
    """Enumeration representing different Pipeline Process Types"""
    MAIN = 0
    MULTI_STREAM = 1
    BRANCH = 2
    QUEUE = 3
