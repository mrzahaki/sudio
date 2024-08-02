from enum import Enum

class PipelineOnBusyType(Enum):
    """Enumeration representing different Pipeline Process Types"""
    DROP = 0
    BLOCK = 1
