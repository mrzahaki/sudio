"""
 W.T.A
 SUDIO (https://github.com/MrZahaki/sudio)
 The Audio Processing Platform
 Mail: mrzahaki@gmail.com
 Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/
"""

from enum import Enum

class StreamMode(Enum):
    normal = 0
    optimized = 1

class RefreshError(Exception):
    pass

