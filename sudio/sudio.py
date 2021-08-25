# the name of allah
from . import _process
from ._register import Members as Mem
import pyaudio

@Mem.sudio.parent
class Sudio:
    formatFloat32 = pyaudio.paFloat32
    formatInt32 = pyaudio.paInt32
    formatInt24 = pyaudio.paInt24
    formatInt16 = pyaudio.paInt16
    formatInt8 = pyaudio.paInt8
    formatUInt8 = pyaudio.paInt8

