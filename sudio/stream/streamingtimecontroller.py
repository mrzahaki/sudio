#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/

class StreamingTimeController:
    def __get__(self, instance, owner):
        assert instance.isready(), PermissionError('current object is not streaming')
        return instance._itime_calculator(instance._stream_file.tell())

    def __set__(self, instance, tim):
        assert abs(tim) < instance.duration, OverflowError('input time must be less than the record duration')
        assert instance.isready(), PermissionError('current object is not streaming')
        seek = instance._time_calculator(abs(tim))
        if tim < 0:
            seek = instance._stream_file_size - seek
        instance._stream_file.seek(seek, 0)
