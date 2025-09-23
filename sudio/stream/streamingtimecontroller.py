
# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


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
