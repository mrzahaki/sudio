
# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


import io
import os
import time
from builtins import ValueError
import numpy as np
from typing import Union,Tuple
import platform

from .sync import synchronize_audio
from sudio.utils.timed_indexed_string import TimedIndexedString
from sudio.types import DecodeError
from sudio.metadata import AudioMetadata
from sudio.io import SampleFormat, get_sample_size

def is_file_locked(file_path):
    """
    Check if a file is currently locked or in use by another process.

    Supports both Windows and Unix-like systems. On Windows, attempts to open 
    the file exclusively. On Unix systems, uses file locking mechanisms.

    Args:
    -----

        - file_path (str): Path to the file to check for lock status.

    Returns:
    --------
    
        - bool: True if the file is locked, False otherwise.
    """
    is_windows = platform.system() == "Windows"

    if not is_windows:
        try:
            import fcntl
        except ImportError:
            fcntl = None

    if is_windows:
        try:
            with open(file_path, 'r+') as f:
                return False
        except IOError:
            return True
    else:
        if fcntl:
            try:
                with open(file_path, 'rb') as file:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                return False
            except IOError:
                return True
        else:
            try:
                with open(file_path, 'r+') as f:
                    return False
            except IOError:
                return True


def write_to_cached_file(*head,
                         head_dtype: str = 'u8',
                         data: Union[bytes, io.BufferedRandom] = None,
                         file_name: str = None,
                         file_mode: str = 'wb+',
                         buffered_random: io.BufferedRandom = None,
                         pre_truncate: bool = False,
                         pre_seek: Tuple[int, int] = None,
                         after_seek: Tuple[int, int] = None,
                         pre_flush: bool = False,
                         after_flush: bool = False,
                         size_on_output: bool = False,
                         data_chunk: int = int(1e6)) -> Union[io.BufferedRandom, Tuple[io.BufferedRandom, int]]:
    """
    Flexibly write data to a file with granular control over the writing process.

    Provides advanced file writing capabilities, allowing precise manipulation 
    of file operations before, during, and after writing data.

    Args:
    -----

        - *head: Variable header arguments to write at the beginning of the file.
        - head_dtype (str, optional): Numpy data type for header arguments. Defaults to 'u8' (64-bit unsigned integer).
        - data (bytes or BufferedRandom, optional): Data to write to the file. Can be bytes or a file-like object.
        - file_name (str, optional): Name of the file to write. Required if no buffered_random is provided.
        - file_mode (str, optional): Mode to open the file. Defaults to 'wb+' (write binary, read/write).
        - buffered_random (BufferedRandom, optional): Existing file object to write to. Overrides file_name if provided.
        - pre_truncate (bool, optional): Truncate the file before writing. Defaults to False.
        - pre_seek (tuple, optional): Seek to this position before writing. Tuple of (offset, whence).
        - after_seek (tuple, optional): Seek to this position after writing. Tuple of (offset, whence).
        - pre_flush (bool, optional): Flush the file before writing. Defaults to False.
        - after_flush (bool, optional): Flush the file after writing. Defaults to False.
        - size_on_output (bool, optional): Return total bytes written along with file object. Defaults to False.
        - data_chunk (int, optional): Size of chunks when reading BufferedRandom data. Defaults to 1,000,000 bytes.

    Returns:
    --------

        io.BufferedRandom or tuple: 
        - File object if size_on_output is False
        - Tuple of (file object, total bytes written) if size_on_output is True
    
    Raises:
    -------

        ValueError: If neither buffered_random nor file_name is provided.
    """
    if buffered_random:
        file: io.BufferedRandom = buffered_random
    elif file_name:
        file = open(file_name, file_mode)
    else:
        raise ValueError("Either buffered_random or file_name must be provided.")
    size = 0

    if pre_seek:
        file.seek(*pre_seek)

    if pre_truncate:
        file.truncate()

    if pre_flush:
        file.flush()

    if head:
        size = file.write(np.asarray(head, dtype=head_dtype))

    if data:
        if isinstance(data, io.BufferedRandom):
            data_position = data.tell()
            buffer = data.read(data_chunk)
            while buffer:
                size += file.write(buffer)
                buffer = data.read(data_chunk)
            data.seek(data_position, 0)
        else:
            size += file.write(data)

    if after_seek:
        file.seek(*after_seek)

    if after_flush:
        file.flush()

    if size_on_output:
        return file, size
    return file


def handle_cached_record(record: Union[AudioMetadata, dict],
                         path_server: TimedIndexedString,
                         orphaned_cache:list,
                         cache_info_size:str,
                         buffer_type:str=None,
                         decoder: callable = None,
                         sync_sample_format_id: int = 1,
                         sync_nchannels: int = 2,
                         sync_sample_rate: int = 44100,
                         safe_load: bool = True,
                         max_attempts=5) -> AudioMetadata:
    """
   Manage audio record caching with advanced synchronization and error recovery.

   Handles complex file caching scenarios for audio records, including 
   format conversion, file locking management, and safe loading strategies.

   Args:
   -----

       - record (AudioMetadata or dict): Audio record to be processed and cached.
       - path_server (TimedIndexedString): Service for generating unique file paths.
       - orphaned_cache (list): List tracking obsolete cache entries.
       - cache_info_size (str): Size of metadata in the cache.
       - buffer_type (str, optional): Type of buffer for filename parsing.
       - decoder (callable, optional): Custom function to decode audio if needed.
       - sync_sample_format_id (int, optional): Target audio sample format. Defaults to 1.
       - sync_nchannels (int, optional): Target number of audio channels. Defaults to 2.
       - sync_sample_rate (int, optional): Target audio sample rate. Defaults to 44100.
       - safe_load (bool, optional): Enable safe loading with error recovery. Defaults to True.
       - max_attempts (int, optional): Maximum retry attempts for file operations. Defaults to 5.

   Returns:
   --------

       - AudioMetadata: Processed audio record with updated metadata and file reference.

   Raises:
   -------

       IOError: If file cannot be accessed after max attempts.
       DecodeError: If audio decoding fails and no decoder is provided.
   """

    path: str = path_server()
    if path in orphaned_cache:
        attempt = 0
        while attempt < max_attempts:
            try:
                if not is_file_locked(path):
                    f = open(path, 'rb+')
                    record.o = f
                    break
                else:
                    new_path = path_server()
                    if not os.path.exists(new_path) or not is_file_locked(new_path):
                        # Write a new file based on the new path
                        data_chunk = int(1e7)
                        with open(path, 'rb') as pre_file:
                            pre_file.seek(0, 0)
                            f = open(new_path, 'wb+')
                            data = pre_file.read(data_chunk)
                            while data:
                                f.write(data)
                                data = pre_file.read(data_chunk)
                            record.o = f
                        break
            except (IOError, OSError) as e:
                attempt += 1
                time.sleep(0.1)  # Short delay before retrying

        if attempt >= max_attempts:
            raise IOError(f"Unable to access the file after {max_attempts} attempts")

        f.seek(0, 0)
        orphaned_cache = f.read(cache_info_size)
        try:
            csize, csample_rate, csample_format_id, cnchannels = np.frombuffer(orphaned_cache, dtype='u8').tolist()
            csample_format = SampleFormat(csample_format_id) if csample_format_id else SampleFormat.UNKNOWN
            
        except ValueError:
            # bad orphaned_cache error
            f.close()
            os.remove(f.name)
            raise DecodeError

        csample_format = csample_format if csample_format else None
        record.size = csize

        record.frameRate = csample_rate
        record.nchannels = cnchannels
        record.sampleFormat = csample_format

        if (cnchannels == sync_nchannels and
                csample_format == sync_sample_format_id and
                csample_rate == sync_sample_rate):
            pass

        elif safe_load:
            record.o = f.read()
            record = synchronize_audio(record,
                                        sync_nchannels,
                                        sync_sample_rate,
                                        sync_sample_format_id)

            record.o = write_to_cached_file(record.size,
                    record.frameRate,
                    record.sampleFormat if record.sampleFormat else 0,
                    record.nchannels,
                    buffered_random=f,
                    data=record.o,
                    pre_seek=(0, 0),
                    pre_truncate=True,
                    pre_flush=True,
                    after_seek=(cache_info_size, 0),
                    after_flush=True
                    )


    else:
         # Handle decoding error
        if decoder is not None:
            record.o = decoder()
        if isinstance(record.o, io.BufferedRandom):
            record.size = os.path.getsize(record.o.name)
        else:
            record.size = len(record.o) + cache_info_size

        record.o = write_to_cached_file(
            record.size,
            record.frameRate,
            record.sampleFormat if record.sampleFormat else 0,
            record.nchannels,
            file_name=path,
            data=record.o,
            pre_seek=(0, 0),
            pre_truncate=True,
            pre_flush=True,
            after_seek=(cache_info_size, 0),
            after_flush=True
            )

    record.duration = record.size / (record.frameRate *
                                           record.nchannels *
                                           get_sample_size(record.sampleFormat))

    if buffer_type is not None:
        post = record.o.name.index(buffer_type)
        pre = max(record.o.name.rfind('\\'), record.o.name.rfind('/'))
        pre = (pre + 1) if pre > 0 else 0
        record.name = record.o.name[pre: post]

    return record

