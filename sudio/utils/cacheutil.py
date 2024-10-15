
#  W.T.A
#  SUDIO (https://github.com/MrZahaki/sudio)
#  The Audio Processing Platform
#  Mail: mrzahaki@gmail.com
#  Software license: "Apache License 2.0". See https://choosealicense.com/licenses/apache-2.0/


import io
import os
import time
from builtins import ValueError
import numpy as np
from typing import Union,Tuple
import platform

from sudio.audiosys.sync import synchronize_audio
from sudio.utils.timed_indexed_string import TimedIndexedString
from sudio.types import DecodeError
from sudio.metadata import AudioMetadata
from sudio.io import SampleFormat, get_sample_size

def is_file_locked(file_path):
    # Determine the operating system
    is_windows = platform.system() == "Windows"

    if not is_windows:
        try:
            import fcntl
        except ImportError:
            fcntl = None

    if is_windows:
        try:
            # On Windows, try to open the file in read-write mode
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
            # Fallback method if fcntl is not available
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
    Writes data to a cached file with optional pre-processing and post-processing steps.

    Parameters:
    - *head: Variable number of arguments to be written at the beginning of the file.
    - head_dtype: Data type for the head arguments (default: 'u8').
    - data: Bytes or BufferedRandom object to be written to the file.
    - file_name: Name of the file. If not provided, a BufferedRandom object must be provided.
    - file_mode: File mode for opening the file (default: 'wb+').
    - buffered_random: BufferedRandom object for writing data.
    - pre_truncate: Truncate the file before writing (default: False).
    - pre_seek: Tuple (offset, whence) to seek to before writing.
    - after_seek: Tuple (offset, whence) to seek to after writing.
    - pre_flush: Flush the file before writing (default: False).
    - after_flush: Flush the file after writing (default: False).
    - size_on_output: If True, returns a tuple of the file object and the total size written (default: False).
    - data_chunk: Size of the data chunks to read when writing BufferedRandom data (default: 1e6).

    Returns:
    - io.BufferedRandom or Tuple of io.BufferedRandom and int (if size_on_output is True).
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
            buffer = data.read(data_chunk)
            while buffer:
                size += file.write(buffer)
                buffer = data.read(data_chunk)
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
                         master_obj,
                         decoder: callable = None,
                         sync_sample_format_id: int = None,
                         sync_nchannels: int = None,
                         sync_sample_rate: int = None,
                         safe_load: bool = True,
                         max_attempts=5) -> AudioMetadata:
    """
    Handles caching of audio records, ensuring synchronization and safe loading.

    Parameters:
    - record: Audio record as AudioMetadata or dictionary.
    - path_server: TimedIndexedString object for generating unique paths.
    - master_obj: Object managing the audio cache.
    - decoder: Callable function for decoding audio if needed.
    - sync_sample_format_id: Synchronized sample format ID.
    - sync_nchannels: Synchronized number of channels.
    - sync_sample_rate: Synchronized sample rate.
    - safe_load: If True, ensures safe loading in case of errors.

    Returns:
    - Modified audio record as AudioMetadata.
    """
    sync_sample_format_id = master_obj._sample_format if sync_sample_format_id is None else sync_sample_format_id
    sync_nchannels = master_obj._nchannels if sync_nchannels is None else sync_nchannels
    sync_sample_rate = master_obj._sample_rate if sync_sample_rate is None else sync_sample_rate

    path: str = path_server()
    cache = master_obj._cache()
    try:
        if path in cache:
            attempt = 0
            while attempt < max_attempts:
                try:
                    if not is_file_locked(path):
                        f = record['o'] = open(path, 'rb+')
                        break
                    else:
                        new_path = path_server()
                        if not os.path.exists(new_path) or not is_file_locked(new_path):
                            # Write a new file based on the new path
                            data_chunk = int(1e7)
                            with open(path, 'rb') as pre_file:
                                f = record['o'] = open(new_path, 'wb+')
                                data = pre_file.read(data_chunk)
                                while data:
                                    f.write(data)
                                    data = pre_file.read(data_chunk)
                            break
                except (IOError, OSError) as e:
                    attempt += 1
                    time.sleep(0.1)  # Short delay before retrying

            if attempt >= max_attempts:
                raise IOError(f"Unable to access the file after {max_attempts} attempts")

            f.seek(0, 0)
            cache_info = f.read(master_obj.__class__.CACHE_INFO)
            try:
                csize, csample_rate, csample_format_id, cnchannels = np.frombuffer(cache_info, dtype='u8').tolist()
                csample_format = SampleFormat(csample_format_id) if csample_format_id else SampleFormat.UNKNOWN
                
            except ValueError:
                # Handle bad cache error
                f.close()
                os.remove(f.name)
                raise DecodeError

            csample_format = csample_format if csample_format else None
            record['size'] = csize

            record['frameRate'] = csample_rate
            record['nchannels'] = cnchannels
            record['sampleFormat'] = csample_format

            if (cnchannels == sync_nchannels and
                    csample_format == sync_sample_format_id and
                    csample_rate == sync_sample_rate):
                pass

            elif safe_load:
                # Load data safely and synchronize audio
                record['o'] = f.read()
                record = synchronize_audio(record,
                                            sync_nchannels,
                                            sync_sample_rate,
                                            sync_sample_format_id)

                f = record['o'] = write_to_cached_file(record['size'],
                                                       record['frameRate'],
                                                       record['sampleFormat'] if record['sampleFormat'] else 0,
                                                       record['nchannels'],
                                                       buffered_random=f,
                                                       data=record['o'],
                                                       pre_seek=(0, 0),
                                                       pre_truncate=True,
                                                       pre_flush=True,
                                                       after_seek=(master_obj.__class__.CACHE_INFO, 0),
                                                       after_flush=True)

        else:
            raise DecodeError

    except DecodeError:
        # Handle decoding error
        if decoder is not None:
            record['o'] = decoder()
        if isinstance(record['o'], io.BufferedRandom):
            record['size'] = os.path.getsize(record['o'].name)
        else:
            record['size'] = len(record['o']) + master_obj.CACHE_INFO

        f = record['o'] = write_to_cached_file(record['size'],
                                               record['frameRate'],
                                               record['sampleFormat'] if record['sampleFormat'] else 0,
                                               record['nchannels'],
                                               file_name=path,
                                               data=record['o'],
                                               pre_seek=(0, 0),
                                               pre_truncate=True,
                                               pre_flush=True,
                                               after_seek=(master_obj.__class__.CACHE_INFO, 0),
                                               after_flush=True)

    record['duration'] = record['size'] / (record['frameRate'] *
                                           record['nchannels'] *
                                           get_sample_size(record['sampleFormat']))
    if isinstance(record, AudioMetadata):
        # Extract name information from file path for Series
        post = record['o'].name.index(master_obj.__class__.BUFFER_TYPE)
        pre = max(record['o'].name.rfind('\\'), record['o'].name.rfind('/'))
        pre = (pre + 1) if pre > 0 else 0
        record.name = record['o'].name[pre: post]

    return record

