I/O module
==========


.. raw:: html

   <script src='https://storage.ko-fi.com/cdn/scripts/overlay-widget.js'></script>
   <script>
     kofiWidgetOverlay.draw('mrzahaki', {
       'type': 'floating-chat',
       'floating-chat.donateButton.text': 'Support me',
       'floating-chat.donateButton.background-color': '#2980b9',
       'floating-chat.donateButton.text-color': '#fff'
     });
   </script>



The **sudio.io** module provides bindings for working with audio streams, codecs, and device management in Python. It includes functionality for reading, writing, and processing audio data, as well as managing audio devices. This module is part of the **SUDIO** platform.

Module Contents
---------------

Enums
-----

sudio.io.FileFormat
^^^^^^^^^^^^^^^^^^^

An enum representing different audio file formats.

- **UNKNOWN**: Unknown format.
- **WAV**: WAV audio file format.
- **FLAC**: FLAC audio file format.
- **VORBIS**: Vorbis audio file format.
- **MP3**: MP3 audio file format.

sudio.io.SampleFormat
^^^^^^^^^^^^^^^^^^^^^

An enum representing different sample formats.

- **UNKNOWN**: Unknown sample format.
- **UNSIGNED8**: Unsigned 8-bit samples.
- **SIGNED16**: Signed 16-bit samples.
- **SIGNED24**: Signed 24-bit samples.
- **SIGNED32**: Signed 32-bit samples.
- **FLOAT32**: 32-bit floating-point samples.

sudio.io.DitherMode
^^^^^^^^^^^^^^^^^^^

An enum representing the dither modes available during audio processing.

- **NONE**: No dither.
- **RECTANGLE**: Rectangular dither.
- **TRIANGLE**: Triangular dither.

Classes
-------

sudio.io.AudioFileInfo
^^^^^^^^^^^^^^^^^^^^^^

Represents information about an audio file.

Attributes:

- **name** (str): Name of the audio file.
- **file_format** (FileFormat): Format of the audio file.
- **nchannels** (int): Number of audio channels.
- **sample_rate** (int): Sample rate in Hz.
- **sample_format** (SampleFormat): Format of audio samples.
- **num_frames** (int): Total number of frames in the file.
- **duration** (float): Duration of the audio file in seconds.

sudio.io.codec.AudioFileStream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Represents a stream for reading audio data from a file.

Methods:

- **read_frames(frames_to_read=0)**: Reads a specified number of frames from the stream. If **frames_to_read** is 0, it uses the default value set during initialization.

sudio.io.codec.PyAudioStreamIterator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An iterator to read frames from an **AudioFileStream**.

Methods:
- **__next__()**: Returns the next set of audio frames as bytes.

sudio.io.AudioDeviceInfo
^^^^^^^^^^^^^^^^^^^^^^^^

Represents information about an audio device.

Attributes:

- **index** (int): Index of the audio device.
- **name** (str): Name of the audio device.
- **max_input_channels** (int): Maximum number of input channels.
- **max_output_channels** (int): Maximum number of output channels.
- **default_sample_rate** (int): Default sample rate of the device.
- **is_default_input** (bool): Whether this device is the default input.
- **is_default_output** (bool): Whether this device is the default output.

sudio.io.AudioStream
^^^^^^^^^^^^^^^^^^^^

Represents an audio stream for reading and writing audio data.

Methods:

- open(input_dev_index: int=None, output_dev_index:int=None, sample_rate:float=0.0, format:SampleFormat=ma_format_s16, input_channels: int=0, output_channels:int=0, frames_per_buffer:int=None, enable_input:bool=True, enable_output:bool=True, stream_flags=None, input_callback=None, output_callback=None): 

  Opens an audio stream with the given parameters.

  Parameters:
    - **input_dev_index** (int, optional): Index of the input device. If None and input is enabled, it uses the default standard streaming source.
    - **output_dev_index** (int, optional): Index of the output device. If None and output is enabled, it uses the default standard streaming destination.
    - **sample_rate** (float, optional): Sample rate in Hz. If 0.0, it uses the default sample rate of the selected device.
    - **format** (SampleFormat, optional): Sample format. Defaults to SIGNED16.
    - **input_channels** (int, optional): Number of input channels. If 0, it uses the maximum number of input channels for the selected device.
    - **output_channels** (int, optional): Number of output channels. If 0, it uses the maximum number of output channels for the selected device.
    - **frames_per_buffer** (int, optional): Number of frames per buffer. If None, it uses a default value appropriate for the host system.
    - **enable_input** (bool, optional): Whether to enable input. Defaults to True.
    - **enable_output** (bool, optional): Whether to enable output. Defaults to True.
    - **stream_flags** (int, optional): Stream flags. Reserved for future use.
    - **input_callback** (callable, optional): Callback function for input processing. If not provided and input is enabled, the stream operates in blocking mode.
    - **output_callback** (callable, optional): Callback function for output processing. If not provided and output is enabled, the stream operates in blocking mode.

  Note:
    If input_callback and output_callback are not provided, and enable_input/enable_output is True, the data is read/written in blocking mode using the read_stream/write_stream methods.

- **start()**: Starts the audio stream.

- **stop()**: Stops the audio stream.

- **close()**: Closes the audio stream.

- **read_stream(frames)**: Reads a number of frames from the stream. Returns a tuple **(data: bytes, frames_read: int)**.

- **write_stream(data)**: Writes audio data to the stream. Returns the number of frames written.

- **get_stream_read_available()**: Returns the number of frames that can be read without waiting.

- **get_stream_write_available()**: Returns the number of frames that can be written without waiting.

Static Methods:

- **get_input_devices()**: Returns a list of available input devices as **AudioDeviceInfo** objects.
- **get_output_devices()**: Returns a list of available output devices as **AudioDeviceInfo** objects.
- **get_default_input_device()**: Returns the default input device as an **AudioDeviceInfo** object.
- **get_default_output_device()**: Returns the default output device as an **AudioDeviceInfo** object.
- **get_device_count()**: Returns the total number of audio devices.
- **get_device_info_by_index(index)**: Returns an **AudioDeviceInfo** object for the device at the specified index.

Functions
---------

sudio.io.codec.decode_audio_file(filename, output_format=SampleFormat.SIGNED16, nchannels=2, sample_rate=44100, dither=DitherMode.NONE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Decodes an audio file into raw audio data.

- **filename** (str): Path to the audio file.
- **output_format** (SampleFormat): Desired output sample format.
- **nchannels** (int): Number of output channels.
- **sample_rate** (int): Output sample rate in Hz.
- **dither** (DitherMode): Dither mode.

Returns: Raw audio data as bytes.


sudio.io.codec.encode_wav_file(filename, data, format, nchannels, sample_rate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Encodes raw audio data into a WAV file.

Parameters:
  - **filename** (str): Path to the output WAV file.
  - **data** (bytes): Raw audio data.
  - **format** (SampleFormat): Audio sample format.
  - **nchannels** (int): Number of audio channels.
  - **sample_rate** (int): Sample rate in Hz.

Returns:
  uint64_t: Number of frames written to the file.

sudio.io.codec.encode_mp3_file(filename, data, format, nchannels, sample_rate, bitrate=128, quality=2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Encodes raw audio data into an MP3 file.

Parameters:
  - **filename** (str): Path to the output MP3 file.
  - **data** (bytes): Raw audio data.
  - **format** (SampleFormat): Audio sample format.
  - **nchannels** (int): Number of audio channels.
  - **sample_rate** (int): Sample rate in Hz.
  - **bitrate** (int, optional): MP3 encoding bitrate in kbps. Defaults to 128.
  - **quality** (int, optional): MP3 encoding quality (0-9, where 0 is best and 9 is worst). Defaults to 2.

Returns:
  uint64_t: Number of frames written to the file.

sudio.io.codec.encode_flac_file(filename, data, format, nchannels, sample_rate, compression_level=5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Encodes raw audio data into a FLAC file.

Parameters:
  - **filename** (str): Path to the output FLAC file.
  - **data** (bytes): Raw audio data.
  - **format** (SampleFormat): Audio sample format.
  - **nchannels** (int): Number of audio channels.
  - **sample_rate** (int): Sample rate in Hz.
  - **compression_level** (int, optional): FLAC compression level (0-8, where 0 is least compressed and 8 is most compressed). Defaults to 5.

Returns:
  uint64_t: Number of frames written to the file.

sudio.io.codec.encode_vorbis_file(filename, data, format, nchannels, sample_rate, quality=0.4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Encodes raw audio data into an Ogg Vorbis file.

Parameters:
  - **filename** (str): Path to the output Ogg Vorbis file.
  - **data** (bytes): Raw audio data.
  - **format** (SampleFormat): Audio sample format.
  - **nchannels** (int): Number of audio channels.
  - **sample_rate** (int): Sample rate in Hz.
  - **quality** (float, optional): Vorbis encoding quality (-0.1 to 1.0, where -0.1 is lowest quality/smallest file and 1.0 is highest quality/largest file). Defaults to 0.4.

Returns:
  uint64_t: Number of frames written to the file.

Notes
-----

- All encoding functions accept raw audio data as bytes and return the number of frames written to the output file.
- The input `format` parameter should match the format of the raw audio data provided.
- For best results, ensure that the `nchannels` and `sample_rate` parameters match the characteristics of your input audio data.
- When encoding to lossy formats like MP3 and Vorbis, adjusting the `quality` or `bitrate` parameters allows you to balance between file size and audio quality.
- FLAC is a lossless format, so the `compression_level` parameter only affects file size and encoding speed, not audio quality.

Here are some examples of how to use these encoding functions:

```python
  import sudio.io.codec as codec
  from sudio.io import SampleFormat

  # Example: Encoding a WAV file
  wav_data = b'...'  # Your raw audio data here
  codec.encode_wav_file("output.wav", wav_data, SampleFormat.SIGNED16, 2, 44100)

  # Example: Encoding an MP3 file with custom bitrate
  mp3_data = b'...'  # Your raw audio data here
  codec.encode_mp3_file("output.mp3", mp3_data, SampleFormat.SIGNED16, 2, 44100, bitrate=192, quality=0)

  # Example: Encoding a FLAC file with high compression
  flac_data = b'...'  # Your raw audio data here
  codec.encode_flac_file("output.flac", flac_data, SampleFormat.SIGNED24, 2, 48000, compression_level=8)

  # Example: Encoding an Ogg Vorbis file with high quality
  vorbis_data = b'...'  # Your raw audio data here
  codec.encode_vorbis_file("output.ogg", vorbis_data, SampleFormat.FLOAT32, 2, 44100, quality=0.8)
```

These examples demonstrate basic usage of each encoding function. Remember to replace the placeholder audio data with your actual raw audio bytes.



sudio.io.codec.get_file_info(filename)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns information about an audio file as an **AudioFileInfo** object.

- **filename** (str): Path to the audio file.

sudio.io.codec.stream_audio_file(filename, output_format=SampleFormat.SIGNED16, nchannels=2, sample_rate=44100, frames_to_read=1024, dither=DitherMode.NONE, seek_frame=0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streams audio data from a file.

- **filename** (str): Path to the audio file.
- **output_format** (SampleFormat): Desired output sample format.
- **nchannels** (int): Number of output channels.
- **sample_rate** (int): Output sample rate in Hz.
- **frames_to_read** (int): Number of frames to read per iteration.
- **dither** (DitherMode): Dither mode.
- **seek_frame** (int): Frame to start reading from.

Returns: A **PyAudioStreamIterator** object.

sudio.io.write_to_default_output(data, format=SampleFormat.FLOAT32, channels=None, sample_rate=None)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writes raw audio data to the default audio output device.

- **data** (bytes): Raw audio data.
- **format** (SampleFormat): Sample format.
- **channels** (int): Number of audio channels. If None, uses the maximum number of output channels for the default device.
- **sample_rate** (int): Sample rate in Hz. If None, uses the default sample rate of the output device.

sudio.io.get_sample_size(format)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the sample size in bytes for a given sample format.

- **format** (SampleFormat): Audio sample format.

Returns: The sample size in bytes.

Callback Function Signatures
----------------------------

Input Callback
^^^^^^^^^^^^^^

The input callback function should have the following signature:

.. code-block:: python

    def input_callback(input_buffer: bytes, frames: int, format: SampleFormat) -> bool:
        # Process input_buffer
        # Return True to continue streaming, False to stop


- **input_buffer**: The input audio data as bytes.
- **frames**: The number of frames in the input buffer.
- **format**: The sample format of the input data.

The callback should return **True** to continue streaming or **False** to stop.

Output Callback
^^^^^^^^^^^^^^^

The output callback function should have the following signature:

.. code-block:: python

    def output_callback(frames: int, format: SampleFormat) -> tuple[bytes, bool]:
        # Generate or process output data
        # Return (output_buffer, continue_flag)

- **frames**: The number of frames to generate.
- **format**: The required sample format for the output data.

The callback should return a tuple containing:

1. The output audio data as bytes.
2. A boolean flag indicating whether to continue streaming (**True**) or stop (**False**).
