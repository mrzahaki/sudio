# Sudio
 
`sudio` is a `powerful`, `Open-source`, `fast`, `cross-platform` and `easy-to-use` digital audio processing library featuring both a **real-time**, **non-real-time** mix/edit platform. 

## Abstract

Audio processing is a subset of Digital Signal Processing (DSP). 
Audio signals are captured in digital format and certain properties of the human ear are
used along with some mathematical techniques to achieve compression in order to store studio quality, 
high fidelity audio data on small disks or stream the data at lower bit rates.

#### Real-time audio processing

For live digital audio systems with high-resolution multichannel functionalities, it is desirable to have accurate latency
control and estimation over all the stages of digital audio processing chain. 

The sudio is written flexible can used for high level audio processing algorithm 
and other system factors, which might cause the latency effects. 
It also can estimate the synchronization and delay of multiple channels.

#### Non-Realtime processing:

Sudio is a comprehensive library for mixing, editing, 
and recording audio content.


#### Audio data maintaining process:

The sudio used additional cached files to reduce dynamic memory usage and 
improve performance, meaning that audio data storage methods could have 
different execution times based on the stored files. Thanks to that, 
Sudo can manage various audio streams from audio files or operating systems 
without any size limitation.

 ---

##### Table of contents:

- [Sudio](#sudio)
  - [Abstract](#abstract)
      - [Real-time audio processing](#real-time-audio-processing)
      - [Non-Realtime processing:](#non-realtime-processing)
      - [Audio data maintaining process:](#audio-data-maintaining-process)
        - [Table of contents:](#table-of-contents)
    - [Installation](#installation)
        - [Latest PyPI stable release](#latest-pypi-stable-release)
        - [Latest development release on GitHub](#latest-development-release-on-github)
  - [Quick start](#quick-start)
      - [Audio playback](#audio-playback)
      - [Audio slicing](#audio-slicing)
        - [Time domain](#time-domain)
          - [simple slicing](#simple-slicing)
          - [slice & merge](#slice--merge)
        - [Frequency domain](#frequency-domain)
          - [LPF 100Hz](#lpf-100hz)
          - [HPF 1KHz](#hpf-1khz)
          - [BPF 500Hz - 1KHz](#bpf-500hz---1khz)
        - [Complex Slicing](#complex-slicing)
    - [Examples and Advanced Usage](#examples-and-advanced-usage)
        - [STFT](#stft)
    - [Usage](#usage)
    - [Prerequisites](#prerequisites)
  - [API Documentation](#api-documentation)
    - [Master](#master)
      - [Parameters](#parameters)
      - [Notes](#notes)
      - [Methods](#methods)
        - [add_file](#add_file)
        - [add](#add)
        - [start](#start)
        - [recorder](#recorder)
        - [load_all](#load_all)
        - [load](#load)
        - [get_record_info](#get_record_info)
        - [get_exrecord_info](#get_exrecord_info)
        - [syncable](#syncable)
        - [sync](#sync)
        - [del_record](#del_record)
        - [save_as](#save_as)
        - [save](#save)
        - [save](#save-1)
        - [save_all](#save_all)
        - [get_exrecord_names](#get_exrecord_names)
        - [get_record_names](#get_record_names)
        - [get_nperseg](#get_nperseg)
        - [get_nchannels](#get_nchannels)
        - [get_sample_rate](#get_sample_rate)
        - [stream](#stream)
        - [mute](#mute)
        - [unmute](#unmute)
        - [echo](#echo)
        - [wrap](#wrap)
        - [clean_cache](#clean_cache)
        - [add_pipeline](#add_pipeline)
        - [set_pipeline](#set_pipeline)
        - [set_window](#set_window)
    - [StreamControl](#streamcontrol)
      - [Methods](#methods-1)
        - [isready](#isready)
        - [is_streaming](#is_streaming)
        - [start](#start-1)
        - [resume](#resume)
        - [stop](#stop)
        - [pause](#pause)
        - [enable_loop](#enable_loop)
        - [disable_loop](#disable_loop)
      - [Properties](#properties)
        - [time](#time)
          - [Getter](#getter)
          - [Setter](#setter)
    - [WrapGenerator](#wrapgenerator)
      - [Parameters](#parameters-1)
      - [Methods](#methods-2)
        - [get_data](#get_data)
        - [get](#get)
        - [set_data](#set_data)
        - [get_sample_format](#get_sample_format)
        - [get_sample_width](#get_sample_width)
        - [get_master](#get_master)
        - [get_size](#get_size)
        - [get_cache_size](#get_cache_size)
        - [get_nchannels](#get_nchannels-1)
        - [get_duration](#get_duration)
        - [join](#join)
      - [Magic methods](#magic-methods)
        - [getitem](#getitem)
        - [del](#del)
        - [str](#str)
        - [mul](#mul)
        - [truediv](#truediv)
        - [pow](#pow)
        - [add](#add-1)
        - [sub](#sub)
        - [call](#call)
    - [Wrap](#wrap-1)
      - [Parameters](#parameters-2)
      - [Notes](#notes-1)
      - [Methods](#methods-3)
        - [get_sample_format](#get_sample_format-1)
        - [get_sample_width](#get_sample_width-1)
        - [get_master](#get_master-1)
        - [get_size](#get_size-1)
        - [get_frame_rate](#get_frame_rate)
        - [get_nchannels](#get_nchannels-2)
        - [get_duration](#get_duration-1)
        - [join](#join-1)
        - [unpack](#unpack)
        - [get_data](#get_data-1)
        - [is_packed](#is_packed)
        - [get](#get-1)
        - [set_data](#set_data-1)
        - [getitem](#getitem-1)
      - [Magic methods](#magic-methods-1)
        - [del](#del-1)
        - [str](#str-1)
        - [mul](#mul-1)
        - [truediv](#truediv-1)
        - [pow](#pow-1)
        - [add](#add-2)
        - [sub](#sub-1)
    - [Pipeline](#pipeline)
      - [Parameters](#parameters-3)
      - [Methods](#methods-4)
        - [clear](#clear)
        - [run](#run)
        - [insert](#insert)
          - [Parameters](#parameters-4)
          - [Reterns](#reterns)
        - [append](#append)
        - [sync](#sync-1)
        - [aasync](#aasync)
        - [delay](#delay)
        - [set_timeout](#set_timeout)
        - [get_timeout](#get_timeout)
        - [put](#put)
        - [get](#get-2)
      - [Magic methods](#magic-methods-2)
        - [call](#call-1)
        - [delitem](#delitem)
        - [len](#len)
        - [getitem](#getitem-2)


### [Installation](#installation)


##### [Latest PyPI stable release](#latest-pypi-stable-release)

    pip install sudio

##### [Latest development release on GitHub](#github_installation_link)

Pull and install pre-release `main` branch:

    pip install git+https://github.com/MrZahaki/sudio.git


## Quick start

#### Audio playback

```python
import sudio

su = sudio.Master()
su.add('baroon.mp3')
su.echo('baroon')
``` 

the record with the name of baroon will be played on the stdout audio stream. 

#### Audio slicing

##### Time domain

###### simple slicing

The following example is used to play the clipped record in the stdout audio stream.

```python
su = sudio.Master()
baroon = su.add('baroon.mp3')
su.echo(baroon[12: 27.66])
```

###### slice & merge

```python
su = sudio.Master()
baroon = su.add('baroon.mp3')
su.echo(baroon[12: 27.66, 65: 90])
```

The audio record is split into two parts, the first one 12-27.66 seconds, and the last one 65-90 seconds, then the sliced records are merged and played in the stream.

##### Frequency domain

###### LPF 100Hz

```python
su = sudio.Master()
baroon = su.add('baroon.mp3')
su.echo(baroon[: '100'])
```

###### HPF 1KHz

```python
su = sudio.Master()
baroon = su.add('baroon.mp3')
su.echo(baroon['1000':])
```

###### BPF 500Hz - 1KHz

```python
su = sudio.Master()
baroon = su.add('baroon.mp3')
su.echo(baroon['500':'1000'])
```

##### Complex Slicing

```python
su = sudio.Master()
baroon = su.add('baroon.mp3')
su.echo(baroon[5:10, :'1000', 10: 20, '1000': '5000'])
```

In the example above, a low-pass filter with a cutoff frequency of 1 kHz is applied to the record from 5-10 seconds, then a band-pass filter is applied from 10-20 seconds, and finally they are merged.


### [Examples and Advanced Usage](#examples-and-advanced-usage)

All your files and folders are presented as a tree in the file explorer. You can switch from one to another by clicking a file in the tree.
##### [STFT](#examples-stft)


<br />

### [Usage](#usage)

### Prerequisites

Sudio has written in the python language, you can see the python documentation from this link. 
This library used scientific computing packages to manipulate data like the numpy and scipy.


## API Documentation

### Master

```py
sudio.Master
```


#### Parameters

- **std_input_dev_id**: int, optional
    os standard input device id. If not given, then the input device id will be selected automatically(ui_mode=False) or manually by the user(ui_mode=True)

    <br />
    
- **std_output_dev_id**: int, optional
    os standard output device id. If not given, then the output device id will
    be selected automatically(ui_mode=False) or manually by the user(ui_mode=True)

- **frame_rate**:  int, optional
    Input channel sample rate. If std_input_dev_id is selected as None, the value will be selected automatically.

    <br />
    
- **nchannels**: int, optional
    Number of input channels (inner channels). If std_input_dev_id is selected as None, the value will be selected automatically.

    <br />
    
- **data_format**: SampleFormat
    Specifies the audio bit depths. Supported data format (from sudio):
    **formatFloat32**, **formatInt32**, **formatInt24**, **formatInt16** (default), **formatInt8**, **formatUInt8**

    <br />
    
- **mono_mode**: bool, optional
    If True, then all the input channels will be mixed into one channel.

- **ui_mode**: bool, optional
    If Enabled then user interface mode will be activated.

    <br />
    
- **nperseg**: int,optional
    number of samples per each data frame (in one channel)

    <br />
    
- **noverlap**: int, default None

    The noverlap defines the number of overlap between defined windows. If not given then it's value will be selected as
    
    ```math
    SE = \frac{nperseg}{2}.
    ```

    When the length of a data set to be transformed is larger than necessary to provide the desired frequency resolution, a common practice is to subdivide it into smaller sets and window them individually.

    To mitigate the "loss" at the edges of the window, the individual sets may overlap in time.

    

    <br />
    
- **NOLA_check**: bool, optional
    Check whether the Nonzero Overlap Add (NOLA) constraint is met.(If true)

    <br />
    
- **IO_mode**: str, optional
    Input/Output processing mode that can be:
    
    <br />
    
    - "**stdInput2stdOutput**":(default)
        default mode; standard input stream (system audio or any other windows defined streams) to standard defined output stream (system speakers).

    - "**usrInput2usrOutput**":
        user defined input stream (callback function defined with input_dev_callback) to  user defined output stream (callback function defined with output_dev_callback).

    - "**usrInput2stdOutput**":
        user defined input stream (callback function defined with input_dev_callback) to  user defined output stream (callback function defined with output_dev_callback).

    - "**stdInput2usrOutput**":
        standard input stream (system audio or any other windows defined streams) to  user defined output stream (callback function defined with output_dev_callback).

    <br />
    
- **input_dev_callback**: callable;
    user input callback; default None; 
    this function used in "**usrInput2usrOutput**" and "**usrInput2stdOutput**" IO modes and called by the sudio core every 1/frame_rate second and must returns frame data as a numpy array with the dimensions of the (Number of channels, Number of data per segment).
    <br />

    Format: 
    
    ```py
    :inputs:    frame_count, time_info, status
    :outputs:   Numpy N-channel data frame in  (Number of channels, Number of data per segment)
                dimensions.
    ```                
    :memo: **Note:** The data frame in single channel mode has the shape of (Number of data per segment, ).

    <br />
    
- **output_dev_callback**: callable;

    User input callback; default None;
    This function used in "**usrInput2usrOutput**" and "**stdInput2usrOutput**" IO modes and called by the sudio core every 1 / frame_rate second after processing frames and takes the frame data as a numpy array in (Number of channels, Number of data per segment) dimensions numpy array.

    format: 
    
    ```py
    :inputs:    Numpy N-channel data frame in  (Number of channels, Number of data per segment) dimensions.
    :outputs: customized audio stream.
    ```

    :memo: **Note:** The data frame in single channel mode has the shape of (Number of data per segment, ).

    <br />
    
- **master_mix_callback**: callable, optional
    This callback is used before the main-processing stage in the master for controlling and mixing all slave channels to a ndarray of shape (master.nchannels, 2, master.nperseg).

    If this parameter is not defined, then the number of audio channels in all slaves must be the same as the master.

    <br />
    
- **window**:  string, float, tuple, ndarray, optional
    The type of window, to create(string) or pre designed window in numpy array format. use None to disable windowing process.
    Default ("**hann**") Hanning function.

#### Notes
    
-    **If** the window function requires no parameters, then `window` can be a string otherwise the `window` must be a tuple with the first argument the string name of the window, and the next
    arguments the needed parameters.
    <br />
    If `window` is a floating point number, it is interpreted as the beta parameter of the `kaiser` window.
    Each of the window types listed above is also the name of a function that can be called directly to create a window of that type.
    <br />
    window types:
        ```py
        boxcar, triang , blackman, hamming, hann(default), bartlett, flattop, parzen, bohman, blackmanharris
        nuttall, barthann, cosine, exponential, tukey, taylor, kaiser (needs beta),
        gaussian (needs standard deviation), general_cosine (needs weighting coefficients),
        general_gaussian (needs power, width), general_hamming (needs window coefficient),
        dpss (needs normalized half-bandwidth), chebwin (needs attenuation)
        ```
 -   "**nonzero** overlap add" (NOLA):
    This ensures that the normalization factors in the denominator of the overlap-add inversion equation are not zero. Only very pathological windows will fail the NOLA constraint.


#### Methods

<br />
     
##### add_file

```py
sudio.Master.add_file(  self, filename: str, sample_format: SampleFormat = SampleFormat.formatUnknown, 
                        nchannels: int = None, sample_rate: int = None ,safe_load=True)
```

The add_file method used to Add an audio file to the local database. None value for the parameters means that they are automatically selected based on the properties of the audio file.

- **parameters**:

    - **filename**: path/name of the audio file
    - **sample_format**: optional; sample format (refer to the SampleFormat data field).
    - **nchannels**: optional;  number of audio channels
    - **sample_rate**: optional;  sample rate
    - **safe_load**: optional;  load an audio file and modify it according to the 'Master' attributes. (sample rate, sample format, number of channels, etc).

- **returns** WrapGenerator object


:memo: **Note:** supported files format: WAV, FLAC, VORBIS, MP3

:memo: **Note:** The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance, meaning that, The audio data storage methods can have different execution times based on the cached files.


<br />

##### add

```py
sudio.Master.add(self, record, safe_load=True)
```

Add new record to the local database.

- **parameters**:

    - **record**: can be an wrapped record, Record object or an audio file in mp3, WAV, FLAC or VORBIS format.
    - **safe_load**: optional; load an audio file and modify it according to the 'Master' attributes.

- **returns** Wrapped(WrapGenerator) object

:memo: **Note:** The name of the new record can be changed with the following tuple that is passed to the function: (record object, new name of ), otherwise automatically generated.

:memo: **Note:** The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance, meaning that, The audio data storage methods can have different execution times based on the cached files.


<br />

##### start

start audio streaming, must be called before audio streaming process. 
**returns** master object.





<br />

##### recorder

```py
   sudio.Master.recorder(self,
                        record_duration: Union[int, float] = 10,
                        name: str=None,
                        enable_compressor: bool=False,
                        noise_sampling_duration: Union[int, float]=1,
                        enable_ui: bool=False,
                        play_recorded: bool=False,
                        catching_precision: float=0.1,
                        echo_mode: bool=False,
                        on_start_callback: callable=None,
                        on_sampling_callback: callable=None)
```

record from main stream for a while.


- **parameters**:

    - **record_duration**: determines duration of the recording process.
    - **name**: optional; name of new record.
    - **enable_compressor**: optional; enable to compress the recorded data. The compressor deletes part of the signal that has lower energy than the sampled noise.
    - **noise_sampling_duration**: optional; noise sampling duration used by the compressor(if enabled).
    - **enable_ui**: optional; user inteface mode
    - **play_recorded**: optional; Determines whether the recorded file will be played after the recording process.
    - **catching_precision**: optional; Signal compression accuracy, more accuracy can be achieved with a smaller number (default 0.1).
    - **echo_mode**: It can disable the mainstream's echo mode, when the recorder is online.
    - **on_start_callback**: called after noise sampling. 
        ```py
        parameters: master object, sampled noise level, record duration time. 
        return true to continiue recording.
        ```
- **returns** Wrapped(WrapGenerator) object



<br />

##### load_all

```py
   sudio.Master.load_all(self, safe_load=True)
```

load all of the saved records from the external database(static memory) to the local database(dynamic memory).


- **parameters**:

    - **safe_load**: if safe load is enabled then load function tries to load a record in the local database based on the master settings, like the frame rate and etc.
        ```
- **returns** None

<br />

##### load

```py
   sudio.Master.load(self, name: str, load_all: bool=False, safe_load: bool=True,
                    series: bool=False) -> Union[WrapGenerator, Record])
```

The load method used to load a predefined recoed from the external database (static memoty) to
the local database(dynamic memory). Trying to load a record that was previously loaded, outputs a wrapped version of the named record.


- **parameters**:

  - **name**: The name of the predefined record.
  - **load_all**: used to load all of the saved records from the external database to the local database.
  - **safe_load**: if safe load is enabled this method tries to load a record in the local database, based on the master settings, like the frame rate and etc.
  - **series**: If enabled, attempting to load a record that has already been loaded will output the data series of the named record.

- **returns** optional; Wrapped object, Record obejct.

<br />


##### get_record_info

```py
   sudio.Master.get_record_info(self, name: str) -> Record
```

get extra info about the record 


- **parameters**:

  - **name**: name of the registered record on the local or external 

- **returns** information about saved record ['noiseLevel' 'frameRate'  'sizeInByte' 'duration' 'nchannels' 'nperseg' 'name'].

<br />


##### get_exrecord_info

```py
   sudio.Master.get_exrecord_info(self, name: str) -> Record
```

get extra info about the record 


- **parameters**:

  - **name**: name of the registered record on the external database(if exists).

- **returns** information about saved record ['noiseLevel' 'frameRate'  'sizeInByte' 'duration' 'nchannels' 'nperseg' 'name'].




##### syncable

```py
   sudio.Master.syncable(self,
                        * target,
                        nchannels: int = None,
                        sample_rate: int = None,
                        sample_format: SampleFormat = SampleFormat.formatUnknown)
```

Determines whether the wrapped record/s can be synced with specified properties.

- **parameters**:

  - **target**: target: wrapped record\s.(regular records)

  - **nchannels**: number of channels; if the value is None, the target will be compared to the 'self' properties.

  - **sample_rate**: sample rate; if the value is None, the target will be compared to the 'self' properties.

  - **sample_format**: if the value is None, the target will be compared to the 'self' properties.
  - 
- **returns** returns only objects that need to be synchronized.

<br />

##### sync

```py
   sudio.Master.sync(self,
                    * targets,
                    nchannels: int=None,
                    sample_rate: int=None,
                    sample_format: SampleFormat=SampleFormat.formatUnknown,
                    output='wrapped')
```

This method used to Synchronize wrapped record/s with the specified properties.


- **parameters**:

  - **targets**: wrapped records\s. 
  - **nchannels**: number of channels; if the value is None, the target will be synced to the 'self' properties.
  - **sample_rate**: if the value is None, the target will be synced to the 'self' properties.
  - **sample_format**: if the value is None, the target will be synced to the 'self' properties.
  - **output**: can be 'wrapped'(regular records), 'series'(dict type) or 'ndarray_data'
-  **returns** returns synchronized objects.


<br />

##### del_record

```py
sudio.Master.del_record(self, name: str, deep: bool=False)

```

The del_record method used to delete a record from the internal/external database.


- **parameters**:

- **name** str: the name of preloaded record.
- **deep** bool: deep delete mode is used to remove the record  and its corresponding caches from the external database.
-  **returns** None

<br />

##### save_as

```py
sudio.Master.save_as(self, record: Union[str, Record, Wrap, WrapGenerator], file_path: str=SAVE_PATH)
```
Convert the record to the wav audio format.

- **Parameters**:
   - **record**: name of the registered (wrapped) record, a 'Record' object or a (customized) wrapped record.
   - **file_path**: The name or path of the wav file.
        A new name for the record to be converted can be placed at the end of the address.
-  **returns** None


<br />

##### save

```py
sudio.Master.save(self, name: str='None', save_all: bool=False)
```
Save the preloaded record to the external database.

- **Parameters**:
   - **name**: name of the preloaded record
    - **save_all**: if true then it's tries to save all of the preloaded records
-  **returns** None

<br />

##### save

```py
sudio.Master.save(self, name: str='None', save_all: bool=False)
```
Save the preloaded record to the external database.

- **Parameters**:
   - **name**: name of the preloaded record
    - **save_all**: if true then it's tries to save all of the preloaded records
-  **returns** None
    
<br />

##### save_all

```py
sudio.Master.save_all(self)
```

Save all of the preloaded records to the external database
-  **returns** None    

<br />

##### get_exrecord_names

```py
sudio.Master.get_exrecord_names(self) -> list
```

:return: list of the saved records in the external database


<br />

##### get_record_names

```py
sudio.Master.get_record_names(self, local_database: bool=True) -> list
```

**param** local_database: if false then external database will be selected
**returns**: a list of the saved records in the external or internal database

<br />

##### get_nperseg

```py
sudio.Master.get_nperseg(self)
```

**returns**:  number of samples per each data frame (single channel)

<br />

##### get_nchannels

```py
sudio.Master.get_nchannels(self)
```

**returns**:  number of input channels for current master object.

<br />

##### get_sample_rate

```py
sudio.Master.get_sample_rate(self)
```

**returns**: current master sample rate


<br />

##### stream


```py
sudio.Master.stream(self, 
                    record: Union[str, Wrap, Record, WrapGenerator],
                    block_mode: bool=False,
                    safe_load: bool=False,
                    on_stop: callable=None,
                    loop_mode: bool=False,
                    use_cached_files=True,
                    stream_mode:StreamMode = StreamMode.optimized
                    ) -> StreamControl
```
'Record' playback on the mainstream.


- **Parameters**:

  - **record**: predefined record name, (customized) wrapped record, or a 'Record' object.
  - **block_mode**: This can be true, in which case the current thread will be
      blocked as long as the stream is busy.
  - **safe_load**: load an audio file and modify it according to the 'Master' attributes(like the frame rate, number oof channels, etc).
  - **on_stop**: An optional callback is called at the end of the streaming process.
  - **loop_mode**: playback continuously.
  - **use_cached_files**: enable additional cache maintaining process.
  - **stream_mode**: (see StreamMode enum).

-  **returns** StreamControl object
    

:memo: **Note:** The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance, meaning that, The audio data storage methods can have different execution times based on the cached files.

:memo: **Note:** The recorder can only capture normal streams(Non-optimized streams)


<br />

##### mute

```py
sudio.Master.mute(self)
```
mute the mainstream


<br />

##### unmute

```py
sudio.Master.unmute(self)
```
disable mute mode of the mainstream



<br />

##### echo


```py
sudio.Master.echo(  self, 
                    record: Union[Wrap, str, Record, WrapGenerator]=None,
                    enable: bool=None, 
                    main_output_enable: bool=False)
```
start to play "Record" on the operating system's default audio output.          

- **Parameters**:

  - **record**: optional, default None;
      It could be a predefined record name, a wrapped record,
      or a 'Record' object.

  - **enable**: optional, default None(trigger mode)
      determines that the standard output of the master is enable or not.

  - **main_output_enable**:
      when the 'record' is not None, controls the standard output activity of the master

-  **returns** self object
    

:memo: **Note:** If the 'record' argument takes the value None, the method controls the standard output activity of the master with the 'enable' argument.

:memo: **Note:** If the 'record' argument takes the value None, the kernel starts streaming the std input to the std output.



<br />

##### wrap

```py
sudio.Master.wrap(self, record: Union[str, Record])
```

Create a Wrap object.

**param** record: preloaded record or a Record object
**returns**: Wrap object

<br />

##### clean_cache

```py
sudio.Master.clean_cache(self)
```

The audio data maintaining process has additional cached files to reduce dynamic memory usage and improve performance, meaning that, The audio data storage methods can have different execution times based on the cached files. This function used to clean additional cached files.

**returns**: self object

<br />

##### add_pipeline

```py
sudio.Master.add_pipeline(  self, 
                            name: str, 
                            pip: Union[Pipeline, list],
                            process_type: str='main', 
                            channel: int=None)

```

Add a new process pipeline to the master object.

- **Parameters**:

  - **name**: string; represents the new pipeline
     <br/>
  
  - **pip**: obj; Pipeline object/s
             In the multi_stream process type, this argument must be a list of the defined pipelines, with the size equal to the nchannel.
     <br/>

  - **process_type**: string; 'main', 'branch', 'multi_stream'
     The sudio kernel inject audio data to the activated pipeline[if exist] and all of the branch type pipelines then takes output from the primary one.
     
     <br />

     :memo: **Note:** Use set_pipeline to activate a main branch or a multi_stream one.
     <br />

     :memo: **Note:** A branch pipeline is used for data monitoring(GUI) purposes.

     <br/>
        

  - **channel**: obj; None or [0 to nchannel]; just activated in branched pipelines;

     The input  data passed to a pipeline can be an numpy array with the shape of the (number of the audio channels, 2 [number of the windows per each frame], nperseg) (in mono mode (2, self._nperseg)).
     <br/>

:memo: **Note:** the  pipeline used to process data and return it to the kernel with the  dimensions as same as the input.

:memo: **Note:** Each pipeline used to process data in different threads, so the the performance will be improved. 

<br />

##### set_pipeline

```py
sudio.Master.set_pipeline(self, name: str, enable: bool=True)
```

activate the registered pipeline on the process stream. 

- **Parameters**:

  - **name**: string; A name that represents the new pipeline
  - **enable**: bool; state of the primary pipeline.

:memo: **Note:** Only multi_stream and main branches are allowed for activation.

<br />

##### set_window


```py
sudio.Master.set_window(self,
                        window: object = 'hann',
                        noverlap: int = None,
                        NOLA_check: bool = True)
```
      
change type of the current processing window.

- **Parameters**:

  - **window**: string, float, tuple, ndarray, optional
    The type of window, to create (string) or pre designed window in numpy array format. use None to disable windowing process.
    Default ("**hann**") Hanning function.

  - **noverlap**: int, default None

    The noverlap defines the number of overlap between defined windows. If not given then it's value will be selected as
    
    ```math
    SE = \frac{nperseg}{2}.
    ```

    When the length of a data set to be transformed is larger than necessary to provide the desired frequency resolution, a common practice is to subdivide it into smaller sets and window them individually.

    To mitigate the "loss" at the edges of the window, the individual sets may overlap in time.

  - **NOLA_check**: bool, optional
    Check whether the Nonzero Overlap Add (NOLA) constraint is met.(If true)


<br />

### StreamControl

```py
sudio.Master.StreamControl
```

The StreamControl class used to control data flow of an audio record (live control on audio streaming).

<br />

#### Methods



<br />

##### isready

```py
sudio.Master.StreamControl.isready()
```
check current stream compatibility with the Master object and return true on ready to streaming.


<br />

##### is_streaming

```py
sudio.Master.StreamControl.is_streaming()
```

return true if current stream was started before. 

<br />

##### start

```py
sudio.Master.StreamControl.start()
```

check current stream compatibility with the Master object and start to streaming.

<br />

##### resume

```py
sudio.Master.StreamControl.resume()
```

resume current streaming if current stream was paused.

<br />

##### stop

```py
sudio.Master.StreamControl.stop()
```
stop current streaming if current stream is activated.


<br />

##### pause

```py
sudio.Master.StreamControl.pause()
```

pause current streaming if current stream is activated.


<br />

##### enable_loop

```py
sudio.Master.StreamControl.enable_loop()
```

enable to restart at the end of streaming. 

<br />

##### disable_loop

```py
sudio.Master.StreamControl.disable_loop()
```

disable audio streaming loop mode.

<br />


#### Properties



<br />

##### time

```py
sudio.Master.StreamControl.time
```

###### Getter

used to retrive elapsed time of the current streamed record.

###### Setter

Set the current time of streamed record.


<br />
<br />

### WrapGenerator

```py
sudio.WrapGenerator
```

#### Parameters


```py
__init__(self, other, record)
```


#### Methods

##### get_data

```py
sudio.WrapGenerator.
```

<br />

##### get

```py
sudio.WrapGenerator.
```

<br />

##### set_data

```py
sudio.WrapGenerator.
```

<br />

##### get_sample_format

```py
sudio.WrapGenerator.
```

<br />

##### get_sample_width

```py
sudio.WrapGenerator.
```

<br />

##### get_master

```py
sudio.WrapGenerator.
```

<br />

##### get_size

```py
sudio.WrapGenerator.
```

<br />

##### get_cache_size

```py
sudio.WrapGenerator.
```

<br />

##### get_nchannels

```py
sudio.WrapGenerator.
```

<br />

##### get_duration

```py
sudio.WrapGenerator.
```

<br />

##### join

```py
sudio.WrapGenerator.
```

<br />



#### Magic methods




##### getitem

```py
sudio.WrapGenerator.
```

<br />

##### del

```py
sudio.WrapGenerator.
```

<br />

##### str

```py
sudio.WrapGenerator.
```

<br />

##### mul

```py
sudio.WrapGenerator.
```

<br />

##### truediv

```py
sudio.WrapGenerator.
```

<br />

##### pow

```py
sudio.WrapGenerator.
```

<br />

##### add

```py
sudio.WrapGenerator.
```

<br />

##### sub

```py
sudio.WrapGenerator.
```

<br />

##### call

```py
sudio.WrapGenerator.
```

<br />




<br />
<br />

### Wrap

```py
sudio.Wrap
```

#### Parameters


```py
__init__(self, other, record, generator)
```

#### Notes



#### Methods

##### get_sample_format

##### get_sample_width

##### get_master

##### get_size

##### get_frame_rate

##### get_nchannels

##### get_duration

##### join

##### unpack

##### get_data

##### is_packed

##### get

##### set_data

##### getitem


#### Magic methods

##### del

##### str

##### mul

##### truediv

##### pow

##### add

##### sub




<br />
<br />

### Pipeline

A pipeline is a system of pipes used to transport data, each pipe is a method that processes data and pass it to the next one.  

Pipeline helps the audio processing algorithms to break the complexity into smaller blocks, and the use of threading techniques improves the overall performance of the system.

```py
sudio.Pipeline
```

```mermaid 
  graph LR;
      A(Input Queue)-->B(Pipe 0);
      B-->C(Pipe 1);
      C-->D(...);
      D-->E(Output Queue);
```





#### Parameters



- **max_size**: int, optional
    Maximum number of callable per Pipeline object 

    <br />
    
- **io_buffer_size**: int, optional
    Maximum size of the I/O queues.

- **on_busy**:  [float, str], optional
    Determines the behavior of the pipeline after the I/O queues are full. (busy state).

    Please use:
    -  "drop" to drop old data from the output queue and allocate new space for new ones.
    -  "block" to block the pipeline until the last data arrives.
    -  timeout in float type to block the pipeline until the end time.

  <br />
    
- **list_dispatch**: bool, optional
    dispatch the input list type, to the arguments of the first pipe.

    <br />


#### Methods


##### clear


```py
sudio.Pipeline.clear(self)
```
      
Remove all of the items from the I/O queues in block mode.

<br />


##### run

```py
sudio.Pipeline.run(self)
```

Start data injection into pipeline. 

<br />


##### insert

```py
sudio.Pipeline.insert(self,
                      index: int,
                      *func: callable,
                      args: Union[list, tuple, object] = (),
                      init: Union[list, tuple, callable] = ())
```

Insert callable/s before index.

###### Parameters



- **index**: int
  
    index value.

    <br />
    
- **func**: callable
- 
    Pipeline callable element/s.

  <br />

- **args**:  (tuple, list, object), optional
  
    the static argument/s, will be passed to the pipeline element/s.

    example:
    - multiple arguments for multiple callables

      ```py
      def f0(arg0, arg1, data):
        return data

      def f1(arg, data):
        return data

      pip = sudio.Pipeline()
      pip.insert(2, f0, f1, args=(['arg0 for f0', 'arg1 for f0'], 'single arg for f1'))
      ```

    - single argument for multiple callables

      ```py
      
      def f0(arg, data):
        return data

      def f1(arg, data):
        return data

      pip = sudio.Pipeline()
      pip.insert(2, f0, f1, args='single arg for all')
      ```
      
      ```py
      def f0(data):
        return data

      def f1(arg, data):
        return data

      pip = sudio.Pipeline()
      pip.insert(2, f0, f1, args=(None, 'single arg for f1'))
      ```

    - single argument for single callable

      ```py
      
      def f0(arg, data):
        return data

      pip = sudio.Pipeline()
      pip.insert(2, f0, args='single arg for f0')
      ```

  <br />
    
- **init**: (list, tuple, callable), optional
 
    single or multiple callables suggested for access to the pipeline's shared memory, that called after starting pipeline thread execution.


###### Reterns

  - self object


<br />


##### append

```py
sudio.Pipeline.append(self,
                      *func: callable,
                      args: Union[list, tuple, object] = (),
                      init: Union[list, tuple, callable] = ())
```

Append callable/s to the end of the pipeline.
For more detailes Please refer to the  sudio.Pipeline.insert method.

<br />

##### sync

```py
sudio.Pipeline.sync(self, barrier: threading.Barrier)
```
Synchronize current pipeline with others using a barrier.


<br />

##### aasync

```py
sudio.Pipeline.aasync(self)
```

Disable pipeline synchronization.

<br />

##### delay

```py
sudio.Pipeline.delay(self)
```

used to retrive pipeline execution delay in us.

<br />

##### set_timeout

```py
sudio.Pipeline.set_timeout(self, t: Union[float, int])
```
Determines the blocking timeout of the pipeline after the I/O queues are full.

<br />

##### get_timeout

```py
sudio.Pipeline.get_timeout(self)
```

used to retrive the blocking timeout of the pipeline after the I/O queues are full.


<br />

##### put

```py
sudio.Pipeline.put(self, data)
```

Inject new data into the pipeline object 

<br />


##### get

```py
sudio.Pipeline.get(self, block=True, timeout=None):
```

Remove and return an item from the output queue.

If optional args 'block' is true and 'timeout' is None (the default), block if necessary until an item is available. If 'timeout' is a non-negative number, it blocks at most 'timeout' seconds and raises
the Empty exception if no item was available within that time.
Otherwise ('block' is false), return an item if one is immediately available, else raise the Empty exception ('timeout' is ignored in that case).

<br />

#### Magic methods


<br />

##### call

```py
sudio.Pipeline.__call__(self, data)
```

The __ call __ magic method used to Inject data into the current pipeline object 


<br />

##### delitem

```py
sudio.Pipeline.__delitem__(self, key)
```

Delete self[key].

<br />

##### len

```py
sudio.Pipeline.__len__(self)
```

Return len(self).

<br />

##### getitem

```py
sudio.Pipeline.__getitem__(self, key)
```

Return self[key].

<br />















