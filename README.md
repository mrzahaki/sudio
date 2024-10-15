# Welcome to the Sudio 🎵
 
`sudio` is an `Open-source`,  `easy-to-use` digital audio processing library featuring both a **real-time**, **non-real-time** mix/edit platform. 



</br>
</br>
</br>
</br>
<div align="center"> 🎵 <b>Sudio</b> is like a <b>budding musician</b> finding its rhythm. Your support can help it <b>compose a symphony</b>. 🎶</div>
</br>
</br>
</br>
</br>

## Installation


##### Latest PyPI stable release (previous version)

    pip install sudio


## Quick start

#### Audio playback

```python
import sudio

su = sudio.Master()
su.add('baroon.mp3')
su.echo('baroon')
``` 

the record with the name of baroon will be played on the stdout audio stream. 

#### Audio Manipulation

##### Time Domain Slicing


You can easily slice audio files to play specific segments:

```python
su = sudio.Master()
song = su.add('baroon.mp3')
su.echo(song[12: 27.66])

# Play from 30 seconds to the end
su.echo(song[30:])

# Play the first 15 seconds
su.echo(song[:15])
```

##### Combining Audio Segments

You can join multiple segments of audio: 

```python
su = sudio.Master()
rec = su.add('baroon.mp3')

# method 1
su.echo(rec[12: 27.66, 65: 90])

# method 2
result = rec[12: 27.66].join(rec[65: 90])

# Combine multiple segments
medley = song[10:20].join(song[40:50], song[70:80])
su.echo(medley)
```

The audio record is split into two parts, the first one 12-27.66 seconds, and the last one 65-90 seconds, then the sliced records are merged and played in the stream.

##### Audio Basic Effects

###### Volume Adjustment

Adjust the volume of an audio segment:

```python
su = sudio.Master()
song = su.add('song.mp3')

# Double the volume
loud_segment = song[10:20] * 2

# Halve the volume
quiet_segment = song[30:40] / 2

su.echo(loud_segment.join(quiet_segment))
```

###### Applying Filters

Apply frequency filters to audio:


```python
su = sudio.Master()
song = su.add('song.mp3')

# Apply a low-pass filter (keep frequencies below 1000 Hz)
low_pass = song[:'1000']

# Apply a high-pass filter (keep frequencies above 500 Hz)
high_pass = song['500':]

# Apply a band-pass filter (keep frequencies between 500 Hz and 2000 Hz)
band_pass = song['500':'2000']

su.echo(low_pass.join(high_pass, band_pass))
```

###### Complex Slicing

```python
su = sudio.Master()
baroon = su.add('baroon.mp3')
su.echo(baroon[5:10, :'1000', 10: 20, '1000': '5000'])
```

In the example above, a low-pass filter with a cutoff frequency of 1 kHz is applied to the record from 5 to 10 seconds, then a band-pass filter is applied from 10 to 20 seconds, and finally they are merged.

######  Custom Fade-In and Mixing

```python
import sudio
from sudio.types import SampleFormat
import numpy as np

su = sudio.Master()
song = su.add('example.mp3')

fade_length = int(song.get_sample_rate() * 5)  # 5-second fade
fade_in = np.linspace(0, 1, fade_length)

with song.unpack(astype=SampleFormat.FLOAT32) as data:
    data[:, :fade_length] *= fade_in
    song.set_data(data)

modified = song[:30] + song[:15, :'100'] * 3
su.echo(modified)
su.export(modified, 'modified_song.wav')
```

This example demonstrates advanced audio manipulation techniques:

1. **Fade-In Effect**: We create a 5-second linear fade-in effect.

2. **Data Type Conversion**: The audio data is converted to float32 for precise calculations.

3. **Unpacking and Repacking**: We use `unpack()` to access raw audio data, modify it, and then `set_data()` to apply changes.

4. **Audio Slicing and Mixing**:
   - `song[:30]`: Takes the first 30 seconds of the audio.
   - `song[:15, :'100']`: Takes the first 15 seconds and applies a low-pass filter at 100 Hz.
   - The `* 3` multiplies the amplitude of the filtered segment.
   - These segments are then added together.

5. **Playback and Export**: The modified audio is played and exported to a new file.

This example showcases the library's capability to perform complex audio manipulations, combining time-domain operations (fade-in, slicing) with frequency-domain filtering and amplitude adjustments.


##### Audio Analysis

Perform simple analysis on audio files:

```python
su = sudio.Master()
song = su.add('song.mp3')

# Get audio duration
duration = song.get_duration()
print(f"Song duration: {duration} seconds")

# Get sample rate
sample_rate = song.get_sample_rate()
print(f"Sample rate: {sample_rate} Hz")

# Get number of channels
channels = song.get_nchannels()
print(f"Number of channels: {channels}")
```

##### Exporting Modified Audio

After manipulating audio, you can save the results:

```python
su = sudio.Master()
song = su.add('song.mp3')

# Create a modified version
modified = song[30:60] * 1.5  # 30 seconds from 30s mark, amplified

# Export to a new file
su.export(modified, 'modified_song.wav')
```

##### Mixing and Shifting Tracks

When adding two Wrap objects, the combined audio will be as long as the longer one, mixing overlapping parts. Adding a constant shifts the waveform up while keeping the original duration. This allows for flexible audio mixing and simple DC offset adjustments.

```python
import sudio
import numpy as np

su = sudio.Master()

# Add two audio files
song1 = su.add('song1.mp3')  # Assuming this is a 30-second audio
song2 = su.add('song2.mp3')  # Assuming this is a 40-second audio

# Add the two songs
combined = song1 + song2

# Play the combined audio
su.echo(combined)

# Add a constant value to shift the audio
shifted = song1 + 0.1

# Play the shifted audio
su.echo(shifted)

# Print durations
print(f"Song1 duration: {song1.get_duration()} seconds")
print(f"Song2 duration: {song2.get_duration()} seconds")
print(f"Combined duration: {combined.get_duration()} seconds")
print(f"Shifted duration: {shifted.get_duration()} seconds")
```




#### Audio Streaming

##### Basic Streaming with Pause and Resume

This example demonstrates how to control audio playback using the sudio library, including starting, pausing, resuming, and stopping a stream.

```python
import sudio
import time

# Initialize the audio master
su = sudio.Master()
su.start()

# Add an audio file to the master
record = su.add('example.mp3')
stream = su.stream(record)

# Enable stdout echo
su.echo()

# Start the audio stream
stream.start()
print(f"Current playback time: {stream.time} seconds")

# Pause the playback after 5 seconds
time.sleep(5)
stream.pause()
print("Paused playback")

# Resume playback after 2 seconds
time.sleep(2)
stream.resume()
print("Resumed playback")

# Stop playback after 5 more seconds
time.sleep(5)
stream.stop()
print("Stopped playback")
```

This script showcases basic audio control operations, allowing you to manage playback with precise timing.


##### Basic Streaming with Jumping to Specific Times in the Audio

This example illustrates how to start playback and jump to a specific time in an audio file.

```python
import sudio
import time

# Initialize the audio master
su = sudio.Master()
su.start()

# Add a long audio file to the master
record = su.add('long_audio.mp3')
stream = su.stream(record)

# Enable stdout echo
su.echo()

# Start the audio stream
stream.start()
print(f"Starting playback at: {stream.time} seconds")

# Jump to 30 seconds into the audio after 5 seconds of playback
time.sleep(5)
stream.time = 30
print(f"Jumped to: {stream.time} seconds")

# Continue playback for 10 more seconds
time.sleep(10)
print(f"Current playback time: {stream.time} seconds")

# Stop the audio stream
stream.stop()
```

This script demonstrates how to navigate within an audio file, which is useful for long audio content or when specific sections need to be accessed quickly.

##### Streaming with Volume Control

This example shows how to dynamically control the volume of an audio stream using a custom pipeline.

```python
import sudio
import time
import sudio.types

# Initialize the audio master with a specific input device
su = sudio.Master(std_input_dev_id=2)
su.start()

# Add an audio file to the master
record = su.add('example.mp3')
stream = su.stream(record)

# Define a function to adjust the volume
def adjust_volume(data, args):
    return data * args['volume']

# Create a pipeline and append the volume adjustment function
pip = sudio.Pipeline()
pip.append(adjust_volume, args={'volume': 1.0})

# Start the pipeline
pip.start()

# Add the pipeline to the master
pipeline_id = su.add_pipeline(pip, process_type=sudio.types.PipelineProcessType.MAIN)
su.set_pipeline(pipeline_id)

# Enable stdout echo
su.echo()

# Start the audio stream
stream.start()
print("Playing at normal volume")
time.sleep(10)

# Adjust the volume to 50%
pip.update_args(adjust_volume, {'volume': 0.5})
print("Reduced volume to 50%")
time.sleep(10)

# Restore the volume to normal
pip.update_args(adjust_volume, {'volume': 1.0})
print("Restored normal volume")
time.sleep(10)

# Stop the audio stream
stream.stop()
```

This example introduces a more complex setup using a custom pipeline to dynamically adjust the audio volume during playback. It's particularly useful for applications requiring real-time audio processing or user-controlled volume adjustments.


LICENCE
-------

Open Source (OSI approved): Apache License 2.0











