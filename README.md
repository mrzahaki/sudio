<div align="center">
    <picture style="pointer-events: none; user-select: none;">
        <img src="https://raw.githubusercontent.com/mrzahaki/sudio/Master/docs/_static/sudio.png" alt="sudio" width="60%" height="60%">
    </picture>
</div>


# Sudio 🎵

[![PyPI version](https://badge.fury.io/py/sudio.svg)](https://badge.fury.io/py/sudio)
[![PyPI Downloads](https://static.pepy.tech/badge/sudio)](https://www.pepy.tech/projects/sudio)
[![Documentation Status](https://img.shields.io/badge/docs-github%20pages-blue)](https://mrzahaki.github.io/sudio/)
[![Build Status](https://github.com/mrzahaki/sudio/actions/workflows/python-package.yml/badge.svg)](https://github.com/mrzahaki/sudio/actions/workflows/python-package.yml)
[![Python Version](https://img.shields.io/pypi/pyversions/sudio.svg)](https://pypi.org/project/sudio/)
[![Supported OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)](https://shields.io/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrzahaki/sudio/blob/Master/docs/_static/sudio.ipynb)  


use case of audio processing and manipulation, providing set of tools for working with digital audio files. supports operations like time-domain slicing, frequency filtering, audio mixing, streaming, and effect application across various audio formats, making complex audio engineering tasks accessible through a pythonic interface.


## 🚀 Quick Start

### Installation

install Sudio using pip:

```bash
pip install sudio==1.0.10
```

### Basic Usage

an example to get you started with sudio:

```python
import sudio
from sudio.process.fx import (
    PitchShifter, 
    Tempo, 
    ChannelMixer, 
    FadeEnvelope, 
    FadePreset
)
su = sudio.Master()

song = su.add('./Farhad Jahangiri - 5 Sobh (320).mp3')

cool_remix = (
    song[:40]
    .afx(
        PitchShifter, 
        semitones=-3
    ).afx(
        PitchShifter, 
        start=2,
        duration=0.8,
        envelope=[0.8, 2, 1]
    ).afx(
        PitchShifter, 
        start=10,
        duration=0.8,
        envelope=[0.65, 3, 1]
    ).afx(
        PitchShifter, 
        start=20,
        duration=0.8,
        envelope=[2, 0.7, 1]
    ).afx(
        PitchShifter, 
        start=30,
        duration=4,
        envelope=[1, 3, 1, 1]
    ).afx(
        Tempo,
        envelope=[1, 0.95, 1.2, 1]
    ).afx(
        FadeEnvelope, 
        start=0,
        stop=10,
        preset=FadePreset.SMOOTH_FADE_IN
    )
)

side_slide  = (
    song[:10].afx(
        ChannelMixer, 
        correlation=[[0.4, -0.6], [0, 1]]
    ).afx(
        FadeEnvelope, 
        preset=FadePreset.SMOOTH_FADE_OUT
    )
)

cool_remix = side_slide  + cool_remix 

# simple 4 band EQ
cool_remix = cool_remix[
        : '200': 'order=6, scale=0.7', 
        '200':'800':'scale=0.5', 
        '1000':'4000':'scale=0.4', 
        '4000'::'scale=0.6'
    ] 

su.export(
    cool_remix, 
    'remix.mp3', 
    quality=.8, 
    bitrate=256
    )

su.echo(cool_remix)
```

#### Remix
[Listen to the remix](https://raw.githubusercontent.com/mrzahaki/sudio/Master/docs/_static/remix.mp3)

#### Original
[Listen to the main track](https://raw.githubusercontent.com/mrzahaki/sudio/Master/docs/_static/main.mp3)




 it used effects like PitchShifter, which allows dynamic pitch alterations through static semitone shifts and dynamic pitch envelopes, Tempo for seamless time-stretching without pitch distortion, ChannelMixer to rebalance and spatialize audio channels, and FadeEnvelope for nuanced amplitude shaping. The remix workflow shows the library's flexibility by applying multiple pitch-shifting effects with varying start times and envelopes, dynamically adjusting tempo, adding a smooth fade-in, creating a side-slide effect through channel mixing, and scaling different remix sections. By chaining these effects together you can craft complex audio transformations, enabling audio remixing with just a few lines of code.




### Explore Sudio

Get started with `Sudio` processing in minutes using [Google Colab](https://colab.research.google.com/github/mrzahaki/sudio/blob/Master/docs/_static/sudio.ipynb)!


## 🎹 Key Features
 
 Handles both real-time streaming and offline processing, allowing for dynamic applications like live audio effects as well as batch processing of audio files.
 
 Allows integration of custom processing modules.
 
 Flexible audio playback, precise time-domain slicing, and Comprehensive filtering options
 
 Advanced audio manipulation (joining, mixing, shifting)
 
 Real-time audio streaming with dynamic control (pause, resume, jump)
 
 Custom audio processing pipelines for complex effects

 Multi-format support with quality-controlled encoding/decoding


## 📚 Documentation

for detailed documentation and examples, visit the [Sudio Documentation](https://mrzahaki.github.io/sudio/).


## 💖 [Support Sudio](https://donate.webmoney.com/w/OTmd0tU8H4gRUG4eXokeQb)

**Audio software shouldn’t lock you in.** Sudio offers free, open-source tools for developers and creators—built to replace proprietary apps without the fees or restrictions. To stay viable, it relies on community support: code contributions, testing, donations, or simply spreading the word. If open tools matter to you, let’s keep improving them.

## 📄 License

released under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3. See the [LICENSE](https://github.com/mrzahaki/sudio/blob/Master/LICENSE) file for details.
