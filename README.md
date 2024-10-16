# Sudio 🎵

Sudio is an open-source digital audio processing library that offers advanced functionality through an intuitive interface. It supports both real-time and non-real-time audio manipulation, making it versatile for a wide range of audio applications, from simple playback to complex audio transformations.

[![PyPI version](https://badge.fury.io/py/sudio.svg)](https://badge.fury.io/py/sudio)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/sudio)](https://pepy.tech/project/sudio)


## 🚀 Quick Start

### Installation

Install Sudio using pip:

```bash
pip install sudio
```

### Basic Usage

Here's a simple example to get you started with audio playback:

```python
import sudio

su = sudio.Master()
song = su.add('example.mp3')
su.echo(song[0:15, :'1000'])
```

This will play the first 15 seconds of the audio file ‘example.mp3’, filtering out frequencies below 1000 Hz, on the standard output audio stream.

## 🎹 Key Features
- Handles both real-time streaming and offline processing, allowing for dynamic applications like live audio effects as well as batch processing of audio files.
- Allows integration of custom processing modules.
- Flexible audio playback, precise time-domain slicing, and Comprehensive filtering options
- Advanced audio manipulation (joining, mixing, shifting)
- Real-time audio streaming with dynamic control (pause, resume, jump)
- Custom audio processing pipelines for complex effects
- Multi-format support with quality-controlled encoding/decoding


## 📚 Documentation

For detailed documentation and examples, visit our [GitHub Pages site](https://mrzahaki.github.io/sudio).

## 🤝 Contributing

Sudio is like a symphony in progress, and we'd love for you to join the orchestra! If you're interested in contributing, please check out our [contribution guidelines](CONTRIBUTING.md).

## 💖 Support Sudio

If Sudio has been helpful to you, consider supporting its development:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/X8X8LYHJQ)

## 📄 License

Sudio is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

🎵 Let's compose the future of audio processing together with Sudio! 🎶
