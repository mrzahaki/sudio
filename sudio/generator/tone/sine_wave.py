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

from .tone_generator import ToneGenerator
from ..generator import Generator
from sudio.utils.functional import filter_kwargs as fw
from sudio.utils.functional import exclude_kwargs as ekw
from sudio.utils.functional import type_check
from .sinewave import generate_wave
import numpy as _np

class SineWave(ToneGenerator):
    @type_check(check_args=False)
    def __init__(
            self, 
            *args, 
            type: int = 1,
            **kwargs
            ):
        """
        Generates sine wave signals with harmonic overtones.

        Implements both pure sine waves and harmonic-rich waveforms approaching square waves through odd harmonic summation. 
        Also see the ToneGenerator, and Generator classes for additional details.

        Parameters:
        -----------
        *args : float | list | np.ndarray
            Harmonic frequencies in Hz (positional arguments)
        type : int, default=1
            Waveform type:
            - 1: Pure sine wave
            - >1: Add harmonics up to Nth order (square wave approximation)
        **kwargs: dict
            Strictly validated parameters forwarded to ToneGenerator

        Raises:
        -------
        ValueError
            If type <1
        TypeError: If invalid keyword arguments are provided to the ToneGenerator initializer.

        Example:
        --------
        **Square Wave Approximation**
        Generates a 1-second waveform transitioning from sine to square wave:
        ```python
        ms.generate(SineWave, [1, 30],  # Frequency components sweep 1-30Hz
                   duration=1,
                   base_freq=5,        # Fundamental frequency
                   type=10)            # 10 harmonics for square wave shape
        ```
        Uses the harmonic series formula with 10 terms (type=10) to approximate
        a square wave. The [1,30] frequency components create dynamic harmonic
        content that evolves over time.

        **Musical Chords**
        Creates minor and major chords through harmonic superposition:
        ```python
        # A minor chord (220Hz base + harmonics at 261.58, 329.63, 440Hz)
        ms.generate(SineWave, 261.58, 329.63, 440,
                   duration=5,
                   base_freq=220,      # Base A3 note
                   amplitude=-12,      -12dB master volume
                   frequency_components_amplitude=-12)  # Equal harmonic levels

        # C major chord (220Hz base + harmonics at 277, 329.63, 440Hz)
        ms.generate(SineWave, 277, 329.63, 440,
                   duration=5,
                   base_freq=220,
                   type=1)            # Pure sine components
        ```
        Demonstrates musical chord construction through precise frequency
        ratios. The frequency_components_amplitude parameter controls harmonic
        balance while maintaining phase coherence.

        **Bell Sound Synthesis**
        Creates complex metallic timbre with stereo imaging:
        ```python
        ms.generate(SineWave,
                   224.4, 215.6, 880, 1540,  # Bell harmonics
                   duration=5,
                   base_freq=220,
                   frequency_components_amplitude=[-30, -34.4, -28.7, -14.4],
                   modulation_depth=6,       # Frequency modulation depth
                   modulation_freq=1.5,       # Slow modulation rate
                   phase_different_map=[0, [0, 10]])  # Stereo phase sweep
        ```
        Uses inharmonic frequencies (224.4Hz, 215.6Hz) for metallic character.
        The phase_different_map creates stereo movement by applying different
        phase progressions (0 vs 0-10 radians) to left/right channels.

        **Frequency Sweep**
        Linear frequency sweep from 20Hz to 20kHz:
        ```python
        ms.generate(SineWave,
                   duration=5,
                   base_freq=[20, 20000],  # Sweep range
                   amplitude=-20,          # Lower amplitude protection
                   type=1)                 # Pure sine sweep
        ```
        Demonstrates dynamic parameter control through array inputs. The
        base_freq parameter linearly interpolates between 20Hz and 20kHz
        over 5 seconds.

        **Evolving Texture**
        Complex harmonic movement with stereo effects:
        ```python
        ms.generate(SineWave,
                   [120.12, 180], [120.36, 279.96],  # Harmonic sweeps
                   [120.6, 300], [120.24, 399.6],
                   duration=10,
                   frequency_components_amplitude=[-16, -20, -20, -20],
                   phase_different_map=[0, [0, 40]],  # Fast stereo rotation
                   normalize=True)        # Peak normalization
        ```
        Uses multiple sweeping harmonics with spline interpolation
        (frequency_components_spline_sigma=0.2) for smooth transitions.
        The phase_different_map's 40 radian range creates rapid stereo
        panning effects.

        **Experimental Synthesis**
        Aggressive FM modulation with dynamic parameters:
        ```python
        ms.generate(SineWave,
                   [440, 453.2], [440, 3520],  # Glissando harmonics
                   duration=10,
                   type=10,                    # 10-harmonic richness
                   modulation_depth=[1, 100],  # Depth sweep 1-100Hz
                   modulation_freq=[1, 10],    # Rate sweep 1-10Hz
                   frequency_components_amplitude=[[-96, -24], ...], # Amp envelopes
                   phase_different_map=[0, [0, 40]])
        ```
        Combines multiple dynamic parameters: harmonic frequencies glide
        between values, modulation depth/rate evolve over time, and
        amplitude uses envelope shaping ([-96, -24] dB sweeps). The
        type=10 parameter adds harmonic complexity approaching square
        wave characteristics.
        """
        if ekw(
            cls=ToneGenerator,
            property='__init__', 
            **kwargs
            ):
            raise TypeError('Invalid keyword argument(s) for ToneGenerator')
        
        super().__init__(
            *args, 
            interleave=False,
            **fw(
                cls=ToneGenerator, 
                property='__init__', 
                **kwargs
                )
            )
        
        if type < 0:
            raise ValueError("tone_type must be greater than 0")
        elif type == 0:
            type = 1
        self._tone_type = type
        
        
    @type_check()
    def generate(
        self, 
        *args,
        )->_np.ndarray:
        """
        Synthesizes audio buffer from configured parameters.

        Combines base frequency, harmonics, and modulation effects into final
        multi-channel output. Applies phase differentiation for stereo imaging. 
        Also see the ToneGenerator.*, and Generator.* methods for additional details.

        Returns:
        --------
        np.ndarray
            Shaped (nchannels, samples) array of synthesized audio
        """
        
        wave = generate_wave(
            int(self._tone_type),
            self._phase,
            self._modulation_freq,
            self._modulation_depth,
            self._time_vector,
            self._base_freq,
            self._amplitude,
            self._frequency_components_amplitude,
            self._frequency_components,
            self._phase_different_map,
            self._nchannels
        )

        return wave

        
            