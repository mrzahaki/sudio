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

from sudio.generator import Generator
from sudio.types import EnvelopeType as ETYPE
from sudio.utils.functional import filter_kwargs as fw
from sudio.utils.functional import exclude_kwargs as ekw
from sudio.utils.functional import type_check
from sudio.utils.math import db2amp
from sudio.process.fx._fade_envelope import prepare_envelope
from typing import Union
import numpy as _np


class ToneGenerator(Generator):
    @type_check(check_args=False)
    def __init__(
            self, 
            *args, 
            duration: float | int = 5.0,
            base_freq: ETYPE = 3e3,
            base_freq_spline: bool = True,
            base_freq_spline_sigma: float | int = .1,
            frequency_components_spline: bool | list[bool] | tuple[bool] | _np.ndarray[bool] = True, # bool applies to all, list or tuple or nd array applies to each(with same length)
            frequency_components_spline_sigma: ETYPE = .1, # int or float applies to all, list or tuple or nd array applies to each(with same length)
            frequency_components_amplitude: ETYPE = 0.0, # int or float applies to all, list or tuple or nd array applies to each (dynamic mode supported)(with same length)
            amplitude: ETYPE = 0, # in db
            amplitude_spline: bool = True,
            amplitude_spline_sigma: float | int = .1,
            phase: ETYPE = 0.0,  # start phase
            phase_different_map: list[ETYPE] | tuple[ETYPE] | _np.ndarray[ETYPE] | None = None, # list or tuple or nd array applies to each channel (with length as nchannels)
            phase_spline: bool = True,
            phase_spline_sigma: float | int = .1,
            modulation_depth: ETYPE = 0.0,
            modulation_depth_spline: bool = True,
            modulation_depth_spline_sigma: float | int = .1,
            modulation_freq: ETYPE = 0.0,
            modulation_freq_spline: bool = True,
            modulation_freq_spline_sigma: float | int = .1,
            buffer_size: int = 512,
            **kwargs
            ):
        """
        Generates tonal audio signals with dynamic parameter control.

        Supports complex waveforms with frequency modulation, harmonic components,
        and multi-channel phase manipulation. Accepts both static values and 
        time-varying parameters through spline interpolation.

        Parameters:
        -----------
        *args : float | list | np.ndarray
            Frequency components in Hz (positional arguments).
        duration : float, default=5.0
            Audio duration in seconds (>0).
        base_freq : dynamic, default=3000
            Fundamental frequency (Hz). Accepts:
            - Single value (static)
            - [start, end] array (sweep)
            - Arbitrary array (custom envelope)
        base_freq_spline : bool, default=True
            Enable spline interpolation for base frequency.
        frequency_components_* : dynamic
            Configuration for harmonic components (same syntax as base_freq).
        amplitude : dynamic, default=0
            Base amplitude in dBFS. Use -∞ for silence.
        phase : dynamic, default=0.0
            Initial phase offset in radians (0-2π).
        phase_different_map : list, optional
            Per-channel phase variations for stereo effects.
        modulation_depth : dynamic, default=0.0
            Frequency modulation depth in Hz.
        modulation_freq : dynamic, default=0.0
            Modulation frequency in Hz.
        buffer_size : int, default=512
            Processing block size for real-time generation.

        Raises:
        -------
        ValueError
            For invalid frequency ranges or duration <=0
        TypeError
            For invalid keyword arguments or type mismatches

        Example:
        --------
        Generate sweeping FM tone:
        >>> ToneGenerator(2000, duration=3,
                        base_freq=[100, 1000],
                        modulation_depth=50,
                        modulation_freq=2)
        """
        if ekw(
            super().__init__, 
            **kwargs
            ):
            raise TypeError('Invalid keyword argument(s) for ToneGenerator')
        
        super().__init__(
            **fw(super().__init__, **kwargs)
            )

        if duration <= 0:
            raise ValueError("duration must be greater than 0")
               
        base_freq = base_freq  if  isinstance(base_freq, (int, float)) else _np.array(base_freq, dtype=_np.double)
        phase = phase if  isinstance(phase, (int, float)) else _np.array(phase, dtype=_np.double)
        modulation_depth = modulation_depth if  isinstance(modulation_depth, (int, float)) else _np.array(modulation_depth, dtype=_np.double)
        modulation_freq = modulation_freq if  isinstance(modulation_freq, (int, float)) else _np.array(modulation_freq, dtype=_np.double)
        amplitude = amplitude if  isinstance(amplitude, (int, float)) else _np.array(amplitude, dtype=_np.double)
        frequency_components_amplitude = frequency_components_amplitude if  isinstance(frequency_components_amplitude, (int, float)) else _np.array(frequency_components_amplitude, dtype=_np.double)

        frequency_components = []
        for fc in args:
            fc = fc if isinstance(fc, (int, float)) else _np.array(fc, dtype=_np.double)
            frequency_components.append(fc)
            
            if not _np.all((0 <= fc) & (fc <= self._sample_rate / 2)):
                raise ValueError(f"frequency component must be in the range 0 to {self._sample_rate / 2}")
            
        if not _np.all((0.0 <= modulation_depth) & (modulation_depth <= self._sample_rate / 2)):
            raise ValueError(f"modulation_depth must be in the range 0.0 to {self._sample_rate / 2}")
        if not _np.all((0 <= base_freq) & (base_freq <= self._sample_rate / 2)):
            raise ValueError(f"base_freq must be in the range 0 to {self._sample_rate / 2}")
        if not _np.all((0 <= modulation_freq) & (modulation_freq <= self._sample_rate / 2)):
            raise ValueError(f"modulation_freq must be in the range 0 to {self._sample_rate / 2}")
        if not _np.all((0.0 <= phase) & (phase <= 2.0 * _np.pi)):
            raise ValueError("phase must be in the range 0.0 to 2π")


        
        self._duration = duration
        self._size = int(self._duration * self._sample_rate)
        comargs = (buffer_size, 0.0, 1.0)

        self._base_freq = self.prepare(
            base_freq,
            base_freq_spline,
            base_freq_spline_sigma,
            *comargs
            ) + self._eps 
        self._modulation_freq = self.prepare(
            modulation_freq,
            modulation_freq_spline,
            modulation_freq_spline_sigma,
            *comargs
            )  + self._eps
        self._modulation_depth = self.prepare(
            modulation_depth,
            modulation_depth_spline,
            modulation_depth_spline_sigma,
            *comargs
            ) + self._eps 
        
        self._modulation_depth /= self._modulation_freq

        self._amplitude = self.prepare(
            db2amp(amplitude),
            amplitude_spline,
            amplitude_spline_sigma,
            *comargs
            ) + self._eps 
        self._phase = self.prepare(
            phase,
            phase_spline,
            phase_spline_sigma,
            *comargs
            ) + self._eps 

        for idx, fc in enumerate(frequency_components):
            frequency_components[idx] = self.prepare(
                fc,
                frequency_components_spline if isinstance(frequency_components_spline, bool) else frequency_components_spline[idx],
                frequency_components_spline_sigma if isinstance(frequency_components_spline_sigma, (int, float)) else frequency_components_spline_sigma[idx],
                *comargs
                ) + self._eps
            
        self._frequency_components = frequency_components

        if isinstance(frequency_components_amplitude, (int, float)):
            fca = db2amp(frequency_components_amplitude)
            frequency_components_amplitude = [_np.array([fca,], dtype=self._sample_type_descriptor),]
        else:
            new_component_amplitude = []
            for idx, fca in enumerate(frequency_components_amplitude):
                nca = self.prepare(
                    db2amp(fca),
                    frequency_components_spline if isinstance(frequency_components_spline, bool) else frequency_components_spline[idx],
                    frequency_components_spline_sigma if isinstance(frequency_components_spline_sigma, (int, float)) else frequency_components_spline_sigma[idx],
                    *comargs
                    ) + self._eps
                new_component_amplitude.append(nca)                
            frequency_components_amplitude = new_component_amplitude

        pdm = []
        if phase_different_map is not None:
            if len(phase_different_map) != self._nchannels:
                raise ValueError("phase_different_map must have the same length as the number of channels")
            for pdm_ in phase_different_map:
                pdm_ = pdm_ if isinstance(pdm_, (int, float)) else _np.array(pdm_, dtype=_np.double)
                pdm.append(self.prepare(
                    pdm_,
                    phase_spline,
                    phase_spline_sigma,
                    *comargs
                    ) + self._eps)

        self._phase_different_map = pdm
        self._frequency_components_amplitude= frequency_components_amplitude
        self._time_vector = 2.0 * _np.pi * _np.linspace(0, self._duration, self._size, dtype=self._sample_type_descriptor)
        # self._modulation_vector = 2 * _np.pi * self._modulation_freq * _np.linspace(0, self._duration, self._size)

    @type_check()
    def prepare(
        self, 
        arg: ETYPE, 
        arg_spline: bool,
        arg_spline_sigma: float | int,
        buffer_size: int,
        min: float,
        max: float,
        ) -> _np.ndarray | float | int | None:
        """
        Processes dynamic parameters into time-domain envelopes.

        Applies spline interpolation and resampling to create smooth parameter
        transitions. Converts static values into constant arrays.

        Parameters:
        -----------
        arg : dynamic
            Input parameter (scalar or array)
        arg_spline : bool
            Enable cubic spline smoothing
        arg_spline_sigma : float
            Spline filter strength (0.1=soft ~1.0=aggressive)
        buffer_size : int
            Minimum length for spline processing
        min : float
            Envelope minimum value (normalized)
        max : float
            Envelope maximum value (normalized)

        Returns:
        --------
        np.ndarray
            Processed parameter envelope ready for audio generation
        """
        if arg is None:
            return None
        elif isinstance(arg, (int, float)):
            return  _np.array([arg,], dtype=self._sample_type_descriptor)
        
        env = prepare_envelope(
            self._size,
            arg,
            arg_spline,
            arg_spline_sigma,
            max,
            min,
            buffer_size
            )
        return env.astype(self._sample_type_descriptor)
    
    def generate(self, *args, **kwargs):
        ...

        
            
            