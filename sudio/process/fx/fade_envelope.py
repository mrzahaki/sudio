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


import numpy as np
from sudio.process.fx import FX
from sudio.io import SampleFormat
from sudio.process.fx._fade_envelope import generate_envelope, prepare_envelope_db
from sudio.process.fx._fade_envelope import FadePreset as FP
from enum import Enum
from typing import Union

class FadePreset(Enum):
    SMOOTH_ENDS = FP.SMOOTH_ENDS
    BELL_CURVE = FP.BELL_CURVE
    KEEP_ATTACK_ONLY = FP.KEEP_ATTACK_ONLY
    LINEAR_FADE_IN = FP.LINEAR_FADE_IN
    LINEAR_FADE_OUT = FP.LINEAR_FADE_OUT
    PULSE = FP.PULSE
    REMOVE_ATTACK = FP.REMOVE_ATTACK
    SMOOTH_ATTACK = FP.SMOOTH_ATTACK
    SMOOTH_FADE_IN = FP.SMOOTH_FADE_IN
    SMOOTH_FADE_OUT = FP.SMOOTH_FADE_OUT
    SMOOTH_RELEASE = FP.SMOOTH_RELEASE
    TREMORS = FP.TREMORS
    ZIGZAG_CUT = FP.ZIGZAG_CUT


class FadeEnvelope(FX):
    def __init__(self, *args, **kwargs) -> None:
        features = {
            'streaming_feature': True, 
            'offline_feature': True,
            'preferred_datatype': SampleFormat.FLOAT32
        }
        """
        Initialize the FadeEnvelope audio effect processor.

        This method configures the FadeEnvelope effect with specific processing features,
        setting up support for both streaming and offline audio processing.

        Parameters:
        -----------
        *args : Variable positional arguments
            Allows passing additional arguments to the parent FX class.

        **kwargs : Variable keyword arguments
            Additional configuration parameters for the effect.

        Features:
        ---------
        - Supports streaming audio processing
        - Supports offline audio processing
        - Prefers 32-bit floating-point audio format for high-precision dynamics manipulation

        Notes:
        ------
        The FadeEnvelope effect provides versatile amplitude shaping capabilities,
        enabling complex audio envelope transformations with minimal computational overhead.
        """
        super().__init__(*args, **kwargs, **features)


    def process(
        self, 
        data: np.ndarray, 
        preset: Union[FadePreset, np.ndarray] = FadePreset.SMOOTH_ENDS, 
        **kwargs
    ) -> np.ndarray:
        """
        Shape your audio's dynamics with customizable envelope effects!

        This method allows you to apply various envelope shapes to your audio signal, 
        transforming its amplitude characteristics with precision and creativity. 
        Whether you want to smooth out transitions, create pulsing effects, 
        or craft unique fade patterns, this method has you covered.

        Parameters:
        -----------
        data : numpy.ndarray
            Your input audio data. Can be a single channel or multi-channel array.
            The envelope will be applied across the last dimension of the array.

        preset : FadePreset or numpy.ndarray, optional
            Define how you want to shape your audio's amplitude:
            
            - If you choose a FadePreset (default: SMOOTH_ENDS):
            Select from predefined envelope shapes like smooth fades, 
            bell curves, pulse effects, tremors, and more. Each preset 
            offers a unique way to sculpt your sound.
            
            - If you provide a custom numpy array:
            Create your own bespoke envelope by passing in a custom amplitude array. 
            This gives you ultimate flexibility in sound design.

        Additional keyword arguments (optional):
        ----------------------------------------
        Customize envelope generation with these powerful parameters:

        Envelope Generation Parameters:
        - enable_spline : bool
        Smoothen your envelope with spline interpolation. Great for creating 
        more organic, natural-feeling transitions.

        - spline_sigma : float, default varies
        Control the smoothness of spline interpolation. Lower values create 
        sharper transitions, higher values create more gradual blends.

        - fade_max_db : float, default 0.0
        Set the maximum amplitude in decibels. Useful for controlling peak loudness.

        - fade_max_min_db : float, default -60.0
        Define the minimum amplitude in decibels. Helps create subtle or dramatic fades.

        - fade_attack : float, optional
        Specify the proportion of the audio dedicated to the attack phase. 
        Influences how quickly the sound reaches its peak volume.

        - fade_release : float, optional
        Set the proportion of the audio dedicated to the release phase. 
        Controls how the sound tapers off.

        - buffer_size : int, default 400
        Adjust the internal buffer size for envelope generation.

        - sawtooth_freq : float, default 37.7
        For presets involving sawtooth wave modulation, control the frequency 
        of the underlying oscillation.

        Returns:
        --------
        numpy.ndarray
            Your processed audio data with the envelope applied.
            Maintains the same shape and type as the input data.

        Examples:
        ---------
            >>> from sudio.process.fx import FadeEnvelope, FadePreset
            >>> su = sudio.Master()        
            >>> rec = su.add('file.mp3')
            >>> rec.afx(FadeEnvelope, preset=FadePreset.PULSE, start=0, stop=10)
        """
        if isinstance(preset, (np.ndarray, list, tuple)):
            
            envelope = prepare_envelope_db(
            data.shape[-1], 
            np.array(preset, dtype=np.double), 
            **kwargs
            )
        elif isinstance(preset, FadePreset):
            envelope = generate_envelope(
                data.shape[-1], 
                preset.value, 
                **kwargs
                )
        else:
            raise TypeError('Invalid preset')
        
        processed_data = data * envelope
        return processed_data

