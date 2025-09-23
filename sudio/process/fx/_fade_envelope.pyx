# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True


# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


import numpy as np
cimport numpy as np
cimport cython
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import sawtooth
from sudio.utils.math cimport db2amp


cpdef np.ndarray[double, ndim=1] generate_envelope(
    int envlen,
    FadePreset preset = FadePreset.SMOOTH_ENDS,
    object enable_spline = None,
    double spline_sigma = -1.0,
    double fade_max_db = 0.0,
    double fade_min_db = -60.0,
    double fade_attack = -1.0,
    double fade_release = -1.0,
    int buffer_size = 400,
    double sawtooth_freq = 37.7,
):
    cdef:
        int length = buffer_size if buffer_size > 0 else envlen
        double fade_max = db2amp(fade_max_db)
        double fade_min = db2amp(fade_min_db)
        np.ndarray[double, ndim=1] linear_array = np.linspace(fade_min, fade_max, length)
        np.ndarray[double, ndim=1] envelope
        double sigma = 0.1
        bint use_spline = False
        int fade_attack_samples, fade_release_samples
        int remained, r1, r2

    if fade_attack < 0:
        if preset == FadePreset.SMOOTH_ENDS:
            fade_attack = 0.03
        elif preset == FadePreset.BELL_CURVE:
            fade_attack = 0.2
        elif preset == FadePreset.KEEP_ATTACK_ONLY:
            fade_attack = 0.1
        elif preset == FadePreset.REMOVE_ATTACK:
            fade_attack = 0.0
        elif preset == FadePreset.SMOOTH_ATTACK:
            fade_attack = 0.0
        elif preset == FadePreset.SMOOTH_FADE_IN:
            fade_attack = 0.3
        elif preset == FadePreset.SMOOTH_FADE_OUT:
            fade_attack = 0.3
        elif preset == FadePreset.SMOOTH_RELEASE:
            fade_attack = 0.94
        else: # LINEAR_FADE_IN, LINEAR_FADE_OUT, PULSE, TREMORS, ZIGZAG_CUT
            fade_attack = 0.5

    if fade_release < 0:
        if preset == FadePreset.SMOOTH_ENDS:
            fade_release = 0.03
        elif preset == FadePreset.BELL_CURVE:
            fade_release = 0.3
        elif preset == FadePreset.KEEP_ATTACK_ONLY:
            fade_release = 0.3
        elif preset == FadePreset.REMOVE_ATTACK:
            fade_release = 0.7
        elif preset == FadePreset.SMOOTH_ATTACK:
            fade_release = 0.94
        elif preset == FadePreset.SMOOTH_FADE_IN:
            fade_release = 0.3
        elif preset == FadePreset.SMOOTH_FADE_OUT:
            fade_release = 0.3
        elif preset == FadePreset.SMOOTH_RELEASE:
            fade_release = 0.0
        else: 
            fade_release = 0.5

    fade_attack_samples = int(fade_attack * length)
    fade_release_samples = int(fade_release * length)
    assert (length - fade_attack_samples - fade_release_samples) >= 0, f"fade_attack + fade_release should be lower than or equal to one but {fade_attack + fade_release}"

    if preset == FadePreset.SMOOTH_ENDS:
        envelope = np.concatenate([
            np.linspace(fade_min, fade_max, fade_attack_samples),
            np.full(length - fade_attack_samples - fade_release_samples, fade_max),
            np.linspace(fade_max, fade_min, fade_release_samples)
        ])
        sigma = .01

    elif preset == FadePreset.BELL_CURVE:
        envelope = np.concatenate([
            np.full(fade_attack_samples, fade_max),
            np.linspace(fade_max, fade_min, length - fade_attack_samples - fade_release_samples),
            np.full(fade_release_samples, fade_min)
        ])
        use_spline = True

    elif preset == FadePreset.KEEP_ATTACK_ONLY:
        remained = length - fade_attack_samples - fade_release_samples
        r1 = int(remained * 0.15)
        r2 = int(remained * 0.3)
        envelope = np.concatenate([
            np.full(fade_attack_samples, fade_max),
            np.linspace(fade_max, 0.45, r1),
            np.linspace(0.45, 0.18, r2),
            np.linspace(0.18, fade_min, remained - r1 - r2),
            np.full(fade_release_samples, fade_min)
        ])

    elif preset == FadePreset.LINEAR_FADE_IN:
        envelope = linear_array

    elif preset == FadePreset.LINEAR_FADE_OUT:
        envelope = 1 - linear_array

    elif preset == FadePreset.PULSE:
        envelope = np.hstack((
                np.linspace(fade_min, fade_max, fade_attack_samples),
                np.full(length - fade_attack_samples - fade_release_samples, fade_max),
                np.linspace(fade_max, fade_min, fade_release_samples),
            ))
        sigma = .2
        use_spline = True

    elif preset == FadePreset.REMOVE_ATTACK:
        remained = length - fade_attack_samples - fade_release_samples
        r1 = int(remained * 0.3)
        r2 = int(remained * 0.6)
        envelope = np.concatenate([
            np.full(fade_attack_samples, fade_min),
            np.linspace(fade_min, 0.5, r1),
            np.linspace(0.5, 0.82, r2),
            np.linspace(0.82, fade_max, remained - r1 - r2),
            np.full(fade_release_samples, fade_max)
        ])


    elif preset == FadePreset.SMOOTH_ATTACK:
        envelope = np.concatenate([
            np.full(fade_attack_samples, fade_min),
            np.linspace(fade_min, fade_max, length - fade_attack_samples - fade_release_samples),
            np.full(fade_release_samples, fade_max)
        ])
        use_spline = True
        sigma = 0.13

    elif preset == FadePreset.SMOOTH_FADE_IN:
        envelope = np.concatenate([
            np.full(fade_attack_samples, fade_min),
            np.linspace(fade_min, fade_max, length - fade_attack_samples - fade_release_samples),
            np.full(fade_release_samples, fade_max)
        ])
        use_spline = True

    elif preset == FadePreset.SMOOTH_FADE_OUT:
        envelope = np.concatenate([
            np.full(fade_attack_samples, fade_max),
            np.linspace(fade_max, fade_min, length - fade_attack_samples - fade_release_samples),
            np.full(fade_release_samples, fade_min)
        ])
        use_spline = True

    elif preset == FadePreset.SMOOTH_RELEASE:
        envelope = np.concatenate([
            np.full(fade_attack_samples, fade_max),
            np.linspace(fade_max, fade_min, length - fade_attack_samples - fade_release_samples),
            np.full(fade_release_samples, fade_min)
        ])
        use_spline = True
        sigma = 0.13

    elif preset == FadePreset.TREMORS:
        envelope = (1 - sawtooth(linear_array * sawtooth_freq, 0.5)) / 2 * fade_max
        use_spline = True
        sigma = 0.01

    elif preset == FadePreset.ZIGZAG_CUT:
        envelope = (1 - sawtooth(linear_array * sawtooth_freq, 0.5)) / 2 * fade_max
        sigma = 0.02

    else:
        raise ValueError(f"Unknown preset: {preset}")

    use_spline = enable_spline if enable_spline is not None else use_spline
    if use_spline:
        sigma = spline_sigma if spline_sigma > 0 else sigma
        envelope = gaussian_filter1d(envelope, length * sigma)

    if envlen != length:
        x_new = np.linspace(fade_min, fade_max, envlen)
        envelope = interp1d(linear_array, envelope)(x_new)

    return envelope




cpdef np.ndarray[double, ndim=1] prepare_envelope(
    int envlen,
    np.ndarray[double, ndim=1] envelope,
    bint enable_spline = False,
    double spline_sigma = 0.1,
    double fade_max_db = 0.0,
    double fade_min_db = -80.0,
    int buffer_size = 400,
):
    cdef:
        int length = len(envelope)
        double fade_max = db2amp(fade_max_db)
        double fade_min = db2amp(fade_min_db)
        np.ndarray[double, ndim=1] linear_array = np.linspace(fade_min, fade_max, length)

    if enable_spline:
        if length < 100:
            length = buffer_size
            new_rep = np.linspace(fade_min, fade_max, length)
            envelope = interp1d(linear_array, envelope)(new_rep)
            linear_array = new_rep

        envelope = gaussian_filter1d(envelope, length * spline_sigma)

    if envlen != length:
        x_new = np.linspace(fade_min, fade_max, envlen)
        envelope = interp1d(linear_array, envelope)(x_new)

    return envelope

