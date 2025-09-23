/*
 * SUDIO - Audio Processing Platform
 * Copyright (C) 2024 Hossein Zahaki
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * - GitHub: https://github.com/MrZahaki/sudio
 */

#include "alsa_suppressor.hpp"

#if HAS_ALSA
    #include <alsa/asoundlib.h>
#endif

namespace suio {

#if HAS_ALSA

AlsaErrorSuppressor::AlsaErrorSuppressor() {
    original_handler = (error_handler_t)snd_lib_error_set_handler((snd_lib_error_handler_t)silent_error_handler);
}

AlsaErrorSuppressor::~AlsaErrorSuppressor() {
    snd_lib_error_set_handler((snd_lib_error_handler_t)original_handler);
}

void AlsaErrorSuppressor::silent_error_handler(const char* file, int line, 
                                             const char* function, int err, 
                                             const char* fmt, ...) {
}

#else // Non-Linux systems or no ALSA

AlsaErrorSuppressor::AlsaErrorSuppressor() {
}

AlsaErrorSuppressor::~AlsaErrorSuppressor() {
}

#endif

} // namespace suio