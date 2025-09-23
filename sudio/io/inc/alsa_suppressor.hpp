
#pragma once

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

namespace suio {
    class AlsaErrorSuppressor {
    public:
        AlsaErrorSuppressor();
        ~AlsaErrorSuppressor();

    private:
        #ifdef __has_include
            #if __has_include(<alsa/asoundlib.h>)
                #define HAS_ALSA 1
                using error_handler_t = void(*)(const char*, int, const char*, int, const char*, ...);
                error_handler_t original_handler;
                
                static void silent_error_handler(const char* file, int line, const char* function, 
                                              int err, const char* fmt, ...) 
                                              __attribute__((format(printf, 5, 6)));
            #else
                #define HAS_ALSA 0
            #endif
        #else
            #define HAS_ALSA 0
        #endif
    };
}