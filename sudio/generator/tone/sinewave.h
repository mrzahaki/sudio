/*
 * SUDIO - Audio Processing Platform
 * Copyright (C) 2024 Hossein Zahaki
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * - GitHub: https://github.com/MrZahaki/sudio
 */

#include <vector>
#include <cmath>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

template<typename T>
T get_value_at(const T& val, size_t /*i*/) {
    return val;
}

template<typename T>
T get_value_at(const std::vector<T>& vec, size_t i) {
    if(vec.size() == 1) {
        return vec[0];
    }
    return vec[i];
}


template<typename AmplitudeT,
         typename BaseFreqT,
         typename ModDepthT,
         typename ModFreqT,
         typename PhaseT,
         typename T>
void integrate_harmonics(
    std::vector<T>& output,
    int32_t size,
    const AmplitudeT& amplitude,
    const BaseFreqT& base_freq,
    const ModDepthT& mod_depth,
    const ModFreqT& mod_freq,
    const std::vector<T>& time_vector,
    const PhaseT& phase,
    int32_t tonetype
) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] += get_value_at(amplitude, i) * std::sin(
            get_value_at(base_freq, i) * time_vector[i] +
            get_value_at(mod_depth, i) * std::sin(get_value_at(mod_freq, i) * time_vector[i]) +
            get_value_at(phase, i)
        );
    }

    #pragma omp parallel for collapse(2)
    for (int h = 1; h < tonetype; ++h) {
        for (int i = 0; i < size; ++i) {
            T harmonic_index = static_cast<T>(h * 2 + 1);
            T harmonic_time = time_vector[i] * harmonic_index;
            
            // atomic addition for safety
            #pragma omp atomic
            output[i] += get_value_at(amplitude, i) * std::sin(
                get_value_at(base_freq, i) * harmonic_time +
                get_value_at(mod_depth, i) * std::sin(get_value_at(mod_freq, i) * harmonic_time) +
                get_value_at(phase, i)
            ) / harmonic_index;
        }
    }
}

template<typename AmplitudeT,
         typename BaseFreqT,
         typename ModDepthT,
         typename ModFreqT,
         typename PhaseT,
         typename T>
void calculate_harmonics(
    std::vector<T>& output,
    int32_t size,
    const AmplitudeT& amplitude,
    const BaseFreqT& base_freq,
    const ModDepthT& mod_depth,
    const ModFreqT& mod_freq,
    const std::vector<T>& time_vector,
    const PhaseT& phase,
    int32_t tonetype
) {
    std::fill(output.begin(), output.end(), 0.0);
    integrate_harmonics(output, size, amplitude, base_freq, mod_depth, mod_freq, time_vector, phase, tonetype);
}


template <typename T>
std::vector<T> add2vector(const std::vector<T>& v1, const std::vector<T>& v2) {
    size_t v2size = v2.size();
    size_t v1size = v1.size();
    std::vector<T> result;

    if(v1size == 1){
        result.resize(v2size);
        for (size_t i = 0; i < v2size; i++) {
            result[i] = v1[0] + v2[i];
        }
    }
    else if(v2size == 1){
        result.resize(v1size);
        for (size_t i = 0; i < v1size; i++) {
            result[i] = v1[i] + v2[0];
        }
    }
    else{
        size_t size = std::min(v2size, v1size);
        result.resize(size);
        for (size_t i = 0; i < size; i++) {
            result[i] = v1[i] + v2[i];
        }

    }
    
    return result;
}
