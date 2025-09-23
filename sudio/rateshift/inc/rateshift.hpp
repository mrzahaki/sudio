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

#pragma once

#include <samplerate.h>
#include <functional>
#include <vector>
#include <cstddef>

namespace rateshift {

enum class ConverterType {
  sinc_best,
  sinc_medium,
  sinc_fastest,
  zero_order_hold,
  linear
};

class Resampler {
public:
  Resampler(ConverterType converter_type, int channels);
  ~Resampler();
  
  std::vector<float> process(const std::vector<float>& input, double sr_ratio, bool end_of_input);
  void set_ratio(double new_ratio);
  void reset();

private:
  SRC_STATE* _state;
  int _converter_type;
  int _channels;
};

class CallbackResampler {
public:
  using callback_t = std::function<std::vector<float>()>;
  
  CallbackResampler(callback_t callback_func, double ratio, ConverterType converter_type, size_t channels);
  ~CallbackResampler();

  std::vector<float> read(size_t frames);
  void set_starting_ratio(double new_ratio);
  void reset();

private:
  SRC_STATE* _state;
  callback_t _callback;
  double _ratio;
  int _converter_type;
  size_t _channels;
};

std::vector<float> resample(const std::vector<float>& input, double sr_ratio, ConverterType converter_type, int channels);

}
