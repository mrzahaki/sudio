/*
 -- W.T.A
 -- SUDIO (https://github.com/MrZahaki/sudio)
 -- The Audio Processing Platform
 -- Mail: mrzahaki@gmail.com
 -- Software license: "Apache License 2.0". 
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
