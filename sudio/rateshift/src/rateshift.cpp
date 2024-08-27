/*
 -- W.T.A
 -- SUDIO (https://github.com/MrZahaki/sudio)
 -- The Audio Processing Platform
 -- Mail: mrzahaki@gmail.com
 -- Software license: "Apache License 2.0". 
*/
#include "rateshift.hpp"
#include <stdexcept>
#include <cmath>

namespace rateshift {

namespace {
    void error_handler(int errnum) {
        if (errnum != 0) {
            throw std::runtime_error(src_strerror(errnum));
        }
    }

    int convert_type(ConverterType type) {
        switch (type) {
            case ConverterType::sinc_best: return SRC_SINC_BEST_QUALITY;
            case ConverterType::sinc_medium: return SRC_SINC_MEDIUM_QUALITY;
            case ConverterType::sinc_fastest: return SRC_SINC_FASTEST;
            case ConverterType::zero_order_hold: return SRC_ZERO_ORDER_HOLD;
            case ConverterType::linear: return SRC_LINEAR;
            default: throw std::invalid_argument("Invalid converter type");
        }
    }
}

Resampler::Resampler(ConverterType converter_type, int channels)
    : _converter_type(convert_type(converter_type)), _channels(channels) {
    int error;
    _state = src_new(_converter_type, _channels, &error);
    error_handler(error);
}

Resampler::~Resampler() {
    if (_state) {
        src_delete(_state);
    }
}

std::vector<float> Resampler::process(const std::vector<float>& input, double sr_ratio, bool end_of_input) {
    size_t input_frames = input.size() / _channels;
    size_t output_frames = static_cast<size_t>(std::ceil(input_frames * sr_ratio));
    std::vector<float> output(output_frames * _channels);

    SRC_DATA src_data = {
        const_cast<float*>(input.data()),
        const_cast<float*>(output.data()),
        static_cast<long>(input_frames),
        static_cast<long>(output_frames),
        0, 0,
        end_of_input ? 1 : 0,
        sr_ratio
    };

    error_handler(src_process(_state, &src_data));

    output.resize(src_data.output_frames_gen * _channels);
    return output;
}

void Resampler::set_ratio(double new_ratio) {
    error_handler(src_set_ratio(_state, new_ratio));
}

void Resampler::reset() {
    error_handler(src_reset(_state));
}

CallbackResampler::CallbackResampler(callback_t callback_func, double ratio, ConverterType converter_type, size_t channels)
    : _callback(std::move(callback_func)), _ratio(ratio), _converter_type(convert_type(converter_type)), _channels(channels) {
    int error;
    _state = src_callback_new(
        [](void* cb_data, float** data) -> long {
            auto* self = static_cast<CallbackResampler*>(cb_data);
            auto input = self->_callback();
            if (input.empty()) return 0;
            *data = input.data();
            return static_cast<long>(input.size() / self->_channels);
        },
        _converter_type,
        static_cast<int>(_channels),
        &error,
        this
    );
    error_handler(error);
}

CallbackResampler::~CallbackResampler() {
    if (_state) {
        src_delete(_state);
    }
}

std::vector<float> CallbackResampler::read(size_t frames) {
    std::vector<float> output(frames * _channels);
    long frames_read = src_callback_read(_state, _ratio, static_cast<long>(frames), output.data());
    
    if (frames_read == 0) {
        error_handler(src_error(_state));
    }
    
    output.resize(frames_read * _channels);
    return output;
}

void CallbackResampler::set_starting_ratio(double new_ratio) {
    error_handler(src_set_ratio(_state, new_ratio));
    _ratio = new_ratio;
}

void CallbackResampler::reset() {
    error_handler(src_reset(_state));
}

std::vector<float> resample(const std::vector<float>& input, double sr_ratio, ConverterType converter_type, int channels) {
    size_t input_frames = input.size() / channels;
    size_t output_frames = static_cast<size_t>(std::ceil(input_frames * sr_ratio));
    std::vector<float> output(output_frames * channels);

    SRC_DATA src_data = {
        const_cast<float*>(input.data()),
        static_cast<float *>(output.data()),
        static_cast<long>(input_frames),
        static_cast<long>(output_frames),
        0, 0, 0,
        sr_ratio
    };

    error_handler(src_simple(&src_data, convert_type(converter_type), channels));

    output.resize(src_data.output_frames_gen * channels);
    return output;
}

} 