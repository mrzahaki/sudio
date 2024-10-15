/*
 -- W.T.A
 -- SUDIO (https://github.com/MrZahaki/sudio)
 -- The Audio Processing Platform
 -- Mail: mrzahaki@gmail.com
 -- Software license: "Apache License 2.0". 
 -- file suiobind.cpp
*/
#include <sstream>
#include <iostream>
#include <iomanip>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "stdstream.hpp"
#include "codec.hpp"

namespace py = pybind11;

class PyAudioStreamIterator {
public:
    PyAudioStreamIterator(std::shared_ptr<suio::AudioCodec::AudioFileStream> stream, uint64_t frames_to_read)
        : m_stream(stream), m_frames_to_read(frames_to_read) {}

    py::bytes next() {
        auto data = m_stream->readFrames(m_frames_to_read);
        if (data.empty()) {
            throw py::stop_iteration();
        }
        return py::bytes(reinterpret_cast<char*>(data.data()), data.size());
    }

private:
    std::shared_ptr<suio::AudioCodec::AudioFileStream> m_stream;
    uint64_t m_frames_to_read;
};

std::string getSampleFormatName(ma_format format) {
    switch (format) {
        case ma_format_unknown: return "UNKNOWN";
        case ma_format_u8: return "UNSIGNED8";
        case ma_format_s16: return "SIGNED16";
        case ma_format_s24: return "SIGNED24";
        case ma_format_s32: return "SIGNED32";
        case ma_format_f32: return "FLOAT32";
        default: return "INVALID";
    }
}

PYBIND11_MODULE(suio, m) {


    py::enum_<suio::FileFormat>(
        m, 
        "FileFormat"
        )
        .value("UNKNOWN", suio::FileFormat::UNKNOWN)
        .value("WAV", suio::FileFormat::WAV)
        .value("FLAC", suio::FileFormat::FLAC)
        .value("VORBIS", suio::FileFormat::VORBIS)
        .value("MP3", suio::FileFormat::MP3);

    py::enum_<ma_format>(
        m, 
        "SampleFormat"
        )
        .value("UNKNOWN", ma_format_unknown)
        .value("UNSIGNED8", ma_format_u8)
        .value("SIGNED16", ma_format_s16)
        .value("SIGNED24", ma_format_s24)
        .value("SIGNED32", ma_format_s32)
        .value("FLOAT32", ma_format_f32);

    py::enum_<ma_dither_mode>(
        m, 
        "DitherMode"
        )
        .value("NONE", ma_dither_mode_none)
        .value("RECTANGLE", ma_dither_mode_rectangle)
        .value("TRIANGLE", ma_dither_mode_triangle);

    py::class_<suio::AudioFileInfo>(
        m, 
        "AudioFileInfo"
        )
        .def_readonly("name", &suio::AudioFileInfo::name)
        .def_readonly("file_format", &suio::AudioFileInfo::fileFormat)
        .def_readonly("nchannels", &suio::AudioFileInfo::nchannels)
        .def_readonly("sample_rate", &suio::AudioFileInfo::sampleRate)
        .def_readonly("sample_format", &suio::AudioFileInfo::sampleFormat)
        .def_readonly("num_frames", &suio::AudioFileInfo::numFrames)
        .def_readonly("duration", &suio::AudioFileInfo::duration)
        .def("__str__", [](const suio::AudioFileInfo &self) {
            std::stringstream ss;
            ss << "<AudioFileInfo: '" << self.name << "' " << self.nchannels << " ch, "
               << self.sampleRate << " Hz, " << getSampleFormatName(self.sampleFormat) << ", "
               << self.numFrames << " frames=" << std::fixed << std::setprecision(2) << self.duration << " sec.>";
            return py::str(ss.str());
        })
        .def("__repr__", [](const suio::AudioFileInfo &self) {
            std::stringstream ss;
            ss << "<AudioFileInfo: '" << self.name << "' " << self.nchannels << " ch, "
               << self.sampleRate << " Hz, " << getSampleFormatName(self.sampleFormat) << ", "
               << self.numFrames << " frames=" << std::fixed << std::setprecision(2) << self.duration << " sec.>";
            return py::str(ss.str());
        });

    py::module codec = m.def_submodule("codec", "Audio codec submodule");

    codec.def("decode_audio_file", [](const std::string& filename, ma_format outputFormat, uint32_t nchannels, uint32_t sampleRate, ma_dither_mode dither) {
        auto data = suio::AudioCodec::decodeAudioFile(filename, outputFormat, nchannels, sampleRate, dither);
        return py::bytes(reinterpret_cast<char*>(data.data()), data.size());
    }, 
    py::arg("filename"), 
    py::arg("output_format") = ma_format_s16, 
    py::arg("nchannels") = 2, 
    py::arg("sample_rate") = 44100, 
    py::arg("dither") = ma_dither_mode_none
    );

    codec.def("encode_wav_file", [](const std::string& filename, py::bytes data, ma_format format, uint32_t nchannels, uint32_t sampleRate) {
        py::buffer_info info(py::buffer(data).request());
        std::vector<uint8_t> vec(static_cast<uint8_t*>(info.ptr), static_cast<uint8_t*>(info.ptr) + info.size);
        suio::AudioCodec::encodeWavFile(filename, vec, format, nchannels, sampleRate);
    });

    codec.def("encode_mp3_file", [](const std::string& filename, py::bytes data, ma_format format, uint32_t nchannels, uint32_t sampleRate, int bitrate, int quality) {
        py::buffer_info info(py::buffer(data).request());
        std::vector<uint8_t> vec(static_cast<uint8_t*>(info.ptr), static_cast<uint8_t*>(info.ptr) + info.size);
        return suio::AudioCodec::encodeMP3File(filename, vec, format, nchannels, sampleRate, bitrate, quality);
    }, py::arg("filename"), py::arg("data"), py::arg("format"), py::arg("nchannels"), py::arg("sample_rate"), py::arg("bitrate") = 128, py::arg("quality") = 2);

    codec.def("encode_flac_file", [](const std::string& filename, py::bytes data, ma_format format, uint32_t nchannels, uint32_t sampleRate, int compressionLevel) {
        py::buffer_info info(py::buffer(data).request());
        std::vector<uint8_t> vec(static_cast<uint8_t*>(info.ptr), static_cast<uint8_t*>(info.ptr) + info.size);
        return suio::AudioCodec::encodeFlacFile(filename, vec, format, nchannels, sampleRate, compressionLevel);
    }, py::arg("filename"), py::arg("data"), py::arg("format"), py::arg("nchannels"), py::arg("sample_rate"), py::arg("compression_level") = 5);

    codec.def("encode_vorbis_file", [](const std::string& filename, py::bytes data, ma_format format, uint32_t nchannels, uint32_t sampleRate, float quality) {
        py::buffer_info info(py::buffer(data).request());
        std::vector<uint8_t> vec(static_cast<uint8_t*>(info.ptr), static_cast<uint8_t*>(info.ptr) + info.size);
        return suio::AudioCodec::encodeVorbisFile(filename, vec, format, nchannels, sampleRate, quality);
    }, py::arg("filename"), py::arg("data"), py::arg("format"), py::arg("nchannels"), py::arg("sample_rate"), py::arg("quality") = 0.4);


    codec.def(
        "get_file_info", 
        &suio::AudioCodec::getFileInfo
        );

    py::class_<suio::AudioCodec::AudioFileStream>(
        codec, 
        "AudioFileStream"
        )
        .def(py::init<const std::string&, ma_format, uint32_t, uint32_t, uint64_t, ma_dither_mode, uint64_t>(),
             py::arg("filename"),
             py::arg("output_format") = ma_format_s16,
             py::arg("nchannels") = 2,
             py::arg("sample_rate") = 44100,
             py::arg("frames_to_read") = 1024,
             py::arg("dither") = ma_dither_mode_none,
             py::arg("seek_frame") = 0)
        .def("read_frames", &suio::AudioCodec::AudioFileStream::readFrames,
             py::arg("frames_to_read") = 0);

    py::class_<PyAudioStreamIterator>(codec, "PyAudioStreamIterator")
        .def("__iter__", [](PyAudioStreamIterator &it) -> PyAudioStreamIterator& { return it; })
        .def("__next__", &PyAudioStreamIterator::next);

    codec.def("stream_audio_file", [](const std::string& filename,
                                      ma_format output_format,
                                      uint32_t nchannels,
                                      uint32_t sample_rate,
                                      uint64_t frames_to_read,
                                      ma_dither_mode dither,
                                      uint64_t seek_frame) {
        auto stream = std::make_shared<suio::AudioCodec::AudioFileStream>(
            filename, output_format, nchannels, sample_rate, frames_to_read, dither, seek_frame);

        return PyAudioStreamIterator(stream, frames_to_read);
    }, py::arg("filename"),
       py::arg("output_format") = ma_format_s16,
       py::arg("nchannels") = 2,
       py::arg("sample_rate") = 44100,
       py::arg("frames_to_read") = 1024,
       py::arg("dither") = ma_dither_mode_none,
       py::arg("seek_frame") = 0);

 

    py::class_<stdstream::AudioDeviceInfo>(
        m, 
        "AudioDeviceInfo"
        )
        .def_readonly("index", &stdstream::AudioDeviceInfo::index)
        .def_readonly("name", &stdstream::AudioDeviceInfo::name)
        .def_readonly("max_input_channels", &stdstream::AudioDeviceInfo::maxInputChannels)
        .def_readonly("max_output_channels", &stdstream::AudioDeviceInfo::maxOutputChannels)
        .def_readonly("default_sample_rate", &stdstream::AudioDeviceInfo::defaultSampleRate)
        .def_readonly("is_default_input", &stdstream::AudioDeviceInfo::isDefaultInput)
        .def_readonly("is_default_output", &stdstream::AudioDeviceInfo::isDefaultOutput)
        .def("__str__", [](const stdstream::AudioDeviceInfo &adi) {
            return "AudioDeviceInfo(index=" + std::to_string(adi.index) +
                ", name=" + adi.name +
                ", max_input_channels=" + std::to_string(adi.maxInputChannels) +
                ", max_output_channels=" + std::to_string(adi.maxOutputChannels) +
                ", default_sample_rate=" + std::to_string(adi.defaultSampleRate) +
                ", is_default_input=" + std::to_string(adi.isDefaultInput) +
                ", is_default_output=" + std::to_string(adi.isDefaultOutput) + ")";
        })
        .def("__repr__", [](const stdstream::AudioDeviceInfo &adi) {
            return "<AudioDeviceInfo(index=" + std::to_string(adi.index) +
                ", name='" + adi.name + "'" +
                ", max_input_channels=" + std::to_string(adi.maxInputChannels) +
                ", max_output_channels=" + std::to_string(adi.maxOutputChannels) +
                ", default_sample_rate=" + std::to_string(adi.defaultSampleRate) +
                ", is_default_input=" + std::to_string(adi.isDefaultInput) +
                ", is_default_output=" + std::to_string(adi.isDefaultOutput) + ")>";
        });


    py::class_<stdstream::AudioStream>(
        m, 
        "AudioStream"
        )
            .def(py::init<>())
            .def("open", [](stdstream::AudioStream& self, 
            py::object input_dev_index,
            py::object output_dev_index,
            py::object sample_rate,
            py::object format,
            py::object input_channels,
            py::object output_channels,
            py::object frames_per_buffer,
            py::object enable_input,
            py::object enable_output,
            py::object stream_flags,
            py::object input_callback,
            py::object output_callback) {
    
        const int inputDeviceIndex = input_dev_index.is_none() ? -1 : input_dev_index.cast<int>();
        const int outputDeviceIndex = output_dev_index.is_none() ? -1 : output_dev_index.cast<int>();
        const double sampleRate = sample_rate.is_none() ? 0 : sample_rate.cast<double>();
        const ma_format maFormat = format.is_none() ? ma_format_s16 : format.cast<ma_format>();
        const int inputChannels = input_channels.is_none() ? 0 : input_channels.cast<int>();
        const int outputChannels = output_channels.is_none() ? 0 : output_channels.cast<int>();
        const unsigned long framesPerBuffer = frames_per_buffer.is_none() ? paFramesPerBufferUnspecified : frames_per_buffer.cast<unsigned long>();
        const bool enableInput = enable_input.is_none() ? true : enable_input.cast<bool>();
        const bool enableOutput = enable_output.is_none() ? true : enable_output.cast<bool>();
        const PaStreamFlags streamFlags = stream_flags.is_none() ? paNoFlag : stream_flags.cast<PaStreamFlags>();

        PaSampleFormat paFormat = (
            maFormat == ma_format_u8 ? paUInt8 : (
                maFormat == ma_format_s16 ? paInt16 : (
                    maFormat == ma_format_s24 ? paInt24 : (
                        maFormat == ma_format_s32 ? paInt32 : paFloat32
                    )
                )
            )
        );
 
        stdstream::AudioStream::InputCallback cppInputCallback = nullptr;
        if (!input_callback.is_none()) {
                cppInputCallback = [&self, input_callback, maFormat](const char* inputBuffer, unsigned long framesPerBuffer, PaSampleFormat format) {
                    py::gil_scoped_acquire acquire;

                    size_t bytes_size = framesPerBuffer * Pa_GetSampleSize(format) * self.inputChannels;
                    try {
                        py::bytes input_bytes(reinterpret_cast<const char*>(inputBuffer), bytes_size);
                        py::object result = input_callback(input_bytes, framesPerBuffer, maFormat);
                        return result.cast<py::tuple>()[1].cast<bool>();
                    } catch (const std::exception& e) {
                        std::cerr << "Exception in input callback: " << e.what() << std::endl;
                        return false;
                    }
                };
            }

            stdstream::AudioStream::OutputCallback cppOutputCallback = nullptr;
            if (!output_callback.is_none()) {
                cppOutputCallback = [&self, output_callback, maFormat](char* outputBuffer, unsigned long framesPerBuffer, PaSampleFormat format) {
                    py::gil_scoped_acquire acquire;
                    try {
                        py::object result = output_callback(framesPerBuffer, maFormat);
                        if (py::isinstance<py::tuple>(result)) {
                            py::tuple tuple_result = result.cast<py::tuple>();
                            if (tuple_result.size() == 2) {
                                py::bytes output_bytes = tuple_result[0].cast<py::bytes>();
                                std::string output_str = output_bytes;
                                size_t expected_size = framesPerBuffer * Pa_GetSampleSize(format) * self.outputChannels;
                                if (output_str.size() == expected_size) {
                                    std::memcpy(outputBuffer, output_str.data(), output_str.size());
                                } else {
                                    throw std::runtime_error(
                                        "Returned bytes object has incorrect size, expected " + 
                                        std::to_string(expected_size) + 
                                        ", but got " + 
                                        std::to_string(output_str.size()) + 
                                        ", frames " + std::to_string(framesPerBuffer) +
                                        ", sample " + std::to_string(Pa_GetSampleSize(format)) +
                                        ", ochannels " + std::to_string(self.outputChannels)
                                        );
                                }
                                return tuple_result[1].cast<bool>();
                            }
                        }
                        throw std::runtime_error("Output callback should return a tuple (bytes, bool)");
                    } catch (const std::exception& e) {
                        py::print("Exception in output callback: ", e.what());
                        return false;
                    }
                };
            }
        

        self.open(inputDeviceIndex, outputDeviceIndex, sampleRate, paFormat, 
                inputChannels, outputChannels, framesPerBuffer, enableInput, enableOutput, streamFlags,
                cppInputCallback, cppOutputCallback);
                
    }, py::arg("input_dev_index") = py::none(),
    py::arg("output_dev_index") = py::none(),
    py::arg("sample_rate") = py::none(),
    py::arg("format") = py::none(),
    py::arg("input_channels") = py::none(),
    py::arg("output_channels") = py::none(),
    py::arg("frames_per_buffer") = py::none(),
    py::arg("enable_input") = py::none(),
    py::arg("enable_output") = py::none(),
    py::arg("stream_flags") = py::none(),
    py::arg("input_callback") = py::none(),
    py::arg("output_callback") = py::none())

    .def("start", &stdstream::AudioStream::start)
    .def("stop", &stdstream::AudioStream::stop)
    .def("close", &stdstream::AudioStream::close)
    .def_static("get_input_devices", []() {
        stdstream::AudioStream temp;
        return temp.getInputDevices();
    })
    .def_static("get_output_devices", []() {
        stdstream::AudioStream temp;
        return temp.getOutputDevices();
    })
    .def_static("get_default_input_device", []() {
        stdstream::AudioStream temp;
        return temp.getDefaultInputDevice();
    })
    .def_static("get_default_output_device", []() {
        stdstream::AudioStream temp;
        return temp.getDefaultOutputDevice();
    })
    .def_static("get_device_count", []() {
        stdstream::AudioStream temp;
        return temp.getDeviceCount();
    })
    .def_static("get_device_info_by_index", [](int index) {
        stdstream::AudioStream temp;
        return temp.getDeviceInfoByIndex(index);
    })
    
    .def("read_stream", [](stdstream::AudioStream &self, unsigned long frames) {
        try {
            size_t buffer_size = frames * Pa_GetSampleSize(self.streamFormat) * self.inputChannels;
            std::vector<uint8_t> buffer(buffer_size);
            long framesRead = self.readStream(buffer.data(), frames);
            return py::make_tuple(py::bytes(reinterpret_cast<char*>(buffer.data()), framesRead * Pa_GetSampleSize(self.streamFormat) * self.inputChannels), framesRead);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error in read_stream: ") + e.what());
        }
    })

    .def("write_stream", [](stdstream::AudioStream &self, py::bytes data) {
        try {

            py::buffer_info info(py::buffer(data).request());
            size_t frame_size = Pa_GetSampleSize(self.streamFormat) * self.outputChannels;
            return self.writeStream(static_cast<const uint8_t*>(info.ptr), frame_size? info.size / frame_size: 0);
            
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error in write_stream: ") + e.what());
        }
    })
    .def("get_stream_read_available", &stdstream::AudioStream::getStreamReadAvailable)
    .def("get_stream_write_available", &stdstream::AudioStream::getStreamWriteAvailable);


    m.def("write_to_default_output", [](py::bytes data, 
                                        ma_format format, py::object channels, py::object sample_rate) {
        // Convert py::bytes to std::vector<uint8_t>
        py::buffer_info info(py::buffer(data).request());
        std::vector<uint8_t> byteData(static_cast<uint8_t*>(info.ptr), static_cast<uint8_t*>(info.ptr) + info.size);

        PaSampleFormat paFormat = (
            format == ma_format_u8 ? paUInt8 : (
                format == ma_format_s16 ? paInt16 : (
                    format == ma_format_s24 ? paInt24 : (
                        format == ma_format_s32 ? paInt32 : paFloat32
                    )
                )
            )
        );

        int nChannels = channels.is_none() ? 0 : channels.cast<int>();
        double sampleRate = sample_rate.is_none() ? 0.0 : sample_rate.cast<double>();

        stdstream::writeToDefaultOutput(byteData, paFormat, nChannels, sampleRate);
    }, py::arg("data"), py::arg("format") = ma_format_f32, py::arg("channels") = py::none(), py::arg("sample_rate") = py::none());


    m.def("get_sample_size", [](ma_format format) {
        switch (format) {
            case ma_format_u8:
                return 1;
            case ma_format_s16:
                return 2;
            case ma_format_s24:
                return 3;
            case ma_format_s32:
            case ma_format_f32:
                return 4;
            default:
                throw std::runtime_error("Unknown sample format");
        }
    });

}