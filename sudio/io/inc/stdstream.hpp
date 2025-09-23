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

#include <portaudio.h>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <stdexcept>


# ifdef PA_ENABLE_DEBUG_OUTPUT
    # undef PA_ENABLE_DEBUG_OUTPUT
# endif

namespace stdstream {

// Base exception class for all audio-related errors
class AudioException : public std::runtime_error {
public:
    explicit AudioException(const std::string& message) : std::runtime_error(message) {}
};

// Initialization and setup errors
class AudioInitException : public AudioException {
public:
    explicit AudioInitException(const std::string& message) : AudioException(message) {}
};

// Device-related errors
class DeviceException : public AudioException {
public:
    explicit DeviceException(const std::string& message) : AudioException(message) {}
};

// Invalid parameter errors
class InvalidParameterException : public AudioException {
public:
    explicit InvalidParameterException(const std::string& message) : AudioException(message) {}
};

// Stream operation errors
class StreamException : public AudioException {
public:
    explicit StreamException(const std::string& message) : AudioException(message) {}
};

// Resource unavailable errors
class ResourceException : public AudioException {
public:
    explicit ResourceException(const std::string& message) : AudioException(message) {}
};


struct AudioDeviceInfo {
    int index;
    std::string name;
    int maxInputChannels;
    int maxOutputChannels;
    double defaultSampleRate;
    bool isDefaultInput;
    bool isDefaultOutput;
};

class AudioStream {
public:
    AudioStream();
    ~AudioStream();

    using InputCallback = std::function<bool(const char*, unsigned long, PaSampleFormat)>;
    using OutputCallback = std::function<bool(char*, unsigned long, PaSampleFormat)>;

    void open(int inputDeviceIndex = -1, int outputDeviceIndex = -1, 
              double sampleRate = 0, PaSampleFormat format = paFloat32, 
              int inputChannels = 0, int outputChannels = 0, 
              unsigned long framesPerBuffer = paFramesPerBufferUnspecified,
              bool enableInput = true, bool enableOutput = true,
              PaStreamFlags streamFlags = paNoFlag,
              InputCallback inputCallback = nullptr,
              OutputCallback outputCallback = nullptr);
    void start();
    void stop();
    void close();

    std::vector<AudioDeviceInfo> getInputDevices();
    std::vector<AudioDeviceInfo> getOutputDevices();
    AudioDeviceInfo getDefaultInputDevice();
    AudioDeviceInfo getDefaultOutputDevice();
    int getDeviceCount();
    AudioDeviceInfo getDeviceInfoByIndex(int index);

    long readStream(uint8_t* buffer, unsigned long frames);
    long writeStream(const uint8_t* buffer, unsigned long frames);
    long getStreamReadAvailable();
    long getStreamWriteAvailable();
    int outputChannels;
    int inputChannels;
    PaSampleFormat streamFormat;

private:
    PaStream* stream;
    std::atomic<bool> continueStreaming;
    InputCallback userInputCallback;
    OutputCallback userOutputCallback;
    static int paCallback(const void* inputBuffer, void* outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void* userData);
    int handleCallback(const void* inputBuffer, void* outputBuffer,
                       unsigned long framesPerBuffer,
                       const PaStreamCallbackTimeInfo* timeInfo,
                       PaStreamCallbackFlags statusFlags);
    bool isBlockingMode;
    bool inputEnabled;
    bool outputEnabled;
};

void writeToDefaultOutput(
    const std::vector<uint8_t>& data, 
    PaSampleFormat sampleFormat, 
    int channels, 
    double sampleRate,
    int outputDeviceIndex = -1
    );

} // namespace stdstream