/*
 -- W.T.A
 -- SUDIO (https://github.com/MrZahaki/sudio)
 -- The Audio Processing Platform
 -- Mail: mrzahaki@gmail.com
 -- Software license: "Apache License 2.0". 
 -- file stdstream.hpp
*/
#pragma once

#include <portaudio.h>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>

namespace stdstream {

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

void writeToDefaultOutput(const std::vector<uint8_t>& data, PaSampleFormat sampleFormat, 
                          int channels, double sampleRate);

} // namespace stdstream