/*
 -- W.T.A
 -- SUDIO (https://github.com/MrZahaki/sudio)
 -- The Audio Processing Platform
 -- Mail: mrzahaki@gmail.com
 -- Software license: "Apache License 2.0". 
 -- file stdstream.cpp
*/
#include "stdstream.hpp"
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>

namespace stdstream {

AudioStream::AudioStream() : stream(nullptr), isBlockingMode(false), inputEnabled(false), outputEnabled(false) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw std::runtime_error("PortAudio initialization failed: " + std::string(Pa_GetErrorText(err)));
    }
}

AudioStream::~AudioStream() {
    if (stream) {
        close();
    }
    Pa_Terminate();
}

void AudioStream::open(int inputDeviceIndex, int outputDeviceIndex, 
                       double sampleRate, PaSampleFormat format, 
                       int inputChannels, int outputChannels, 
                       unsigned long framesPerBuffer, bool enableInput, 
                       bool enableOutput, PaStreamFlags streamFlags,
                       InputCallback inputCallback,
                       OutputCallback outputCallback) {
    PaStreamParameters inputParameters, outputParameters;
    PaStreamParameters *inputParamsPtr = nullptr;
    PaStreamParameters *outputParamsPtr = nullptr;
    
    if (!enableInput && !enableOutput) {
        throw std::runtime_error("At least one of input or output must be enabled");
    }

    inputEnabled = enableInput;
    outputEnabled = enableOutput;
    
    if (enableInput) {
        if (inputDeviceIndex == -1) inputDeviceIndex = Pa_GetDefaultInputDevice();
        const PaDeviceInfo* inputInfo = Pa_GetDeviceInfo(inputDeviceIndex);
        inputParameters.device = inputDeviceIndex;
        inputParameters.channelCount = inputChannels > 0 ? inputChannels : inputInfo->maxInputChannels;
        inputParameters.sampleFormat = format;
        inputParameters.suggestedLatency = inputInfo->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;
        inputParamsPtr = &inputParameters;
    }

    if (enableOutput) {
        if (outputDeviceIndex == -1) outputDeviceIndex = Pa_GetDefaultOutputDevice();
        const PaDeviceInfo* outputInfo = Pa_GetDeviceInfo(outputDeviceIndex);
        outputParameters.device = outputDeviceIndex;
        outputParameters.channelCount = outputChannels > 0 ? outputChannels : outputInfo->maxOutputChannels;
        outputParameters.sampleFormat = format;
        outputParameters.suggestedLatency = outputInfo->defaultHighOutputLatency;
        outputParameters.hostApiSpecificStreamInfo = nullptr;
        outputParamsPtr = &outputParameters;
    }

    if (sampleRate == 0) {
        sampleRate = enableInput ? Pa_GetDeviceInfo(inputDeviceIndex)->defaultSampleRate :
                                   Pa_GetDeviceInfo(outputDeviceIndex)->defaultSampleRate;
    }

    userInputCallback = inputCallback;
    userOutputCallback = outputCallback;
    this->outputChannels =  outputParameters.channelCount;
    this->inputChannels = inputParameters.channelCount;


    isBlockingMode = !(inputCallback || outputCallback);
    PaStreamCallback *callbackPtr = isBlockingMode ? nullptr : &AudioStream::paCallback;

    PaError err = Pa_OpenStream(&stream, inputParamsPtr, outputParamsPtr, sampleRate, framesPerBuffer, 
                                streamFlags, callbackPtr, this);
    if (err != paNoError) {
        throw std::runtime_error("Failed to open PortAudio stream: " + std::string(Pa_GetErrorText(err)));
    }

    if (!stream) {
        throw std::runtime_error("Failed to create a valid PortAudio stream");
    }

    streamFormat = format;
    continueStreaming.store(true);
}

void AudioStream::start() {
    if (!stream) {
        throw std::runtime_error("Stream is not open");
    }
    PaError err = Pa_StartStream(stream);
    if (err != paNoError) {
        throw std::runtime_error("Failed to start PortAudio stream: " + std::string(Pa_GetErrorText(err)));
    }
}

void AudioStream::stop() {
    if (stream) {
        PaError err = Pa_StopStream(stream);
        if (err != paNoError) {
            std::cerr << "Warning: Failed to stop PortAudio stream: " << Pa_GetErrorText(err) << std::endl;
        }
    }
}

void AudioStream::close() {
    if (stream) {
        stop();
        PaError err = Pa_CloseStream(stream);
        if (err != paNoError) {
            std::cerr << "Warning: Failed to close PortAudio stream: " << Pa_GetErrorText(err) << std::endl;
        }
        stream = nullptr;
    }
}


std::vector<AudioDeviceInfo> AudioStream::getInputDevices() {
    std::vector<AudioDeviceInfo> devices;
    int numDevices = Pa_GetDeviceCount();
    int defaultInputDevice = Pa_GetDefaultInputDevice();

    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            devices.push_back({
                i,
                deviceInfo->name,
                deviceInfo->maxInputChannels,
                deviceInfo->maxOutputChannels,
                deviceInfo->defaultSampleRate,
                (i == defaultInputDevice),
                false
            });
        }
    }
    return devices;
}

std::vector<AudioDeviceInfo> AudioStream::getOutputDevices() {
    std::vector<AudioDeviceInfo> devices;
    int numDevices = Pa_GetDeviceCount();
    int defaultOutputDevice = Pa_GetDefaultOutputDevice();

    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxOutputChannels > 0) {
            devices.push_back({
                i,
                deviceInfo->name,
                deviceInfo->maxInputChannels,
                deviceInfo->maxOutputChannels,
                deviceInfo->defaultSampleRate,
                false,
                (i == defaultOutputDevice)
            });
        }
    }
    return devices;
}

AudioDeviceInfo AudioStream::getDefaultInputDevice() {
    int defaultInputDevice = Pa_GetDefaultInputDevice();
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(defaultInputDevice);
    return {
        defaultInputDevice,
        deviceInfo->name,
        deviceInfo->maxInputChannels,
        deviceInfo->maxOutputChannels,
        deviceInfo->defaultSampleRate,
        true,
        false
    };
}

AudioDeviceInfo AudioStream::getDefaultOutputDevice() {
    int defaultOutputDevice = Pa_GetDefaultOutputDevice();
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(defaultOutputDevice);
    return {
        defaultOutputDevice,
        deviceInfo->name,
        deviceInfo->maxInputChannels,
        deviceInfo->maxOutputChannels,
        deviceInfo->defaultSampleRate,
        false,
        true
    };
}


long AudioStream::readStream(uint8_t* buffer, unsigned long frames) {
    if (!stream) {
        throw std::runtime_error("Stream is not open");
    }
    if (!isBlockingMode) {
        throw std::runtime_error("Write operation is only available in blocking mode");
    }
    if (!outputEnabled) {
        throw std::runtime_error("Output is not enabled for this stream");
    }
    if (frames == 0) {
        return 0;  // No frames to write
    }
    if (!buffer) {
        throw std::runtime_error("Invalid buffer pointer");
    }
    
    PaError err = Pa_ReadStream(stream, buffer, frames);
    if (err != paNoError) {
        if (err == paOutputUnderflowed) {
            return 0;  // Indicate no frames were written
        } else {
            throw std::runtime_error("Error writing to stream: " + std::string(Pa_GetErrorText(err)));
        }
    }

    return frames;
}

long AudioStream::writeStream(const uint8_t* buffer, unsigned long frames) {
    if (!stream) {
        throw std::runtime_error("Stream is not open");
    }
    if (!isBlockingMode) {
        throw std::runtime_error("Write operation is only available in blocking mode");
    }
    if (!outputEnabled) {
        throw std::runtime_error("Output is not enabled for this stream");
    }
    if (frames == 0) {
        return 0;  // No frames to write
    }
    if (!buffer) {
        throw std::runtime_error("Invalid buffer pointer");
    }

    unsigned long framesWritten = 0;
    const uint8_t* currentBuffer = buffer;

    while (framesWritten < frames) {
        long availableFrames = Pa_GetStreamWriteAvailable(stream);

        if (availableFrames == 0) {
            // No space available, wait a bit
            Pa_Sleep(1);
            continue;
        }
        unsigned long framesToWrite = std::min(static_cast<unsigned long>(availableFrames), frames - framesWritten);
        
        PaError err = Pa_WriteStream(stream, currentBuffer, framesToWrite);
        if (err != paNoError) {
            if (err == paOutputUnderflowed) {
                std::cerr << "Warning: Output underflowed" << std::endl;
                // In case of underflow, we'll try to continue
                Pa_Sleep(1);
                continue;
            } else {
                throw std::runtime_error("Error writing to stream: " + std::string(Pa_GetErrorText(err)));
            }
        }

        framesWritten += framesToWrite;
        currentBuffer += framesToWrite * outputChannels * Pa_GetSampleSize(streamFormat);
    }

    return framesWritten;
}

long AudioStream::getStreamReadAvailable() {
    return Pa_GetStreamReadAvailable(stream);
}

long AudioStream::getStreamWriteAvailable() {
    return Pa_GetStreamWriteAvailable(stream);
}

int AudioStream::paCallback(const void* inputBuffer, void* outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void* userData) {
    AudioStream* stream = static_cast<AudioStream*>(userData);
    return stream->handleCallback(inputBuffer, outputBuffer, framesPerBuffer, timeInfo, statusFlags);
}


int AudioStream::handleCallback(const void* inputBuffer, void* outputBuffer,
                                unsigned long framesPerBuffer,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags) {
    bool shouldContinue = true;

    if (inputEnabled && userInputCallback) {
        shouldContinue = userInputCallback((const char*)(inputBuffer),
                                           framesPerBuffer, streamFormat);
    }

    if (shouldContinue && outputEnabled && userOutputCallback) {
        shouldContinue = userOutputCallback((char*)(outputBuffer),
                                            framesPerBuffer, streamFormat);
    }

    if (!shouldContinue) {
        continueStreaming.store(false);
    }

    return continueStreaming.load() ? paContinue : paComplete;
}


int AudioStream::getDeviceCount() {
    return Pa_GetDeviceCount();
}

AudioDeviceInfo AudioStream::getDeviceInfoByIndex(int index) {
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(index);
    if (!deviceInfo) {
        throw std::runtime_error("Invalid device index");
    }

    AudioDeviceInfo info;
    info.index = index;
    info.name = deviceInfo->name;
    info.maxInputChannels = deviceInfo->maxInputChannels;
    info.maxOutputChannels = deviceInfo->maxOutputChannels;
    info.defaultSampleRate = deviceInfo->defaultSampleRate;
    info.isDefaultInput = (index == Pa_GetDefaultInputDevice());
    info.isDefaultOutput = (index == Pa_GetDefaultOutputDevice());

    return info;
}



void writeToDefaultOutput(const std::vector<uint8_t>& data, PaSampleFormat sampleFormat, 
                          int channels, double sampleRate) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw std::runtime_error("PortAudio initialization failed: " + std::string(Pa_GetErrorText(err)));
    }

    int outputDeviceIndex = Pa_GetDefaultOutputDevice();
    if (outputDeviceIndex == paNoDevice) {
        Pa_Terminate();
        throw std::runtime_error("No default output device found");
    }

    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(outputDeviceIndex);
    if (!deviceInfo) {
        Pa_Terminate();
        throw std::runtime_error("Failed to get device info for default output device");
    }

    if (channels <= 0) channels = deviceInfo->maxOutputChannels;
    if (sampleRate <= 0) sampleRate = deviceInfo->defaultSampleRate;

    PaStreamParameters outputParameters;
    outputParameters.device = outputDeviceIndex;
    outputParameters.channelCount = channels;
    outputParameters.sampleFormat = sampleFormat;
    outputParameters.suggestedLatency = deviceInfo->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    PaStream* stream;
    err = Pa_OpenStream(&stream, nullptr, &outputParameters, sampleRate, paFramesPerBufferUnspecified, 
                        paClipOff, nullptr, nullptr);
    if (err != paNoError) {
        Pa_Terminate();
        throw std::runtime_error("Failed to open PortAudio stream: " + std::string(Pa_GetErrorText(err)));
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        Pa_CloseStream(stream);
        Pa_Terminate();
        throw std::runtime_error("Failed to start PortAudio stream: " + std::string(Pa_GetErrorText(err)));
    }

    const uint8_t* buffer = data.data();
    unsigned long totalFrames = data.size() / (channels * Pa_GetSampleSize(sampleFormat));
    unsigned long framesWritten = 0;

    while (framesWritten < totalFrames) {
        long availableFrames = Pa_GetStreamWriteAvailable(stream);
        if (availableFrames < 0) {
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            Pa_Terminate();
            throw std::runtime_error("Error getting available write frames: " + std::string(Pa_GetErrorText(availableFrames)));
        }

        unsigned long framesToWrite = std::min(static_cast<unsigned long>(availableFrames), totalFrames - framesWritten);
        if (framesToWrite == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        err = Pa_WriteStream(stream, buffer + framesWritten * channels * Pa_GetSampleSize(sampleFormat), framesToWrite);
        if (err != paNoError) {
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            Pa_Terminate();
            throw std::runtime_error("Error writing to stream: " + std::string(Pa_GetErrorText(err)));
        }

        framesWritten += framesToWrite;
    }

    err = Pa_StopStream(stream);
    if (err != paNoError) {
        Pa_CloseStream(stream);
        Pa_Terminate();
        throw std::runtime_error("Error stopping stream: " + std::string(Pa_GetErrorText(err)));
    }

    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        Pa_Terminate();
        throw std::runtime_error("Error closing stream: " + std::string(Pa_GetErrorText(err)));
    }

    Pa_Terminate();
}

} // namespace stdstream