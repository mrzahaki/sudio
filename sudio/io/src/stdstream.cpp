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

#include "stdstream.hpp"
#include "alsa_suppressor.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>


namespace stdstream {


inline std::string createErrorMessage(const std::string& context, PaError error) {
    std::stringstream ss;
    ss << context << ": " << Pa_GetErrorText(error);
    return ss.str();
}

AudioStream::AudioStream() : stream(nullptr), isBlockingMode(false), inputEnabled(false), outputEnabled(false) {
    suio::AlsaErrorSuppressor suppressor;
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw AudioInitException(createErrorMessage(" initialization failed", err));
    }
}

AudioStream::~AudioStream() {
    if (stream) {
        try {
            close();
        } catch (const AudioException& e) {
            // pass
        }
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
    double inputSampleRate, outputSampleRate;
    
    if (!enableInput && !enableOutput) {
        throw InvalidParameterException("At least one of input or output must be enabled");
    }

    inputEnabled = enableInput;
    outputEnabled = enableOutput;
    
    if (enableInput) {
        if (inputDeviceIndex == -1) {
            inputDeviceIndex = Pa_GetDefaultInputDevice();
            if (inputDeviceIndex == paNoDevice) {
                throw DeviceException("No default input device available");
            }
        } else if (inputDeviceIndex >= Pa_GetDeviceCount()) {
            throw InvalidParameterException("Invalid input device index: " + std::to_string(inputDeviceIndex));
        }

        const PaDeviceInfo* inputInfo = Pa_GetDeviceInfo(inputDeviceIndex);
        if (!inputInfo) {
            throw DeviceException("Failed to get input device info for index: " + std::to_string(inputDeviceIndex));
        }
        
        if (inputInfo->maxInputChannels == 0) {
            throw ResourceException("Selected device does not support input: " + std::string(inputInfo->name));
        }

        if (inputChannels > inputInfo->maxInputChannels) {
            throw InvalidParameterException("Requested input channels (" + 
                std::to_string(inputChannels) + ") exceed device maximum (" + 
                std::to_string(inputInfo->maxInputChannels) + ")");
        }

        inputParameters.device = inputDeviceIndex;
        inputParameters.channelCount = inputChannels > 0 ? inputChannels : inputInfo->maxInputChannels;
        inputParameters.sampleFormat = format;
        inputParameters.suggestedLatency = inputInfo->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;
        inputParamsPtr = &inputParameters;
        inputSampleRate = inputInfo->defaultSampleRate;
    }

    if (enableOutput) {
        if (outputDeviceIndex == -1) {
            outputDeviceIndex = Pa_GetDefaultOutputDevice();
            if (outputDeviceIndex == paNoDevice) {
                throw DeviceException("No default output device available");
            }
        } else if (outputDeviceIndex >= Pa_GetDeviceCount()) {
            throw InvalidParameterException("Invalid output device index: " + std::to_string(outputDeviceIndex));
        }

        const PaDeviceInfo* outputInfo = Pa_GetDeviceInfo(outputDeviceIndex);
        if (!outputInfo) {
            throw DeviceException("Failed to get output device info for index: " + std::to_string(outputDeviceIndex));
        }
        
        if (outputInfo->maxOutputChannels == 0) {
            throw ResourceException("Selected device does not support output: " + std::string(outputInfo->name));
        }

        if (outputChannels > outputInfo->maxOutputChannels) {
            throw InvalidParameterException("Requested output channels (" + 
                std::to_string(outputChannels) + ") exceed device maximum (" + 
                std::to_string(outputInfo->maxOutputChannels) + ")");
        }

        outputParameters.device = outputDeviceIndex;
        outputParameters.channelCount = outputChannels > 0 ? outputChannels : outputInfo->maxOutputChannels;
        outputParameters.sampleFormat = format;
        outputParameters.suggestedLatency = outputInfo->defaultHighOutputLatency;
        outputParameters.hostApiSpecificStreamInfo = nullptr;
        outputParamsPtr = &outputParameters;
        outputSampleRate = outputInfo->defaultSampleRate;

    }

    if (sampleRate < 0){
        throw InvalidParameterException("Invalid sample rate: " + std::to_string(sampleRate));
    }
    else if (sampleRate == 0) {
        sampleRate = enableInput ? inputSampleRate: outputSampleRate;
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
        throw StreamException(createErrorMessage("Failed to open  stream", err));
    }

    if (!stream) {
        throw StreamException("Failed to create a valid  stream");
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
        throw std::runtime_error("Failed to start  stream: " + std::string(Pa_GetErrorText(err)));
    }
}

void AudioStream::stop() {
    if (stream) {
        PaError err = Pa_StopStream(stream);
        if (err != paNoError) {
            std::cerr << "Warning: Failed to stop  stream: " << Pa_GetErrorText(err) << std::endl;
        }
    }
}

void AudioStream::close() {
    if (stream) {
        stop();
        PaError err = Pa_CloseStream(stream);
        if (err != paNoError) {
            std::cerr << "Warning: Failed to close  stream: " << Pa_GetErrorText(err) << std::endl;
        }
        stream = nullptr;
    }
}


std::vector<AudioDeviceInfo> AudioStream::getInputDevices() {
    std::vector<AudioDeviceInfo> devices;
    int numDevices = Pa_GetDeviceCount();
    if (numDevices <= 0) {
        throw DeviceException(createErrorMessage("Failed to get device count or no device", numDevices));
    }
    int defaultInputDevice = Pa_GetDefaultInputDevice();
    if (defaultInputDevice == paNoDevice) {
        throw DeviceException("No default input device");
    }

    for (int i = 0; i < numDevices; i++) {
        try {
            const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
            if (!deviceInfo) {
                throw DeviceException("Unable to get device info for index " + std::to_string(i));
            }

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
        } catch (const DeviceException& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
            continue;
        }
    }
    return devices;
}

std::vector<AudioDeviceInfo> AudioStream::getOutputDevices() {
    std::vector<AudioDeviceInfo> devices;
    int numDevices = Pa_GetDeviceCount();
    if (numDevices <= 0) {
        throw DeviceException(createErrorMessage("Failed to get device count or no device", numDevices));
    }
    int defaultOutputDevice = Pa_GetDefaultOutputDevice();
    if (defaultOutputDevice == paNoDevice) {
        throw DeviceException("No default output device");
    }
    
    for (int i = 0; i < numDevices; i++) {
        try {
            const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
            if (!deviceInfo) {
                throw DeviceException("Unable to get device info for index " + std::to_string(i));
            }

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
        } catch (const DeviceException& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
            continue;
        }
    }
    return devices;
}

AudioDeviceInfo AudioStream::getDefaultInputDevice() {
    int defaultInputDevice = Pa_GetDefaultInputDevice();
    if (defaultInputDevice == paNoDevice) {
        throw DeviceException("No default input device");
    }
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(defaultInputDevice);
    if (!deviceInfo) {
        throw DeviceException("Unable to get device info for index " + std::to_string(defaultInputDevice));
    }
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
    if (defaultOutputDevice == paNoDevice) {
        throw DeviceException("No default output device");
    }
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(defaultOutputDevice);
    if (!deviceInfo) {
        throw DeviceException("Unable to get device info for index " + std::to_string(defaultOutputDevice));
    }
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
        throw StreamException("Stream is not open");
    }
    else if (!isBlockingMode) {
        throw InvalidParameterException("Write operation is only available in blocking mode");
    }
    else if (!outputEnabled) {
        throw InvalidParameterException("Output is not enabled for this stream");
    }
    else if (!buffer) {
        throw InvalidParameterException("Invalid buffer pointer");
    }
    else if (frames == 0) {
        return 0;  // No frames to write
    }

    unsigned long framesWritten = 0;
    const uint8_t* currentBuffer = buffer;

    while (framesWritten < frames) {
        long availableFrames = Pa_GetStreamWriteAvailable(stream);

        if (availableFrames < 0) {
            throw StreamException(createErrorMessage("Error getting available write frames", availableFrames));
        }
        else if (availableFrames == 0) {
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
                throw StreamException(createErrorMessage("Error writing to stream", err));
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

    if(index < 0){
        throw std::range_error("Index is out of range");
    }
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(index);
    if (!deviceInfo) {
        throw DeviceException("Unable to get device info for index " + std::to_string(index));
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



void writeToDefaultOutput(
    const std::vector<uint8_t>& data, 
    PaSampleFormat sampleFormat, 
    int channels, 
    double sampleRate,
    int outputDeviceIndex
    ){

    suio::AlsaErrorSuppressor suppressor;
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw AudioInitException(createErrorMessage("initialization failed", err));
    }

    if(outputDeviceIndex < 0)
        outputDeviceIndex = Pa_GetDefaultOutputDevice();

    if (outputDeviceIndex == paNoDevice) {
        Pa_Terminate();
        throw DeviceException("No default input device available");
    }

    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(outputDeviceIndex);
    if (!deviceInfo) {
        Pa_Terminate();
        throw DeviceException("Failed to get device info");
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
        throw StreamException(createErrorMessage("Failed to open stream", err));
    }

    if (!stream) {
        throw StreamException("Failed to create a valid stream");
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        Pa_CloseStream(stream);
        Pa_Terminate();

        throw StreamException("Failed to start  stream: " + std::string(Pa_GetErrorText(err)));
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
        throw StreamException("Error stopping stream: " + std::string(Pa_GetErrorText(err)));
    }

    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        Pa_Terminate();
        throw StreamException("Error closing stream: " + std::string(Pa_GetErrorText(err)));
    }

    Pa_Terminate();
}

} // namespace stdstream