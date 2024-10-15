/*
 -- W.T.A
 -- SUDIO (https://github.com/MrZahaki/sudio)
 -- The Audio Processing Platform
 -- Mail: mrzahaki@gmail.com
 -- Software license: "Apache License 2.0". 
 -- file codec.hpp
*/
#pragma once

#include <vector>
#include <string>
#include <functional>
#include <memory>



extern "C" {
#include <miniaudio.h>
}

namespace suio {

    enum class FileFormat {
        UNKNOWN,
        WAV,
        FLAC,
        VORBIS,
        MP3
    };

    struct AudioFileInfo {
        std::string name;
        FileFormat fileFormat;
        uint32_t nchannels;
        uint32_t sampleRate;
        ma_format sampleFormat;
        uint64_t numFrames;
        float duration;
    };

    class AudioCodec {
    public:
        static std::vector<uint8_t> decodeAudioFile(const std::string& filename,
                                                    ma_format outputFormat = ma_format_s16,
                                                    uint32_t nchannels = 2,
                                                    uint32_t sampleRate = 44100,
                                                    ma_dither_mode dither = ma_dither_mode_none);

        static std::vector<uint8_t> decodeVorbisFile(const std::string& filename,
                                                  ma_format format,
                                                  uint32_t nchannels,
                                                  uint32_t sampleRate);

        static uint64_t encodeWavFile(const std::string& filename,
                                const std::vector<uint8_t>& data,
                                ma_format format,
                                uint32_t nchannels,
                                uint32_t sampleRate);


        static uint64_t encodeMP3File(  const std::string& filename,
                                        const std::vector<uint8_t>& data,
                                        ma_format format,
                                        uint32_t nchannels,
                                        uint32_t sampleRate,
                                        int bitrate,
                                        int quality
                                        );                            

         static uint64_t encodeFlacFile(const std::string& filename,
                                    const std::vector<uint8_t>& data,
                                    ma_format format,
                                    uint32_t nchannels,
                                    uint32_t sampleRate,
                                    int compressionLevel
                                    );

        static uint64_t encodeVorbisFile(const std::string& filename,
                                      const std::vector<uint8_t>& data,
                                      ma_format format,
                                      uint32_t nchannels,
                                      uint32_t sampleRate,
                                      float quality
                                      );

        static AudioFileInfo getFileInfo(const std::string& filename);


        static std::unique_ptr<ma_decoder> initializeDecoder(const std::string& filename,
                                                            ma_format outputFormat,
                                                            uint32_t nchannels,
                                                            uint32_t sampleRate,
                                                            ma_dither_mode dither);

        static std::vector<uint8_t> readDecoderFrames(ma_decoder* decoder,
                                                    uint64_t framesToRead);

        class AudioFileStream {
        public:
            AudioFileStream(const std::string& filename,
                            ma_format outputFormat = ma_format_s16,
                            uint32_t nchannels = 2,
                            uint32_t sampleRate = 44100,
                            uint64_t framesToRead = 1024,
                            ma_dither_mode dither = ma_dither_mode_none,
                            uint64_t seekFrame = 0);

            ~AudioFileStream();

            std::vector<uint8_t> readFrames(uint64_t framesToRead = 0);

        private:
            std::unique_ptr<ma_decoder> m_decoder;
            uint64_t m_framesToRead;
            uint32_t m_nchannels;
            ma_format m_outputFormat;
        };
    };

}  // namespace suio