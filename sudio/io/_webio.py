
# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio


from sudio.io import codec, SampleFormat
import base64

class WebAudioIO:
    """
    Web-based Audio I/O class for environments without native device support
    """
    @staticmethod
    def is_web_audio_supported() -> bool:
        try:
            from IPython.core.interactiveshell import InteractiveShell

            if InteractiveShell.initialized():
                InteractiveShell.instance()
            else:
                raise
        except:
            return False
        return True


    @staticmethod
    def play_audio_data(data: bytes, sample_format: SampleFormat, channels: int, frame_rate: int) -> bool:
        """
        Play audio data by creating an HTML5 audio element
        
        Parameters:
        -----------
        data : bytes
            Raw audio data
        sample_format : SampleFormat
            Sample format from sudio.SampleFormat
        channels : int
            Number of channels
        frame_rate : int
            Sample rate in Hz
            
        Returns:
        --------
        bool
            True if playback was successful
        """
        try:
            wav_data = codec.encode_to_mp3(
                data,
                format=sample_format,
                nchannels=channels,
                sample_rate=frame_rate
            )
            
            base64_data = base64.b64encode(wav_data).decode('ascii')
            audio_html = f"""
                <audio controls="controls" autoplay="autoplay">
                    <source src="data:audio/wav;base64,{base64_data}" type="audio/mp3" />
                    Your browser does not support the audio element.
                </audio>
            """
            
            from IPython.display import display, HTML
            display(HTML(audio_html))
            return True
            
        except:
            return False
        
        