"""Media utilities and enums for Smart Media Cutter.

This module contains all media-related enums, codec compatibility helpers,
and low-level media knowledge. All media format details should be centralized here.
"""

from enum import Enum


class VideoExportMode(Enum):
    """Video export modes."""
    SMARTCUT = 1    # Recode only around cutpoints (fast and accurate)
    KEYFRAMES = 2   # Cut on keyframes (inaccurate timing, lossless, very fast)
    RECODE = 3      # Recode the whole video (slow)


class VideoExportQuality(Enum):
    """Video quality presets."""
    LOW = 1                    # "Low"
    NORMAL = 2                 # "Normal" 
    HIGH = 3                   # "High"
    INDISTINGUISHABLE = 4      # "Almost indistinguishable (Large file size)"
    NEAR_LOSSLESS = 5          # "Near lossless (Huge file size)"
    LOSSLESS = 6               # "Lossless (Extremely large file size)"


class VideoCodec(Enum):
    """Video codecs with PyAV-compatible string values."""
    COPY = "copy"     # No re-encoding
    H264 = "h264"     # H.264 encoding
    HEVC = "hevc"     # H.265/HEVC encoding
    VP9 = "vp9"       # VP9 encoding
    AV1 = "av1"       # AV1 encoding


class AudioCodec(Enum):
    """Audio codecs with PyAV-compatible string values."""
    LIBOPUS = "libopus"         # Opus codec
    LIBVORBIS = "libvorbis"     # Vorbis codec
    AAC = "aac"                 # AAC codec
    MP3 = "libmp3lame"          # MP3 codec (correct PyAV codec name)
    FLAC = "flac"               # FLAC lossless codec
    PCM_S16LE = "pcm_s16le"     # 16-bit PCM
    PCM_F32LE = "pcm_f32le"     # 32-bit float PCM
    PASSTHRU = "passthru"       # Pass through without re-encoding


class AudioChannels(Enum):
    """Audio channel configuration."""
    MONO = "mono"
    STEREO = "stereo"
    SURROUND_5_1 = "5.1"


def get_crf_for_quality(quality: VideoExportQuality) -> int:
    """Get CRF value for the selected quality preset.
    
    Args:
        quality: Video quality enum value
        
    Returns:
        CRF value (lower = higher quality)
    """
    crf_map = {
        VideoExportQuality.LOW: 23,
        VideoExportQuality.NORMAL: 18,
        VideoExportQuality.HIGH: 14,
        VideoExportQuality.INDISTINGUISHABLE: 8,
        VideoExportQuality.NEAR_LOSSLESS: 3,
        VideoExportQuality.LOSSLESS: 0
    }
    return crf_map.get(quality, 18)


def get_compatible_codec_for_format(user_codec: AudioCodec, file_extension: str) -> str:
    """Get compatible audio codec for the given file format.
    
    Args:
        user_codec: User's preferred codec
        file_extension: Output file extension (e.g., 'mp3', 'flac')
        
    Returns:
        Compatible codec string for PyAV
    """
    # Map extensions to required codecs for compatibility
    extension_codec_map = {
        'mp3': AudioCodec.MP3.value,
        'flac': AudioCodec.FLAC.value, 
        'ogg': AudioCodec.LIBOPUS.value,
        'wav': AudioCodec.PCM_S16LE.value,
        'm4a': AudioCodec.AAC.value,
        'ipod': AudioCodec.AAC.value,  # iPod format (alternative M4A name)
    }
    
    # If extension requires specific codec, use it
    if file_extension.lower() in extension_codec_map:
        return extension_codec_map[file_extension.lower()]
    
    # Otherwise use user's choice
    return user_codec.value


def get_audio_only_formats() -> list[str]:
    """Get list of audio-only container formats.
    
    Returns:
        List of file extensions that are audio-only
    """
    return ['mp3', 'flac', 'ogg', 'wav', 'm4a', 'ipod']


def is_audio_only_format(file_extension: str) -> bool:
    """Check if the given file extension is audio-only.
    
    Args:
        file_extension: File extension (with or without dot)
        
    Returns:
        True if format is audio-only
    """
    ext = file_extension.lower().lstrip('.')
    return ext in get_audio_only_formats()