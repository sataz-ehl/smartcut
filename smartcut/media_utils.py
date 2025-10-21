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

# --- Validation helpers centralizing media rules ---

def _normalize_video_codec_name(name: str) -> str:
    """Normalize user-provided video encoder name to canonical form.

    Accept common synonyms (e.g., 'h265' -> 'hevc').
    """
    if not name:
        return ""
    n = name.strip().lower()
    if n == 'h265':
        return 'hevc'
    return n

def validate_video_container_compat(encoder_name: str, container_ext: str) -> list[str]:
    """Validate video encoder vs. container compatibility.

    Returns a list of error strings if incompatible.
    """
    errors: list[str] = []
    enc = _normalize_video_codec_name(encoder_name)
    ext = container_ext.lower().lstrip('.')

    # H.264 in OGG is not a supported combination
    if enc == 'h264' and ext == 'ogg':
        errors.append("H.264 video codec is not supported in OGG containers")

    # H.265/HEVC not supported in MP3 or OGG
    if enc == 'hevc' and ext in ['mp3', 'ogg']:
        errors.append(f"H.265 video codec is not supported in {ext.upper()} containers")

    return errors

def infer_audio_codec_warnings(container_ext: str, has_any_audio: bool, has_mix_track: bool) -> list[str]:
    """Return informational warnings about implied audio codecs by container.

    This does not block export and is intended for user visibility.
    """
    warnings: list[str] = []
    ext = container_ext.lower().lstrip('.')

    if not has_any_audio:
        return warnings

    if ext == 'mp3':
        if has_mix_track:
            warnings.append("Mixed audio will be re-encoded to MP3 format")
    elif ext == 'ogg':
        warnings.append("Audio will be encoded using Vorbis or Opus codec")

    return warnings

def validate_audio_track_limits_for_container(container_ext: str, total_audio_tracks: int) -> list[str]:
    """Validate total audio track count for a given container.

    Returns list of error strings if over the limit.
    """
    errors: list[str] = []
    if total_audio_tracks <= 1:
        return errors

    ext = container_ext.lower().lstrip('.')
    single_track_formats = ['ogg', 'mp3', 'm4a', 'flac', 'wav']
    if ext in single_track_formats:
        errors.append(f"{ext.upper()} format can only have 1 audio track, but {total_audio_tracks} were selected")
    return errors
