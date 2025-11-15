from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class MixInfo:
    track_levels: list[float]

@dataclass
class AudioExportSettings:
    codec: str
    channels: str | None = None
    bitrate: int | None = None
    sample_rate: int | None = None
    denoise: int = -1

@dataclass
class AudioExportInfo:
    mix_info: MixInfo | None = None
    mix_export_settings: AudioExportSettings | None = None
    output_tracks: list[AudioExportSettings | None] = field(default_factory = lambda: [])

@dataclass
class FadeInfo:
    """Information about fade-in and fade-out effects for a segment."""
    fadein_duration: Fraction | None = None  # None means no fade-in
    fadeout_duration: Fraction | None = None  # None means no fade-out

@dataclass
class SegmentWithFade:
    """A time segment with optional fade-in/out effects."""
    start_time: Fraction
    end_time: Fraction
    fade_info: FadeInfo

@dataclass
class CutSegment:
    require_recode: bool
    start_time: Fraction
    end_time: Fraction
    gop_start_dts: int = -1
    gop_end_dts: int = -1
    gop_index: int = -1
    fade_info: FadeInfo | None = None  # Fade information for this segment
    orig_segment_start: Fraction | None = None  # Original segment start for fade calculation
    orig_segment_end: Fraction | None = None  # Original segment end for fade calculation
