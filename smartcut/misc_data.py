from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class AudioExportSettings:
    codec: str
    channels: str | None = None
    bitrate: int | None = None
    sample_rate: int | None = None
    denoise: int = -1

@dataclass
class AudioExportInfo:
    output_tracks: list[AudioExportSettings | None] = field(default_factory=lambda: [])

@dataclass
class FadeInfo:
    """Information about fade-in and fade-out effects for a segment.

    Supports independent control of video and audio fades.
    When video_* or audio_* fields are None, no fade is applied to that media type.
    """
    video_fadein_duration: Fraction | None = None  # None means no video fade-in
    video_fadeout_duration: Fraction | None = None  # None means no video fade-out
    audio_fadein_duration: Fraction | None = None  # None means no audio fade-in
    audio_fadeout_duration: Fraction | None = None  # None means no audio fade-out

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
    # Where the segment this piece belongs to starts / ends at a join, i.e. its neighbour in the output file comes from
    # elsewhere in the source: its first piece's start, its last piece's end (whole GOPs in KEYFRAMES mode); None if
    # not at a join (fork addition; re-encoded audio gets a short fade there, see RecodeAudioCutter)
    join_start: Fraction | None = None
    join_end: Fraction | None = None
