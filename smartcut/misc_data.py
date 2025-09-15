from dataclasses import dataclass, field

@dataclass
class MixInfo():
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
class CutSegment:
    require_recode: bool
    start_time: int
    end_time: int
    gop_start_dts: int = -1
    gop_end_dts: int = -1
    gop_index: int = -1
