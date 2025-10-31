from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from av import AudioStream, Packet, VideoStream
from av import open as av_open
from av.container.input import InputContainer
from av.stream import Stream

from smartcut.nal_tools import (
    get_h264_nal_unit_type,
    get_h265_nal_unit_type,
    is_safe_h264_keyframe_nal,
    is_safe_h265_keyframe_nal,
)


def ts_to_time(ts: float) -> Fraction:
    return Fraction(round(ts*1000), 1000)

AV_TIME_BASE: int = 1000000

@dataclass
class AudioTrack:
    media_container: "MediaContainer"
    av_stream: AudioStream
    path: str
    index: int

    packets: list[Packet] = field(default_factory = lambda: [])
    frame_times: np.ndarray | list[int] = field(default_factory = lambda: [])

    def start_time(self) -> Fraction:
        return self.media_container.start_time

class MediaContainer:
    av_container: InputContainer
    video_stream: VideoStream | None
    path: str

    video_frame_times: np.ndarray
    video_keyframe_indices: list[int]
    gop_start_times_pts_s: list[int] # Smallest pts in a GOP, in seconds

    gop_start_times_dts: list[int]
    gop_end_times_dts: list[int]
    gop_start_nal_types: list[int | None]  # NAL type of first picture frame after each GOP boundary

    audio_tracks: list[AudioTrack]
    subtitle_tracks: list

    duration: Fraction
    start_time: Fraction

    def __init__(self, path: str) -> None:
        self.path = path

        frame_pts = []
        self.video_keyframe_indices = []

        self.av_container = av_container = av_open(path, 'r', metadata_errors='ignore')

        self.chat_url = None
        self.chat_history = None
        self.chat_visualize = True
        self.start_time = Fraction(av_container.start_time, AV_TIME_BASE) if av_container.start_time is not None else Fraction(0)
        manual_duration_calc = av_container.duration is None
        self.duration = Fraction(av_container.duration , AV_TIME_BASE) if av_container.duration is not None else Fraction(0)

        is_h264 = False
        is_h265 = False

        streams: list[Stream]

        if len(av_container.streams.video) == 0:
            self.video_stream = None
            streams = [*av_container.streams.audio]
        else:
            self.video_stream = av_container.streams.video[0]
            assert self.video_stream is not None
            self.video_stream.thread_type = "FRAME"
            streams = [self.video_stream, *av_container.streams.audio]

            assert self.video_stream.codec_context is not None
            if self.video_stream.codec_context.name == 'hevc':
                is_h265 = True
            if self.video_stream.codec_context.name == 'h264':
                is_h264 = True

        self.audio_tracks = []
        stream_index_to_audio_track = {}
        for i, audio_stream in enumerate(av_container.streams.audio):
            audio_stream.codec_context.thread_type = "FRAME"
            track = AudioTrack(self, audio_stream, path, i)
            self.audio_tracks.append(track)
            stream_index_to_audio_track[audio_stream.index] = track

        self.subtitle_tracks = []
        stream_index_to_subtitle_track = {}
        for i, s in enumerate(av_container.streams.subtitles):
            streams.append(s)
            stream_index_to_subtitle_track[s.index] = i
            self.subtitle_tracks.append([])

        first_keyframe = True  # Always allow the first keyframe regardless of NAL type

        self.gop_start_times_dts = []
        self.gop_end_times_dts = []
        self.gop_start_nal_types = []
        last_seen_video_dts = None

        for packet in av_container.demux(streams):
            if packet.pts is None:
                continue

            if manual_duration_calc and (packet.pts is not None and packet.duration is not None):
                self.duration = max(self.duration, (packet.pts + packet.duration) * packet.time_base)
            if packet.stream.type == 'video' and self.video_stream:

                if packet.is_keyframe:
                    nal_type = None
                    if is_h265:
                        nal_type = get_h265_nal_unit_type(bytes(packet))
                    elif is_h264:
                        nal_type = get_h264_nal_unit_type(bytes(packet))

                    # Always allow the first keyframe regardless of NAL type (may be SEI, parameter sets, etc.)
                    is_safe_keyframe = True
                    if first_keyframe:
                        first_keyframe = False  # Only apply to the very first keyframe
                    # Use centralized helper functions for NAL type safety checks
                    elif is_h265:
                        is_safe_keyframe = is_safe_h265_keyframe_nal(nal_type)
                    elif is_h264:
                        is_safe_keyframe = is_safe_h264_keyframe_nal(nal_type)
                    if is_safe_keyframe:
                        self.video_keyframe_indices.append(len(frame_pts))
                        dts = packet.dts if packet.dts is not None else -100_000_000
                        self.gop_start_times_dts.append(dts)
                        self.gop_start_nal_types.append(nal_type)

                        if last_seen_video_dts is not None:
                            self.gop_end_times_dts.append(last_seen_video_dts)
                last_seen_video_dts = packet.dts
                frame_pts.append(packet.pts)
            elif packet.stream.type == 'audio':
                track = stream_index_to_audio_track[packet.stream_index]
                track.last_packet = packet

                # NOTE: storing the audio packets like this keeps the whole compressed audio loaded in RAM
                track.packets.append(packet)
                track.frame_times.append(packet.pts)
            elif packet.stream.type == 'subtitle':
                self.subtitle_tracks[stream_index_to_subtitle_track[packet.stream_index]].append(packet)

        if self.video_stream is not None:
            if last_seen_video_dts is not None:
                self.gop_end_times_dts.append(last_seen_video_dts)
            self.video_frame_times = np.sort(np.array(frame_pts)) * self.video_stream.time_base

            self.gop_start_times_pts_s = list(self.video_frame_times[self.video_keyframe_indices])

        for t in self.audio_tracks:
            frame_times = np.array(t.frame_times)
            t.frame_times = frame_times * t.av_stream.time_base

    def close(self):
        self.av_container.close()

    def get_next_frame_time(self, t):
        t += self.start_time
        idx = np.searchsorted(self.video_frame_times, t)
        if idx == len(self.video_frame_times):
            return self.duration
        elif idx == 0:
            return self.video_frame_times[0] - self.start_time
        # Otherwise, find the closest of the two possible candidates: arr[idx-1] and arr[idx]
        else:
            prev_val = self.video_frame_times[idx - 1]
            next_val = self.video_frame_times[idx]
            if t - prev_val <= next_val - t:
                return prev_val - self.start_time
            else:
                return next_val - self.start_time
