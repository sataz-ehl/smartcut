import heapq
import os
import sys
from collections.abc import Generator
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, cast

import av
import av.bitstream
import numpy as np
from av import VideoCodecContext, VideoStream
from av.codec.context import CodecContext
from av.container.input import InputContainer
from av.container.output import OutputContainer
from av.packet import Packet
from av.stream import Disposition
from av.video.frame import PictureType, VideoFrame

from smartcut.media_container import MediaContainer
from smartcut.media_utils import VideoExportMode, VideoExportQuality, get_crf_for_quality
from smartcut.misc_data import AudioExportInfo, AudioExportSettings, CutSegment, FadeInfo, SegmentWithFade
from smartcut.nal_tools import convert_hevc_cra_to_bla


class ProgressCallback(Protocol):
    """Protocol for progress callback objects."""
    def emit(self, value: int) -> None:
        """Emit progress update."""
        ...


class CancelObject:
    cancelled: bool = False

@dataclass
class FrameHeapItem:
    """Wrapper for frames in the heap, sorted by PTS"""
    pts: int | None
    frame: VideoFrame

    def __lt__(self, other: 'FrameHeapItem') -> bool:
        # Handle None PTS values by treating them as -1 (earliest)
        self_pts = self.pts if self.pts is not None else -1
        other_pts = other.pts if other.pts is not None else -1
        return self_pts < other_pts

def is_annexb(packet: Packet | bytes | None) -> bool:
        if packet is None:
            return False
        data = bytes(packet)
        return data[:3] == b'\0\0\x01' or data[:4] == b'\0\0\0\x01'

def copy_packet(p: Packet) -> Packet:
    # return p
    packet = Packet(bytes(p))
    packet.pts = p.pts
    packet.dts = p.dts
    packet.duration = p.duration
    # packet.pos = p.pos
    packet.time_base = p.time_base
    packet.stream = p.stream
    packet.is_keyframe = p.is_keyframe
    for side_data in p.iter_sidedata():
        packet.set_sidedata(side_data)
    # packet.is_discard = p.is_discard

    return packet

def make_adjusted_segment_times(positive_segments: list[tuple[Fraction, Fraction]], media_container: MediaContainer) -> list[tuple[Fraction, Fraction]]:
    adjusted_segment_times = []
    EPSILON = Fraction(1, 1_000_000)
    for (s, e) in positive_segments:
        if s <= EPSILON:
            s = -10
        if e >= media_container.duration - EPSILON:
            e = media_container.duration + 10
        adjusted_segment_times.append((s + media_container.start_time, e + media_container.start_time))
    return adjusted_segment_times

def make_cut_segments(media_container: MediaContainer,
        positive_segments: list[tuple[Fraction, Fraction]],
        keyframe_mode: bool = False
        ) -> list[CutSegment]:
    cut_segments = []
    if media_container.video_stream is None:
        first_audio_track = media_container.audio_tracks[0]
        min_time = first_audio_track.frame_times[0]
        max_time = first_audio_track.frame_times[-1] + Fraction(1,10000)
        for p in positive_segments:
            s = max(p[0], min_time)
            e = min(p[1], max_time)
            while s + 20 < e:
                cut_segments.append(CutSegment(False, s, s + 19))
                s += 19
            cut_segments.append(CutSegment(False, s, e))
        return cut_segments

    source_cutpoints = [*media_container.gop_start_times_pts_s, media_container.start_time + media_container.duration + Fraction(1,10000)]
    p = 0
    for gop_idx, (i, o, i_dts, o_dts) in enumerate(zip(source_cutpoints[:-1], source_cutpoints[1:], media_container.gop_start_times_dts, media_container.gop_end_times_dts)):
        while p < len(positive_segments) and positive_segments[p][1] <= i:
            p += 1

        # Three cases: no overlap, complete overlap, and partial overlap
        if p == len(positive_segments) or o <= positive_segments[p][0]:
            pass
        elif keyframe_mode or (i >= positive_segments[p][0] and o <= positive_segments[p][1]):
            cut_segments.append(CutSegment(False, i, o, i_dts, o_dts, gop_idx))
        else:
            if i > positive_segments[p][0]:
                cut_segments.append(CutSegment(True, i, positive_segments[p][1], i_dts, o_dts, gop_idx))
                p += 1
            while p < len(positive_segments) and positive_segments[p][1] < o:
                cut_segments.append(CutSegment(True, positive_segments[p][0], positive_segments[p][1], i_dts, o_dts, gop_idx))
                p += 1
            if p < len(positive_segments) and positive_segments[p][0] < o:
                cut_segments.append(CutSegment(True, positive_segments[p][0], o, i_dts, o_dts, gop_idx))

    return cut_segments

class PassthruAudioCutter:
    def __init__(self, media_container: MediaContainer, output_av_container: OutputContainer,
                track_index: int, export_settings: AudioExportSettings) -> None:
        self.track = media_container.audio_tracks[track_index]

        self.out_stream = output_av_container.add_stream_from_template(self.track.av_stream, options={'x265-params': 'log_level=error'})

        self.out_stream.metadata.update(self.track.av_stream.metadata)
        self.out_stream.disposition = cast(Disposition, self.track.av_stream.disposition.value)
        self.segment_start_in_output = 0
        self.prev_dts = -100_000
        self.prev_pts = -100_000

    def segment(self, cut_segment: CutSegment) -> list[Packet]:
        in_tb = cast(Fraction, self.track.av_stream.time_base)
        if cut_segment.start_time <= 0:
            start = 0
        else:
            start_pts = round(cut_segment.start_time / in_tb)
            start = np.searchsorted(self.track.frame_times_pts, start_pts)
        end_pts = round(cut_segment.end_time / in_tb)
        end = np.searchsorted(self.track.frame_times_pts, end_pts)
        in_packets = self.track.packets[start : end]
        packets = []
        for p in in_packets:
            if p.dts is None or p.pts is None:
                continue
            packet = copy_packet(p)
            # packet = p
            packet.stream = self.out_stream
            packet.pts = int(p.pts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
            packet.dts = int(p.dts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
            if packet.pts <= self.prev_pts:
                print("Correcting for too low pts in audio passthru")
                packet.pts = self.prev_pts + 1
            if packet.dts <= self.prev_dts:
                print("Correcting for too low dts in audio passthru")
                packet.dts = self.prev_dts + 1
            self.prev_pts = packet.pts
            self.prev_dts = packet.dts
            packets.append(packet)

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
        return packets

    def finish(self) -> list[Packet]:
        return []

class RecodeAudioCutter:
    """Audio cutter that re-encodes audio with fade effects."""

    def __init__(self, media_container: MediaContainer, output_av_container: OutputContainer,
                track_index: int, export_settings: AudioExportSettings) -> None:
        self.track = media_container.audio_tracks[track_index]
        self.media_container = media_container

        # Create output stream
        self.out_stream = output_av_container.add_stream_from_template(self.track.av_stream, options={'x265-params': 'log_level=error'})
        self.out_stream.metadata.update(self.track.av_stream.metadata)
        self.out_stream.disposition = cast(Disposition, self.track.av_stream.disposition.value)

        self.segment_start_in_output = 0
        self.prev_dts = -100_000
        self.prev_pts = -100_000

        # Decoder
        self.decoder = self.track.av_stream.codec_context

        # Encoder will be created on demand
        self.encoder = None

    def _apply_audio_fade(self, audio_arr: np.ndarray, sample_idx: int, samples_per_second: int,
                         segment_start_sample: int, segment_end_sample: int, fade_info: FadeInfo | None) -> np.ndarray:
        """
        Apply fade-in/out to audio samples.

        Args:
            audio_arr: Audio array (samples x channels)
            sample_idx: Global sample index (start of this chunk)
            samples_per_second: Sample rate
            segment_start_sample: Start sample of segment
            segment_end_sample: End sample of segment
            fade_info: Fade information

        Returns:
            Audio array with fade applied

        Environment Variables:
            SMARTCUT_DEBUG_AUDIO_FADE: Set to '1' to enable debug output showing
                                       amplitude reduction during fades
        """
        if fade_info is None:
            return audio_arr

        # Debug mode: capture original amplitude for comparison
        debug_mode = os.environ.get('SMARTCUT_DEBUG_AUDIO_FADE') == '1'
        if debug_mode:
            orig_peak = np.max(np.abs(audio_arr))

        num_samples = audio_arr.shape[0]
        result = audio_arr.copy()

        for i in range(num_samples):
            current_sample = sample_idx + i
            alpha = 1.0

            # Fade-in
            if fade_info.fadein_duration is not None:
                fadein_samples = int(float(fade_info.fadein_duration) * samples_per_second)
                samples_from_start = current_sample - segment_start_sample
                if samples_from_start < fadein_samples:
                    alpha = min(1.0, samples_from_start / fadein_samples)

            # Fade-out
            if fade_info.fadeout_duration is not None:
                fadeout_samples = int(float(fade_info.fadeout_duration) * samples_per_second)
                samples_from_end = segment_end_sample - current_sample
                if samples_from_end < fadeout_samples:
                    fadeout_alpha = min(1.0, samples_from_end / fadeout_samples)
                    alpha = min(alpha, fadeout_alpha)

            # Apply fade
            if alpha < 1.0:
                result[i] = result[i] * alpha

        # Debug output: show amplitude reduction
        if debug_mode:
            result_peak = np.max(np.abs(result))
            reduction_pct = (1 - result_peak / orig_peak) * 100 if orig_peak > 0 else 0
            time_in_segment = (sample_idx - segment_start_sample) / samples_per_second
            print(f"[AUDIO_FADE] t={time_in_segment:.3f}s | fade_info={fade_info} | "
                  f"orig_peak={orig_peak:.6f} → result_peak={result_peak:.6f} "
                  f"({reduction_pct:.1f}% reduction)", file=sys.stderr)

        return result

    def segment(self, cut_segment: CutSegment) -> list[Packet]:
        """Process a segment with optional fade effects."""
        in_tb = cast(Fraction, self.track.av_stream.time_base)

        # Calculate sample indices
        sample_rate = self.track.av_stream.codec_context.sample_rate
        if sample_rate is None:
            sample_rate = 48000  # Default

        start_sample = int(float(cut_segment.start_time) * sample_rate) if cut_segment.start_time > 0 else 0
        end_sample = int(float(cut_segment.end_time) * sample_rate)

        # Check if we need to recode for fades
        needs_fade = (cut_segment.fade_info is not None and
                     (cut_segment.fade_info.fadein_duration is not None or
                      cut_segment.fade_info.fadeout_duration is not None))

        if not needs_fade:
            # No fade - use passthrough
            if cut_segment.start_time <= 0:
                start = 0
            else:
                start_pts = round(cut_segment.start_time / in_tb)
                start = np.searchsorted(self.track.frame_times_pts, start_pts)
            end_pts = round(cut_segment.end_time / in_tb)
            end = np.searchsorted(self.track.frame_times_pts, end_pts)
            in_packets = self.track.packets[start : end]
            packets = []
            for p in in_packets:
                if p.dts is None or p.pts is None:
                    continue
                packet = copy_packet(p)
                packet.stream = self.out_stream
                packet.pts = int(p.pts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
                packet.dts = int(p.dts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
                if packet.pts <= self.prev_pts:
                    packet.pts = self.prev_pts + 1
                if packet.dts <= self.prev_dts:
                    packet.dts = self.prev_dts + 1
                self.prev_pts = packet.pts
                self.prev_dts = packet.dts
                packets.append(packet)

            self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
            return packets

        # Recode with fade effects
        # Initialize encoder if needed
        if self.encoder is None:
            # Get channel layout
            in_layout = self.track.av_stream.codec_context.layout
            if in_layout is None:
                # Default to stereo if no layout specified
                in_layout = 'stereo' if self.track.av_stream.codec_context.channels == 2 else 'mono'

            self.encoder = av.codec.CodecContext.create(
                self.track.av_stream.codec_context.name,
                'w'
            )
            self.encoder.sample_rate = sample_rate
            self.encoder.layout = in_layout
            self.encoder.format = self.track.av_stream.codec_context.format
            self.encoder.time_base = in_tb

            # Match source bitrate to maintain quality consistency
            if self.track.av_stream.bit_rate:
                self.encoder.bit_rate = self.track.av_stream.bit_rate

        # Decode, apply fade, and re-encode
        packets = []
        current_sample = start_sample

        # Get packets for this segment
        if cut_segment.start_time <= 0:
            start_pkt_idx = 0
        else:
            start_pts = round(cut_segment.start_time / in_tb)
            start_pkt_idx = np.searchsorted(self.track.frame_times_pts, start_pts)
        end_pts = round(cut_segment.end_time / in_tb)
        end_pkt_idx = np.searchsorted(self.track.frame_times_pts, end_pts)

        for pkt in self.track.packets[start_pkt_idx:end_pkt_idx]:
            for audio_frame in self.decoder.decode(pkt):
                # Convert to numpy array
                audio_arr = audio_frame.to_ndarray()

                # Apply fade
                if cut_segment.fade_info is not None:
                    audio_arr = self._apply_audio_fade(
                        audio_arr, current_sample, sample_rate,
                        start_sample, end_sample, cut_segment.fade_info
                    )

                # Create new frame from modified array
                new_frame = av.AudioFrame.from_ndarray(audio_arr, format=audio_frame.format.name, layout=audio_frame.layout.name)
                new_frame.sample_rate = audio_frame.sample_rate
                new_frame.pts = int(current_sample - start_sample + self.segment_start_in_output * sample_rate)

                # Encode
                for encoded_pkt in self.encoder.encode(new_frame):
                    encoded_pkt.stream = self.out_stream
                    if encoded_pkt.pts <= self.prev_pts:
                        encoded_pkt.pts = self.prev_pts + 1
                    if encoded_pkt.dts is not None and encoded_pkt.dts <= self.prev_dts:
                        encoded_pkt.dts = self.prev_dts + 1
                    self.prev_pts = encoded_pkt.pts
                    if encoded_pkt.dts is not None:
                        self.prev_dts = encoded_pkt.dts
                    packets.append(encoded_pkt)

                current_sample += audio_arr.shape[0]

        # Flush encoder after this recoded segment to prevent stale state
        # when mixing recoded and passthrough segments
        if self.encoder is not None:
            for pkt in self.encoder.encode(None):
                pkt.stream = self.out_stream
                if pkt.pts <= self.prev_pts:
                    pkt.pts = self.prev_pts + 1
                if pkt.dts is not None and pkt.dts <= self.prev_dts:
                    pkt.dts = self.prev_dts + 1
                self.prev_pts = pkt.pts
                if pkt.dts is not None:
                    self.prev_dts = pkt.dts
                packets.append(pkt)
            # Reset encoder so it's recreated fresh for next recoded segment
            self.encoder = None

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
        return packets

    def finish(self) -> list[Packet]:
        """Flush encoder if still active (should rarely happen with per-segment flushing)."""
        packets = []
        if self.encoder is not None:
            for pkt in self.encoder.encode(None):
                pkt.stream = self.out_stream
                if pkt.pts <= self.prev_pts:
                    pkt.pts = self.prev_pts + 1
                if pkt.dts is not None and pkt.dts <= self.prev_dts:
                    pkt.dts = self.prev_dts + 1
                self.prev_pts = pkt.pts
                if pkt.dts is not None:
                    self.prev_dts = pkt.dts
                packets.append(pkt)
        return packets

class SubtitleCutter:
    def __init__(self, media_container: MediaContainer, output_av_container: OutputContainer, subtitle_track_index: int) -> None:
        self.track_i = subtitle_track_index
        self.packets = media_container.subtitle_tracks[subtitle_track_index]

        self.in_stream = media_container.av_container.streams.subtitles[subtitle_track_index]
        self.out_stream = output_av_container.add_stream_from_template(self.in_stream)
        self.out_stream.metadata.update(self.in_stream.metadata)
        self.out_stream.disposition = cast(Disposition, self.in_stream.disposition.value)
        self.segment_start_in_output = 0
        self.prev_pts = -100_000

        self.current_packet_i = 0

    def segment(self, cut_segment: CutSegment) -> list[Packet]:
        in_tb = cast(Fraction, self.in_stream.time_base)
        segment_start_pts = int(cut_segment.start_time / in_tb)
        segment_end_pts = int(cut_segment.end_time / in_tb)

        out_packets = []

        # TODO: This is the simplest implementation of subtitle cutting. Investigate more complex logic.
        # We include subtitles for the whole original time if the subtitle start time is included in the output
        # Good: simple, Bad: 1) if start is cut it's not shown at all 2) we can show a subtitle for too long if there is cut after it's shown
        while self.current_packet_i < len(self.packets):
            p = self.packets[self.current_packet_i]
            if p.pts < segment_start_pts:
                self.current_packet_i += 1
            elif p.pts >= segment_start_pts and p.pts < segment_end_pts:
                out_packets.append(p)
                self.current_packet_i += 1
            else:
                break

        for packet in out_packets:
            packet.stream = self.out_stream
            packet.pts = int(packet.pts - segment_start_pts + self.segment_start_in_output / in_tb)

            if packet.pts < self.prev_pts:
                print("Correcting for too low pts in subtitle passthru. This should not happen.")
                packet.pts = self.prev_pts + 1
            packet.dts = packet.pts
            self.prev_pts = packet.pts
            self.prev_dts = packet.dts

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
        return out_packets

    def finish(self) -> list[Packet]:
        return []



@dataclass
class VideoSettings:
    mode: VideoExportMode
    quality: VideoExportQuality
    codec_override: str = 'copy'

class VideoCutter:
    def __init__(self, media_container: MediaContainer, output_av_container: OutputContainer, video_settings: VideoSettings, log_level: str | None) -> None:
        self.media_container = media_container
        self.log_level = log_level
        self.encoder_inited = False
        self.video_settings = video_settings

        self.enc_codec = None

        self.in_stream = cast(VideoStream, media_container.video_stream)
        # Assert time_base is not None once at initialization
        assert self.in_stream.time_base is not None, "Video stream must have a time_base"
        self.in_time_base: Fraction = self.in_stream.time_base

        # Open another container because seeking to beginning of the file is unreliable...
        self.input_av_container: InputContainer = av.open(media_container.path, 'r', metadata_errors='ignore')

        self.demux_iter = self.input_av_container.demux(self.in_stream)
        self.demux_saved_packet = None

        # Frame buffering for fetch_frame (using heap for efficient PTS ordering)
        self.frame_buffer = []
        self.frame_buffer_gop_dts = -1
        self.decoder = self.in_stream.codec_context

        if video_settings.mode == VideoExportMode.RECODE and video_settings.codec_override != 'copy':
            self.out_stream = cast(VideoStream, output_av_container.add_stream(video_settings.codec_override, rate=self.in_stream.guessed_rate, options={'x265-params': 'log_level=error'}))
            self.out_stream.width = self.in_stream.width
            self.out_stream.height = self.in_stream.height
            if self.in_stream.sample_aspect_ratio is not None:
                self.out_stream.sample_aspect_ratio = self.in_stream.sample_aspect_ratio
            self.out_stream.metadata.update(self.in_stream.metadata)
            self.out_stream.disposition = cast(Disposition, self.in_stream.disposition.value)
            self.out_stream.time_base = self.in_time_base
            self.codec_name = video_settings.codec_override

            self.init_encoder()
            self.enc_codec = self.out_stream.codec_context
            self.enc_codec.options.update(self.encoding_options)
            self.enc_codec.time_base = self.in_time_base
            self.enc_codec.thread_type = "FRAME"
            self.enc_last_pts = -1
        else:
            # Map codec name for decoder to encoder compatibility (AV1 case)
            original_codec_name = self.in_stream.codec_context.name

            codec_mapping = {
                'libdav1d': 'libaom-av1',  # AV1 decoder to encoder
            }

            mapped_codec_name = codec_mapping.get(original_codec_name, original_codec_name)

            if mapped_codec_name != original_codec_name:
                # Need to create stream with mapped codec name. Can't use copy from template b/c codec name has changed
                self.out_stream = cast(VideoStream, output_av_container.add_stream(mapped_codec_name, rate=self.in_stream.guessed_rate))
                self.out_stream.width = self.in_stream.width
                self.out_stream.height = self.in_stream.height
                if self.in_stream.sample_aspect_ratio is not None:
                    self.out_stream.sample_aspect_ratio = self.in_stream.sample_aspect_ratio
                self.out_stream.metadata.update(self.in_stream.metadata)
                self.out_stream.disposition = cast(Disposition, self.in_stream.disposition.value)
                self.out_stream.time_base = self.in_stream.time_base
                self.codec_name = mapped_codec_name
            else:
                # Copy the stream if no mapping needed
                self.out_stream = output_av_container.add_stream_from_template(self.in_stream, options={'x265-params': 'log_level=error'})
                self.out_stream.metadata.update(self.in_stream.metadata)
                self.out_stream.disposition = cast(Disposition, self.in_stream.disposition.value)
                self.out_stream.time_base = self.in_stream.time_base
                self.codec_name = original_codec_name


            # if self.codec_name == 'mpeg2video':
                # self.out_stream.average_rate = self.in_stream.average_rate
                # self.out_stream.base_rate = self.in_stream.base_rate

            self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('null', self.in_stream, self.out_stream)
            if self.in_stream.codec_context.name == 'h264' and not is_annexb(self.in_stream.codec_context.extradata):
                self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('h264_mp4toannexb', self.in_stream, self.out_stream)
            elif self.in_stream.codec_context.name == 'hevc' and not is_annexb(self.in_stream.codec_context.extradata):
                self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('hevc_mp4toannexb', self.in_stream, self.out_stream)
            # MPEG-4 Visual family: optional filters for robustness (ASF/AVI tend to need this)
            elif self.in_stream.codec_context.name in {'mpeg4', 'msmpeg4v3', 'msmpeg4v2', 'msmpeg4v1'}:
                self.remux_bitstream_filter = av.bitstream.BitStreamFilterContext('dump_extra', self.in_stream, self.out_stream)

        self._normalize_output_codec_tag(output_av_container)

        # Assert out_stream time_base is not None once at initialization
        assert self.out_stream.time_base is not None, "Output stream must have a time_base"
        self.out_time_base: Fraction = self.out_stream.time_base

        self.last_dts = -100_000_000

        self.segment_start_in_output = 0

        # Track stream continuity for CRA to BLA conversion
        self.last_remuxed_segment_gop_index = None
        self.is_first_remuxed_segment = True
        # Track decoder continuity between GOPs: last GOP end DTS consumed
        self._last_fetch_end_dts: int | None = None

    def _normalize_output_codec_tag(self, output_av_container: OutputContainer) -> None:
        """Ensure codec tags are compatible with MP4/MOV style containers."""

        in_codec_ctx = self.in_stream.codec_context
        codec_name = in_codec_ctx.name
        container_name = output_av_container.format.name.lower() if output_av_container.format.name else ''

        if not any(name in container_name for name in ('mp4', 'mov', 'matroska', 'webm')):
            return

        out_codec_ctx = cast(CodecContext, self.out_stream.codec_context)
        if codec_name == 'h264' and self._is_mpegts_h264_tag(out_codec_ctx.codec_tag):
            out_codec_ctx.codec_tag = 'avc1'
        elif codec_name in ('hevc', 'h265') and self._is_mpegts_hevc_tag(out_codec_ctx.codec_tag):
            out_codec_ctx.codec_tag = 'hvc1'

    @staticmethod
    def _is_mpegts_h264_tag(codec_tag: int | bytes | str) -> bool:
        if isinstance(codec_tag, int):
            return codec_tag == 27
        if isinstance(codec_tag, bytes):
            return codec_tag == b'\x1b\x00\x00\x00'
        if isinstance(codec_tag, str):
            return codec_tag == '\x1b\x00\x00\x00'
        return False

    @staticmethod
    def _is_mpegts_hevc_tag(codec_tag: int | bytes | str) -> bool:
        if isinstance(codec_tag, int):
            return codec_tag == 36
        if isinstance(codec_tag, bytes):
            return codec_tag in (b'HEVC', b'\x24\x00\x00\x00')
        if isinstance(codec_tag, str):
            return codec_tag in ('HEVC', '\x24\x00\x00\x00')
        return False

    def init_encoder(self) -> None:
        self.encoder_inited = True
        # v_codec = self.in_stream.codec_context
        profile = self.out_stream.codec_context.profile

        codec_name = self.codec_name or ''
        if 'av1' in codec_name:
            self.codec_name = 'av1'
            profile = None
        if self.codec_name == 'vp9':
            if profile is not None:
                profile = profile[-1:]
                if int(profile) > 1:
                    raise ValueError("VP9 Profile 2 and Profile 3 are not supported by the encoder. Please select cutting on keyframes mode.")
        elif profile is not None:
            if 'Baseline' in profile:
                profile = 'baseline'
            elif 'High 4:4:4' in profile:
                profile = 'high444'
            elif 'Rext' in profile or 'Simple' in profile: # This is some sort of h265 extension. This might be the source of some issues I've had?
                profile = None
            else:
                profile = profile.lower().replace(':', '').replace(' ', '')

        # Get CRF value for quality setting
        crf_value = get_crf_for_quality(self.video_settings.quality)

        # Adjust CRF for newer codecs that are more efficient
        if self.codec_name in ['hevc', 'av1', 'vp9']:
            crf_value += 4
        if self.video_settings.quality == VideoExportQuality.LOSSLESS:
            crf_value = 0

        self.encoding_options = {'crf': str(crf_value)}
        if self.codec_name == 'vp9' and self.video_settings.quality == VideoExportQuality.LOSSLESS:
            self.encoding_options['lossless'] = '1'
        # encoding_options = {}
        if profile is not None:
            self.encoding_options['profile'] = profile

        if self.codec_name == 'h264':
            # sps-id = 3. We try to avoid collisions with the existing SPS ids.
            # Particularly 0 is very commonly used. Technically we should probably try
            # to dynamically set this to a safe number, but it can be difficult to know
            # our detection is robust / correct.
            self.encoding_options['x264-params'] = 'sps-id=3'

        elif self.codec_name == 'hevc':
            # Get the encoder settings from input stream extradata.
            # In theory this should not work. The stuff in extradata is technically just comments set by the encoder.
            # Another issue is that the extradata format is going to be different depending on the encoder.
            # So this will likely only work if the input stream is encoded with x265 ¯\_(ツ)_/¯
            # However, this does make the testcases from fails -> passes.
            # And I've tested that it works on some real videos as well.
            # Maybe there is some option that I'm not setting correctly and there is a better way to get the correct value?

            assert self.in_stream is not None
            assert self.in_stream.codec_context is not None
            extradata = self.in_stream.codec_context.extradata
            x265_params = []
            try:
                if extradata is None:
                    raise ValueError("No extradata")
                options_str = str(extradata.split(b'options: ')[1][:-1], 'ascii')
                x265_params = options_str.split(' ')
                for i, o in enumerate(x265_params):
                    if ':' in o:
                        x265_params[i] = o.replace(':', ',')
                    if '=' not in o:
                        x265_params[i] = o + '=1'
            except Exception:
                pass

            # Repeat headers. This should be the same as `global_headers = False`,
            # but for some reason setting this explicitly is necessary with x265.
            x265_params.append('repeat-headers=1')

            if self.log_level is not None:
                x265_params.append(f'log_level={self.log_level}')

            if self.video_settings.quality == VideoExportQuality.LOSSLESS:
                x265_params.append('lossless=1')

            self.encoding_options['x265-params'] = ':'.join(x265_params)


    def _fix_packet_timestamps(self, packet: Packet) -> None:
        """Fix packet DTS/PTS to ensure monotonic increase and PTS >= DTS."""
        packet.stream = self.out_stream
        packet.time_base = self.out_time_base
        if packet.dts is not None:
            if packet.dts <= self.last_dts:
                packet.dts = self.last_dts + 1
                # Ensure PTS >= DTS (required by all container formats)
                if packet.pts is not None and packet.pts < packet.dts:
                    packet.pts = packet.dts
            self.last_dts = packet.dts
        else:
            # When DTS is None, use PTS as fallback (common for keyframes without B-frame reordering)
            # Ensure we don't use the sentinel value to avoid extremely negative DTS
            pts_value = packet.pts if packet.pts is not None else 0
            if self.last_dts < 0:
                # First packet with None DTS, use PTS
                packet.dts = pts_value
            else:
                # Subsequent packets, ensure monotonic increase
                packet.dts = max(pts_value, self.last_dts + 1)
            self.last_dts = packet.dts

    def segment(self, cut_segment: CutSegment) -> list[Packet]:
        if cut_segment.require_recode:
            packets = self.recode_segment(cut_segment)
        else:
            packets = self.flush_encoder()
            packets.extend(self.remux_segment(cut_segment))
            # Update tracking variables for CRA to BLA conversion
            self.last_remuxed_segment_gop_index = cut_segment.gop_index
            self.is_first_remuxed_segment = False

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time

        for packet in packets:
            self._fix_packet_timestamps(packet)
        return packets

    def finish(self) -> list[Packet]:
        packets = self.flush_encoder()
        for packet in packets:
            self._fix_packet_timestamps(packet)

        self.input_av_container.close()

        return packets

    def _apply_fade_to_frame(self, frame: VideoFrame, frame_time: Fraction, segment_start: Fraction,
                            segment_end: Fraction, fade_info: FadeInfo | None) -> VideoFrame:
        """
        Apply fade-in/out effects to a frame.

        Args:
            frame: The video frame to apply fade to
            frame_time: The absolute time of this frame in the segment
            segment_start: Start time of the segment
            segment_end: End time of the segment
            fade_info: Fade information (None if no fade)

        Returns:
            The frame with fade applied (may modify in place)
        """
        if fade_info is None:
            return frame

        # Save original frame attributes
        original_pts = frame.pts
        original_time_base = frame.time_base
        original_pix_fmt = frame.format.name

        # Convert frame to numpy array for processing
        arr = frame.to_ndarray(format='rgb24')
        alpha = 1.0

        # Calculate relative time within segment
        relative_time = frame_time - segment_start
        segment_duration = segment_end - segment_start

        # Apply fade-in
        if fade_info.fadein_duration is not None and relative_time < fade_info.fadein_duration:
            alpha = min(1.0, float(relative_time / fade_info.fadein_duration))

        # Apply fade-out
        if fade_info.fadeout_duration is not None:
            time_from_end = segment_duration - relative_time
            if time_from_end < fade_info.fadeout_duration:
                fadeout_alpha = min(1.0, float(time_from_end / fade_info.fadeout_duration))
                alpha = min(alpha, fadeout_alpha)

        # Apply alpha to frame
        if alpha < 1.0:
            arr = (arr * alpha).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
            # Convert back to original pixel format to match encoder expectations
            # This avoids implicit conversions that can cause playback stuttering
            frame = frame.reformat(format=original_pix_fmt)
            # Restore original attributes
            frame.pts = original_pts
            if original_time_base is not None:
                frame.time_base = original_time_base

        return frame

    def recode_segment(self, s: CutSegment) -> list[Packet]:
        if not self.encoder_inited:
            self.init_encoder()
        result_packets = []

        if self.enc_codec is None:
            muxing_codec = self.out_stream.codec_context
            enc_codec = cast(VideoCodecContext, CodecContext.create(self.codec_name, 'w'))

            if muxing_codec.rate is not None:
                enc_codec.rate = muxing_codec.rate
            enc_codec.options.update(self.encoding_options)

            enc_codec.width = muxing_codec.width
            enc_codec.height = muxing_codec.height
            enc_codec.pix_fmt = muxing_codec.pix_fmt

            if muxing_codec.sample_aspect_ratio is not None:
                enc_codec.sample_aspect_ratio = muxing_codec.sample_aspect_ratio
            if self.codec_name == 'mpeg2video':
                enc_codec.time_base = Fraction(1, muxing_codec.rate)
            else:
                enc_codec.time_base = self.out_time_base
            #enc_codec.flags = muxing_codec.flags # This was here, but it's a bit sus. Disabling doesn't break any tests
            #enc_codec.flags ^= Flags.global_header # either doesn't help or doesn't work

            if muxing_codec.bit_rate is not None:
                enc_codec.bit_rate = muxing_codec.bit_rate
            if muxing_codec.bit_rate_tolerance is not None:
                enc_codec.bit_rate_tolerance = muxing_codec.bit_rate_tolerance
            enc_codec.codec_tag = muxing_codec.codec_tag
            enc_codec.thread_type = "FRAME"
            self.enc_last_pts = -1
            self.enc_codec = enc_codec

        # Determine if we should prime decoder from previous GOP (HEVC CRA at GOP start)
        start_override = None
        if (
            self.codec_name == 'hevc'
            and s.gop_index > 0
            and self.media_container.gop_start_nal_types[s.gop_index] == 21
        ):
            start_override = self.media_container.gop_start_times_dts[s.gop_index - 1]

        for frame in self.fetch_frame(s.gop_start_dts, s.gop_end_dts, s.end_time, start_override):
            assert frame.pts is not None, "Frame pts should not be None after decoding"
            in_tb = frame.time_base if frame.time_base is not None else self.in_time_base
            frame_abs_time = frame.pts * in_tb

            if frame_abs_time < s.start_time:
                continue
            if frame_abs_time >= s.end_time:
                break

            # Apply fade effects if specified
            if s.fade_info is not None and (s.fade_info.fadein_duration is not None or s.fade_info.fadeout_duration is not None):
                frame = self._apply_fade_to_frame(frame, frame_abs_time, s.start_time, s.end_time, s.fade_info)

            out_tb = self.out_time_base if self.codec_name != 'mpeg2video' else self.enc_codec.time_base

            frame.pts = int(frame.pts - s.start_time / in_tb)

            frame.pts = int(frame.pts * in_tb / out_tb)
            frame.time_base = out_tb
            frame.pts = int(frame.pts + self.segment_start_in_output / out_tb)

            if frame.pts <= self.enc_last_pts:
                frame.pts = int(self.enc_last_pts + 1)
            self.enc_last_pts = frame.pts

            frame.pict_type = PictureType.NONE
            result_packets.extend(self.enc_codec.encode(frame))

        if self.codec_name == 'mpeg2video':
            for p in result_packets:
                p.pts = p.pts * p.time_base / self.out_time_base
                p.dts = p.dts * p.time_base / self.out_time_base
                p.time_base = self.out_time_base
        return result_packets

    def remux_segment(self, s: CutSegment) -> list[Packet]:
        result_packets = []
        segment_start_pts = int(s.start_time / self.in_time_base)

        # Check if we need CRA to BLA conversion for this segment
        should_convert_cra = self._should_convert_cra_for_segment(s)
        first_packet = True

        for packet in self.fetch_packet(s.gop_start_dts, s.gop_end_dts):

            # Apply CRA to BLA conversion only to the first packet if needed
            if first_packet and should_convert_cra:
                converted_data = convert_hevc_cra_to_bla(bytes(packet))
                if converted_data != bytes(packet):
                    # Create new packet with converted data using same pattern as copy_packet
                    new_packet = Packet(converted_data)
                    new_packet.pts = packet.pts
                    new_packet.dts = packet.dts
                    new_packet.duration = packet.duration
                    new_packet.time_base = packet.time_base
                    new_packet.stream = packet.stream
                    new_packet.is_keyframe = packet.is_keyframe
                    for side_data in packet.iter_sidedata():
                        new_packet.set_sidedata(side_data)

                    packet = new_packet

            # Mark that we've processed the first packet
            if first_packet:
                first_packet = False

            # Apply timing adjustments
            segment_start_offset = self.segment_start_in_output / self.out_time_base
            pts = packet.pts if packet.pts else 0
            packet.pts = int((pts - segment_start_pts) * self.in_time_base / self.out_time_base + segment_start_offset)
            if packet.dts is not None:
                packet.dts = int((packet.dts - segment_start_pts) * self.in_time_base / self.out_time_base + segment_start_offset)

            result_packets.extend(self.remux_bitstream_filter.filter(packet))

        result_packets.extend(self.remux_bitstream_filter.filter(None))

        self.remux_bitstream_filter.flush()
        return result_packets

    def _should_convert_cra_for_segment(self, s: CutSegment) -> bool:
        """
        Check if this segment needs CRA to BLA conversion.
        This is needed when:
        1. The stream is HEVC
        2. The GOP starts with a CRA frame (NAL type 21)
        3. The stream was cut before this point. Meaning, we had discontinuity in the remux sequence right before this point.
          Examples:
            beginning was skipped: -k 8,12. Current segment is the first segment without recoding (e.g. starting at 8.5 with a little bit of recoding before this).
            discontinuity in the stream: -c 20,30. Current segment is the first segment without recoding after the cut that happened at 30.
        """
        # Check 1: Must be HEVC
        assert self.in_stream is not None
        assert self.in_stream.codec_context is not None
        if self.in_stream.codec_context.name != 'hevc':
            return False

        # Check 2: GOP must start with CRA (NAL type 21)
        if s.gop_index >= 0 and s.gop_index < len(self.media_container.gop_start_nal_types):
            nal_type = self.media_container.gop_start_nal_types[s.gop_index]
            if nal_type != 21:  # Not a CRA frame
                return False
        else:
            return False  # Invalid GOP index

        # Check 3: Check for discontinuity (missing stream content)
        # Case 1: First segment doesn't start at beginning (content was skipped)
        if self.is_first_remuxed_segment and s.gop_index >0:
            return True

        # Case 2: Gap between current segment and previous segment (content was cut)
        return self.last_remuxed_segment_gop_index is not None and s.gop_index > self.last_remuxed_segment_gop_index + 1

    def _apply_cra_to_bla_conversion(self, packets: list[Packet]) -> list[Packet]:
        """
        Apply CRA to BLA conversion to the packet list.
        This modifies the packet data to convert CRA frames to BLA frames.
        """

        converted_packets = []
        for packet in packets:
            if packet.stream.type == 'video':
                # Convert the packet data
                converted_data = convert_hevc_cra_to_bla(bytes(packet))
                if converted_data != bytes(packet):
                    # Create a new packet with converted data
                    new_packet = Packet(converted_data)
                    # Copy all attributes from original packet
                    new_packet.stream = packet.stream
                    new_packet.pts = packet.pts
                    new_packet.dts = packet.dts
                    new_packet.time_base = packet.time_base
                    converted_packets.append(new_packet)
                else:
                    converted_packets.append(packet)
            else:
                converted_packets.append(packet)

        return converted_packets

    def flush_encoder(self) -> list[Packet]:
        if self.enc_codec is None:
            return []

        result_packets = self.enc_codec.encode()

        if self.codec_name == 'mpeg2video':
            for p in result_packets:
                if p.time_base is not None:
                    if p.pts is not None:
                        p.pts = int(p.pts * p.time_base / self.out_time_base)
                    if p.dts is not None:
                        p.dts = int(p.dts * p.time_base / self.out_time_base)
                p.time_base = self.out_time_base

        self.enc_codec = None
        return result_packets

    def fetch_packet(self, target_dts: int, end_dts: int) -> Generator[Packet, None, None]:
        # First, check if we have a saved packet from previous call
        if self.demux_saved_packet is not None:
            saved_dts = self.demux_saved_packet.dts if self.demux_saved_packet.dts is not None else -100_000_000
            if saved_dts >= target_dts:
                if saved_dts <= end_dts:
                    packet = self.demux_saved_packet
                    self.demux_saved_packet = None
                    yield packet
                else:
                    # Saved packet is beyond our end range, don't yield it
                    return
            else:
                # Saved packet is before our target, clear it
                self.demux_saved_packet = None

        for packet in self.demux_iter:
            in_dts = packet.dts if packet.dts is not None else -100_000_000

            # Skip packets before target_dts
            if packet.pts is None or in_dts < target_dts:
                diff = (target_dts - in_dts) * self.in_time_base
                if in_dts > 0 and diff > 120:
                    t = int(target_dts - 30 / self.in_time_base)
                    # print(f"Seeking to skip a gap: {float(t * tb)}")
                    self.input_av_container.seek(t, stream = self.in_stream)
                    # Clear saved packet after seek since iterator position changed
                    self.demux_saved_packet = None
                continue

            # Check if packet exceeds end_dts
            if in_dts > end_dts:
                # Save this packet for next call and stop iteration
                self.demux_saved_packet = packet
                return

            # Packet is in our target range, yield it
            yield packet

    def fetch_frame(self, gop_start_dts: int, gop_end_dts: int, end_time: Fraction, start_dts_override: int | None = None) -> Generator[VideoFrame, None, None]:
        # Check if previous iteration consumed exactly to this GOP start
        continuous = self._last_fetch_end_dts is not None and (self._last_fetch_end_dts in (gop_end_dts, gop_start_dts))
        self._last_fetch_end_dts = gop_end_dts

        # Choose actual start DTS. Allow priming from previous GOP unless we're either still in the same GOP or continuing to the next one.
        start_dts = gop_start_dts if continuous else (start_dts_override if start_dts_override is not None else gop_start_dts)

        # Initialize or reset for new GOP boundary unless continuous
        if self.frame_buffer_gop_dts != gop_start_dts and not continuous:
            self.frame_buffer = []
            self.frame_buffer_gop_dts = gop_start_dts
            self.decoder.flush_buffers()

        # If asked to start earlier than GOP, seek and clear state (skip if continuous)
        if start_dts < gop_start_dts and not continuous:
            try:
                self.decoder.flush_buffers()
                self.frame_buffer = []
                self.input_av_container.seek(start_dts, stream=self.in_stream)
                self.demux_saved_packet = None
              # Recreate demux iterator after an explicit seek to ensure position is honored
                self.demux_iter = self.input_av_container.demux(self.in_stream)

            except Exception:
                pass

        # Process packets and yield frames when safe
        current_dts = gop_start_dts

        for packet in self.fetch_packet(start_dts, gop_end_dts):
            current_dts = packet.dts if packet.dts is not None else current_dts

            # Decode packet and add frames to buffer
            for frame in self.decoder.decode(packet):
                heap_item = FrameHeapItem(frame.pts, frame)
                heapq.heappush(self.frame_buffer, heap_item)

            # Release frames that are safe (buffer_lowest_pts <= current_dts)
            BUFFERED_FRAMES_COUNT = 15 # We need this to be quite high, b/c GENPTS is on and we can't know if the pts values are real or fake
            while len(self.frame_buffer) > BUFFERED_FRAMES_COUNT:
                lowest_heap_item = self.frame_buffer[0]  # Peek at heap minimum
                frame = lowest_heap_item.frame
                frame_pts = lowest_heap_item.pts if lowest_heap_item.pts is not None else -1
                frame_time_base = frame.time_base if frame.time_base is not None else self.in_time_base

                # Only process frames that are safe to release (frame_pts <= current_dts)
                if frame_pts <= current_dts:
                    if frame_pts * frame_time_base < end_time:
                        heapq.heappop(self.frame_buffer)  # Remove from heap
                        yield frame
                    else:
                        # Safe frame is beyond end_time - we're done since all frames from now would be beyond end time
                       return
                else:
                    break

        # Final flush of the decoder
        try:
            for frame in self.decoder.decode(None):
                heap_item = FrameHeapItem(frame.pts, frame)
                heapq.heappush(self.frame_buffer, heap_item)
        except Exception:
            pass

        # Yield remaining frames within time range
        while self.frame_buffer:
            # Peek at the next frame without popping it
            next_frame = self.frame_buffer[0]
            frame = next_frame.frame
            frame_time_base = frame.time_base if frame.time_base is not None else self.in_time_base

            if (next_frame.pts is not None and
                next_frame.pts * frame_time_base < end_time):
                # Frame is within time range, pop and yield it
                heapq.heappop(self.frame_buffer)
                yield frame
            else:
                # Frame is outside time range, stop processing (leave it in buffer)
                break

def expand_segments_with_fades(segments_with_fade: list[SegmentWithFade]) -> tuple[list[tuple[Fraction, Fraction]], list[FadeInfo | None]]:
    """
    Expand segments with fade information into basic segments and their corresponding fade info.

    This function doesn't split segments for minimal re-encoding yet - that will be done
    at the CutSegment level. It just prepares the data structure.

    Args:
        segments_with_fade: List of segments with fade information

    Returns:
        Tuple of (basic_segments, fade_infos) where fade_infos[i] corresponds to basic_segments[i]
    """
    basic_segments = []
    fade_infos = []

    for seg in segments_with_fade:
        basic_segments.append((seg.start_time, seg.end_time))
        fade_infos.append(seg.fade_info)

    return basic_segments, fade_infos

def smart_cut(media_container: MediaContainer, positive_segments: list[tuple[Fraction, Fraction]] | list[SegmentWithFade],
              out_path: str, audio_export_info: AudioExportInfo | None = None, log_level: str | None = None, progress: ProgressCallback | None = None,
              video_settings: VideoSettings | None = None, segment_mode: bool = False, cancel_object: CancelObject | None = None) -> Exception | None:
    if video_settings is None:
        video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)

    # Handle both legacy tuple format and new SegmentWithFade format
    if positive_segments and isinstance(positive_segments[0], SegmentWithFade):
        basic_segments, fade_infos = expand_segments_with_fades(positive_segments)
    else:
        # Legacy format - no fades
        basic_segments = positive_segments
        fade_infos = [None] * len(positive_segments)

    adjusted_segment_times = make_adjusted_segment_times(basic_segments, media_container)
    cut_segments = make_cut_segments(media_container, adjusted_segment_times, video_settings.mode == VideoExportMode.KEYFRAMES)

    # Attach fade information to cut segments and split for minimal re-encoding
    segment_idx = 0
    new_cut_segments = []

    for i, cut_seg in enumerate(cut_segments):
        # Find which original segment this cut_segment belongs to
        while segment_idx < len(adjusted_segment_times) and cut_seg.start_time >= adjusted_segment_times[segment_idx][1]:
            segment_idx += 1

        if segment_idx < len(fade_infos) and fade_infos[segment_idx] is not None:
            fade_info = fade_infos[segment_idx]

            # Check if this segment has fades
            has_fadein = fade_info.fadein_duration is not None and fade_info.fadein_duration > 0
            has_fadeout = fade_info.fadeout_duration is not None and fade_info.fadeout_duration > 0

            if has_fadein or has_fadeout:
                # Split segment for minimal re-encoding
                seg_duration = cut_seg.end_time - cut_seg.start_time
                segments_to_add = []

                current_time = cut_seg.start_time

                # Fade-in portion (needs re-encoding)
                if has_fadein:
                    fadein_end = min(cut_seg.start_time + fade_info.fadein_duration, cut_seg.end_time)
                    fadein_seg = CutSegment(
                        require_recode=True,
                        start_time=current_time,
                        end_time=fadein_end,
                        gop_start_dts=cut_seg.gop_start_dts,
                        gop_end_dts=cut_seg.gop_end_dts,
                        gop_index=cut_seg.gop_index,
                        fade_info=FadeInfo(fadein_duration=fade_info.fadein_duration, fadeout_duration=None)
                    )
                    segments_to_add.append(fadein_seg)
                    current_time = fadein_end

                # Middle portion (passthrough if possible)
                fadeout_start = cut_seg.end_time
                if has_fadeout:
                    fadeout_start = max(cut_seg.end_time - fade_info.fadeout_duration, current_time)

                if current_time < fadeout_start:
                    # There's a middle portion that doesn't need fading
                    middle_seg = CutSegment(
                        require_recode=cut_seg.require_recode,  # Keep original recode status
                        start_time=current_time,
                        end_time=fadeout_start,
                        gop_start_dts=cut_seg.gop_start_dts,
                        gop_end_dts=cut_seg.gop_end_dts,
                        gop_index=cut_seg.gop_index,
                        fade_info=None  # No fade for middle portion
                    )
                    segments_to_add.append(middle_seg)
                    current_time = fadeout_start

                # Fade-out portion (needs re-encoding)
                if has_fadeout and current_time < cut_seg.end_time:
                    fadeout_seg = CutSegment(
                        require_recode=True,
                        start_time=current_time,
                        end_time=cut_seg.end_time,
                        gop_start_dts=cut_seg.gop_start_dts,
                        gop_end_dts=cut_seg.gop_end_dts,
                        gop_index=cut_seg.gop_index,
                        fade_info=FadeInfo(fadein_duration=None, fadeout_duration=fade_info.fadeout_duration)
                    )
                    segments_to_add.append(fadeout_seg)

                new_cut_segments.extend(segments_to_add)
            else:
                # No fades, keep original segment
                cut_seg.fade_info = None
                new_cut_segments.append(cut_seg)
        else:
            # No fade info for this segment
            cut_seg.fade_info = None
            new_cut_segments.append(cut_seg)

    cut_segments = new_cut_segments

    if video_settings.mode == VideoExportMode.RECODE:
        for c in cut_segments:
            c.require_recode = True

    if segment_mode:
        output_files = []
        padding = len(str(len(adjusted_segment_times)))
        for i, s in enumerate(adjusted_segment_times):
            segment_index = str(i + 1).zfill(padding)  # Zero-pad the segment index
            if "#" in out_path:
                pound_index = out_path.rfind("#")
                output_file = out_path[:pound_index] + segment_index + out_path[pound_index + 1:]
            else:
                # Insert the segment index right before the last '.'
                dot_index = out_path.rfind(".")
                output_file = out_path[:dot_index] + segment_index + out_path[dot_index:] if dot_index != -1 else f"{out_path}{segment_index}"

            output_files.append((output_file, s))

    else:
        output_files = [(out_path, adjusted_segment_times[-1])]
    previously_done_segments = 0
    for output_path_segment in output_files:
        if cancel_object is not None and cancel_object.cancelled:
            break
        with av.open(output_path_segment[0], 'w') as output_av_container:

            include_video = True
            if output_av_container.format.name in ['ogg', 'mp3', 'm4a', 'ipod', 'flac', 'wav']: #ipod is the real name for m4a, I guess
                include_video = False

                        # Preserve container attachments (e.g., MKV attachments) when supported by the output format
            container_name = (output_av_container.format.name or "").lower()
            supports_attachments = any(x in container_name for x in ("matroska", "webm"))

            if supports_attachments:
                # Copy attachment streams from the primary input container
                for in_stream in media_container.av_container.streams:
                    if getattr(in_stream, "type", None) != "attachment":
                        continue

                    output_av_container.add_stream_from_template(in_stream)

            generators = []
            if media_container.video_stream is not None and include_video:
                generators.append(VideoCutter(media_container, output_av_container, video_settings, log_level))

            # Check if any segment has fades - if so, we need to use RecodeAudioCutter
            has_any_fades = any(seg.fade_info is not None and
                               (seg.fade_info.fadein_duration is not None or seg.fade_info.fadeout_duration is not None)
                               for seg in cut_segments if seg.fade_info is not None)

            if audio_export_info is not None:
                for track_i, track_export_settings in enumerate(audio_export_info.output_tracks):
                    if track_export_settings is not None and  track_export_settings.codec == 'passthru':
                        # Use RecodeAudioCutter if fades are present, otherwise use PassthruAudioCutter
                        if has_any_fades:
                            generators.append(RecodeAudioCutter(media_container, output_av_container, track_i, track_export_settings))
                        else:
                            generators.append(PassthruAudioCutter(media_container, output_av_container, track_i, track_export_settings))

            for sub_track_i in range(len(media_container.subtitle_tracks)):
                generators.append(SubtitleCutter(media_container, output_av_container, sub_track_i))

            output_av_container.start_encoding()
            if progress is not None:
                progress.emit(len(cut_segments))
            for s in cut_segments[previously_done_segments:]:
                if cancel_object is not None and cancel_object.cancelled:
                    break
                if s.start_time >= output_path_segment[1][1]: # Go to the next output file
                    break

                if progress is not None:
                    progress.emit(previously_done_segments)
                previously_done_segments += 1
                assert s.start_time < s.end_time, f"Invalid segment: start_time {s.start_time} >= end_time {s.end_time}"
                for g in generators:
                    for packet in g.segment(s):
                        if packet.dts < -900_000:
                            packet.dts = None
                        output_av_container.mux(packet)
            for g in generators:
                for packet in g.finish():
                    output_av_container.mux(packet)
            if progress is not None:
                progress.emit(previously_done_segments)

        if cancel_object is not None and cancel_object.cancelled:
            last_file_path = output_path_segment[0]

            if os.path.exists(last_file_path):
                os.remove(last_file_path)
