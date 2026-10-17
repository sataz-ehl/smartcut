import os
import sys
from fractions import Fraction
from typing import cast

import av
import numpy as np
from av import AudioStream
from av.container.output import OutputContainer
from av.packet import Packet
from av.stream import Disposition, Stream

from smartcut.media_container import MediaContainer
from smartcut.misc_data import CutSegment, FadeInfo
from smartcut.video_cutter import copy_packet


def create_audio_output_stream(
    media_container: MediaContainer,
    output_av_container: OutputContainer,
    track_index: int,
) -> AudioStream:
    """
    Create output audio stream from source media container.

    Separated from PassthruAudioCutter to allow reuse in joining operations.
    """
    track = media_container.audio_tracks[track_index]
    out_stream = output_av_container.add_stream_from_template(
        track.av_stream,
        options={'x265-params': 'log_level=error'}
    )
    out_stream.metadata.update(track.av_stream.metadata)
    out_stream.disposition = cast(Disposition, track.av_stream.disposition.value)
    return out_stream


class PassthruAudioCutter:
    def __init__(
        self,
        media_container: MediaContainer,
        out_stream: AudioStream,
        track_index: int,
        initial_position: Fraction = Fraction(0),
        initial_prev_dts: int = -100_000,
        initial_prev_pts: int = -100_000,
    ) -> None:
        self.track = media_container.audio_tracks[track_index]
        self.out_stream = out_stream

        # Output position state - can be set for joining multiple files
        self.segment_start_in_output = initial_position
        self.prev_dts = initial_prev_dts
        self.prev_pts = initial_prev_pts

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

    def __init__(self, media_container: MediaContainer, out_stream: AudioStream, track_index: int) -> None:
        self.track = media_container.audio_tracks[track_index]
        self.media_container = media_container
        self.out_stream = out_stream

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

            # Audio fade-in
            if fade_info.audio_fadein_duration is not None:
                fadein_samples = int(float(fade_info.audio_fadein_duration) * samples_per_second)
                samples_from_start = current_sample - segment_start_sample
                if samples_from_start < fadein_samples:
                    alpha = min(1.0, samples_from_start / fadein_samples)

            # Audio fade-out
            if fade_info.audio_fadeout_duration is not None:
                fadeout_samples = int(float(fade_info.audio_fadeout_duration) * samples_per_second)
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

        # Check if we need to recode for audio fades
        needs_fade = (cut_segment.fade_info is not None and
                     (cut_segment.fade_info.audio_fadein_duration is not None or
                      cut_segment.fade_info.audio_fadeout_duration is not None))

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
            # Use floating point planar format for better fade processing
            self.encoder.format = 'fltp'
            self.encoder.time_base = in_tb

            # Match source bitrate to maintain quality consistency
            if self.track.av_stream.bit_rate:
                self.encoder.bit_rate = self.track.av_stream.bit_rate

            # Open encoder before use (required in PyAV)
            self.encoder.open()

        # Decode, apply fade, and re-encode
        packets = []
        current_sample = start_sample

        # Calculate original segment sample boundaries for fade calculation
        # This ensures all GOPs in a segment use the same fade curve
        if cut_segment.orig_segment_start is not None:
            orig_start_sample = int(float(cut_segment.orig_segment_start) * sample_rate)
        else:
            orig_start_sample = start_sample

        if cut_segment.orig_segment_end is not None:
            orig_end_sample = int(float(cut_segment.orig_segment_end) * sample_rate)
        else:
            orig_end_sample = end_sample

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
                # Convert to numpy array (as float32 for better precision)
                audio_arr = audio_frame.to_ndarray()

                # Ensure audio_arr is float32 for fade processing
                if audio_arr.dtype != np.float32:
                    audio_arr = audio_arr.astype(np.float32)

                # Apply fade using ORIGINAL segment boundaries to avoid duplicate fades
                if cut_segment.fade_info is not None:
                    audio_arr = self._apply_audio_fade(
                        audio_arr, current_sample, sample_rate,
                        orig_start_sample, orig_end_sample, cut_segment.fade_info
                    )

                # Create new frame from modified array in fltp format
                layout_name = audio_frame.layout.name if audio_frame.layout is not None else str(in_layout)
                new_frame = av.AudioFrame.from_ndarray(audio_arr.astype(np.float32), format='fltp', layout=layout_name)
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

                # Increment by number of samples (shape[1] for planar format (channels, samples))
                current_sample += audio_arr.shape[1] if len(audio_arr.shape) > 1 else audio_arr.shape[0]

        # Don't flush encoder between segments to avoid audio artifacts at GOP boundaries
        # The encoder will be flushed once at the end in finish()

        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
        return packets

    def finish(self) -> list[Packet]:
        """Flush encoder at the end of all segments."""
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


def create_subtitle_output_stream(
    media_container: MediaContainer,
    output_av_container: OutputContainer,
    track_index: int,
) -> Stream:
    """
    Create output subtitle stream from source media container.

    Separated from SubtitleCutter to allow reuse in joining operations.
    """
    in_stream = media_container.av_container.streams.subtitles[track_index]
    out_stream = output_av_container.add_stream_from_template(in_stream)
    out_stream.metadata.update(in_stream.metadata)
    out_stream.disposition = cast(Disposition, in_stream.disposition.value)
    return out_stream


class SubtitleCutter:
    def __init__(
        self,
        media_container: MediaContainer,
        out_stream: Stream,
        track_index: int,
        initial_position: Fraction = Fraction(0),
        initial_prev_pts: int = -100_000,
    ) -> None:
        self.track_i = track_index
        self.packets = media_container.subtitle_tracks[track_index]
        self.in_stream = media_container.av_container.streams.subtitles[track_index]
        self.out_stream = out_stream

        # Output position state - can be set for joining multiple files
        self.segment_start_in_output = initial_position
        self.prev_pts = initial_prev_pts

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
