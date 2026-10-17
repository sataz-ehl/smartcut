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


# Encoder to use when re-encoding audio, keyed by FFmpeg decoder name (some decoders and encoders differ)
_AUDIO_ENCODER_FOR_DECODER = {
    'aac': 'aac', 'aac_fixed': 'aac', 'aac_latm': 'aac',
    'opus': 'libopus',
    'mp3': 'libmp3lame', 'mp3float': 'libmp3lame',
    'vorbis': 'libvorbis',
}
_LOSSLESS_AUDIO_PREFIXES = ('flac', 'alac', 'pcm_')
_MIN_RECODE_AUDIO_BITRATE = 128_000  # HE-AAC sources (~64 kbps) become AAC-LC, which needs more bits
# LAME's presets scale the input down below 256 kbps to leave headroom (0.95 up to 160 kbps, 0.97 at 192, 0.98 at
# 224), and the ffmpeg CLI does the same: each re-encode would lower the level ~0.45 dB (fork addition)
_MIN_RECODE_AUDIO_BITRATE_FOR = {'libmp3lame': 256_000}
# Decoders whose copied packets break after a jump: an MP3 frame can take bits from earlier frames (the bit
# reservoir), so after a join ~50 ms decode as silence. Such tracks are re-encoded when the output has a join.
_JOIN_UNSAFE_AUDIO_DECODERS = ('mp3', 'mp3float')


def _recode_audio_encoder_name(decoder_name: str, webm: bool = False) -> str:
    name = _AUDIO_ENCODER_FOR_DECODER.get(decoder_name, decoder_name)
    try:
        av.Codec(name, 'w')
    except Exception:  # no encoder for this codec in this FFmpeg build
        name = 'aac'
    if webm and name not in _WEBM_AUDIO_ENCODERS:
        name = 'libopus'
    return name


# WebM holds only Opus and Vorbis audio: other tracks are re-encoded to Opus for .webm outputs (fork addition)
_WEBM_AUDIO_DECODERS = ('opus', 'libopus', 'vorbis', 'libvorbis')
_WEBM_AUDIO_ENCODERS = ('libopus', 'libvorbis')


def _is_webm(output_av_container: OutputContainer) -> bool:
    return (output_av_container.format.name or '') == 'webm'


def audio_needs_recode(media_container: MediaContainer, output_av_container: OutputContainer, track_index: int,
                       has_joins: bool = False) -> bool:
    """True if the track can't be copied cleanly into this output (fork addition): Opus/Vorbis-only WebM and another
    codec, or MP3 when the output joins pieces (or starts mid-stream)."""
    name = media_container.audio_tracks[track_index].av_stream.codec_context.name
    return ((_is_webm(output_av_container) and name not in _WEBM_AUDIO_DECODERS)
            or (has_joins and name in _JOIN_UNSAFE_AUDIO_DECODERS))


def create_recode_audio_output_stream(
    media_container: MediaContainer,
    output_av_container: OutputContainer,
    track_index: int,
) -> AudioStream:
    """
    Create the output stream for RecodeAudioCutter from its encoder.

    Unlike create_audio_output_stream this doesn't copy the source's codec parameters: an encoder writes
    what it supports (FFmpeg's AAC encoder writes AAC-LC, never HE-AAC), and the stream header has to
    describe the packets that are actually written.
    """
    track = media_container.audio_tracks[track_index]
    in_ctx = track.av_stream.codec_context
    codec_name = _recode_audio_encoder_name(in_ctx.name, webm=_is_webm(output_av_container))

    rate = in_ctx.sample_rate or 48000
    supported_rates = av.Codec(codec_name, 'w').audio_rates
    if supported_rates and rate not in supported_rates:
        rate = 48000 if 48000 in supported_rates else max(supported_rates)

    out_stream = cast(AudioStream, output_av_container.add_stream(codec_name, rate=rate))
    out_stream.codec_context.layout = in_ctx.layout
    if not codec_name.startswith(_LOSSLESS_AUDIO_PREFIXES):
        source_bitrate = track.av_stream.bit_rate or in_ctx.bit_rate or 0
        out_stream.codec_context.bit_rate = max(source_bitrate, _MIN_RECODE_AUDIO_BITRATE_FOR.get(codec_name, _MIN_RECODE_AUDIO_BITRATE))
    out_stream.time_base = Fraction(1, rate)
    out_stream.metadata.update(track.av_stream.metadata)
    out_stream.disposition = cast(Disposition, track.av_stream.disposition.value)
    return out_stream


class RecodeAudioCutter:
    """
    Re-encodes a whole audio track with one continuous encoder, applying the segments' audio fades.

    Used for a track when any segment has an audio fade. Decoded audio is cut sample-accurately to each
    cut segment, faded per sample and fed to a single encoder. Mixing re-encoded and copied packets in one
    stream instead leaves the encoder's buffered samples misplaced at segment joins (an audible tick) and
    flushed at the very end of the file.

    Environment Variables:
        SMARTCUT_DEBUG_AUDIO_FADE: Set to '1' to print the gain range and peak reduction of each faded
                                   cut segment to stderr
    """

    # After a jump, decode a few earlier packets so the kept samples decode cleanly: 4 cover the 80 ms Opus decoders
    # need, and the earlier MP3 frames a frame takes bits from (with 2, a cut MP3 started with 13-25 ms of silence)
    PREROLL_PACKETS = 4

    def __init__(self, media_container: MediaContainer, out_stream: AudioStream, track_index: int) -> None:
        self.track = media_container.audio_tracks[track_index]
        self.out_stream = out_stream
        self.encoder = out_stream.codec_context
        self.decoder = self.track.av_stream.codec_context
        self.in_time_base = cast(Fraction, self.track.av_stream.time_base)
        self.rate: int = self.encoder.sample_rate
        self.layout_name = self.encoder.layout.name
        self.channels = len(self.encoder.layout.channels)
        self.debug = os.environ.get('SMARTCUT_DEBUG_AUDIO_FADE') == '1'

        # Output timeline position (seconds of video so far) and samples sent to the encoder
        self.segment_start_in_output = Fraction(0)
        self.samples_out = 0

        # Decoded but not yet used audio, as float planar at the output rate.
        # buf[:, 0] is absolute sample buf_start (source timeline, in output-rate samples).
        self.buf = np.zeros((self.channels, 0), np.float32)
        self.buf_start = 0
        self.next_packet: int | None = None  # None: the decoder must be (re)started before decoding
        self.resampler: av.AudioResampler | None = None

    def segment(self, cut_segment: CutSegment) -> list[Packet]:
        packets: list[Packet] = []

        # Keep the audio locked to the video timeline: pad with silence where the source had no audio,
        # drop samples if rounding put us ahead.
        target = round(self.segment_start_in_output * self.rate)
        self.segment_start_in_output += cut_segment.end_time - cut_segment.start_time
        if self.samples_out < target:
            packets.extend(self._encode(np.zeros((self.channels, target - self.samples_out), np.float32)))
        drop = self.samples_out - target

        start = round(cut_segment.start_time * self.rate)
        chunk = self._take(start, round(cut_segment.end_time * self.rate))
        if drop > 0:
            chunk, start = chunk[:, drop:], start + drop

        fade = cut_segment.fade_info
        if fade is not None and (fade.audio_fadein_duration or fade.audio_fadeout_duration) and chunk.shape[1]:
            chunk = self._apply_fade(chunk, start, cut_segment)

        packets.extend(self._encode(chunk))
        return packets

    def finish(self) -> list[Packet]:
        """Flush the encoder at the end of all segments."""
        return self._mux_ready(self.encoder.encode(None))

    def _apply_fade(self, chunk: np.ndarray, start: int, cut_segment: CutSegment) -> np.ndarray:
        """Multiply by a per-sample gain ramp. Fades use the original segment bounds, so a segment that
        spans several cut segments gets one continuous fade."""
        fade = cast(FadeInfo, cut_segment.fade_info)
        seg_start = float(cut_segment.orig_segment_start if cut_segment.orig_segment_start is not None else cut_segment.start_time)
        seg_end = float(cut_segment.orig_segment_end if cut_segment.orig_segment_end is not None else cut_segment.end_time)
        t = (start + np.arange(chunk.shape[1])) / self.rate
        gain = np.ones(chunk.shape[1])
        if fade.audio_fadein_duration:
            gain = np.minimum(gain, np.clip((t - seg_start) / float(fade.audio_fadein_duration), 0, 1))
        if fade.audio_fadeout_duration:
            gain = np.minimum(gain, np.clip((seg_end - t) / float(fade.audio_fadeout_duration), 0, 1))
        result = chunk * gain.astype(np.float32)
        if self.debug:
            orig_peak = float(np.abs(chunk).max())
            peak = float(np.abs(result).max())
            reduction = (1 - peak / orig_peak) * 100 if orig_peak > 0 else 0
            print(f"[AUDIO_FADE] t={t[0] - seg_start:.3f}..{t[-1] - seg_start:.3f}s | gain {gain.min():.3f}..{gain.max():.3f} | "
                  f"peak {orig_peak:.6f} → {peak:.6f} ({reduction:.1f}% reduction)", file=sys.stderr)
        return result

    def _take(self, start: int, end: int) -> np.ndarray:
        """Return the source audio for absolute samples [start, end). Shorter if the source audio ends."""
        buf_end = self.buf_start + self.buf.shape[1]
        if self.next_packet is None or start < self.buf_start or start > buf_end:
            self._restart_decoder(start)

        # Decode until `end` is covered; new samples are collected and joined once
        pending: list[np.ndarray] = []
        buf_end = self.buf_start + self.buf.shape[1]
        while buf_end < end and self.next_packet is not None and self.next_packet < len(self.track.packets):
            buf_end = self._decode_packet(self.track.packets[self.next_packet], pending, buf_end)
            self.next_packet += 1
        if pending:
            self.buf = np.concatenate([self.buf, *pending], axis=1)

        if self.buf.shape[1] == 0:
            return np.zeros((self.channels, 0), np.float32)
        lead = self.buf_start - start  # audio starts after the requested start: fill with silence
        out = self.buf[:, max(start - self.buf_start, 0): max(end - self.buf_start, 0)]
        if lead > 0:
            out = np.concatenate([np.zeros((self.channels, min(lead, end - start)), np.float32), out], axis=1)
        consumed = min(max(end - self.buf_start, 0), self.buf.shape[1])
        self.buf = self.buf[:, consumed:]
        self.buf_start += consumed
        return out

    def _restart_decoder(self, start: int) -> None:
        """Jump: restart decoding a few packets before the packet that contains `start`."""
        start_pts = start / self.rate / self.in_time_base
        first = int(np.searchsorted(self.track.frame_times_pts, start_pts, side='right')) - 1
        self.next_packet = max(first - self.PREROLL_PACKETS, 0)
        self.decoder.flush_buffers()
        self.resampler = av.AudioResampler(format='fltp', layout=self.layout_name, rate=self.rate)
        self.buf = np.zeros((self.channels, 0), np.float32)
        self.buf_start = start

    def _decode_packet(self, packet: Packet, pending: list[np.ndarray], buf_end: int) -> int:
        """Decode one packet, append its samples to `pending` and return the new buffer end."""
        for frame in self.decoder.decode(packet):
            if frame.pts is None:
                continue
            time_base = frame.time_base if frame.time_base is not None else self.in_time_base
            frame_start = round(frame.pts * time_base * self.rate)
            if frame.format.name == 'fltp' and frame.sample_rate == self.rate and frame.layout.name == self.layout_name:
                samples = frame.to_ndarray()  # already float planar at the output rate (always for AAC)
            else:
                assert self.resampler is not None
                arrays = [f.to_ndarray() for f in self.resampler.resample(frame)]
                if not arrays:
                    continue
                samples = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)

            if self.buf.shape[1] == 0 and not pending:
                self.buf_start = buf_end = frame_start  # first frame after a restart (may precede the requested start)
            elif frame_start > buf_end:  # gap in the source audio: keep timing with silence
                samples = np.concatenate([np.zeros((self.channels, frame_start - buf_end), np.float32), samples], axis=1)
            elif frame_start < buf_end:  # overlap (timestamp rounding): skip what we already have
                samples = samples[:, buf_end - frame_start:]
            if samples.shape[1]:
                pending.append(samples)
                buf_end += samples.shape[1]
        return buf_end

    def _encode(self, samples: np.ndarray) -> list[Packet]:
        if samples.shape[1] == 0:
            return []
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples, np.float32), format='fltp', layout=self.layout_name)
        frame.sample_rate = self.rate
        frame.time_base = Fraction(1, self.rate)
        frame.pts = self.samples_out
        self.samples_out += samples.shape[1]
        return self._mux_ready(self.encoder.encode(frame))

    def _mux_ready(self, packets: list[Packet]) -> list[Packet]:
        for p in packets:
            p.stream = self.out_stream
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
