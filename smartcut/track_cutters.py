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

from smartcut.media_container import AudioTrack, MediaContainer
from smartcut.misc_data import CutSegment
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


_OPUS_DECODERS = ('opus', 'libopus')
_OPUS_PREROLL = Fraction(80, 1000)  # RFC 7845: decode at least 80 ms before the first sample that is used
# Codecs whose frames overlap their neighbours (MDCT), so a copy needs the frame before its first one
_OVERLAP_DECODERS = ('aac', 'aac_fixed', 'ac3', 'ac3_fixed', 'eac3')
_OVERLAP_PREROLL = Fraction(50, 1000)  # the packet holding the cut point and at least one before it
EDIT_LIST_FORMATS = ('mov', 'mp4', 'ipod')  # FFmpeg names for .mov/.mp4/.m4a outputs


def _first_copied_packet(track: AudioTrack, start_time: Fraction) -> int:
    """Index of the first packet PassthruAudioCutter.segment copies for a piece that starts at start_time: the first
    whole packet at or after it (fork addition)."""
    if start_time <= 0:
        return 0
    in_tb = cast(Fraction, track.av_stream.time_base)
    return int(np.searchsorted(track.frame_times_pts, round(start_time / in_tb)))


def _copy_preroll(track: AudioTrack, format_name: str) -> Fraction | None:
    """How much earlier audio a copy of this track needs to start exactly mid-stream in this output format, or None
    if it can't (fork addition). Only MP4-family outputs can: their edit list skips the pre-roll packets (and Opus
    decoders skip them through the pre-skip, which needs an OpusHead to set it in). Matroska has no edit list, and
    other codecs (MP3, FLAC, PCM, ...) have no pre-roll cutter."""
    if format_name not in EDIT_LIST_FORMATS:
        return None
    ctx = track.av_stream.codec_context
    if ctx.name in _OPUS_DECODERS:
        return _OPUS_PREROLL if ctx.extradata and bytes(ctx.extradata[:8]) == b'OpusHead' else None
    if ctx.name in _OVERLAP_DECODERS:
        return _OVERLAP_PREROLL
    return None


def _preroll_start(track: AudioTrack, start: int, preroll: Fraction) -> tuple[int, Fraction]:
    """First packet and length of the packets just before packet `start` that cover `preroll` (fork addition). Fewer
    near the start of the stream: from its first packet the decoder starts as it does for the source."""
    in_tb = cast(Fraction, track.av_stream.time_base)
    first, length = start, Fraction(0)
    while first > 0 and length < preroll:
        first -= 1
        length += (track.packets[first].duration or 0) * in_tb
    return first, length


class PrerollPassthruAudioCutter(PassthruAudioCutter):
    """
    Copies a track into an MP4-family output, timed exactly to the cut by the MP4 edit list (fork addition).

    Start: a copy that started mid-stream was decoded from a cold decoder: its first frames came out low (AAC frames
    overlap their neighbours, as in AAC and AC-3; Opus decoders need 80 ms of pre-roll). The earlier packets covering
    `preroll` are copied too, with timestamps before the start of the output. MP4 writes an edit list that skips them,
    so the audio also starts exactly at the cut point instead of at the first whole packet after it.

    End: the last packet kept runs up to a frame past the end of the output (8 ms in one AAC-LC cut, 28 ms in an
    HE-AAC one). It is held back until finish(), which shortens its duration to end at the output's end, so the track
    and its edit list end there, as FFmpeg's own encoders' last packets do. Players that honour the edit list's end
    stop the audio exactly there; FFmpeg ignores it and decodes the whole last frame (re-encoded tracks likewise).
    """

    def __init__(self, media_container: MediaContainer, out_stream: AudioStream, track_index: int,
                 first_segment: CutSegment, preroll: Fraction = Fraction(0)) -> None:
        super().__init__(media_container, out_stream, track_index)
        self.preroll: list[Packet] = []
        self.preroll_length = Fraction(0)
        self.held: Packet | None = None  # the last packet so far, see finish()
        start = _first_copied_packet(self.track, first_segment.start_time)
        if start == 0 or preroll <= 0:  # starts with the stream: nothing to run before it
            return
        first, self.preroll_length = _preroll_start(self.track, start, preroll)
        self.preroll = [p for p in self.track.packets[first:start] if p.pts is not None and p.dts is not None]

    def segment(self, cut_segment: CutSegment) -> list[Packet]:
        packets = [] if self.held is None else [self.held]
        in_tb = cast(Fraction, self.track.av_stream.time_base)
        for p in self.preroll:  # first piece only: the pre-roll packets, timed like segment() times the others
            assert p.pts is not None and p.dts is not None  # filtered in __init__
            packet = copy_packet(p)
            packet.stream = self.out_stream
            packet.pts = int(p.pts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
            packet.dts = int(p.dts + (self.segment_start_in_output - cut_segment.start_time) / in_tb)
            self.prev_pts, self.prev_dts = packet.pts, packet.dts
            packets.append(packet)
        self.preroll = []
        packets += super().segment(cut_segment)
        self.held = packets.pop() if packets else None
        return packets

    def finish(self) -> list[Packet]:
        packet, self.held = self.held, None
        if packet is None or packet.pts is None:
            return [] if packet is None else [packet]
        end = round(self.segment_start_in_output / cast(Fraction, self.track.av_stream.time_base))  # end of the output
        if packet.duration and packet.pts + packet.duration > end:
            packet.duration = max(1, end - packet.pts)
        return [packet]


class OpusPassthruAudioCutter(PrerollPassthruAudioCutter):
    """
    Copies an Opus track, without a ramp at the start (fork addition).

    A copy that starts mid-stream rose from ~40 % of the level over its first 80 ms, and the source's pre-skip
    (meant for the stream start) also dropped ~6.5 ms of real audio. The pre-roll packets cover the 80 ms Opus
    decoders need, and the stream's pre-skip is set to their length, so decoders discard them. Used for MP4-family
    outputs only (see audio_needs_recode): in Matroska / WebM the muxer shifted all streams by ~73 ms. It allows
    Opus timestamps only down to minus the stream's CodecDelay, which it takes from the source's initial padding
    (6.5 ms; PyAV can't set it), and the pre-roll starts 80 ms before the cut.
    """

    def __init__(self, media_container: MediaContainer, out_stream: AudioStream, track_index: int,
                 first_segment: CutSegment) -> None:
        super().__init__(media_container, out_stream, track_index, first_segment, _OPUS_PREROLL)
        head = out_stream.codec_context.extradata
        if not head or bytes(head[:8]) != b'OpusHead':
            self.preroll = []  # no pre-skip to set: decoders would play the pre-roll
        elif self.preroll:
            head = bytearray(bytes(head))
            head[10:12] = round(self.preroll_length * 48000).to_bytes(2, 'little')  # pre-skip, in 48 kHz samples
            out_stream.codec_context.extradata = bytes(head)


def passthru_audio_cutter(media_container: MediaContainer, out_stream: AudioStream, track_index: int,
                          first_segment: CutSegment, output_av_container: OutputContainer | None = None) -> PassthruAudioCutter:
    """The cutter that copies this track (fork addition). MP4-family outputs get PrerollPassthruAudioCutter: its end
    follows the cut, and Opus, AAC and AC-3 get pre-roll packets when the output starts mid-stream."""
    track = media_container.audio_tracks[track_index]
    format_name = (output_av_container.format.name or '') if output_av_container is not None else ''
    if format_name not in EDIT_LIST_FORMATS:
        return PassthruAudioCutter(media_container, out_stream, track_index)
    preroll = _copy_preroll(track, format_name)
    if preroll is not None and track.av_stream.codec_context.name in _OPUS_DECODERS:
        return OpusPassthruAudioCutter(media_container, out_stream, track_index, first_segment)
    return PrerollPassthruAudioCutter(media_container, out_stream, track_index, first_segment, preroll or Fraction(0))


# Encoder to use when re-encoding audio, keyed by FFmpeg decoder name (some decoders and encoders differ)
_AUDIO_ENCODER_FOR_DECODER = {
    'aac': 'aac', 'aac_fixed': 'aac', 'aac_latm': 'aac',
    'opus': 'libopus',
    'mp3': 'libmp3lame', 'mp3float': 'libmp3lame',
    'vorbis': 'libvorbis',
}
_LOSSLESS_AUDIO_PREFIXES = ('flac', 'alac', 'pcm_')
# Re-encoded audio bitrate (fork addition): the source's, at least a floor, within what the encoder accepts (see
# _recode_audio_bitrate). The floor is per channel, counting mono as two: 128 kb/s mono and stereo, 384 kb/s 5.1. HE-AAC
# sources (~48-64 kb/s stereo) become AAC-LC, which needs more bits. Mono at 64 kb/s fared worse: re-encoding a 64 kb/s
# mono source at its own rate moved the level up to x1.20 in 5 ms windows of music (at 128 kb/s, x1.04).
_RECODE_AUDIO_FLOOR_PER_CHANNEL = 64_000
# LAME's presets scale the input down below 256 kbps to leave headroom (0.95 up to 160 kbps, 0.97 at 192, 0.98 at
# 224), and the ffmpeg CLI does the same: each re-encode would lower the level ~0.45 dB. Whatever the channels.
_RECODE_AUDIO_FLOOR_FOR = {'libmp3lame': 256_000}
# Highest rates the encoders don't enforce themselves: LAME takes any rate, MP3's highest is 320 kb/s
_RECODE_AUDIO_MAX_FOR = {'libmp3lame': 320_000}


def _source_audio_bitrate(track: AudioTrack) -> int:
    """The source track's average bitrate (fork addition): as its stream reports it, else from its packets (total
    size x 8 / the time they span; Matroska / WebM usually report none, e.g. for Opus). 0 for lossless codecs: their
    rate says nothing about a lossy one (PCM into WebM asked libopus for 1536 kb/s, which it refuses)."""
    ctx = track.av_stream.codec_context
    if (ctx.name or '').startswith(_LOSSLESS_AUDIO_PREFIXES):
        return 0
    reported = track.av_stream.bit_rate or ctx.bit_rate
    if reported:
        return reported
    timed = [(p.pts, p.duration or 0, p.size) for p in track.packets if p.pts is not None]
    if not timed:
        return 0
    span = (max(pts + duration for pts, duration, _ in timed) - min(pts for pts, _, _ in timed)) * cast(Fraction, track.av_stream.time_base)
    return round(sum(size for _, _, size in timed) * 8 / span) if span > 0 else 0


def _accepted_audio_bitrate(codec_name: str, rate: int, layout: av.AudioLayout, wanted: int) -> int:
    """The highest bitrate up to `wanted` the encoder opens with, as it reports it after opening (fork addition):
    aac and ac3 lower a too-high rate themselves (aac: 6144 bits per channel per frame, ac3: 640 kb/s), libopus
    (256 kb/s per channel) and libvorbis (~89-249 kb/s per channel, by sample rate) refuse it."""

    def opened(bit_rate: int) -> int | None:
        ctx = cast(av.AudioCodecContext, av.CodecContext.create(codec_name, 'w'))
        ctx.sample_rate, ctx.layout, ctx.bit_rate = rate, layout, bit_rate
        formats = av.Codec(codec_name, 'w').audio_formats
        if formats:
            ctx.format = formats[0].name
        ctx.time_base = Fraction(1, rate)
        try:
            ctx.open()
        except Exception:
            return None
        return ctx.bit_rate or bit_rate

    got = opened(wanted)
    if got is not None:
        return got
    low, high = 0, wanted  # high is refused; find the highest accepted, to 1 kb/s
    while high - low > 1000:
        mid = (low + high) // 2
        if opened(mid) is None:
            high = mid
        else:
            low = mid
    return (opened(low) or low) if low else wanted


def _recode_audio_bitrate(track: AudioTrack, codec_name: str, rate: int, layout: av.AudioLayout) -> int:
    """Bitrate for re-encoding the track with this encoder (fork addition): the source's (_source_audio_bitrate), at
    least _RECODE_AUDIO_FLOOR_FOR or the floor per channel (mono counted as two), at most what the encoder accepts."""
    floor = _RECODE_AUDIO_FLOOR_FOR.get(codec_name, _RECODE_AUDIO_FLOOR_PER_CHANNEL * max(len(layout.channels), 2))
    wanted = max(_source_audio_bitrate(track), floor)
    wanted = min(wanted, _RECODE_AUDIO_MAX_FOR.get(codec_name, wanted))
    return _accepted_audio_bitrate(codec_name, rate, layout, wanted)


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


def audio_needs_recode(media_container: MediaContainer, format_name: str, track_index: int, first_piece: CutSegment,
                       has_joins: bool) -> bool:
    """
    True if a copy of the track into this output wouldn't be exact, so it is re-encoded as a whole (fork addition).

    format_name: the output's FFmpeg format name. first_piece: the output's first cut piece. has_joins: the output
    joins segments that aren't contiguous in the source.

    A copy keeps whole packets, so it is exact only
    - without a join. At a join the copied frames end and start up to a frame off the cut: each join shifted A/V by
      up to a frame (AAC-LC after 5 joins: -15.5 ms; HE-AAC: -27 ms, a random walk). Frames that overlap their
      neighbours (AAC, AC-3) decode as a mix of both sides (level in 5 ms windows x0.48-x2.0 for AAC-LC, x0.13-x9.9
      for HE-AAC), and MP3 frames that take bits from earlier frames (the bit reservoir) decoded as ~50 ms of
      silence.
    - starting with the stream, or mid-stream where a pre-roll copy starts exactly: MP4-family outputs with Opus, AAC
      or AC-3 (see _copy_preroll). Otherwise the copy started at the first whole packet after the cut (up to a frame
      late, and low for MDCT codecs). Re-encoding FLAC or PCM is lossless anyway.
    - and in WebM only for Opus and Vorbis, the only audio WebM holds.
    """
    track = media_container.audio_tracks[track_index]
    name = track.av_stream.codec_context.name
    if (format_name == 'webm' and name not in _WEBM_AUDIO_DECODERS) or has_joins:
        return True
    if _first_copied_packet(track, first_piece.start_time) == 0:
        return False  # the copy starts with the stream
    return _copy_preroll(track, format_name) is None


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
        out_stream.codec_context.bit_rate = _recode_audio_bitrate(track, codec_name, rate, in_ctx.layout)
    out_stream.time_base = Fraction(1, rate)
    out_stream.metadata.update(track.av_stream.metadata)
    out_stream.disposition = cast(Disposition, track.av_stream.disposition.value)
    return out_stream


class RecodeAudioCutter:
    """
    Re-encodes a whole audio track with one continuous encoder, applying the segments' audio fades.

    Used for a track when any segment has an audio fade, or when a copy wouldn't be exact (see audio_needs_recode).
    Decoded audio is cut sample-accurately to each cut segment, faded per sample and fed to a single encoder, so A/V
    stay in sync at every join. Mixing re-encoded and copied packets in one stream instead leaves the encoder's
    buffered samples misplaced at segment joins (an audible tick) and flushed at the very end of the file.

    At each join (fork addition) the earlier segment fades out over its last JOIN_FADE and the later one fades in over
    its first, unless the user's audio fade on that side is longer: a hard cut between unrelated audio clicks. Not at
    the very start or end of the output, and not between segments that are contiguous in the source.

    Environment Variables:
        SMARTCUT_DEBUG_AUDIO_FADE: Set to '1' to print the gain range and peak reduction of each faded
                                   cut segment to stderr
    """

    # After a jump, decode a few earlier packets so the kept samples decode cleanly: 4 cover the 80 ms Opus decoders
    # need, and the earlier MP3 frames a frame takes bits from (with 2, a cut MP3 started with 13-25 ms of silence)
    PREROLL_PACKETS = 4
    # On a steady tone the largest sample step at a hard-cut join was 5-11x the tone's own largest step; with 3 ms
    # ramps on both sides, 1.0x (as at a copied join: 1.3-1.4x)
    JOIN_FADE = Fraction(3, 1000)

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

        if chunk.shape[1]:
            chunk = self._apply_fade(chunk, start, cut_segment)

        packets.extend(self._encode(chunk))
        return packets

    def finish(self) -> list[Packet]:
        """Flush the encoder at the end of all segments."""
        return self._mux_ready(self.encoder.encode(None))

    def _fade_durations(self, cut_segment: CutSegment) -> tuple[Fraction, Fraction]:
        """Fade-in and fade-out of the segment this piece belongs to: the user's audio fades, and at least JOIN_FADE
        where the segment starts or ends at a join (0: none)."""
        fade = cut_segment.fade_info
        fade_in = (fade.audio_fadein_duration if fade is not None else None) or Fraction(0)
        fade_out = (fade.audio_fadeout_duration if fade is not None else None) or Fraction(0)
        if cut_segment.join_before:
            fade_in = max(fade_in, self.JOIN_FADE)
        if cut_segment.join_after:
            fade_out = max(fade_out, self.JOIN_FADE)
        return fade_in, fade_out

    def _apply_fade(self, chunk: np.ndarray, start: int, cut_segment: CutSegment) -> np.ndarray:
        """Multiply by a per-sample gain ramp. Fades use the original segment bounds, so a segment that
        spans several cut segments gets one continuous fade. Returns chunk itself when nothing fades."""
        fade_in, fade_out = self._fade_durations(cut_segment)
        if not (fade_in or fade_out):
            return chunk
        seg_start = float(cut_segment.orig_segment_start if cut_segment.orig_segment_start is not None else cut_segment.start_time)
        seg_end = float(cut_segment.orig_segment_end if cut_segment.orig_segment_end is not None else cut_segment.end_time)
        t = (start + np.arange(chunk.shape[1])) / self.rate
        gain = np.ones(chunk.shape[1])
        if fade_in:
            gain = np.minimum(gain, np.clip((t - seg_start) / float(fade_in), 0, 1))
        if fade_out:
            gain = np.minimum(gain, np.clip((seg_end - t) / float(fade_out), 0, 1))
        if gain.min() >= 1.0:  # a piece away from the segment's ends
            return chunk
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
            elif abs(frame_start - buf_end) <= self.rate * time_base:
                # Contiguous: timestamps rounded to the time base (1 ms in Matroska: AAC frames of 21.33 ms came at
                # 0, 21, 43 ms, ..., and the 16-32 sample "gaps" and "overlaps" clicked, fork fix)
                pass
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
