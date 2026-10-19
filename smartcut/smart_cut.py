import os
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from fractions import Fraction
from typing import Protocol, TypeAlias, runtime_checkable

import av
import numpy as np
from av.container.output import OutputContainer
from av.packet import Packet

from smartcut.media_container import MediaContainer
from smartcut.media_utils import VideoExportMode
from smartcut.misc_data import AudioExportInfo, CutSegment, FadeInfo, SegmentWithFade
from smartcut.track_cutters import (
    EDIT_LIST_FORMATS,
    RecodeAudioCutter,
    SubtitleCutter,
    audio_needs_recode,
    create_audio_output_stream,
    create_recode_audio_output_stream,
    create_subtitle_output_stream,
    passthru_audio_cutter,
)
from smartcut.video_cutter import (
    VideoCutter,
    VideoSettings,
    create_video_output_stream,
)

__version__ = "1.7"


class ProgressCallback(Protocol):
    """Protocol for progress callback objects."""
    def emit(self, value: int) -> None:
        """Emit progress update."""
        ...


@runtime_checkable
class WorkProgressCallback(ProgressCallback, Protocol):
    """A progress callback that also gets the share of the estimated work done (fork addition, see _WorkProgress)."""
    def emit_fraction(self, fraction: float) -> None:
        """Emit the share of the estimated work done, 0.0 to 1.0."""
        ...


class StreamGenerator(Protocol):
    """Protocol for stream generators that produce packets for output."""
    def segment(self, cut_segment: CutSegment) -> list[Packet]: ...
    def finish(self) -> list[Packet]: ...


StreamGeneratorFactory: TypeAlias = Callable[[OutputContainer], StreamGenerator]


class CancelObject:
    cancelled: bool = False


# Progress by estimated work (fork addition). emit() counts cut pieces, but re-encoding a piece can take a
# thousand times longer than copying one, so a progress object that also has emit_fraction(fraction) gets the
# share of the estimated work done. Re-encoded frames count when the encoder outputs them, in the order they were
# fed: encoders with lookahead (libaom, libvpx, x264, x265) take in their first frames instantly and finish the last
# ones in a flush, which outputs them one by one. The same per-frame hook stops a re-encoded piece midway when
# cancel_object is set, with or without progress.
# Cost in seconds on one desktop core, measured with tests/regression/progress_timeline.py --one-core --calibrate
# (Oct 2026: 29 edits of real clips, 360x360 to 1080p; synthetic clips re-encode 2-3x cheaper). Only the ratios matter.
# - A re-encoded frame costs (fixed + megapixels x per_mp) x (1 - busy + busy x busyness) + megapixels x fade if its
#   video fades, fitted over the encoder runs (each run's frames, from the encoder's start to the end of its flush):
#   - fixed + megapixels x per_mp: a frame of average busyness. 720p: x264 25 ms, x265 60 ms, libvpx 53 ms, libaom
#     129 ms.
#   - An encoder's time per frame follows the content. In one 720p clip x264 took 14 ms per frame on a static scene
#     and 32 ms on a busy one, whose source packets had 1.4 and 5.6 kB per frame (the clip's average: 3.8 kB); libaom
#     118 and 198 ms. Busyness is that ratio: the source's bytes per frame within _BUSY_WINDOW of the frame, against
#     the whole video's. busy is the share of the cost that follows it: x264 0.55, x265 0.75, libaom 0.85; libvpx's
#     time hardly does (0.05).
#   - A faded frame costs x264 and x265 more: every block changes, so none is skipped as unchanged (x264 at 720p:
#     +28 ms, about 2x). libvpx and libaom no more (see progress_timeline.fit_recode_cost).
#   - With one cost per frame and codec, as before, progress was up to 15-20 % off an even pace in common edits
#     (2 segments without fades: H.264 720p 16.8-17.1 %, AV1 720p 14.6 %); now 3-7 % (one core, largest distance at
#     the updates; 26 cases on average 9.5-9.7 % -> 4.4-4.7 %). libvpx's runs vary +-15 % in ways busyness doesn't
#     predict (P2_vp9_720p 8-9 %).
# - hevc: x265 with its default settings, as for camera footage (sources without x265's options in their headers;
#   with them smartcut reuses the source's settings), with the 1-thread pool and 1 frame thread it uses on a 1-CPU
#   machine. The earlier cost (0.005 + 0.020 per megapixel) was measured while x265 escaped the one-core pinning, and
#   the one-core costs with that pool (43-49 / 108-116 / 181 ms per frame at 360p / 720p / 1080p) were mostly encoder
#   closes stalling on Windows (see tests/regression/run_cases.pin_to_one_core); without them x265 takes as long as
#   without a pool. An accurate single cost (0.075 per megapixel) made progress less even than the low one (replayed:
#   14.0 % against 12.2 %), as the errors come from how the cost is spread over the frames: a lower video cost only
#   shrinks re-encoding's share. With busyness and fades the measured level is the most even (5.7 %; half of it 5.7 %,
#   twice 6.6 %).
# - Not counted: decoding the frames before a cut (from the GOP's keyframe; an HEVC open GOP also decodes the GOP
#   before) and starting an encoder (x265: ~1 s per run at 720p). Fitted, they made H.264 progress no more even
#   (replayed: 3.8 % -> 4.0 %) and HEVC more even on average but not in every case (5.9 % -> 4.9 %).
# - A copied frame costs the whole loop: remuxing, audio passthrough and muxing: 33-35 us over the 29 edits (11 884
#   frames; timing the video cutter alone 14-19 us). HEVC pieces that re-encode their leading pictures count as
#   re-encoding (calibrated as copies, they made copied frames look 7x dearer).
# - Re-encoded audio: 0.027-0.029 s per second of audio over three calibrations (903 s each; HE-AAC sources ~0.032).
_RECODE_COST = {  # (fixed, per_mp, busy, fade): s per frame, s per megapixel-frame, share, s per faded megapixel-frame
    'h264': (0.0025, 0.024, 0.55, 0.030),
    'hevc': (0.0035, 0.061, 0.75, 0.034),
    'vp9': (0.011, 0.046, 0.05, 0.0),
    'av1': (0.037, 0.10, 0.85, 0.0),
}
_RECODE_COST_OTHER = (0.005, 0.1, 0.0, 0.0)
# Busyness: the source's bytes per frame within this many seconds either side of the frame, relative to the
# whole video's
_BUSY_WINDOW = 0.5
_COPY_COST_PER_FRAME = 0.000035
_AUDIO_RECODE_COST_PER_SECOND = 0.027


class _Cancelled(Exception):
    """Stops a re-encoded piece midway when the cancel object is set."""


def _cra_leading_frames(media_container: MediaContainer, cut_segments: list[CutSegment]) -> list[int]:
    """Per cut piece, the leading pictures VideoCutter re-encodes in a copied piece: the RASL/RADL frames of an HEVC
    CRA GOP right after a jump (VideoCutter._should_hybrid_recode_cra, same bookkeeping). Counted from the DTS span
    between the CRA and its first trailing picture."""
    has_rasl = getattr(media_container, 'gop_has_rasl', [])
    frame_pts = getattr(media_container, 'video_frame_times_pts', None)
    tick = float(np.median(np.diff(frame_pts))) if any(has_rasl) and frame_pts is not None and len(frame_pts) > 1 else 0.0
    if tick <= 0:  # tick: the DTS step of one frame
        return [0] * len(cut_segments)
    counts, last_copied_gop = [], None
    for s in cut_segments:
        n, g = 0, s.gop_index
        if not s.require_recode:
            jump = (last_copied_gop is None and g > 0) or (last_copied_gop is not None and g > last_copied_gop + 1)
            leading_end = media_container.gop_leading_end_dts[g] if 0 <= g < len(has_rasl) and has_rasl[g] else None
            if jump and leading_end is not None:
                n = max(0, round((leading_end - media_container.gop_start_times_dts[g]) / tick) - 1)
            last_copied_gop = g
        counts.append(n)
    return counts


def _recoded_frames(media_container: MediaContainer, cut_segments: list[CutSegment]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per cut piece, the frames VideoCutter re-encodes, in the order it feeds them (fork addition): each one's
    busyness (see _BUSY_WINDOW; 1.0 without packet sizes) and whether its video fades (as in
    VideoCutter._apply_fade_to_frame). Re-encoded pieces re-encode their frames, copied HEVC pieces right after a jump
    their leading pictures (_cra_leading_frames), other pieces none."""
    stream = media_container.video_stream
    frame_pts = getattr(media_container, 'video_frame_times_pts', None)
    none = (np.zeros(0), np.zeros(0, bool))
    if stream is None or stream.time_base is None or frame_pts is None or not len(frame_pts):
        return [none] * len(cut_segments)
    time_base = stream.time_base
    times = frame_pts * float(time_base)
    sizes = getattr(media_container, 'video_frame_sizes', None)
    total = None  # cumulative bytes
    if sizes is not None and len(sizes) == len(frame_pts) and sizes.sum() > 0:
        total = np.concatenate([[0.0], np.cumsum(sizes, dtype=np.float64)])
    leading = _cra_leading_frames(media_container, cut_segments)
    result = []
    for s, n_leading in zip(cut_segments, leading):
        first = int(np.searchsorted(frame_pts, float(s.start_time / time_base)))
        last = int(np.searchsorted(frame_pts, float(s.end_time / time_base))) if s.require_recode else min(first + n_leading, len(frame_pts))
        if last <= first:
            result.append(none)
            continue
        t = times[first:last]
        if total is not None:
            lo = np.searchsorted(times, t - _BUSY_WINDOW)
            hi = np.searchsorted(times, t + _BUSY_WINDOW, side='right')
            busy = (total[hi] - total[lo]) / np.maximum(hi - lo, 1) / (total[-1] / len(times))
        else:
            busy = np.ones(len(t))
        faded = np.zeros(len(t), bool)
        fade = s.fade_info
        if s.require_recode and fade is not None:
            seg_start = float(s.orig_segment_start if s.orig_segment_start is not None else s.start_time)
            seg_end = float(s.orig_segment_end if s.orig_segment_end is not None else s.end_time)
            if fade.video_fadein_duration:
                faded |= t - seg_start < float(fade.video_fadein_duration)
            if fade.video_fadeout_duration:
                faded |= seg_end - t < float(fade.video_fadeout_duration)
        result.append((busy, faded))
    return result


def _recode_cost_of(media_container: MediaContainer) -> tuple[float, float, float]:
    """For this source (_RECODE_COST): the cost of a re-encoded frame of average busyness, the share of it that
    follows busyness, and the extra cost of a faded frame."""
    stream = media_container.video_stream
    if stream is None:
        return 0.0, 0.0, 0.0
    ctx = stream.codec_context
    name = ctx.name or ''
    family = 'av1' if 'av1' in name else name  # decoder names: h264, hevc, vp9, libdav1d, ...
    fixed, per_mp, busy, fade = _RECODE_COST.get(family, _RECODE_COST_OTHER)
    megapixels = (ctx.width or 0) * (ctx.height or 0) / 1e6
    return fixed + per_mp * megapixels, busy, fade * megapixels


class _WorkProgress:
    def __init__(self, progress: WorkProgressCallback | None, cancel_object: CancelObject | None, media_container: MediaContainer,
                 cut_segments: list[CutSegment], recode_audio: list[int], *, include_video: bool = True) -> None:
        """recode_audio: per cut piece, the number of audio tracks its output file re-encodes as a whole."""
        self.progress = progress
        self.cancel_object = cancel_object
        stream = media_container.video_stream
        frame_pts = getattr(media_container, 'video_frame_times_pts', None)
        frame_cost, busy, fade = _recode_cost_of(media_container)
        none = (np.zeros(0), np.zeros(0, bool))
        recoded = _recoded_frames(media_container, cut_segments) if include_video else [none] * len(cut_segments)

        media_start, media_end = media_container.start_time, media_container.start_time + media_container.duration
        self.frame_costs: list[np.ndarray] = []  # per piece, the cost of each frame it re-encodes
        self.piece_weights: list[float] = []  # everything else (copied frames, audio), counted when a piece is done
        for i, s in enumerate(cut_segments):
            busyness, faded = recoded[i]
            self.frame_costs.append(frame_cost * (1 - busy + busy * busyness) + fade * faded)
            weight = 0.0
            if (not s.require_recode or not include_video) and stream is not None and stream.time_base is not None \
                    and frame_pts is not None and len(frame_pts):
                # copied; or no video in the output (e.g. .m4a), where pieces only cost their audio
                frames = int(np.searchsorted(frame_pts, float(s.end_time / stream.time_base))
                             - np.searchsorted(frame_pts, float(s.start_time / stream.time_base)))
                weight += frames * _COPY_COST_PER_FRAME
            if recode_audio[i]:  # pieces at the file edges reach 10 s past it (make_adjusted_segment_times)
                weight += (max(0.0, float(min(s.end_time, media_end) - max(s.start_time, media_start)))
                           * _AUDIO_RECODE_COST_PER_SECOND * recode_audio[i])
            self.piece_weights.append(weight)
        self.video_total = float(sum(c.sum() for c in self.frame_costs))
        self.total = (self.video_total + sum(self.piece_weights)) or 1.0
        # A frame fed beyond the estimate (it can be a few frames off) costs the average
        frames = sum(len(c) for c in self.frame_costs)
        self.average_cost = self.video_total / frames if frames else frame_cost
        self.feeding: Iterator[float] = iter(())  # the costs of the current piece's frames not fed yet
        self.queued: deque[float] = deque()  # the costs of the frames fed to the encoder and not output yet
        self.video_done = 0.0
        self.pieces_done = 0.0
        self.finishing = False  # no cancelling inside the generators' finish()

    def piece_started(self, index: int) -> None:
        self.feeding = iter(self.frame_costs[index].tolist())

    def frames_encoded(self, fed: int, out: int) -> None:
        """VideoCutter calls this after every frame it feeds to its encoder (fed 1) and after every frame an encoder
        flush outputs (fed 0, out 1), with the number of frames the encoder output."""
        if not self.finishing and self.cancel_object is not None and self.cancel_object.cancelled:
            raise _Cancelled()
        for _ in range(fed):
            self.queued.append(next(self.feeding, self.average_cost))
        for _ in range(min(out, len(self.queued))):
            self.video_done += self.queued.popleft()
        if out:
            self._emit()

    def piece_done(self, index: int) -> None:
        self.pieces_done += self.piece_weights[index]
        self._emit()

    def done(self) -> None:
        if self.progress is not None:
            self.progress.emit_fraction(1.0)  # the frame estimate can be a few frames off

    def _emit(self) -> None:
        if self.progress is not None:
            self.progress.emit_fraction(min(1.0, (min(self.video_done, self.video_total) + self.pieces_done) / self.total))


class OutputFormatError(ValueError):
    """The output container can't hold the input's video codec (fork addition)."""


_CODEC_NAMES = {'h264': 'H.264', 'hevc': 'H.265 (HEVC)', 'mpeg4': 'MPEG-4 Part 2', 'mpeg2video': 'MPEG-2'}


def _check_webm_video(media_container: MediaContainer, out_path: str, video_settings: VideoSettings) -> None:
    """WebM holds only VP8, VP9 and AV1 video, and smartcut copies the video except around cuts (fork addition).
    Say so before writing anything: FFmpeg's error named an encoder ("'webm' format does not support 'libx264'")."""
    stream = media_container.video_stream
    if os.path.splitext(out_path)[1].lower() != '.webm' or stream is None or video_settings.codec_override != 'copy':
        return
    name = stream.codec_context.name or ''
    if name not in ('vp8', 'vp9', 'libvpx', 'libvpx-vp9') and 'av1' not in name:  # AV1 decoders: libdav1d, av1, ...
        raise OutputFormatError(f"WebM output needs VP8, VP9 or AV1 video; this video is {_CODEC_NAMES.get(name, name.upper())}: "
                                "use .mp4 or .mkv")


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


def expand_segments_with_fades(segments: Sequence[tuple[Fraction, Fraction] | SegmentWithFade]) -> tuple[list[tuple[Fraction, Fraction]], list[FadeInfo | None]]:
    """
    Expand segments with fade information into basic segments and their corresponding fade info.

    This function doesn't split segments for minimal re-encoding yet - that will be done
    at the CutSegment level. It just prepares the data structure.

    Args:
        segments: Segments with fade information, or (start, end) tuples (the legacy format, no fades)

    Returns:
        Tuple of (basic_segments, fade_infos) where fade_infos[i] corresponds to basic_segments[i]
    """
    basic_segments = []
    fade_infos: list[FadeInfo | None] = []

    for seg in segments:
        if isinstance(seg, SegmentWithFade):
            basic_segments.append((seg.start_time, seg.end_time))
            fade_infos.append(seg.fade_info)
        else:
            basic_segments.append(seg)
            fade_infos.append(None)

    return basic_segments, fade_infos


def smart_cut(media_container: MediaContainer, positive_segments: list[tuple[Fraction, Fraction]] | list[SegmentWithFade],
              out_path: str, audio_export_info: AudioExportInfo | None = None, log_level: str | None = None, progress: ProgressCallback | None = None,
              video_settings: VideoSettings | None = None, segment_mode: bool = False, cancel_object: CancelObject | None = None,
              external_generator_factories: list[StreamGeneratorFactory] | None = None) -> Exception | None:
    if video_settings is None:
        video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)
    _check_webm_video(media_container, out_path, video_settings)

    # Handle both legacy tuple format and new SegmentWithFade format
    basic_segments, fade_infos = expand_segments_with_fades(positive_segments)

    adjusted_segment_times = make_adjusted_segment_times(basic_segments, media_container)
    cut_segments = make_cut_segments(media_container, adjusted_segment_times, video_settings.mode == VideoExportMode.KEYFRAMES)

    # Attach fade information to cut segments.
    # Video: GOPs that overlap a video fade region are re-encoded in addition to the partial GOPs at cut
    # points that make_cut_segments already marked; GOPs between the fades stay passthrough (fast).
    # Audio: every GOP of a segment with fades carries fade_info for the audio cutter.
    # Every piece gets its segment's bounds: fades, and the short audio fades at joins, are timed from them.
    piece_segment = []  # per cut piece, the index of its segment
    segment_idx = 0
    for cut_seg in cut_segments:
        # Find which original segment this cut_segment belongs to
        while segment_idx < len(adjusted_segment_times) and cut_seg.start_time >= adjusted_segment_times[segment_idx][1]:
            segment_idx += 1
        piece_segment.append(segment_idx)

        cut_seg.fade_info = None
        if segment_idx >= len(basic_segments):
            continue
        # Fade bounds in the same timeline as the cut segments and frame times, i.e. including the
        # container start_time. Use the requested times rather than adjusted_segment_times, which become
        # -10 / duration + 10 at the file edges and would break fades there.
        orig_seg_start = basic_segments[segment_idx][0] + media_container.start_time
        orig_seg_end = basic_segments[segment_idx][1] + media_container.start_time
        cut_seg.orig_segment_start = orig_seg_start
        cut_seg.orig_segment_end = orig_seg_end

        fade_info = fade_infos[segment_idx]
        if fade_info is None:
            continue
        video_fadein = fade_info.video_fadein_duration or 0
        video_fadeout = fade_info.video_fadeout_duration or 0
        if not (video_fadein > 0 or video_fadeout > 0 or (fade_info.audio_fadein_duration or 0) > 0
                or (fade_info.audio_fadeout_duration or 0) > 0):
            continue
        cut_seg.fade_info = fade_info

        overlaps_video_fade = (
            (video_fadein > 0 and cut_seg.start_time < orig_seg_start + video_fadein)
            or (video_fadeout > 0 and cut_seg.end_time > orig_seg_end - video_fadeout)
        )
        # Never turn re-encoding off: partial GOPs at cut points must be re-encoded for a frame-accurate cut
        cut_seg.require_recode = cut_seg.require_recode or overlaps_video_fade

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

    # The cut pieces of each output file, and its joins (fork addition): consecutive pieces that aren't contiguous in
    # the source. The pieces of one segment are, and so are segments that touch.
    file_pieces: list[range] = []
    file_has_joins: list[bool] = []
    for _, (_, file_end) in output_files:
        first = last = file_pieces[-1].stop if file_pieces else 0
        while last < len(cut_segments) and cut_segments[last].start_time < file_end:
            last += 1
        file_pieces.append(range(first, last))
        starts_at_join, ends_at_join = set(), set()  # segment indices
        for i in range(first + 1, last):
            if cut_segments[i].start_time != cut_segments[i - 1].end_time:
                starts_at_join.add(piece_segment[i])
                ends_at_join.add(piece_segment[i - 1])
        for i in range(first, last):
            cut_segments[i].join_before = piece_segment[i] in starts_at_join
            cut_segments[i].join_after = piece_segment[i] in ends_at_join
        file_has_joins.append(bool(starts_at_join))

    def recode_tracks_of(file_index: int, format_name: str) -> set[int]:
        """The audio tracks of output file file_index that are re-encoded as a whole (see RecodeAudioCutter): every one
        if its pieces have audio fades, else those a copy wouldn't be exact for (fork addition)."""
        pieces = [cut_segments[i] for i in file_pieces[file_index]]
        if audio_export_info is None or not pieces:
            return set()
        has_audio_fades = any(seg.fade_info is not None and
                              (seg.fade_info.audio_fadein_duration is not None or seg.fade_info.audio_fadeout_duration is not None)
                              for seg in pieces)
        return {i for i, t in enumerate(audio_export_info.output_tracks[:len(media_container.audio_tracks)])
                if t is not None and t.codec == 'passthru'
                and (has_audio_fades or audio_needs_recode(media_container, format_name, i, pieces[0], file_has_joins[file_index]))}

    work = None
    previously_done_segments = 0
    for file_index, output_path_segment in enumerate(output_files):
        if cancel_object is not None and cancel_object.cancelled:
            break
        with av.open(output_path_segment[0], 'w') as output_av_container, ExitStack() as cleanup:
            output_av_container.metadata['ENCODED_BY'] = f'smartcut {__version__}'

            include_video = True
            if output_av_container.format.name in ['ogg', 'mp3', 'm4a', 'ipod', 'flac', 'wav']: #ipod is the real name for m4a, I guess
                include_video = False
            format_name = output_av_container.format.name or ''
            recode_tracks = recode_tracks_of(file_index, format_name)
            if work is None and (isinstance(progress, WorkProgressCallback) or cancel_object is not None):
                # Per piece, the audio tracks its file re-encodes (segment mode: the other files are named like this one,
                # so they have its format)
                recode_audio = [0] * len(cut_segments)
                for k, pieces in enumerate(file_pieces):
                    count = len(recode_tracks if k == file_index else recode_tracks_of(k, format_name))
                    for i in pieces:
                        recode_audio[i] = count
                work = _WorkProgress(progress if isinstance(progress, WorkProgressCallback) else None, cancel_object,
                                     media_container, cut_segments, recode_audio, include_video=include_video)

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
                video_stream_setup = create_video_output_stream(media_container, output_av_container, video_settings)
                generators.append(VideoCutter(media_container, video_stream_setup, output_av_container, video_settings, log_level))
                # Its own input container: finish() closes it, this also when an error stops the cut (fork addition)
                cleanup.callback(generators[-1].input_av_container.close)
                if work is not None:
                    generators[-1].on_frames_encoded = work.frames_encoded

            if external_generator_factories:
                for factory in external_generator_factories:
                    generators.append(factory(output_av_container))

            # Re-encode the whole audio track with one encoder (see RecodeAudioCutter) with audio fades, or when a copy
            # wouldn't be exact (audio_needs_recode: joins, a mid-stream start without an exact pre-roll copy, WebM).
            # Otherwise the audio is copied untouched.
            if audio_export_info is not None:
                for track_i, track_export_settings in enumerate(audio_export_info.output_tracks):
                    if track_export_settings is not None and track_export_settings.codec == 'passthru':
                        if track_i in recode_tracks:
                            audio_out_stream = create_recode_audio_output_stream(media_container, output_av_container, track_i)
                            generators.append(RecodeAudioCutter(media_container, audio_out_stream, track_i, output_av_container))
                        else:
                            audio_out_stream = create_audio_output_stream(media_container, output_av_container, track_i)
                            generators.append(passthru_audio_cutter(media_container, audio_out_stream, track_i,
                                                                    cut_segments[file_pieces[file_index].start],
                                                                    output_av_container))

            for sub_track_i in range(len(media_container.subtitle_tracks)):
                subtitle_out_stream = create_subtitle_output_stream(media_container, output_av_container, sub_track_i)
                generators.append(SubtitleCutter(media_container, subtitle_out_stream, sub_track_i))

            # MP4 writes the edit lists that place the first video frame and end the audio in the movie timescale:
            # FFmpeg's default of 1000 rounds them down to the ms, so the video started up to 1 ms early (fork fix)
            if format_name in EDIT_LIST_FORMATS:
                timescale = next((s.time_base.denominator for s in output_av_container.streams.video if s.time_base),
                                 next((s.sample_rate for s in output_av_container.streams.audio if s.sample_rate), None))
                if timescale:
                    output_av_container.options['movie_timescale'] = str(timescale)

            output_av_container.start_encoding()
            if progress is not None:
                progress.emit(len(cut_segments))
            for piece_index in file_pieces[file_index]:
                s = cut_segments[piece_index]
                if cancel_object is not None and cancel_object.cancelled:
                    break
                if work is not None:
                    work.piece_started(piece_index)

                if progress is not None:
                    progress.emit(previously_done_segments)
                previously_done_segments += 1
                assert s.start_time < s.end_time, f"Invalid segment: start_time {s.start_time} >= end_time {s.end_time}"
                try:
                    for g in generators:
                        for packet in g.segment(s):
                            if packet.dts is not None and packet.dts < -900_000:
                                packet.dts = None
                            if packet.dts is not None and packet.dts > 1_000_000_000_000:
                                print(f"BAD DTS: seg {s.start_time:.3f}-{s.end_time:.3f} gop={s.gop_index} recode={s.require_recode} pts={packet.pts} dts={packet.dts}")
                            output_av_container.mux(packet)
                except _Cancelled:
                    break  # cancelled midway through a re-encoded piece; the partial output is removed below
                if work is not None:
                    work.piece_done(piece_index)
            if cancel_object is not None and cancel_object.cancelled:
                for g in generators:  # don't drain the encoder into a file that is removed (fork addition)
                    if isinstance(g, VideoCutter):
                        g.enc_codec = None
            if work is not None:
                work.finishing = True
            for g in generators:
                for packet in g.finish():
                    if packet.dts is not None and packet.dts > 1_000_000_000_000:
                        print(f"BAD DTS in finish: pts={packet.pts} dts={packet.dts}", flush=True)
                    output_av_container.mux(packet)
            if work is not None:
                work.finishing = False
            if progress is not None:
                progress.emit(previously_done_segments)

        if cancel_object is not None and cancel_object.cancelled:
            last_file_path = output_path_segment[0]

            if os.path.exists(last_file_path):
                os.remove(last_file_path)
        elif work is not None and output_path_segment is output_files[-1]:
            work.done()


# Re-export commonly used types for convenience
from smartcut.media_utils import VideoExportQuality  # noqa: E402
