import logging
import os
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, TypeAlias, cast, runtime_checkable

import av
import numpy as np
from av.container.output import OutputContainer
from av.packet import Packet

from smartcut import progress_costs
from smartcut.media_container import MediaContainer
from smartcut.media_utils import VideoExportMode
from smartcut.misc_data import AudioExportInfo, CutSegment, FadeInfo, SegmentWithFade
from smartcut.track_cutters import (
    AudioRecode,
    RecodeAudioCutter,
    SubtitleCutter,
    audio_needs_recode,
    create_audio_output_stream,
    create_recode_audio_output_stream,
    create_subtitle_output_stream,
    passthru_audio_cutter,
    recode_setup,
    webm_holds,
)
from smartcut.video_cutter import (
    VideoCutter,
    VideoSettings,
    create_video_output_stream,
)

__version__ = "1.7"

log = logging.getLogger("smartcut")


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
# cancel_object is set, with or without progress, and the encoder's last flush too (in finish()).
# The costs the work is weighed with, and how they were measured: progress_costs.py (built-in, or a machine's own from
# its progress_costs.json).


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
    busyness (see progress_costs.BUSY_WINDOW; 1.0 without packet sizes) and whether its video fades (as in
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
            lo = np.searchsorted(times, t - progress_costs.BUSY_WINDOW)
            hi = np.searchsorted(times, t + progress_costs.BUSY_WINDOW, side='right')
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


def _codec_family(media_container: MediaContainer) -> str:
    """The video's codec family, as progress_costs.RECODE_COST names it: decoder names h264, hevc, vp9, libdav1d, ..."""
    stream = media_container.video_stream
    name = (stream.codec_context.name or '') if stream is not None else ''
    return 'av1' if 'av1' in name else name


@dataclass
class _WorkPlan:
    """What a cut is, as far as its cost goes, without the cost constants (fork addition): calibrate_progress.py records
    it with a cut's progress calls and replays them under other constants. Per cut piece: the busyness and faded flags
    of the frames it re-encodes (_recoded_frames), the frames it copies, and its seconds of re-encoded audio times the
    tracks re-encoded."""
    family: str
    megapixels: float
    busyness: list[np.ndarray]
    faded: list[np.ndarray]
    copied_frames: list[int]
    audio_seconds: list[float]


def _work_plan(media_container: MediaContainer, cut_segments: list[CutSegment], recode_audio: list[int],
               include_video: bool = True) -> _WorkPlan:
    """recode_audio: per cut piece, the number of audio tracks its output file re-encodes as a whole. Without video in
    the output (e.g. .m4a) pieces only cost their audio and copied frames."""
    stream = media_container.video_stream
    frame_pts = getattr(media_container, 'video_frame_times_pts', None)
    none = (np.zeros(0), np.zeros(0, bool))
    recoded = _recoded_frames(media_container, cut_segments) if include_video else [none] * len(cut_segments)
    ctx = stream.codec_context if stream is not None else None
    media_start, media_end = media_container.start_time, media_container.start_time + media_container.duration
    copied, audio = [], []
    for i, s in enumerate(cut_segments):
        frames = 0
        if (not s.require_recode or not include_video) and stream is not None and stream.time_base is not None \
                and frame_pts is not None and len(frame_pts):
            frames = int(np.searchsorted(frame_pts, float(s.end_time / stream.time_base))
                         - np.searchsorted(frame_pts, float(s.start_time / stream.time_base)))
        copied.append(frames)
        # pieces at the file edges reach 10 s past it (make_adjusted_segment_times)
        seconds = max(0.0, float(min(s.end_time, media_end) - max(s.start_time, media_start))) if recode_audio[i] else 0.0
        audio.append(seconds * recode_audio[i])
    return _WorkPlan(_codec_family(media_container), (ctx.width or 0) * (ctx.height or 0) / 1e6 if ctx is not None else 0.0,
                     [r[0] for r in recoded], [r[1] for r in recoded], copied, audio)


class _WorkProgress:
    def __init__(self, progress: WorkProgressCallback | None, cancel_object: CancelObject | None, plan: _WorkPlan,
                 costs: progress_costs.ProgressCosts) -> None:
        self.progress = progress
        self.cancel_object = cancel_object
        fixed, per_mp, busy, fade = costs.recode_for(plan.family)
        frame_cost = fixed + per_mp * plan.megapixels  # a frame of average busyness
        self.frame_costs: list[np.ndarray] = [  # per piece, the cost of each frame it re-encodes
            frame_cost * (1 - busy + busy * busyness) + fade * plan.megapixels * faded
            for busyness, faded in zip(plan.busyness, plan.faded)]
        self.piece_weights: list[float] = [  # everything else (copied frames, audio), counted when a piece is done
            frames * costs.copy_per_frame + seconds * costs.audio_per_second
            for frames, seconds in zip(plan.copied_frames, plan.audio_seconds)]
        self.video_total = float(sum(c.sum() for c in self.frame_costs))
        self.total = (self.video_total + sum(self.piece_weights)) or 1.0
        # A frame fed beyond the estimate (it can be a few frames off) costs the average
        frames = sum(len(c) for c in self.frame_costs)
        self.average_cost = self.video_total / frames if frames else frame_cost
        self.feeding: Iterator[float] = iter(())  # the costs of the current piece's frames not fed yet
        self.queued: deque[float] = deque()  # the costs of the frames fed to the encoder and not output yet
        self.video_done = 0.0
        self.pieces_done = 0.0

    def piece_started(self, index: int) -> None:
        self.feeding = iter(self.frame_costs[index].tolist())

    def frames_encoded(self, fed: int, out: int) -> None:
        """VideoCutter calls this after every frame it feeds to its encoder (fed 1) and after every frame an encoder
        flush outputs (fed 0, out 1), with the number of frames the encoder output. Raises _Cancelled when the cancel
        object is set, also in the last flush (VideoCutter.finish)."""
        if self.cancel_object is not None and self.cancel_object.cancelled:
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


def _check_webm_audio(media_container: MediaContainer, out_path: str, audio_export_info: AudioExportInfo | None) -> None:
    """WebM holds only Opus and Vorbis audio, so other tracks are re-encoded to Opus (fork addition). Say so before
    writing anything if one can't be (e.g. a channel layout without a channel order)."""
    if os.path.splitext(out_path)[1].lower() != '.webm' or audio_export_info is None:
        return
    for i, t in enumerate(audio_export_info.output_tracks[:len(media_container.audio_tracks)]):
        track = media_container.audio_tracks[i]
        if t is not None and t.codec == 'passthru' and not webm_holds(track):
            setup = recode_setup(track, 'webm')
            if isinstance(setup, str):
                raise OutputFormatError(f"WebM output needs Opus or Vorbis audio, and audio track {i + 1} "
                                        f"({track.av_stream.codec_context.name}) can't be re-encoded to Opus: {setup}; use .mkv")


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
    _check_webm_audio(media_container, out_path, audio_export_info)

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
    # the source. The pieces of one segment are, and so are segments that touch. Each piece gets where its segment's
    # pieces start and end at a join: the audio's join fades are timed from what is actually kept there (whole GOPs
    # in KEYFRAMES mode; the requested bounds muted the audio the video kept beyond them).
    file_pieces: list[range] = []
    file_has_joins: list[bool] = []
    for _, (_, file_end) in output_files:
        first = last = file_pieces[-1].stop if file_pieces else 0
        while last < len(cut_segments) and cut_segments[last].start_time < file_end:
            last += 1
        file_pieces.append(range(first, last))
        join_start: dict[int, Fraction] = {}  # per segment index: its first piece's start, if at a join
        join_end: dict[int, Fraction] = {}  # its last piece's end, if at a join
        for i in range(first + 1, last):
            if cut_segments[i].start_time != cut_segments[i - 1].end_time:
                join_start[piece_segment[i]] = cut_segments[i].start_time
                join_end[piece_segment[i - 1]] = cut_segments[i - 1].end_time
        for i in range(first, last):
            cut_segments[i].join_start = join_start.get(piece_segment[i])
            cut_segments[i].join_end = join_end.get(piece_segment[i])
        file_has_joins.append(bool(join_start))

    keyframes = video_settings.mode == VideoExportMode.KEYFRAMES
    setups: dict[tuple[int, str], AudioRecode | str] = {}  # recode_setup per (track, output format)
    warned: set[int] = set()

    def recode_tracks_of(file_index: int, format_name: str) -> set[int]:
        """The audio tracks of output file file_index that are re-encoded as a whole (see RecodeAudioCutter): every one
        if its pieces have audio fades, else those a copy wouldn't be exact for, as long as it can be re-encoded
        faithfully (recode_setup); a track that can't is copied, with a warning (fork addition)."""
        pieces = [cut_segments[i] for i in file_pieces[file_index]]
        if audio_export_info is None or not pieces:
            return set()
        has_audio_fades = any(seg.fade_info is not None and
                              (seg.fade_info.audio_fadein_duration is not None or seg.fade_info.audio_fadeout_duration is not None)
                              for seg in pieces)
        tracks = set()
        for i, t in enumerate(audio_export_info.output_tracks[:len(media_container.audio_tracks)]):
            if t is None or t.codec != 'passthru' or not (
                    has_audio_fades or audio_needs_recode(media_container, format_name, i, pieces[0], file_has_joins[file_index],
                                                          keyframes=keyframes)):
                continue
            setup = setups.get((i, format_name))
            if setup is None:
                setup = setups[(i, format_name)] = recode_setup(media_container.audio_tracks[i], format_name)
            if isinstance(setup, AudioRecode):
                tracks.add(i)
            elif i not in warned:
                warned.add(i)
                ctx = media_container.audio_tracks[i].av_stream.codec_context
                result = ("its audio fades are left out" if has_audio_fades else
                          "A/V may shift by up to a frame at each join" if file_has_joins[file_index] else
                          "it starts at the first whole frame after the cut")
                log.warning(f"smartcut: audio track {i + 1} ({ctx.name}, {ctx.layout.name}) is copied, not re-encoded: "
                            f"{setup}. So {result}.")
        return tracks

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
                                     _work_plan(media_container, cut_segments, recode_audio, include_video),
                                     progress_costs.current())

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

            # Re-encode the whole audio track with one encoder (see RecodeAudioCutter) with audio fades, at joins, for
            # lossless tracks starting mid-stream and in WebM (audio_needs_recode), when it can be done faithfully
            # (recode_setup). Otherwise the audio is copied untouched.
            if audio_export_info is not None:
                for track_i, track_export_settings in enumerate(audio_export_info.output_tracks):
                    if track_export_settings is not None and track_export_settings.codec == 'passthru':
                        if track_i in recode_tracks:
                            audio_out_stream = create_recode_audio_output_stream(
                                media_container, output_av_container, track_i, cast(AudioRecode, setups[(track_i, format_name)]))
                            generators.append(RecodeAudioCutter(media_container, audio_out_stream, track_i))
                        else:
                            audio_out_stream = create_audio_output_stream(media_container, output_av_container, track_i)
                            generators.append(passthru_audio_cutter(media_container, audio_out_stream, track_i,
                                                                    cut_segments[file_pieces[file_index].start],
                                                                    output_av_container))

            for sub_track_i in range(len(media_container.subtitle_tracks)):
                subtitle_out_stream = create_subtitle_output_stream(media_container, output_av_container, sub_track_i)
                generators.append(SubtitleCutter(media_container, subtitle_out_stream, sub_track_i))

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
            for g in generators:
                try:
                    packets = g.finish()
                except _Cancelled:
                    # Cancelled during the video encoder's last flush (fork fix): its other frames are dropped, as above,
                    # and the output is removed below. The input container is closed by cleanup.
                    if isinstance(g, VideoCutter):
                        g.enc_codec = None
                    continue
                for packet in packets:
                    if packet.dts is not None and packet.dts > 1_000_000_000_000:
                        print(f"BAD DTS in finish: pts={packet.pts} dts={packet.dts}", flush=True)
                    output_av_container.mux(packet)
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
