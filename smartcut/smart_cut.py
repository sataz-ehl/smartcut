import os
from collections.abc import Callable
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
    PassthruAudioCutter,
    RecodeAudioCutter,
    SubtitleCutter,
    create_audio_output_stream,
    create_recode_audio_output_stream,
    create_subtitle_output_stream,
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
# share of the estimated work done. Re-encoded frames count when the encoder outputs them: encoders with
# lookahead (libaom, libvpx, x264) take in their first frames instantly and finish the last ones in a flush.
# The same per-frame hook stops a re-encoded piece midway when cancel_object is set, with or without progress.
# Cost in seconds on one desktop core, measured with tests/regression/progress_timeline.py --one-core --calibrate
# (Oct 2026, 360p and 720p clips; hevc is an estimate). Only the ratios matter. A copied frame costs the whole
# loop: remuxing, audio passthrough and muxing (24-48 us measured; timing the video cutter alone gave 14-19 us).
_RECODE_COST_PER_MEGAPIXEL_FRAME = {'h264': 0.04, 'hevc': 0.15, 'vp9': 0.075, 'av1': 0.18}
_RECODE_COST_PER_MEGAPIXEL_FRAME_OTHER = 0.1
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


class _WorkProgress:
    def __init__(self, progress: WorkProgressCallback | None, cancel_object: CancelObject | None, media_container: MediaContainer,
                 cut_segments: list[CutSegment], recode_audio: int, *, include_video: bool = True) -> None:
        """recode_audio: the number of audio tracks re-encoded as a whole (audio fades)."""
        self.progress = progress
        self.cancel_object = cancel_object
        self.frame_cost = 0.0
        stream = media_container.video_stream
        frame_pts = getattr(media_container, 'video_frame_times_pts', None)
        if stream is not None:
            ctx = stream.codec_context
            name = ctx.name or ''
            family = 'av1' if 'av1' in name else name  # decoder names: h264, hevc, vp9, libdav1d, ...
            cost = _RECODE_COST_PER_MEGAPIXEL_FRAME.get(family, _RECODE_COST_PER_MEGAPIXEL_FRAME_OTHER)
            self.frame_cost = cost * (ctx.width or 0) * (ctx.height or 0) / 1e6
        leading = _cra_leading_frames(media_container, cut_segments) if stream is not None and include_video else None

        media_start, media_end = media_container.start_time, media_container.start_time + media_container.duration
        self.video_total = 0.0  # re-encoded frames, counted as the encoder outputs them
        self.piece_weights = []  # everything else (copied frames, audio), counted when a piece is done
        for i, s in enumerate(cut_segments):
            weight = 0.0
            if stream is not None and stream.time_base is not None and frame_pts is not None and len(frame_pts):
                frames = int(np.searchsorted(frame_pts, float(s.end_time / stream.time_base))
                             - np.searchsorted(frame_pts, float(s.start_time / stream.time_base)))
                if s.require_recode and include_video:
                    self.video_total += frames * self.frame_cost
                else:  # copied; or no video in the output (e.g. .m4a), where pieces only cost their audio
                    weight += frames * _COPY_COST_PER_FRAME
                if leading is not None:
                    self.video_total += leading[i] * self.frame_cost
            if recode_audio:  # pieces at the file edges reach 10 s past it (make_adjusted_segment_times)
                weight += (max(0.0, float(min(s.end_time, media_end) - max(s.start_time, media_start)))
                           * _AUDIO_RECODE_COST_PER_SECOND * recode_audio)
            self.piece_weights.append(weight)
        self.total = (self.video_total + sum(self.piece_weights)) or 1.0
        self.video_done = 0.0
        self.pieces_done = 0.0
        self.finishing = False  # no cancelling inside the generators' finish()

    def frames_encoded(self, count: int) -> None:
        """VideoCutter calls this after every frame it feeds to its encoder and after every encoder flush,
        with the number of frames the encoder output."""
        if not self.finishing and self.cancel_object is not None and self.cancel_object.cancelled:
            raise _Cancelled()
        if count:
            self.video_done += count * self.frame_cost
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
              video_settings: VideoSettings | None = None, segment_mode: bool = False, cancel_object: CancelObject | None = None,
              external_generator_factories: list[StreamGeneratorFactory] | None = None) -> Exception | None:
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

    # Attach fade information to cut segments.
    # Video: GOPs that overlap a video fade region are re-encoded in addition to the partial GOPs at cut
    # points that make_cut_segments already marked; GOPs between the fades stay passthrough (fast).
    # Audio: every GOP of a segment with fades carries fade_info for the audio cutter.
    segment_idx = 0
    for cut_seg in cut_segments:
        # Find which original segment this cut_segment belongs to
        while segment_idx < len(adjusted_segment_times) and cut_seg.start_time >= adjusted_segment_times[segment_idx][1]:
            segment_idx += 1

        fade_info = fade_infos[segment_idx] if segment_idx < len(fade_infos) else None
        cut_seg.fade_info = None
        if fade_info is None:
            continue

        has_video_fadein = fade_info.video_fadein_duration is not None and fade_info.video_fadein_duration > 0
        has_video_fadeout = fade_info.video_fadeout_duration is not None and fade_info.video_fadeout_duration > 0
        has_audio_fadein = fade_info.audio_fadein_duration is not None and fade_info.audio_fadein_duration > 0
        has_audio_fadeout = fade_info.audio_fadeout_duration is not None and fade_info.audio_fadeout_duration > 0
        if not (has_video_fadein or has_video_fadeout or has_audio_fadein or has_audio_fadeout):
            continue

        # Fade bounds in the same timeline as the cut segments and frame times, i.e. including the
        # container start_time. Use the requested times rather than adjusted_segment_times, which become
        # -10 / duration + 10 at the file edges and would break fades there.
        orig_seg_start = basic_segments[segment_idx][0] + media_container.start_time
        orig_seg_end = basic_segments[segment_idx][1] + media_container.start_time
        cut_seg.fade_info = fade_info
        cut_seg.orig_segment_start = orig_seg_start
        cut_seg.orig_segment_end = orig_seg_end

        overlaps_video_fade = (
            (has_video_fadein and cut_seg.start_time < orig_seg_start + fade_info.video_fadein_duration)
            or (has_video_fadeout and cut_seg.end_time > orig_seg_end - fade_info.video_fadeout_duration)
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
    has_audio_fades = any(seg.fade_info is not None and
                          (seg.fade_info.audio_fadein_duration is not None or seg.fade_info.audio_fadeout_duration is not None)
                          for seg in cut_segments)
    work = None
    recode_audio = sum(1 for t in audio_export_info.output_tracks[:len(media_container.audio_tracks)]
                       if t is not None and t.codec == 'passthru') if has_audio_fades and audio_export_info is not None else 0
    previously_done_segments = 0
    for output_path_segment in output_files:
        if cancel_object is not None and cancel_object.cancelled:
            break
        with av.open(output_path_segment[0], 'w') as output_av_container, ExitStack() as cleanup:
            output_av_container.metadata['ENCODED_BY'] = f'smartcut {__version__}'

            include_video = True
            if output_av_container.format.name in ['ogg', 'mp3', 'm4a', 'ipod', 'flac', 'wav']: #ipod is the real name for m4a, I guess
                include_video = False
            if work is None and (isinstance(progress, WorkProgressCallback) or cancel_object is not None):
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

            # Any audio fade (has_audio_fades): re-encode the whole audio track with one encoder (see
            # RecodeAudioCutter). Without audio fades the audio is copied untouched.
            if audio_export_info is not None:
                for track_i, track_export_settings in enumerate(audio_export_info.output_tracks):
                    if track_export_settings is not None and track_export_settings.codec == 'passthru':
                        if has_audio_fades:
                            audio_out_stream = create_recode_audio_output_stream(media_container, output_av_container, track_i)
                            generators.append(RecodeAudioCutter(media_container, audio_out_stream, track_i))
                        else:
                            audio_out_stream = create_audio_output_stream(media_container, output_av_container, track_i)
                            generators.append(PassthruAudioCutter(media_container, audio_out_stream, track_i))

            for sub_track_i in range(len(media_container.subtitle_tracks)):
                subtitle_out_stream = create_subtitle_output_stream(media_container, output_av_container, sub_track_i)
                generators.append(SubtitleCutter(media_container, subtitle_out_stream, sub_track_i))

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
                piece_index = previously_done_segments
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
