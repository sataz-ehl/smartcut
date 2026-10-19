"""
smartcut - Smart video cutting with minimal re-encoding

This module provides both low-level and high-level APIs for video processing.

High-level API (recommended for most use cases including Gradio):
    - process(): Process a single segment with fades
    - process_segments(): Process multiple segments with independent fades
    - CutSegmentConfig: Configuration for segment with fade effects

Low-level API (for advanced use):
    - MediaContainer: Video file container
    - smart_cut: Main processing function
    - VideoSettings, AudioExportInfo: Configuration classes
    - FadeInfo, SegmentWithFade: Fade effect data structures
"""

import traceback
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any

from .media_container import MediaContainer
from .media_utils import VideoExportMode, VideoExportQuality
from .misc_data import (
    AudioExportInfo,
    AudioExportSettings,
    FadeInfo,
    SegmentWithFade,
)
from .smart_cut import CancelObject, __version__, smart_cut  # noqa: F401 (__version__: re-exported as smartcut.__version__)
from .video_cutter import VideoSettings

__all__ = [  # noqa: RUF022 (grouped: the high-level API first)
    # High-level API
    "process",
    "process_segments",
    "CutSegmentConfig",
    # Low-level API
    "MediaContainer",
    "smart_cut",
    "VideoSettings",
    "VideoExportMode",
    "VideoExportQuality",
    "AudioExportInfo",
    "AudioExportSettings",
    "FadeInfo",
    "SegmentWithFade",
    "CancelObject",
]

_QUALITY_MAP = {
    "low": VideoExportQuality.LOW,
    "normal": VideoExportQuality.NORMAL,
    "high": VideoExportQuality.HIGH,
    "lossless": VideoExportQuality.LOSSLESS,
}


class CutSegmentConfig:
    """Configuration for a single segment with optional fade effects.

    Args:
        start: Start time in seconds
        end: End time in seconds (None means the end of the video)
        fadein: Fade-in duration for both video and audio (seconds)
        fadeout: Fade-out duration for both video and audio (seconds)
        video_fadein: Fade-in duration for video only (seconds)
        video_fadeout: Fade-out duration for video only (seconds)
        audio_fadein: Fade-in duration for audio only (seconds)
        audio_fadeout: Fade-out duration for audio only (seconds)

    Examples:
        # Both video and audio fade
        CutSegmentConfig(start=10, end=20, fadein=2, fadeout=3)

        # Video fade only
        CutSegmentConfig(start=10, end=20, video_fadein=2, video_fadeout=3)

        # Audio fade only
        CutSegmentConfig(start=10, end=20, audio_fadein=1, audio_fadeout=2)

        # Mixed
        CutSegmentConfig(start=10, end=20, video_fadein=2, audio_fadeout=3)
    """

    def __init__(  # noqa: PLR0917 (public API: the arguments may be passed by position)
        self,
        start: float,
        end: float | None,
        fadein: float | None = None,
        fadeout: float | None = None,
        video_fadein: float | None = None,
        video_fadeout: float | None = None,
        audio_fadein: float | None = None,
        audio_fadeout: float | None = None,
    ) -> None:
        self.start = start
        self.end = end

        # Handle legacy fadein/fadeout (applies to both)
        if fadein is not None:
            self.video_fadein = fadein
            self.audio_fadein = fadein
        else:
            self.video_fadein = video_fadein
            self.audio_fadein = audio_fadein

        if fadeout is not None:
            self.video_fadeout = fadeout
            self.audio_fadeout = fadeout
        else:
            self.video_fadeout = video_fadeout
            self.audio_fadeout = audio_fadeout

    def to_segment_with_fade(self, duration: Fraction | None = None) -> SegmentWithFade:
        """Convert to internal SegmentWithFade representation.

        Args:
            duration: Media duration, used when ``end`` is None (end of the video).
        """
        end = self.end
        if end is None:
            if duration is None:
                raise ValueError("Segment end is None (end of video) but the media duration is unknown")
            end = duration
        fade_info = FadeInfo(
            video_fadein_duration=_exact(self.video_fadein) if self.video_fadein else None,
            video_fadeout_duration=_exact(self.video_fadeout) if self.video_fadeout else None,
            audio_fadein_duration=_exact(self.audio_fadein) if self.audio_fadein else None,
            audio_fadeout_duration=_exact(self.audio_fadeout) if self.audio_fadeout else None,
        )
        return SegmentWithFade(_exact(self.start), _exact(end), fade_info)


def _exact(seconds: float | Fraction) -> Fraction:
    """The decimal a float stands for (3.6 -> 18/5). Fraction(3.6) is the binary value, slightly above 3.6, which
    dropped the frame at exactly 3.6 s from a segment starting there."""
    return Fraction(repr(seconds)) if isinstance(seconds, float) else Fraction(seconds)


class _CallbackCancel(CancelObject):
    """CancelObject whose ``cancelled`` flag asks a callback. smart_cut checks it before every GOP and after
    every re-encoded frame, stops, and removes the partial output file.

    Once the callback returned True the flag stays True without asking again: smart_cut reads it several times
    while stopping, and a callback that answered True only once left a truncated output reported as a success."""

    def __init__(self, cancel_check: Callable[[], bool]) -> None:
        self._cancel_check = cancel_check
        self.requested = False

    @property
    def cancelled(self) -> bool:  # type: ignore[override]
        if not self.requested:
            self.requested = bool(self._cancel_check())
        return self.requested


class _ProgressAdapter:
    """Forwards smart_cut's progress as callback(current, total): thousandths of the estimated work done
    (emit_fraction), called whenever that changes. smart_cut's piece counts (emit) are not used: every piece
    counts the same there, so one slow re-encoded piece held the progress still."""

    def __init__(self, callback: Callable[[int, int], None]) -> None:
        self.callback = callback
        self.last = -1

    def emit(self, value: int) -> None:
        pass

    def emit_fraction(self, fraction: float) -> None:
        current = min(1000, max(0, round(fraction * 1000)))
        if current != self.last:
            self.last = current
            self.callback(current, 1000)


def process(  # noqa: PLR0917 (public API: the arguments may be passed by position)
    input_path: str,
    output_path: str,
    start: float = 0,
    end: float | None = None,
    fadein: float | None = None,
    fadeout: float | None = None,
    video_fadein: float | None = None,
    video_fadeout: float | None = None,
    audio_fadein: float | None = None,
    audio_fadeout: float | None = None,
    quality: str = "normal",
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    """Process a single video segment with optional fade effects.

    This is a high-level convenience function suitable for Gradio and other web apps.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        start: Start time in seconds (default: 0)
        end: End time in seconds (default: end of video)
        fadein: Fade-in duration for both video and audio in seconds
        fadeout: Fade-out duration for both video and audio in seconds
        video_fadein: Fade-in duration for video only in seconds
        video_fadeout: Fade-out duration for video only in seconds
        audio_fadein: Fade-in duration for audio only in seconds
        audio_fadeout: Fade-out duration for audio only in seconds
        quality: Video quality - "normal", "high", "low", or "lossless"
        progress_callback: Optional callback(current, total): the estimated work done, in thousandths
            (total is 1000), called whenever it changes, including during re-encoding
        cancel_check: Optional callback() -> bool; processing stops when it returns True (checked before every
            cut piece and after every re-encoded frame, also without progress_callback; not called again once it
            returned True)

    Returns:
        True if successful, False if an error occurred or processing was cancelled

    Raises:
        ValueError: If the segment is invalid (e.g. start is not before end)

    Examples:
        # Simple cut without fades
        process("input.mp4", "output.mp4", start=10, end=20)

        # With both video and audio fades
        process("input.mp4", "output.mp4", start=10, end=20, fadein=2, fadeout=3)

        # Video fade only
        process("input.mp4", "output.mp4", start=10, end=20,
                video_fadein=2, video_fadeout=3)

        # Audio fade only
        process("input.mp4", "output.mp4", start=10, end=20,
                audio_fadein=1, audio_fadeout=2)

        # With progress callback (Gradio compatible)
        def progress(current, total):
            print(f"Progress: {current}/{total}")

        process("input.mp4", "output.mp4", start=10, end=20,
                fadein=2, progress_callback=progress)
    """
    segment = CutSegmentConfig(
        start=start,
        end=end,
        fadein=fadein,
        fadeout=fadeout,
        video_fadein=video_fadein,
        video_fadeout=video_fadeout,
        audio_fadein=audio_fadein,
        audio_fadeout=audio_fadeout,
    )
    return process_segments(input_path, output_path, [segment], quality=quality,
                            progress_callback=progress_callback, cancel_check=cancel_check)


def process_segments(  # noqa: PLR0917 (public API: the arguments may be passed by position)
    input_path: str,
    output_path: str,
    segments: Sequence[CutSegmentConfig | dict[str, Any]],
    quality: str = "normal",
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    """Process multiple video segments with independent fade effects.

    This is a high-level convenience function suitable for Gradio and other web apps.
    Segments are sorted by start time before processing.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        segments: List of segment configurations (CutSegmentConfig or dict)
        quality: Video quality - "normal", "high", "low", or "lossless"
        progress_callback: Optional callback(current, total): the estimated work done, in thousandths
            (total is 1000), called whenever it changes, including during re-encoding
        cancel_check: Optional callback() -> bool; processing stops when it returns True (checked before every
            cut piece and after every re-encoded frame, also without progress_callback; not called again once it
            returned True) and the partial output file is removed

    Returns:
        True if successful, False if an error occurred or processing was cancelled

    Raises:
        ValueError: If a segment is invalid (e.g. start is not before end)

    Examples:
        # Multiple segments with different fade configurations
        process_segments("input.mp4", "output.mp4", [
            {"start": 10, "end": 20, "video_fadein": 2, "audio_fadeout": 3},
            {"start": 30, "end": 40, "fadein": 1, "fadeout": 2},
            {"start": 50, "end": 60, "audio_fadein": 1.5},
        ])

        # Using CutSegmentConfig objects
        process_segments("input.mp4", "output.mp4", [
            CutSegmentConfig(start=10, end=20, fadein=2, fadeout=3),
            CutSegmentConfig(start=30, end=40, video_fadein=1),
        ])

        # With progress callback
        def progress(current, total):
            print(f"Progress: {current}/{total}")

        process_segments("input.mp4", "output.mp4",
                        segments=[{"start": 10, "end": 20, "fadein": 2}],
                        progress_callback=progress)
    """
    try:
        media_container = MediaContainer(input_path)
    except Exception as e:
        print(f"Error opening video: {e}")
        traceback.print_exc()
        return False

    validated = False
    try:
        segments_with_fade = []
        for seg in segments:
            if isinstance(seg, dict):
                segment_config = CutSegmentConfig(**seg)
            elif isinstance(seg, CutSegmentConfig):
                segment_config = seg
            else:
                raise ValueError(f"Invalid segment type: {type(seg)}")
            segments_with_fade.append(segment_config.to_segment_with_fade(media_container.duration))

        # Validate: start < end for each segment
        for i, seg in enumerate(segments_with_fade):
            if seg.start_time >= seg.end_time:
                raise ValueError(
                    f"Invalid segment {i+1}: start time ({float(seg.start_time):.2f}s) must be before "
                    f"end time ({float(seg.end_time):.2f}s)"
                )

        # Auto-sort segments by start time to ensure chronological order
        segments_with_fade.sort(key=lambda seg: seg.start_time)
        validated = True

        cancel_object = _CallbackCancel(cancel_check) if cancel_check else None
        error = smart_cut(
            media_container=media_container,
            positive_segments=segments_with_fade,
            out_path=output_path,
            # Every audio track, like the CLI (copied when exact, else each one re-encoded as a whole)
            audio_export_info=AudioExportInfo(output_tracks=[AudioExportSettings('passthru')] * len(media_container.audio_tracks)),
            video_settings=VideoSettings(
                mode=VideoExportMode.SMARTCUT,
                quality=_QUALITY_MAP.get(quality.lower(), VideoExportQuality.NORMAL),
            ),
            progress=_ProgressAdapter(progress_callback) if progress_callback else None,
            cancel_object=cancel_object,
        )

        if cancel_object is not None and cancel_object.requested:  # not asking again: the output is complete otherwise
            return False  # smart_cut stopped early and removed the partial output
        return error is None

    except Exception as e:
        if isinstance(e, ValueError) and not validated:
            # Re-raise validation errors (user input errors). Not ValueErrors from processing: FFmpeg's EINVAL
            # (av.error.ValueError) and an unsupported VP9 profile are processing errors.
            raise
        # Catch other exceptions and return False for user-friendly behavior
        print(f"Error processing video: {e}")
        traceback.print_exc()
        return False
    finally:
        media_container.close()
