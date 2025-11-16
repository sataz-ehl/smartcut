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

from fractions import Fraction
from typing import Any, Callable, Optional, Union

from .media_container import MediaContainer
from .cut_video import (
    smart_cut,
    VideoSettings,
    VideoExportMode,
    VideoExportQuality,
    CancelObject,
)
from .misc_data import (
    FadeInfo,
    SegmentWithFade,
    AudioExportInfo,
    AudioExportSettings,
)

__version__ = "1.5.0"
__all__ = [
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


class CutSegmentConfig:
    """Configuration for a single segment with optional fade effects.

    Args:
        start: Start time in seconds
        end: End time in seconds
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

    def __init__(
        self,
        start: float,
        end: float,
        fadein: Optional[float] = None,
        fadeout: Optional[float] = None,
        video_fadein: Optional[float] = None,
        video_fadeout: Optional[float] = None,
        audio_fadein: Optional[float] = None,
        audio_fadeout: Optional[float] = None,
    ):
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

    def to_segment_with_fade(self) -> SegmentWithFade:
        """Convert to internal SegmentWithFade representation."""
        fade_info = FadeInfo(
            video_fadein_duration=Fraction(self.video_fadein) if self.video_fadein else None,
            video_fadeout_duration=Fraction(self.video_fadeout) if self.video_fadeout else None,
            audio_fadein_duration=Fraction(self.audio_fadein) if self.audio_fadein else None,
            audio_fadeout_duration=Fraction(self.audio_fadeout) if self.audio_fadeout else None,
        )
        return SegmentWithFade(Fraction(self.start), Fraction(self.end), fade_info)


def process(
    input_path: str,
    output_path: str,
    start: float = 0,
    end: Optional[float] = None,
    fadein: Optional[float] = None,
    fadeout: Optional[float] = None,
    video_fadein: Optional[float] = None,
    video_fadeout: Optional[float] = None,
    audio_fadein: Optional[float] = None,
    audio_fadeout: Optional[float] = None,
    quality: str = "normal",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
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
        progress_callback: Optional callback(current, total) for progress updates
        cancel_check: Optional callback() -> bool to check if processing should be cancelled

    Returns:
        True if successful, False if error occurred

    Examples:
        # Simple cut without fades
        process("input.mp4", "output.mp4", start=10, end=20)

        # With both video and audio fades
        process("input.mp4", "output.mp4", start=10, end=20, fadein=2, fadeout=3)

        # Video fade only (17x faster audio processing)
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
    try:
        # Load media container
        media_container = MediaContainer(input_path)

        # Determine end time if not specified
        if end is None:
            end = float(media_container.duration)

        # Create segment config
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

        # Convert to internal representation
        segment_with_fade = segment.to_segment_with_fade()

        # Setup audio export (passthrough all tracks)
        audio_export_info = AudioExportInfo(
            mix_info=None,
            mix_export_settings=None,
            output_tracks=[AudioExportSettings('passthru')]
        )

        # Setup video quality
        quality_map = {
            "low": VideoExportQuality.LOW,
            "normal": VideoExportQuality.NORMAL,
            "high": VideoExportQuality.HIGH,
            "lossless": VideoExportQuality.LOSSLESS,
        }
        video_quality = quality_map.get(quality.lower(), VideoExportQuality.NORMAL)
        video_settings = VideoSettings(
            mode=VideoExportMode.SMARTCUT,
            quality=video_quality
        )

        # Setup progress callback wrapper
        progress_wrapper = None
        if progress_callback:
            class ProgressWrapper:
                def __init__(self, callback):
                    self.callback = callback
                    self.total = 0

                def emit(self, value):
                    if self.total == 0:
                        self.total = value
                    else:
                        self.callback(value, self.total)

            progress_wrapper = ProgressWrapper(progress_callback)

        # Setup cancel check wrapper
        cancel_object = None
        if cancel_check:
            cancel_object = CancelObject()
            # Note: Would need to poll cancel_check() in a background thread
            # For now, just create the object

        # Process
        error = smart_cut(
            media_container=media_container,
            positive_segments=[segment_with_fade],
            out_path=output_path,
            audio_export_info=audio_export_info,
            video_settings=video_settings,
            progress=progress_wrapper,
            cancel_object=cancel_object,
        )

        return error is None

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_segments(
    input_path: str,
    output_path: str,
    segments: list[Union[CutSegmentConfig, dict[str, Any]]],
    quality: str = "normal",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bool:
    """Process multiple video segments with independent fade effects.

    This is a high-level convenience function suitable for Gradio and other web apps.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        segments: List of segment configurations (CutSegmentConfig or dict)
        quality: Video quality - "normal", "high", "low", or "lossless"
        progress_callback: Optional callback(current, total) for progress updates
        cancel_check: Optional callback() -> bool to check if processing should be cancelled

    Returns:
        True if successful, False if error occurred

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
        # Load media container
        media_container = MediaContainer(input_path)

        # Convert segments to internal representation
        segments_with_fade = []
        for seg in segments:
            if isinstance(seg, dict):
                # Convert dict to CutSegmentConfig
                segment_config = CutSegmentConfig(**seg)
            elif isinstance(seg, CutSegmentConfig):
                segment_config = seg
            else:
                raise ValueError(f"Invalid segment type: {type(seg)}")

            segments_with_fade.append(segment_config.to_segment_with_fade())

        # Setup audio export (passthrough all tracks)
        audio_export_info = AudioExportInfo(
            mix_info=None,
            mix_export_settings=None,
            output_tracks=[AudioExportSettings('passthru')]
        )

        # Setup video quality
        quality_map = {
            "low": VideoExportQuality.LOW,
            "normal": VideoExportQuality.NORMAL,
            "high": VideoExportQuality.HIGH,
            "lossless": VideoExportQuality.LOSSLESS,
        }
        video_quality = quality_map.get(quality.lower(), VideoExportQuality.NORMAL)
        video_settings = VideoSettings(
            mode=VideoExportMode.SMARTCUT,
            quality=video_quality
        )

        # Setup progress callback wrapper
        progress_wrapper = None
        if progress_callback:
            class ProgressWrapper:
                def __init__(self, callback):
                    self.callback = callback
                    self.total = 0

                def emit(self, value):
                    if self.total == 0:
                        self.total = value
                    else:
                        self.callback(value, self.total)

            progress_wrapper = ProgressWrapper(progress_callback)

        # Setup cancel check wrapper
        cancel_object = None
        if cancel_check:
            cancel_object = CancelObject()

        # Process
        error = smart_cut(
            media_container=media_container,
            positive_segments=segments_with_fade,
            out_path=output_path,
            audio_export_info=audio_export_info,
            video_settings=video_settings,
            progress=progress_wrapper,
            cancel_object=cancel_object,
        )

        return error is None

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False
