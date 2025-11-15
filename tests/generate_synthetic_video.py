#!/usr/bin/env python3
"""
Synthetic test video generator for smartcut fade testing.

Generates MP4 videos with:
- RGB color-changing video (sine wave pattern)
- 440Hz stereo audio (sine wave)
- Configurable duration, fps, and sample rate
- Visual frame markers every 10 frames

Usage:
    python generate_synthetic_video.py output.mp4 [duration] [fps] [sample_rate]

Examples:
    # Generate 10-second video at 30fps
    python generate_synthetic_video.py test.mp4 10

    # Generate 20-second video at 60fps with 48kHz audio
    python generate_synthetic_video.py test.mp4 20 60 48000

    # Import and use in other scripts
    from generate_synthetic_video import generate_test_video
    generate_test_video('output.mp4', duration=15, fps=30)
"""

import sys
import av
import numpy as np
from typing import cast
from av import VideoStream, AudioStream


def generate_test_video(output_path: str, duration: int = 20, fps: int = 30, sample_rate: int = 48000):
    """
    Generate a synthetic test video with audio using PyAV.

    Creates an MP4 file with:
    - Video: RGB color pattern that cycles through colors using sine waves
    - Audio: 440Hz sine wave (stereo)
    - Visual markers: White squares every 10 frames for visual reference

    Args:
        output_path: Path to save the video file
        duration: Duration in seconds (default: 20)
        fps: Frames per second (default: 30)
        sample_rate: Audio sample rate in Hz (default: 48000)

    Returns:
        bool: True if successful, False otherwise

    Technical details:
        - Video codec: H.264 (libx264)
        - Video resolution: 1280x720
        - Pixel format: yuv420p
        - Audio codec: AAC
        - Audio channels: Stereo
        - Audio format: fltp (floating point planar)

    Example:
        >>> generate_test_video('test.mp4', duration=10, fps=30)
        Generating test video: test.mp4
        Video: 1280x720 @ 30 fps, 10 seconds (300 frames)
        Audio: 48000 Hz stereo, 10 seconds
        ✓ Test video generated successfully
        True
    """
    print(f"Generating test video: {output_path}")
    print(f"Video: 1280x720 @ {fps} fps, {duration} seconds ({duration * fps} frames)")
    print(f"Audio: {sample_rate} Hz stereo, {duration} seconds")

    try:
        container = av.open(output_path, mode="w")

        # Create video stream
        video_stream = cast(VideoStream, container.add_stream('libx264', rate=fps))
        video_stream.width = 1280
        video_stream.height = 720
        video_stream.pix_fmt = 'yuv420p'

        # Create audio stream (layout automatically sets channels)
        audio_stream = cast(AudioStream, container.add_stream('aac', rate=sample_rate, layout='stereo'))

        total_frames = int(duration * fps)
        audio_samples_per_frame = int(sample_rate / fps)

        # Generate video frames and audio
        for frame_i in range(total_frames):
            # Generate video frame with changing color (RGB sine wave pattern)
            t = frame_i / fps
            r = int(127 + 127 * np.sin(2 * np.pi * t / 2))
            g = int(127 + 127 * np.sin(2 * np.pi * (t / 2 + 1/3)))
            b = int(127 + 127 * np.sin(2 * np.pi * (t / 2 + 2/3)))

            img = np.full((720, 1280, 3), [r, g, b], dtype=np.uint8)

            # Add frame number visual indicator (white square every 10 frames)
            if frame_i % 10 == 0:
                img[100:200, 100:200] = [255, 255, 255]

            video_frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in video_stream.encode(video_frame):
                container.mux(packet)

            # Generate audio (440 Hz sine wave, stereo)
            t_audio = np.linspace(frame_i * audio_samples_per_frame / sample_rate,
                                 (frame_i + 1) * audio_samples_per_frame / sample_rate,
                                 audio_samples_per_frame, endpoint=False)
            audio_data = np.sin(2 * np.pi * 440 * t_audio)  # 440 Hz (A4 note)
            audio_data = np.stack([audio_data, audio_data])  # Stereo (same signal both channels)

            audio_frame = av.AudioFrame.from_ndarray(audio_data.astype(np.float32), format='fltp', layout='stereo')
            audio_frame.sample_rate = sample_rate

            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

        # Flush streams
        for packet in video_stream.encode():
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

        container.close()
        print(f"✓ Test video generated successfully")
        return True

    except Exception as e:
        print(f"✗ Error generating video: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Command-line interface for generating test videos."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    output_path = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    sample_rate = int(sys.argv[4]) if len(sys.argv) > 4 else 48000

    success = generate_test_video(output_path, duration, fps, sample_rate)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
