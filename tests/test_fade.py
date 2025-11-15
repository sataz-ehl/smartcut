#!/usr/bin/env python3
"""
Test script for fade-in/fade-out functionality.
Generates synthetic test video with audio and tests various fade scenarios.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import av
import numpy as np
from av import VideoStream, AudioStream

def generate_test_video(output_path: str, duration: int = 20, fps: int = 30, sample_rate: int = 48000):
    """
    Generate a synthetic test video with audio using PyAV.

    Args:
        output_path: Path to save the video
        duration: Duration in seconds
        fps: Frames per second
        sample_rate: Audio sample rate
    """
    print(f"Generating test video: {output_path}")

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
            # Generate video frame with changing color
            t = frame_i / fps
            r = int(127 + 127 * np.sin(2 * np.pi * t / 2))
            g = int(127 + 127 * np.sin(2 * np.pi * (t / 2 + 1/3)))
            b = int(127 + 127 * np.sin(2 * np.pi * (t / 2 + 2/3)))

            img = np.full((720, 1280, 3), [r, g, b], dtype=np.uint8)

            # Add frame number text indicator (simple pattern)
            if frame_i % 10 == 0:
                img[100:200, 100:200] = [255, 255, 255]

            video_frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in video_stream.encode(video_frame):
                container.mux(packet)

            # Generate audio (sine wave)
            t_audio = np.linspace(frame_i * audio_samples_per_frame / sample_rate,
                                 (frame_i + 1) * audio_samples_per_frame / sample_rate,
                                 audio_samples_per_frame, endpoint=False)
            audio_data = np.sin(2 * np.pi * 440 * t_audio)  # 440 Hz sine wave
            audio_data = np.stack([audio_data, audio_data])  # Stereo

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
        print(f"Error generating video: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_smartcut(input_path: str, output_path: str, keep_args: list[str]) -> bool:
    """
    Run smartcut with the given arguments.

    Args:
        input_path: Input video path
        output_path: Output video path
        keep_args: List of --keep arguments

    Returns:
        True if successful, False otherwise
    """
    cmd = [sys.executable, '-m', 'smartcut', input_path, output_path]
    for keep_arg in keep_args:
        cmd.extend(['--keep', keep_arg])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running smartcut: {result.stderr}")
        return False

    print(f"✓ Smartcut completed successfully")
    return True

def check_video_exists(path: str, min_size: int = 1000) -> bool:
    """Check if video file exists and has reasonable size."""
    if not os.path.exists(path):
        print(f"✗ Output file does not exist: {path}")
        return False

    size = os.path.getsize(path)
    if size < min_size:
        print(f"✗ Output file too small: {size} bytes")
        return False

    print(f"✓ Output file exists ({size} bytes)")
    return True

def test_no_fade():
    """Test basic cutting without fades (baseline)."""
    print("\n" + "="*60)
    print("TEST: No fade (baseline)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_no_fade.mp4")

        if not generate_test_video(input_video, duration=10):
            return False

        if not run_smartcut(input_video, output_video, ['5,8']):
            return False

        if not check_video_exists(output_video):
            return False

        print("✓ TEST PASSED: No fade")
        return True

def test_fadein_only():
    """Test fade-in only."""
    print("\n" + "="*60)
    print("TEST: Fade-in only (1 second)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_fadein.mp4")

        if not generate_test_video(input_video, duration=10):
            return False

        if not run_smartcut(input_video, output_video, ['5:fadein,8']):
            return False

        if not check_video_exists(output_video):
            return False

        print("✓ TEST PASSED: Fade-in only")
        return True

def test_fadeout_only():
    """Test fade-out only."""
    print("\n" + "="*60)
    print("TEST: Fade-out only (1 second)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_fadeout.mp4")

        if not generate_test_video(input_video, duration=10):
            return False

        if not run_smartcut(input_video, output_video, ['5,8:fadeout']):
            return False

        if not check_video_exists(output_video):
            return False

        print("✓ TEST PASSED: Fade-out only")
        return True

def test_both_fades():
    """Test both fade-in and fade-out."""
    print("\n" + "="*60)
    print("TEST: Both fade-in and fade-out")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_both_fades.mp4")

        if not generate_test_video(input_video, duration=10):
            return False

        if not run_smartcut(input_video, output_video, ['5:fadein,8:fadeout']):
            return False

        if not check_video_exists(output_video):
            return False

        print("✓ TEST PASSED: Both fades")
        return True

def test_custom_fade_duration():
    """Test custom fade durations."""
    print("\n" + "="*60)
    print("TEST: Custom fade durations (1.5s in, 2.0s out)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_custom_duration.mp4")

        if not generate_test_video(input_video, duration=15):
            return False

        if not run_smartcut(input_video, output_video, ['5:fadein:1.5,12:fadeout:2.0']):
            return False

        if not check_video_exists(output_video):
            return False

        print("✓ TEST PASSED: Custom fade durations")
        return True

def test_multiple_segments():
    """Test multiple segments with different fade configurations."""
    print("\n" + "="*60)
    print("TEST: Multiple segments with different fades")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_multiple.mp4")

        if not generate_test_video(input_video, duration=20):
            return False

        # First segment: no fade, second segment: fade-in only, third segment: both fades
        if not run_smartcut(input_video, output_video, ['2,4', '8:fadein,10', '15:fadein,18:fadeout']):
            return False

        if not check_video_exists(output_video):
            return False

        print("✓ TEST PASSED: Multiple segments")
        return True

def create_demo_video():
    """Create a demo video saved to tests folder for manual inspection."""
    print("\n" + "="*60)
    print("DEMO: Creating demo video for manual inspection")
    print("="*60)

    tests_dir = Path(__file__).parent
    input_video = tests_dir / "test_input_fade.mp4"
    output_video = tests_dir / "test_output_fade_demo.mp4"

    # Generate a longer test video
    if not generate_test_video(str(input_video), duration=20):
        return False

    # Create output with fades
    if not run_smartcut(str(input_video), str(output_video), ['3:fadein:2,17:fadeout:2']):
        return False

    if not check_video_exists(str(output_video)):
        return False

    print(f"✓ Demo video created: {output_video}")
    print(f"  Input video: {input_video}")
    print("  You can play these files to visually verify the fade effects")
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("Fade Functionality Test Suite")
    print("="*60)

    tests = [
        ("No Fade (Baseline)", test_no_fade),
        ("Fade-in Only", test_fadein_only),
        ("Fade-out Only", test_fadeout_only),
        ("Both Fades", test_both_fades),
        ("Custom Durations", test_custom_fade_duration),
        ("Multiple Segments", test_multiple_segments),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ TEST FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"✗ TEST FAILED: {test_name} - {e}")
            import traceback
            traceback.print_exc()

    # Create demo video
    print("\n" + "="*60)
    try:
        create_demo_video()
    except Exception as e:
        print(f"✗ Demo creation failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
