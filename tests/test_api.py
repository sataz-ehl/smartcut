#!/usr/bin/env python3
"""
Test the high-level smartcut API for Gradio and web app integration.

Tests:
1. Single segment processing with video-only fade
2. Single segment processing with audio-only fade
3. Single segment processing with both fades (legacy)
4. Multiple segments with mixed fade configurations
5. Using CutSegmentConfig objects
6. Dict-based segment configuration

Usage:
    python tests/test_api.py
"""

import sys
import os
import tempfile

# Add parent directory to path so we can import smartcut
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import smartcut
from tests.generate_synthetic_video import generate_test_video


def test_video_only_fade():
    """Test single segment with video-only fade (should be fast)."""
    print("\n" + "="*60)
    print("Test 1: Single segment with video-only fade")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=10, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process with video-only fade
    print("\nProcessing with video-only fade (audio passthrough)...")
    success = smartcut.process(
        input_path=input_video,
        output_path=output_video,
        start=2,
        end=8,
        video_fadein=1,
        video_fadeout=1.5,
        quality="normal"
    )

    # Check result
    if success and os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"✓ Test passed! Output size: {size} bytes")
        os.remove(output_video)
        os.remove(input_video)
        return True
    else:
        print("✗ Test failed!")
        if os.path.exists(input_video):
            os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_audio_only_fade():
    """Test single segment with audio-only fade."""
    print("\n" + "="*60)
    print("Test 2: Single segment with audio-only fade")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=10, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process with audio-only fade
    print("\nProcessing with audio-only fade (video passthrough)...")
    success = smartcut.process(
        input_path=input_video,
        output_path=output_video,
        start=2,
        end=8,
        audio_fadein=1,
        audio_fadeout=1,
        quality="normal"
    )

    # Check result
    if success and os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"✓ Test passed! Output size: {size} bytes")
        os.remove(output_video)
        os.remove(input_video)
        return True
    else:
        print("✗ Test failed!")
        if os.path.exists(input_video):
            os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_legacy_both_fade():
    """Test single segment with legacy fadein/fadeout (both video and audio)."""
    print("\n" + "="*60)
    print("Test 3: Single segment with legacy fadein/fadeout")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=10, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process with legacy fade (both video and audio)
    print("\nProcessing with legacy fadein/fadeout (both media types)...")
    success = smartcut.process(
        input_path=input_video,
        output_path=output_video,
        start=2,
        end=8,
        fadein=1,
        fadeout=1.5,
        quality="normal"
    )

    # Check result
    if success and os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"✓ Test passed! Output size: {size} bytes")
        os.remove(output_video)
        os.remove(input_video)
        return True
    else:
        print("✗ Test failed!")
        if os.path.exists(input_video):
            os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_mixed_fade():
    """Test single segment with mixed fade (video fadein, audio fadeout)."""
    print("\n" + "="*60)
    print("Test 4: Single segment with mixed fade effects")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=10, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process with mixed fade
    print("\nProcessing with mixed fade (video fadein, audio fadeout)...")
    success = smartcut.process(
        input_path=input_video,
        output_path=output_video,
        start=2,
        end=8,
        video_fadein=1,
        audio_fadeout=1.5,
        quality="normal"
    )

    # Check result
    if success and os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"✓ Test passed! Output size: {size} bytes")
        os.remove(output_video)
        os.remove(input_video)
        return True
    else:
        print("✗ Test failed!")
        if os.path.exists(input_video):
            os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_multiple_segments_dict():
    """Test multiple segments with dict-based configuration."""
    print("\n" + "="*60)
    print("Test 5: Multiple segments with dict configuration")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=15, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process multiple segments with different fade configurations
    print("\nProcessing multiple segments with mixed fades...")
    segments = [
        {"start": 1, "end": 3, "video_fadein": 0.5, "video_fadeout": 0.5},  # Video-only
        {"start": 5, "end": 7, "audio_fadein": 0.5, "audio_fadeout": 0.5},  # Audio-only
        {"start": 9, "end": 11, "fadein": 0.5, "fadeout": 0.5},              # Both (legacy)
        {"start": 12, "end": 14, "video_fadein": 0.5, "audio_fadeout": 0.5}, # Mixed
    ]

    success = smartcut.process_segments(
        input_path=input_video,
        output_path=output_video,
        segments=segments,
        quality="normal"
    )

    # Check result
    if success and os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"✓ Test passed! Output size: {size} bytes")
        os.remove(output_video)
        os.remove(input_video)
        return True
    else:
        print("✗ Test failed!")
        if os.path.exists(input_video):
            os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_multiple_segments_config():
    """Test multiple segments using CutSegmentConfig objects."""
    print("\n" + "="*60)
    print("Test 6: Multiple segments with CutSegmentConfig objects")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=12, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process using CutSegmentConfig objects
    print("\nProcessing with CutSegmentConfig objects...")
    segments = [
        smartcut.CutSegmentConfig(start=1, end=3, video_fadein=0.5, video_fadeout=0.5),
        smartcut.CutSegmentConfig(start=5, end=7, audio_fadein=0.5),
        smartcut.CutSegmentConfig(start=8, end=10, fadein=0.5, fadeout=0.5),
    ]

    success = smartcut.process_segments(
        input_path=input_video,
        output_path=output_video,
        segments=segments,
        quality="normal"
    )

    # Check result
    if success and os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"✓ Test passed! Output size: {size} bytes")
        os.remove(output_video)
        os.remove(input_video)
        return True
    else:
        print("✗ Test failed!")
        if os.path.exists(input_video):
            os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def main():
    """Run all API tests."""
    print("\n" + "="*60)
    print("SMARTCUT API TEST SUITE")
    print("Testing independent video/audio fade controls")
    print("="*60)

    tests = [
        ("Video-only fade", test_video_only_fade),
        ("Audio-only fade", test_audio_only_fade),
        ("Legacy both fade", test_legacy_both_fade),
        ("Mixed fade", test_mixed_fade),
        ("Multiple segments (dict)", test_multiple_segments_dict),
        ("Multiple segments (config)", test_multiple_segments_config),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {name}")

    print("="*60)
    print(f"Total: {passed_count}/{total_count} tests passed")
    print("="*60)

    return 0 if passed_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
