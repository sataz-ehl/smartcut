#!/usr/bin/env python3
"""
Test validation and auto-sorting of segments.

Tests:
1. Validation: start >= end within segment (should raise error)
2. Auto-sorting: segments provided in wrong chronological order (should auto-sort)
3. API validation: same tests for process() and process_segments()
4. API: an error while processing returns False; only invalid segments raise ValueError

Usage (from the repo root):
    python smartcut/tests/test_validation.py
"""

import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import closing  # with closing(av.open()): PyAV types a container's __exit__ as maybe swallowing errors
from pathlib import Path
from typing import NoReturn

import av
import av.error

# Make the repo root importable (tests live in smartcut/tests/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import smartcut
from smartcut.tests.generate_synthetic_video import generate_test_video


def test_cli_validation_reversed_segment() -> bool:
    """Test CLI validation: start > end within segment."""
    print("\n" + "="*60)
    print("Test 1: CLI validation - reversed segment (start > end)")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=60, fps=30):
        print("✗ Failed to create test video")
        return False

    # Try to process with reversed segment (should error)
    print("\nTrying to process with reversed segment (30,12)...")
    result = subprocess.run(
        [sys.executable, "-m", "smartcut", input_video, output_video,
         "--keep", "30:fadein:2,12:fadeout:3"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check that it errored
    if result.returncode != 0 and "Invalid segment" in result.stderr:
        print("✓ Test passed! Validation correctly caught reversed segment")
        print(f"   Error message: {result.stderr.split('ValueError:')[1].split(chr(10))[0]}")
        os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return True
    else:
        print("✗ Test failed! Should have raised validation error")
        os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_cli_auto_sort_segments() -> bool:
    """Test CLI auto-sorting: segments in wrong chronological order."""
    print("\n" + "="*60)
    print("Test 2: CLI auto-sort - segments in wrong order")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=60, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process with segments in wrong order (should auto-sort)
    print("\nProcessing with segments in wrong order...")
    print("  Segment 1: 30-35 (should be second)")
    print("  Segment 2: 10-15 (should be first)")
    result = subprocess.run(
        [sys.executable, "-m", "smartcut", input_video, output_video,
         "--keep", "30:fadein:2,35:fadeout:3",
         "--keep", "10:fadein:1,15:fadeout:2"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check that it succeeded
    if result.returncode == 0 and os.path.exists(output_video):
        # Verify output duration is correct (10 seconds total)
        with closing(av.open(output_video)) as container:
            duration_sec = container.duration / av.time_base if container.duration else 0
            frame_count = sum(1 for _ in container.decode(video=0))

        expected_duration = 10.0  # Two 5-second segments
        expected_frames = 300  # 10 seconds at 30fps

        if abs(duration_sec - expected_duration) < 0.5 and abs(frame_count - expected_frames) < 10:
            print("✓ Test passed! Segments auto-sorted correctly")
            print(f"   Output duration: {duration_sec:.2f}s (expected: {expected_duration}s)")
            print(f"   Frame count: {frame_count} (expected: ~{expected_frames})")
            os.remove(input_video)
            os.remove(output_video)
            return True
        else:
            print("✗ Test failed! Unexpected output")
            print(f"   Duration: {duration_sec:.2f}s (expected: {expected_duration}s)")
            print(f"   Frames: {frame_count} (expected: ~{expected_frames})")
            os.remove(input_video)
            os.remove(output_video)
            return False
    else:
        print("✗ Test failed! Processing error")
        print(f"   stderr: {result.stderr}")
        os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_api_validation_reversed_segment() -> bool:
    """Test API validation: start > end within segment."""
    print("\n" + "="*60)
    print("Test 3: API validation - reversed segment (start > end)")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=60, fps=30):
        print("✗ Failed to create test video")
        return False

    # Try to process with reversed segment (should error)
    print("\nTrying to process with reversed segment (start=30, end=12)...")
    try:
        smartcut.process(
            input_path=input_video,
            output_path=output_video,
            start=30,
            end=12,
            quality="normal"
        )
        print("✗ Test failed! Should have raised validation error")
        os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False
    except ValueError as e:
        if "Invalid segment" in str(e):
            print("✓ Test passed! Validation correctly caught reversed segment")
            print(f"   Error message: {e}")
            os.remove(input_video)
            if os.path.exists(output_video):
                os.remove(output_video)
            return True
        else:
            print(f"✗ Test failed! Wrong error: {e}")
            os.remove(input_video)
            if os.path.exists(output_video):
                os.remove(output_video)
            return False


def test_api_auto_sort_segments() -> bool:
    """Test API auto-sorting: segments in wrong chronological order."""
    print("\n" + "="*60)
    print("Test 4: API auto-sort - segments in wrong order")
    print("="*60)

    # Generate test video
    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")

    print("Creating test video...")
    if not generate_test_video(input_video, duration=60, fps=30):
        print("✗ Failed to create test video")
        return False

    # Process with segments in wrong order (should auto-sort)
    print("\nProcessing with segments in wrong order...")
    print("  Segment 1: 30-35 (should be second)")
    print("  Segment 2: 10-15 (should be first)")

    success = smartcut.process_segments(
        input_path=input_video,
        output_path=output_video,
        segments=[
            {"start": 30, "end": 35, "fadein": 2, "fadeout": 3},
            {"start": 10, "end": 15, "fadein": 1, "fadeout": 2},
        ],
        quality="normal"
    )

    # Check result
    if success and os.path.exists(output_video):
        # Verify output duration is correct (10 seconds total)
        with closing(av.open(output_video)) as container:
            duration_sec = container.duration / av.time_base if container.duration else 0
            frame_count = sum(1 for _ in container.decode(video=0))

        expected_duration = 10.0  # Two 5-second segments
        expected_frames = 300  # 10 seconds at 30fps

        if abs(duration_sec - expected_duration) < 0.5 and abs(frame_count - expected_frames) < 10:
            print("✓ Test passed! Segments auto-sorted correctly")
            print(f"   Output duration: {duration_sec:.2f}s (expected: {expected_duration}s)")
            print(f"   Frame count: {frame_count} (expected: ~{expected_frames})")
            os.remove(input_video)
            os.remove(output_video)
            return True
        else:
            print("✗ Test failed! Unexpected output")
            print(f"   Duration: {duration_sec:.2f}s (expected: {expected_duration}s)")
            print(f"   Frames: {frame_count} (expected: ~{expected_frames})")
            os.remove(input_video)
            os.remove(output_video)
            return False
    else:
        print("✗ Test failed! Processing error")
        os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
        return False


def test_api_processing_error_returns_false() -> bool:
    """Test API: a ValueError raised while processing is not a validation error. FFmpeg's EINVAL is an
    av.error.ValueError (a ValueError), e.g. when libvpx can't open a 10-bit VP9 encoder."""
    print("\n" + "="*60)
    print("Test 5: API - a processing error returns False instead of raising")
    print("="*60)

    input_video = tempfile.mktemp(suffix=".mp4")
    output_video = tempfile.mktemp(suffix=".mp4")
    if not generate_test_video(input_video, duration=10, fps=30):
        print("✗ Failed to create test video")
        return False

    def failing_smart_cut(*args: object, **kwargs: object) -> NoReturn:
        raise av.error.ValueError(22, "Invalid argument", "avcodec_open2(libvpx-vp9)")  # as FFmpeg raises it

    real_smart_cut = smartcut.smart_cut
    smartcut.smart_cut = failing_smart_cut  # the name process_segments() calls
    try:
        result = smartcut.process_segments(input_video, output_video, [{"start": 2, "end": 5}])
    except ValueError as e:
        print(f"✗ Test failed! The processing error was raised like a validation error: {e!r}")
        return False
    finally:
        smartcut.smart_cut = real_smart_cut
        os.remove(input_video)
        if os.path.exists(output_video):
            os.remove(output_video)
    if result is False:
        print("✓ Test passed! process_segments returned False")
        return True
    print(f"✗ Test failed! Expected False, got {result!r}")
    return False


def main() -> int:
    """Run all validation tests."""
    print("\n" + "="*60)
    print("SMARTCUT VALIDATION TEST SUITE")
    print("Testing segment validation and auto-sorting")
    print("="*60)

    tests = [
        ("CLI validation (reversed segment)", test_cli_validation_reversed_segment),
        ("CLI auto-sort (wrong order)", test_cli_auto_sort_segments),
        ("API validation (reversed segment)", test_api_validation_reversed_segment),
        ("API auto-sort (wrong order)", test_api_auto_sort_segments),
        ("API processing error returns False", test_api_processing_error_returns_false),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed with exception: {e}")
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
