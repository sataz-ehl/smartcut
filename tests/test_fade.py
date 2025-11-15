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

# Import synthetic video generator
from generate_synthetic_video import generate_test_video

def run_smartcut(input_path: str, output_path: str, keep_args: list[str]) -> None:
    """
    Run smartcut with the given arguments.

    Args:
        input_path: Input video path
        output_path: Output video path
        keep_args: List of --keep arguments

    Raises:
        AssertionError if smartcut fails
    """
    cmd = [sys.executable, '-m', 'smartcut', input_path, output_path]
    for keep_arg in keep_args:
        cmd.extend(['--keep', keep_arg])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"Smartcut failed: {result.stderr}"

    print(f"✓ Smartcut completed successfully")

def check_video_exists(path: str, min_size: int = 1000) -> None:
    """
    Check if video file exists and has reasonable size.

    Raises:
        AssertionError if video doesn't exist or is too small
    """
    assert os.path.exists(path), f"Output file does not exist: {path}"

    size = os.path.getsize(path)
    assert size >= min_size, f"Output file too small: {size} bytes"

    print(f"✓ Output file exists ({size} bytes)")

def test_no_fade():
    """Test basic cutting without fades (baseline)."""
    print("\n" + "="*60)
    print("TEST: No fade (baseline)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_no_fade.mp4")

        assert generate_test_video(input_video, duration=10), "Failed to generate test video"

        run_smartcut(input_video, output_video, ['5,8'])

        check_video_exists(output_video)

        print("✓ TEST PASSED: No fade")

def test_fadein_only():
    """Test fade-in only."""
    print("\n" + "="*60)
    print("TEST: Fade-in only (1 second)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_fadein.mp4")

        assert generate_test_video(input_video, duration=10), "Failed to generate test video"

        run_smartcut(input_video, output_video, ['5:fadein,8'])

        check_video_exists(output_video)

        print("✓ TEST PASSED: Fade-in only")

def test_fadeout_only():
    """Test fade-out only."""
    print("\n" + "="*60)
    print("TEST: Fade-out only (1 second)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_fadeout.mp4")

        assert generate_test_video(input_video, duration=10), "Failed to generate test video"

        run_smartcut(input_video, output_video, ['5,8:fadeout'])

        check_video_exists(output_video)

        print("✓ TEST PASSED: Fade-out only")

def test_both_fades():
    """Test both fade-in and fade-out."""
    print("\n" + "="*60)
    print("TEST: Both fade-in and fade-out")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_both_fades.mp4")

        assert generate_test_video(input_video, duration=10), "Failed to generate test video"

        run_smartcut(input_video, output_video, ['5:fadein,8:fadeout'])

        check_video_exists(output_video)

        print("✓ TEST PASSED: Both fades")

def test_custom_fade_duration():
    """Test custom fade durations."""
    print("\n" + "="*60)
    print("TEST: Custom fade durations (1.5s in, 2.0s out)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_custom_duration.mp4")

        assert generate_test_video(input_video, duration=15), "Failed to generate test video"

        run_smartcut(input_video, output_video, ['5:fadein:1.5,12:fadeout:2.0'])

        check_video_exists(output_video)

        print("✓ TEST PASSED: Custom fade durations")

def test_multiple_segments():
    """Test multiple segments with different fade configurations."""
    print("\n" + "="*60)
    print("TEST: Multiple segments with different fades")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        output_video = os.path.join(tmpdir, "output_multiple.mp4")

        assert generate_test_video(input_video, duration=20), "Failed to generate test video"

        # First segment: no fade, second segment: fade-in only, third segment: both fades
        run_smartcut(input_video, output_video, ['2,4', '8:fadein,10', '15:fadein,18:fadeout'])

        check_video_exists(output_video)

        print("✓ TEST PASSED: Multiple segments")

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
