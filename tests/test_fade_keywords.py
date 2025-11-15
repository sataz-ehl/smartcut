#!/usr/bin/env python3
"""Test fade feature with keyword time specifications (start, end, -N)."""

import subprocess
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path to import test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_fade import generate_test_video


def run_test(name, input_file, output_file, keep_args):
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    cmd = [sys.executable, '-m', 'smartcut', input_file, output_file] + keep_args
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ FAILED: {name}")
        print(f"Error: {result.stderr}")
        return False

    if not os.path.exists(output_file):
        print(f"✗ FAILED: Output file not created")
        return False

    file_size = os.path.getsize(output_file)
    print(f"✓ Smartcut completed successfully")
    print(f"✓ Output file exists ({file_size} bytes)")
    print(f"✓ TEST PASSED: {name}")

    return True


def main():
    print("\n" + "="*60)
    print("Fade Feature - Keyword Time Specification Tests")
    print("="*60)

    tests_passed = 0
    tests_total = 0

    # Test 1: Using "start" keyword with fade-in
    tests_total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        print("\nGenerating test video...")
        generate_test_video(input_file, duration=20)

        if run_test(
            "start keyword with fade-in",
            input_file,
            output_file,
            ['--keep', 'start:fadein:2,10']
        ):
            tests_passed += 1

    # Test 2: Using "end" keyword with fade-out
    tests_total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        print("\nGenerating test video...")
        generate_test_video(input_file, duration=20)

        if run_test(
            "end keyword with fade-out",
            input_file,
            output_file,
            ['--keep', '10,end:fadeout:2']
        ):
            tests_passed += 1

    # Test 3: Using negative time with fade-out
    tests_total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        print("\nGenerating test video...")
        generate_test_video(input_file, duration=20)

        if run_test(
            "negative time (-5) with fade-out",
            input_file,
            output_file,
            ['--keep', '5,-5:fadeout:1.5']
        ):
            tests_passed += 1

    # Test 4: Entire video with both fades (start to end)
    tests_total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        print("\nGenerating test video...")
        generate_test_video(input_file, duration=20)

        if run_test(
            "entire video (start to end) with both fades",
            input_file,
            output_file,
            ['--keep', 'start:fadein:2,end:fadeout:2']
        ):
            tests_passed += 1

    # Test 5: Complex combination - multiple segments with keywords
    tests_total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        print("\nGenerating test video...")
        generate_test_video(input_file, duration=30)

        if run_test(
            "complex: multiple segments with keywords and numeric times",
            input_file,
            output_file,
            ['--keep', 'start:fadein:1.5,8', '--keep', '12,-8:fadeout:2', '--keep', '-5,end']
        ):
            tests_passed += 1

    # Test 6: Last N seconds with fade-in
    tests_total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        print("\nGenerating test video...")
        generate_test_video(input_file, duration=20)

        if run_test(
            "last 10 seconds with fade-in (-10 to end)",
            input_file,
            output_file,
            ['--keep', '-10:fadein:2,end']
        ):
            tests_passed += 1

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {tests_passed}/{tests_total}")
    print(f"Failed: {tests_total - tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("\n✓ ALL KEYWORD TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {tests_total - tests_passed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
