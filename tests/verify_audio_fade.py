#!/usr/bin/env python3
"""Verify audio fade effects by analyzing volume levels."""

import sys
import os
import subprocess
import tempfile
import av
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.test_fade import generate_test_video


def analyze_audio_volume(video_path, duration_seconds=None):
    """Extract audio RMS levels over time."""
    container = av.open(video_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

    if audio_stream is None:
        print("No audio stream found")
        return None, None

    sample_rate = audio_stream.codec_context.sample_rate or 48000

    all_samples = []

    for frame in container.decode(audio=0):
        audio_arr = frame.to_ndarray()
        # PyAV returns shape (channels, samples), transpose to (samples, channels)
        if audio_arr.ndim == 2:
            audio_arr = audio_arr.T
            # Average channels to mono
            mono = np.mean(audio_arr, axis=1)
        else:
            mono = audio_arr
        all_samples.extend(mono)

    container.close()

    if len(all_samples) == 0:
        return None, None

    # Convert to numpy array
    samples = np.array(all_samples)
    times = np.arange(len(samples)) / sample_rate

    # Calculate RMS in windows
    window_size = int(sample_rate * 0.1)  # 100ms windows
    num_windows = len(samples) // window_size

    rms_values = []
    time_values = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_samples = samples[start_idx:end_idx]
        rms = np.sqrt(np.mean(window_samples**2))
        rms_values.append(rms)
        time_values.append((start_idx + end_idx) / 2 / sample_rate)

    return np.array(time_values), np.array(rms_values)


def test_fade_in():
    """Test fade-in: volume should increase from 0 to full."""
    print("\n" + "="*60)
    print("TEST: Fade-in (2 seconds)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        # Generate 10-second test video
        generate_test_video(input_file, duration=10)

        # Keep segment from 2s to 8s with 2-second fade-in
        subprocess.run([
            sys.executable, '-m', 'smartcut', input_file, output_file,
            '--keep', '2:fadein:2,8'
        ], capture_output=True)

        times, rms = analyze_audio_volume(output_file)

        if times is None or len(times) == 0:
            print("✗ Failed to analyze audio")
            return False

        print(f"\nAnalyzed {len(times)} time windows")
        print("\nVolume over time (first 2s should show fade-in):")
        print("Time (s) | RMS    | Normalized | Visual")
        print("-" * 60)

        max_rms = np.max(rms)

        for i in range(min(25, len(times))):
            normalized = rms[i] / max_rms if max_rms > 0 else 0
            bar_length = int(normalized * 40)
            bar = '█' * bar_length
            print(f"{times[i]:6.2f}   | {rms[i]:.4f} | {normalized:6.2%}     | {bar}")

        # Verify fade-in behavior
        # First 0.5s should be quiet (< 20% of max)
        # Last 2s should be loud (> 80% of max)

        early_samples = rms[times < 0.5]
        late_samples = rms[times > 4.0]  # After fade-in completes

        if len(early_samples) > 0 and len(late_samples) > 0:
            early_avg = np.mean(early_samples) / max_rms
            late_avg = np.mean(late_samples) / max_rms

            print(f"\n{'='*60}")
            print(f"Early volume (0-0.5s): {early_avg:.1%} of max")
            print(f"Late volume (4s+):     {late_avg:.1%} of max")

            if early_avg < 0.3 and late_avg > 0.7:
                print("✓ Fade-in working correctly (starts quiet, ends loud)")
                return True
            else:
                print("✗ Fade-in may not be working (unexpected volume pattern)")
                return False

        return False


def test_fade_out():
    """Test fade-out: volume should decrease from full to 0."""
    print("\n" + "="*60)
    print("TEST: Fade-out (2 seconds)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        # Generate 10-second test video
        generate_test_video(input_file, duration=10)

        # Keep segment from 2s to 8s with 2-second fade-out
        subprocess.run([
            sys.executable, '-m', 'smartcut', input_file, output_file,
            '--keep', '2,8:fadeout:2'
        ], capture_output=True)

        times, rms = analyze_audio_volume(output_file)

        if times is None or len(times) == 0:
            print("✗ Failed to analyze audio")
            return False

        print(f"\nAnalyzed {len(times)} time windows")
        print("\nVolume over time (last 2s should show fade-out):")
        print("Time (s) | RMS    | Normalized | Visual")
        print("-" * 60)

        max_rms = np.max(rms)

        for i in range(min(25, len(times))):
            normalized = rms[i] / max_rms if max_rms > 0 else 0
            bar_length = int(normalized * 40)
            bar = '█' * bar_length
            print(f"{times[i]:6.2f}   | {rms[i]:.4f} | {normalized:6.2%}     | {bar}")

        # Verify fade-out behavior
        duration = times[-1]
        early_samples = rms[times < 2.0]  # Before fade-out starts
        late_samples = rms[times > duration - 0.5]  # Last 0.5s should be quiet

        if len(early_samples) > 0 and len(late_samples) > 0:
            early_avg = np.mean(early_samples) / max_rms
            late_avg = np.mean(late_samples) / max_rms

            print(f"\n{'='*60}")
            print(f"Early volume (0-2s):        {early_avg:.1%} of max")
            print(f"Late volume (last 0.5s):    {late_avg:.1%} of max")

            if early_avg > 0.7 and late_avg < 0.3:
                print("✓ Fade-out working correctly (starts loud, ends quiet)")
                return True
            else:
                print("✗ Fade-out may not be working (unexpected volume pattern)")
                return False

        return False


def test_both_fades():
    """Test both fade-in and fade-out."""
    print("\n" + "="*60)
    print("TEST: Both fade-in and fade-out (1.5s each)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        # Generate 10-second test video
        generate_test_video(input_file, duration=10)

        # Keep segment from 2s to 8s with both fades
        subprocess.run([
            sys.executable, '-m', 'smartcut', input_file, output_file,
            '--keep', '2:fadein:1.5,8:fadeout:1.5'
        ], capture_output=True)

        times, rms = analyze_audio_volume(output_file)

        if times is None or len(times) == 0:
            print("✗ Failed to analyze audio")
            return False

        print(f"\nAnalyzed {len(times)} time windows")
        print("\nVolume over time (should fade in, stay loud, then fade out):")
        print("Time (s) | RMS    | Normalized | Visual")
        print("-" * 60)

        max_rms = np.max(rms)

        for i in range(min(25, len(times))):
            normalized = rms[i] / max_rms if max_rms > 0 else 0
            bar_length = int(normalized * 40)
            bar = '█' * bar_length
            print(f"{times[i]:6.2f}   | {rms[i]:.4f} | {normalized:6.2%}     | {bar}")

        duration = times[-1]
        start_samples = rms[times < 0.5]  # Start should be quiet
        middle_samples = rms[(times > 2.0) & (times < duration - 2.0)]  # Middle should be loud
        end_samples = rms[times > duration - 0.5]  # End should be quiet

        if len(start_samples) > 0 and len(middle_samples) > 0 and len(end_samples) > 0:
            start_avg = np.mean(start_samples) / max_rms
            middle_avg = np.mean(middle_samples) / max_rms
            end_avg = np.mean(end_samples) / max_rms

            print(f"\n{'='*60}")
            print(f"Start volume (0-0.5s):      {start_avg:.1%} of max")
            print(f"Middle volume:              {middle_avg:.1%} of max")
            print(f"End volume (last 0.5s):     {end_avg:.1%} of max")

            if start_avg < 0.3 and middle_avg > 0.7 and end_avg < 0.3:
                print("✓ Both fades working correctly (quiet → loud → quiet)")
                return True
            else:
                print("✗ Fades may not be working (unexpected volume pattern)")
                return False

        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Audio Fade Verification Test Suite")
    print("="*60)

    results = []

    results.append(("Fade-in", test_fade_in()))
    results.append(("Fade-out", test_fade_out()))
    results.append(("Both fades", test_both_fades()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ All audio fade effects verified!")
        sys.exit(0)
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        sys.exit(1)
