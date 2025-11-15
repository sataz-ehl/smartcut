#!/usr/bin/env python3
"""
Test suite to verify audio parameter consistency in fade feature.

Verifies that re-encoded fade sections maintain the same audio parameters
(codec, sample rate, channels, layout, format, bitrate) as the source audio
and that all segments can be properly muxed together.
"""

import sys
import os
import subprocess
import tempfile
import av
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.test_fade import generate_test_video


def check_audio_parameters(video_path):
    """Extract audio stream parameters from a video file."""
    container = av.open(video_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

    if audio_stream is None:
        container.close()
        return None

    params = {
        'codec': audio_stream.codec_context.name,
        'sample_rate': audio_stream.codec_context.sample_rate,
        'channels': audio_stream.codec_context.channels,
        'layout': str(audio_stream.codec_context.layout),
        'format': str(audio_stream.codec_context.format),
        'bitrate': audio_stream.bit_rate,
    }

    container.close()
    return params


def verify_frame_consistency(video_path):
    """Verify all audio frames have consistent parameters."""
    container = av.open(video_path)

    sample_rates = []
    layouts = []
    frame_count = 0

    try:
        for frame in container.decode(audio=0):
            sample_rates.append(frame.sample_rate)
            layouts.append(str(frame.layout))
            frame_count += 1
    except Exception as e:
        container.close()
        return False, f"Error decoding: {e}"

    container.close()

    unique_rates = set(sample_rates)
    unique_layouts = set(layouts)

    if len(unique_rates) != 1 or len(unique_layouts) != 1:
        return False, f"Inconsistent parameters: {unique_rates} rates, {unique_layouts} layouts"

    return True, f"All {frame_count} frames consistent"


def test_fully_reencoded_segment():
    """Test segment that is entirely re-encoded (fade-in + fade-out covers entire segment)."""
    print("\n" + "="*60)
    print("TEST: Fully re-encoded segment (fade-in + fade-out)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        generate_test_video(input_file, duration=10)

        # Get source parameters
        source_params = check_audio_parameters(input_file)
        print(f"\nSource parameters:")
        print(f"  Codec: {source_params['codec']}")
        print(f"  Sample rate: {source_params['sample_rate']} Hz")
        print(f"  Channels: {source_params['channels']}")
        print(f"  Layout: {source_params['layout']}")
        print(f"  Format: {source_params['format']}")
        print(f"  Bitrate: {source_params['bitrate'] // 1000} kbps")

        # Process: 2s segment with 1s fade-in and 1s fade-out = entirely re-encoded
        result = subprocess.run([
            sys.executable, '-m', 'smartcut', input_file, output_file,
            '--keep', '2:fadein:1,4:fadeout:1'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Smartcut failed: {result.stderr}")
            return False

        # Get output parameters
        output_params = check_audio_parameters(output_file)
        print(f"\nOutput parameters:")
        print(f"  Codec: {output_params['codec']}")
        print(f"  Sample rate: {output_params['sample_rate']} Hz")
        print(f"  Channels: {output_params['channels']}")
        print(f"  Layout: {output_params['layout']}")
        print(f"  Format: {output_params['format']}")
        print(f"  Bitrate: {output_params['bitrate'] // 1000} kbps")

        # Verify parameters match
        passed = True

        if source_params['codec'] != output_params['codec']:
            print(f"✗ Codec mismatch: {source_params['codec']} → {output_params['codec']}")
            passed = False
        else:
            print(f"✓ Codec match: {source_params['codec']}")

        if source_params['sample_rate'] != output_params['sample_rate']:
            print(f"✗ Sample rate mismatch: {source_params['sample_rate']} → {output_params['sample_rate']}")
            passed = False
        else:
            print(f"✓ Sample rate match: {source_params['sample_rate']} Hz")

        if source_params['channels'] != output_params['channels']:
            print(f"✗ Channels mismatch: {source_params['channels']} → {output_params['channels']}")
            passed = False
        else:
            print(f"✓ Channels match: {source_params['channels']}")

        if source_params['layout'] != output_params['layout']:
            print(f"✗ Layout mismatch: {source_params['layout']} → {output_params['layout']}")
            passed = False
        else:
            print(f"✓ Layout match: {source_params['layout']}")

        # Bitrate should be very close (allow 10% tolerance for VBR)
        bitrate_diff = abs(source_params['bitrate'] - output_params['bitrate'])
        bitrate_tolerance = source_params['bitrate'] * 0.1
        if bitrate_diff > bitrate_tolerance:
            print(f"✗ Bitrate mismatch: {source_params['bitrate'] // 1000} kbps → {output_params['bitrate'] // 1000} kbps")
            passed = False
        else:
            print(f"✓ Bitrate match: ~{source_params['bitrate'] // 1000} kbps")

        # Verify frame consistency
        consistent, msg = verify_frame_consistency(output_file)
        if consistent:
            print(f"✓ Frame consistency: {msg}")
        else:
            print(f"✗ Frame consistency failed: {msg}")
            passed = False

        return passed


def test_mixed_passthrough_and_reencoded():
    """Test segment with both passthrough and re-encoded sections."""
    print("\n" + "="*60)
    print("TEST: Mixed passthrough and re-encoded sections")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        generate_test_video(input_file, duration=10)

        source_params = check_audio_parameters(input_file)
        print(f"\nSource: {source_params['sample_rate']}Hz, {source_params['channels']}ch, {source_params['bitrate'] // 1000}kbps")

        # Process: 6s segment with 1s fade-in, 4s passthrough, 1s fade-out
        result = subprocess.run([
            sys.executable, '-m', 'smartcut', input_file, output_file,
            '--keep', '2:fadein:1,8:fadeout:1'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Smartcut failed: {result.stderr}")
            return False

        output_params = check_audio_parameters(output_file)
        print(f"Output: {output_params['sample_rate']}Hz, {output_params['channels']}ch, {output_params['bitrate'] // 1000}kbps")

        # Verify critical parameters match
        passed = (
            source_params['codec'] == output_params['codec'] and
            source_params['sample_rate'] == output_params['sample_rate'] and
            source_params['channels'] == output_params['channels'] and
            source_params['layout'] == output_params['layout']
        )

        if passed:
            print("✓ All critical parameters match")
        else:
            print("✗ Parameter mismatch detected")
            return False

        # Verify frame consistency
        consistent, msg = verify_frame_consistency(output_file)
        if consistent:
            print(f"✓ Frame consistency: {msg}")
        else:
            print(f"✗ Frame consistency failed: {msg}")
            return False

        return True


def test_multi_segment_consistency():
    """Test multiple segments with different fade configurations."""
    print("\n" + "="*60)
    print("TEST: Multiple segments with mixed configurations")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        generate_test_video(input_file, duration=20)

        source_params = check_audio_parameters(input_file)
        print(f"\nSource: {source_params['sample_rate']}Hz, {source_params['channels']}ch")

        # Complex case: multiple segments
        # Segment 1: fade-in + passthrough
        # Segment 2: passthrough only (no fade)
        # Segment 3: passthrough + fade-out
        result = subprocess.run([
            sys.executable, '-m', 'smartcut', input_file, output_file,
            '--keep', '2:fadein:1,5',       # 1s re-encode + 2s passthrough
            '--keep', '8,12',                # 4s passthrough
            '--keep', '14,18:fadeout:1'      # 3s passthrough + 1s re-encode
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Smartcut failed: {result.stderr}")
            return False

        output_params = check_audio_parameters(output_file)
        print(f"Output: {output_params['sample_rate']}Hz, {output_params['channels']}ch")

        # Open container and check all frames
        container = av.open(output_file)
        audio_stream = next(s for s in container.streams if s.type == 'audio')

        sample_rates = set()
        layouts = set()
        frame_count = 0

        for frame in container.decode(audio=0):
            sample_rates.add(frame.sample_rate)
            layouts.add(str(frame.layout))
            frame_count += 1

        container.close()

        print(f"\nDecoded {frame_count} frames")
        print(f"Unique sample rates: {sample_rates}")
        print(f"Unique layouts: {layouts}")

        if len(sample_rates) == 1 and len(layouts) == 1:
            print("✓ All frames have consistent parameters across all segments")
            return True
        else:
            print("✗ Inconsistent parameters detected across segments")
            return False


def test_keyword_segments():
    """Test segments using keywords (start, end, -N) with fades."""
    print("\n" + "="*60)
    print("TEST: Keyword segments with fades")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.mp4')
        output_file = os.path.join(tmpdir, 'output.mp4')

        generate_test_video(input_file, duration=20)

        source_params = check_audio_parameters(input_file)

        # Use keywords with fades
        result = subprocess.run([
            sys.executable, '-m', 'smartcut', input_file, output_file,
            '--keep', 'start:fadein:1,8',
            '--keep', '12,end:fadeout:1'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Smartcut failed: {result.stderr}")
            return False

        output_params = check_audio_parameters(output_file)

        # Verify parameters
        passed = (
            source_params['sample_rate'] == output_params['sample_rate'] and
            source_params['channels'] == output_params['channels']
        )

        if passed:
            print(f"✓ Parameters match: {output_params['sample_rate']}Hz, {output_params['channels']}ch")
        else:
            print("✗ Parameter mismatch")
            return False

        # Verify frame consistency
        consistent, msg = verify_frame_consistency(output_file)
        if consistent:
            print(f"✓ {msg}")
            return True
        else:
            print(f"✗ {msg}")
            return False


if __name__ == '__main__':
    print("="*60)
    print("Audio Parameter Consistency Test Suite")
    print("="*60)

    tests = [
        ("Fully re-encoded segment", test_fully_reencoded_segment),
        ("Mixed passthrough/re-encoded", test_mixed_passthrough_and_reencoded),
        ("Multi-segment consistency", test_multi_segment_consistency),
        ("Keyword segments", test_keyword_segments),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ All audio parameter consistency tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        sys.exit(1)
