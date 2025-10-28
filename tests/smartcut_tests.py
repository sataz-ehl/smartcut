import argparse
import glob
import os
import platform
import random
import sys
import traceback
from fractions import Fraction
from time import time

import av
import av.datasets as av_datasets
import av.logging
import ffmpeg
import numpy as np
import requests
import scipy
import scipy.signal
import soundfile as sf

from smartcut.cut_video import AudioExportInfo, AudioExportSettings, VideoExportMode, VideoExportQuality, VideoSettings, make_cut_segments, smart_cut
from smartcut.media_container import AudioTrack, MediaContainer
from smartcut.misc_data import MixInfo

from test_utils import *

DEFAULT_SEED = 12345

# Set the log level to silence the None dts warnings. I believe those can be ignored since
# we do set dts, except when it's not set in the source in which case it's not clear what
# value dts should take. It would be nice to occasionally check that there aren't more warnings.
# UPDATE 2024-09-13: setting logging level to FATAL because mpeg2 tests spam a lot of buffer errors I didn't manage to silence
av.logging.set_level(av.logging.FATAL)

data_dir = 'test_data'

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Smart Media Cutter Tests')
    parser.add_argument('--category', choices=['basic', 'h264', 'h265', 'codecs', 'containers', 'audio', 'mixed', 'transforms', 'long', 'external', 'real_world', 'real_world_h264', 'real_world_h265', 'real_world_av1', 'real_world_vp9', 'all', 'smc_audio', 'smc_mixed'],
                       help='Run tests from specific category')
    parser.add_argument('--single', type=str, help='Run a single specific test function (e.g., test_h264_non_idr_keyframes)')
    parser.add_argument('--list-categories', action='store_true', help='List available test categories')
    parser.add_argument('--list-tests', action='store_true', help='List all available test functions')
    parser.add_argument('--flaky', type=int, help='Repeat each selected test N times with different seeds')
    parser.add_argument('--seed', type=int, help='Base PRNG seed (defaults to 12345)')
    parser.add_argument('files', nargs='*', help='Specific files to test (legacy mode)')

    args = parser.parse_args()

    if args.list_categories:
        print("Available test categories:")
        print("  basic      - Quick fundamental H.264 tests")
        print("  h264       - H.264 codec tests including profiles")
        print("  h265       - H.265/HEVC codec tests")
        print("  codecs     - Other video codecs (VP9, AV1, etc.)")
        print("  containers - Container format tests (MP4, AVI, MOV, etc.)")
        print("  audio      - Audio-only tests (MP3, Vorbis, FLAC)")
        print("  mixed      - Tests with video and audio tracks")
        print("  transforms - Video transformation and recoding tests")
        print("  long       - Long-running or large file tests")
        print("  external   - Tests requiring external downloads")
        print("  real_world - All real-world video tests from public sources")
        print("  real_world_h264 - Real-world H.264 videos (Google, test-videos.co.uk, etc)")
        print("  real_world_h265 - Real-world H.265/HEVC videos")
        print("  real_world_av1  - Real-world AV1 videos")
        print("  real_world_vp9  - Real-world VP9 videos")
        print("  all        - Run all tests (default)")
        sys.exit(0)

    if args.list_tests:
        print("Available test functions:")
        test_categories = get_test_categories()
        all_tests = []
        for cat_tests in test_categories.values():
            all_tests.extend(cat_tests)

        # Remove duplicates while preserving order
        seen = set()
        unique_tests = []
        for test in all_tests:
            if test not in seen:
                seen.add(test)
                unique_tests.append(test)

        for test in sorted(unique_tests, key=lambda t: t.__name__):
            print(f"  {test.__name__}")
        sys.exit(0)

    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_base_seed(args) -> int:
    if args is None or args.seed is None:
        return DEFAULT_SEED
    return args.seed

# Handle legacy argument parsing for backwards compatibility
def setup_legacy_args():
    global manual_input, pixel_color_diff_tolerance

    # Check if we're using new argument format (--single, --category, etc.)
    has_new_args = any(arg.startswith('--') for arg in sys.argv[1:])

    if has_new_args:
        # Using new argument format, no legacy mode
        manual_input = None
        pixel_color_diff_tolerance = 20
        return

    # Check for legacy usage (no -- arguments)
    legacy_args = [arg for arg in sys.argv[1:] if arg not in ['basic', 'h264', 'h265', 'codecs', 'containers', 'audio', 'mixed', 'transforms', 'long', 'external', 'real_world', 'real_world_h264', 'real_world_h265', 'real_world_av1', 'real_world_vp9', 'all']]

    if legacy_args:
        if legacy_args[0] == 'all':
            files = []
            patterns = ["./*.mkv", "./*.ts", "./*.mpg", "./*.webm", "./*.mp4", "./*.mp3"]
            for p in patterns:
                files.extend(glob.glob(p))
            manual_input = [os.path.abspath(x) for x in files if '_edited' not in x]
            pixel_color_diff_tolerance = 50
        else:
            manual_input = [os.path.abspath(x) for x in legacy_args]
            pixel_color_diff_tolerance = 50
    else:
        manual_input = None
        pixel_color_diff_tolerance = 20

setup_legacy_args()

os.chdir(os.path.dirname(__file__))
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)

short_h264_path = 'short_h264.mkv'
short_h265_path = 'short_h265.mkv'


def test_h264_cut_on_keyframes():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_h264_cut_on_keyframes.__name__ + '.mkv'
    run_cut_on_keyframes_test(short_h264_path, output_path)

def test_h264_smart_cut():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_h264_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10, 30, 100]:
        run_smartcut_test(short_h264_path, output_path, c)

def test_h264_multiple_cuts():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    source = MediaContainer(short_h264_path)

    output_path = test_h264_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10, 30, 100]:
        cutpoints = source.video_frame_times
        cutpoints = [0] + list(np.sort(np.random.choice(cutpoints, c, replace=False))) + [source.duration]

        segments = list(zip(cutpoints[:-1], cutpoints[1:]))

        smart_cut(source, segments, output_path, log_level='warning')
        result_container = MediaContainer(output_path)
        check_videos_equal(source, result_container)

def test_h265_cut_on_keyframes():
    create_test_video(short_h265_path, 30, 'hevc', 'yuv422p10le', 60, (256, 144))
    output_path = test_h265_cut_on_keyframes.__name__ + '.mkv'
    run_cut_on_keyframes_test(short_h265_path, output_path)

def test_h265_smart_cut():
    create_test_video(short_h265_path, 30, 'hevc', 'yuv422p10le', 60, (256, 144))
    output_path = test_h265_smart_cut.__name__ + '.mkv'
    for c in [1, 2]:
        run_smartcut_test(short_h265_path, output_path, c)

def test_h265_smart_cut_large():
    input_file = 'h265_large.mkv'
    create_test_video(input_file, 17, 'hevc', 'yuv422p', 25, (1280, 720))
    output_path = test_h265_smart_cut_large.__name__ + '.mkv'
    for c in [1, 2, 5]:
        run_smartcut_test(input_file, output_path, c)

def test_peaks_mkv_memory_usage():
    """Test that peaks.mkv doesn't cause excessive memory usage and has proper GOP detection."""
    import psutil

    # Download peaks.mkv using cached_download utility
    url = "https://raw.githubusercontent.com/skeskinen/media-test-data/refs/heads/main/peaks.mkv"
    peaks_path = cached_download(url, os.path.join(data_dir, "peaks.mkv"))

    # Test GOP detection
    container = MediaContainer(peaks_path)

    actual_gops = len(container.gop_start_times_pts_s)
    expected_gops = 10

    assert actual_gops >= expected_gops, f"Expected {expected_gops} GOPs but found {actual_gops}"

    container.close()

    # Test memory usage during cutting
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    peak_memory = initial_memory

    def memory_monitor():
        nonlocal peak_memory
        while True:
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)
            time.sleep(0.1)

    import threading
    import time
    stop_monitoring = False

    def monitor_thread():
        nonlocal peak_memory
        while not stop_monitoring:
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)
            time.sleep(0.1)

    monitor = threading.Thread(target=monitor_thread)
    monitor.daemon = True
    monitor.start()

    try:
        output_path = test_peaks_mkv_memory_usage.__name__ + '.mp4'
        segments = [(Fraction(0), Fraction(5))]

        container = MediaContainer(peaks_path)
        audio_settings = [AudioExportSettings(codec='passthru')] * len(container.audio_tracks)
        export_info = AudioExportInfo(output_tracks=audio_settings)
        video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL)

        smart_cut(
            container,
            segments,
            output_path,
            audio_export_info=export_info,
            video_settings=video_settings,
            log_level='error'
        )

        container.close()

    finally:
        stop_monitoring = True
        if monitor.is_alive():
            monitor.join(timeout=1)

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)

    # Memory usage check - should not exceed 700MB
    memory_increase = peak_memory - initial_memory
    max_allowed_memory = 700 * 1024 * 1024

    print(f"Peak memory increase: {memory_increase / (1024*1024):.1f} MB")
    assert memory_increase < max_allowed_memory, f"Memory usage too high: {memory_increase / (1024*1024):.1f} MB"

def test_h264_24_fps_long():
    filename = 'long_h264.mkv'
    # 15 mins
    create_test_video(filename, 60 * 15, 'h264', 'yuv420p', 24, (32, 18))
    output_path = test_h264_24_fps_long.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_1080p():
    filename = '1080p_h264.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (1920, 1080))
    output_path = test_h264_1080p.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_baseline():
    filename = 'h264_baseline.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='baseline')
    output_path = test_h264_profile_baseline.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_main():
    filename = 'h264_main.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='main')
    output_path = test_h264_profile_main.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high():
    filename = 'h264_high.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p', 30, (32, 18), profile='high')
    output_path = test_h264_profile_high.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high10():
    filename = 'h264_high10.mkv'
    create_test_video(filename, 15, 'h264', 'yuv420p10le', 30, (32, 18), profile='high10')
    output_path = test_h264_profile_high10.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high422():
    filename = 'h264_high422.mkv'
    create_test_video(filename, 15, 'h264', 'yuv422p', 30, (32, 18), profile='high422')
    output_path = test_h264_profile_high422.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_h264_profile_high444():
    filename = 'h264_high444.mkv'
    create_test_video(filename, 15, 'h264', 'yuv444p', 30, (32, 18), profile='high444')
    output_path = test_h264_profile_high444.__name__ + '.mkv'
    run_smartcut_test(filename, output_path, n_cuts=3)

def test_mp4_cut_on_keyframe():
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_cut_on_keyframe.__name__ + '.mp4'
    run_cut_on_keyframes_test(filename, output_path)

def test_mp4_smart_cut():
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mp4_to_mkv_smart_cut():
    filename = 'basic.mp4'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mp4_to_mkv_smart_cut.__name__ + '.mkv'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mkv_to_mp4_smart_cut():
    create_test_video(short_h264_path, 30, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_mkv_to_mp4_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(short_h264_path, output_path, c)

def test_mp4_h265_smart_cut():
    filename = 'h265.mp4'
    create_test_video(filename, 30, 'hevc', 'yuv420p', 30, (256, 144))
    output_path = test_mp4_h265_smart_cut.__name__ + '.mp4'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mpg_smart_cut():
    filename = cached_download('https://filesamples.com/samples/video/mpg/sample_640x360.mpg', 'mpeg640x360.mpg')
    output_path = test_mpg_smart_cut.__name__ + '.mpg'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_mpg_cut_on_keyframes():
    filename = cached_download('https://filesamples.com/samples/video/mpg/sample_640x360.mpg', 'mpeg640x360.mpg')
    output_path = test_mpg_cut_on_keyframes.__name__ + '.mpg'
    run_cut_on_keyframes_test(filename, output_path)

def test_m2ts_mpeg2_smart_cut():
    filename = cached_download('https://filesamples.com/samples/video/m2ts/sample_960x540.m2ts', 'm2ts960x540.m2ts')
    output_path = test_m2ts_mpeg2_smart_cut.__name__ + '.m2ts'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_m2ts_h264_smart_cut():
    filename = cached_download('https://www.dwsamplefiles.com/?dl_id=311', 'm2ts_h264_636x360.m2ts')
    output_path = test_m2ts_h264_smart_cut.__name__ + '.m2ts'
    for c in [1, 2, 3, 10]:
        run_smartcut_test(filename, output_path, c)

def test_ts_smart_cut():
    filename = cached_download('https://filesamples.com/samples/video/ts/sample_1280x720.ts', 'mpeg2.ts')
    output_path = test_ts_smart_cut.__name__ + '.ts'
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, c)

# Video transform functionality has been removed
# def test_vertical_transform():
#     pass

def test_video_recode_codec_override():
    input_path = 'video_settings_in.mkv'
    file_duration = 10
    n_cuts = 5
    create_test_video(input_path, file_duration, 'h264', 'yuv420p', 30, (854, 480))

    source_container = MediaContainer(input_path)

    cutpoints = source_container.video_frame_times
    cutpoints = [0] + list(np.sort(np.random.choice(cutpoints, n_cuts, replace=False))) + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    output_path_a = test_video_recode_codec_override.__name__ + 'a.mkv'
    output_path_b = test_video_recode_codec_override.__name__ + 'b.mkv'

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, codec_override='hevc')
    smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_a)
    assert output_container.video_stream.codec_context.name == 'hevc', f'codec should be hevc, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, codec_override='hevc')
    smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_b)
    assert output_container.video_stream.codec_context.name == 'hevc', f'codec should be hevc, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, codec_override='vp9')
    smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_a)
    assert output_container.video_stream.codec_context.name == 'vp9', f'codec should be vp9, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, codec_override='vp9')
    smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    output_container = MediaContainer(output_path_b)
    assert output_container.video_stream.codec_context.name == 'vp9', f'codec should be vp9, found {output_container.video_stream.codec_context.name}'
    check_videos_equal(source_container, output_container)

    assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

    # These tests are very slow because the encoders are slow
    # video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.NORMAL, codec_override='av1')
    # smart_cut(source_container, segments, output_path_a, video_settings=video_settings, log_level='warning')

    # output_container = MediaContainer(output_path_a)
    # assert output_container.video_stream.codec_context.name == 'libdav1d', f'codec should be av1, found {output_container.video_stream.codec_context.name}'
    # check_videos_equal(source_container, output_container)

    # video_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH, codec_override='vp9')
    # smart_cut(source_container, segments, output_path_b, video_settings=video_settings, log_level='warning')

    # output_container = MediaContainer(output_path_b)
    # assert output_container.video_stream.codec_context.name == 'libdav1d', f'codec should be av1 found {output_container.video_stream.codec_context.name}'
    # check_videos_equal(source_container, output_container)

    # assert os.path.getsize(output_path_b) > os.path.getsize(output_path_a)

def test_vorbis_passthru():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_vorbis_passthru.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [file_duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    output_container = MediaContainer(output_path)

    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.duration > 14.9 and suffix_container.duration < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000

def test_vorbis_track_cut():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_vorbis_track_cut.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=44100)
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.duration > 14.9 and suffix_container.duration < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000

def test_mp3_track_cut():
    filename = 'basic.mp3'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_mp3_track_cut.__name__ + '.mp3'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='mp3', channels = 'mono', bitrate=128000, sample_rate=44100)
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    # NOTE: mp3 output has a timing issue at the beginning that I can't be arsed to fix.
    # Namely, the mp3 encoder adds some silence to the beginning (encoder delay).
    # Therefore we loosen the rms threshold so the mp3 test pass
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.15)

    # partial file i.e. suffix
    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(output_path)
    assert suffix_container.duration > 14.9 and suffix_container.duration < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000

def test_mp3_passthru():
    filename = 'basic.mp3'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_mp3_passthru.__name__ + '.mp3'

    n_cuts = 5
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [file_duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[settings])

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)
    output_container = MediaContainer(output_path)

    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    # partial file i.e. suffix
    suffix_output_path = test_mp3_passthru.__name__ + '_suffix.mp3'

    cutpoints = [15, file_duration]
    segments = list(zip(cutpoints[:-1], cutpoints[1:]))
    smart_cut(source_container, segments, suffix_output_path, audio_export_info=export_info)
    suffix_container = MediaContainer(suffix_output_path)
    assert suffix_container.duration > 14.8 and suffix_container.duration < 15.1
     # The cut point is not on packet boundary so the audio stream doesn't start at 0
    assert suffix_container.audio_tracks[0].packets[0].pts < 1000


def test_vorbis_encode_mix():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_vorbis_encode_mix.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=44100)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

def test_flac_conversions():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_flac_conversions.__name__ + '.flac'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='flac', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])

    ogg_output_path = test_flac_conversions.__name__ + '.ogg'

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(output_container, segments, ogg_output_path, audio_export_info=export_info)

    vorbis_output_container = MediaContainer(ogg_output_path)
    compare_tracks(source_container.audio_tracks[0], vorbis_output_container.audio_tracks[0], rms_threshold=0.075)

def test_wav_conversions():
    filename = 'basic.ogg'
    freq = 440

    file_duration = 30
    generate_sine_wave(file_duration, filename, frequency=freq)
    output_path = test_wav_conversions.__name__ + '.wav'

    n_cuts = 10
    source_container = MediaContainer(filename)
    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='pcm_f32le', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.075)

    settings = AudioExportSettings(codec='pcm_s16le', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.075)

    # convert back to vorbis from s16
    ogg_output_path = test_wav_conversions.__name__ + '.ogg'

    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', sample_rate=44_100, bitrate=64_000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(output_container, segments, ogg_output_path, audio_export_info=export_info)

    ogg_output_container = MediaContainer(ogg_output_path)
    # TODO: flaky rms test
    compare_tracks(source_container.audio_tracks[0], ogg_output_container.audio_tracks[0], rms_threshold=0.15)


def make_video_and_audio_mkv(path, file_duration):

    audio_file_440 = 'tmp_440.ogg'
    # audio_file_440 = 'tmp_440.aac'
    generate_sine_wave(file_duration, audio_file_440, frequency=440)

    audio_file_630 = 'tmp_630.ogg'
    # audio_file_630 = 'tmp_630.aac'
    generate_sine_wave(file_duration, audio_file_630, frequency=630)

    tmp_video = 'tmp_video.mkv'
    create_test_video(tmp_video, file_duration, 'h264', 'yuv420p', 30, (32, 18))

    (
        ffmpeg
        .input(tmp_video)
        .output(ffmpeg.input(audio_file_440), ffmpeg.input(audio_file_630),
                path, vcodec='copy', acodec='aac', audio_bitrate=92_000, y=None)
        .run(quiet=True)
    )

def make_video_with_subtitles(path, file_duration, subtitle_configs):
    """
    Create a video with multiple subtitle tracks.

    Args:
        path: Output file path
        file_duration: Duration in seconds
        subtitle_configs: List of dicts with keys:
            - 'content': SRT content string
            - 'language': Language code (e.g., 'en', 'fi')
            - 'disposition': Disposition string (e.g., 'forced+default', 'default', '')
            - 'temp_file': Temporary SRT file path
    """
    if os.path.exists(path):
        return path

    # Create base video
    tmp_video = 'tmp_video_for_sub.mkv'
    create_test_video(tmp_video, file_duration, 'h264', 'yuv420p', 30, (32, 18))

    # Create subtitle files
    subtitle_inputs = []
    for config in subtitle_configs:
        with open(config['temp_file'], 'w', encoding='utf-8') as f:
            f.write(config['content'])
        subtitle_inputs.append(ffmpeg.input(config['temp_file']))

    # Build ffmpeg command
    video_input = ffmpeg.input(tmp_video)

    # Prepare output options
    output_options = {
        'vcodec': 'copy',
        'scodec': 'subrip',
        'y': None
    }

    # Add disposition and language options for each subtitle stream
    for i, config in enumerate(subtitle_configs):
        if config.get('disposition'):
            output_options[f'disposition:s:{i}'] = config['disposition']
        if config.get('language'):
            output_options[f'metadata:s:s:{i}'] = f'language={config["language"]}'

    # Run ffmpeg command with proper input structure
    all_inputs = [video_input] + subtitle_inputs
    (
        ffmpeg
        .output(*all_inputs, path, **output_options)
        .run(quiet=True)
    )

    return path

def make_video_with_attachment(path, file_duration=3,
                               attachment_filename='smartcut_attachment.txt',
                               attachment_payload=b'SmartCutAttachmentTest'):
    """Create a small MKV file that carries a single attachment stream."""
    if os.path.exists(path):
        return path

    base_video = 'tmp_attachment_base.mkv'
    create_test_video(base_video, file_duration, 'h264', 'yuv420p', 25, (32, 18))

    attachment_path = 'tmp_attachment_payload.bin'
    with open(attachment_path, 'wb') as fh:
        fh.write(attachment_payload)

    output_options = {
        'c': 'copy',
        'attach': attachment_path,
        'metadata:s:t': f'filename={attachment_filename}',
        'metadata:s:t:0': 'mimetype=text/plain',
        'y': None,
    }

    (
        ffmpeg
        .output(ffmpeg.input(base_video), path, **output_options)
        .run(quiet=True)
    )

    return path

def get_attachment_stream_metadata(path):
    """Extract attachment metadata and raw bytes using PyAV.

    Returns a list of dicts with keys:
      - 'filename': attachment filename (if present)
      - 'mimetype': attachment mimetype (if present)
      - 'extradata_size': size of raw data
      - 'data': raw attachment bytes
    """
    attachments = []
    with av.open(path) as container:
        # Use the dedicated attachments accessor; matches PyAV's own tests
        for att in container.streams.attachments:
            # For MKV, PyAV exposes name/mimetype/data directly
            name = getattr(att, 'name', None)
            mimetype = getattr(att, 'mimetype', None)
            data_bytes = getattr(att, 'data', None)

            # Also pick filename/mimetype from metadata if present
            md = dict(getattr(att, 'metadata', {}) or {})
            filename = name or md.get('filename')
            mimetype = mimetype or md.get('mimetype')

            attachments.append({
                'filename': filename,
                'mimetype': mimetype,
                'extradata_size': (len(data_bytes) if data_bytes is not None else None),
                'data': data_bytes,
            })

    attachments.sort(key=lambda info: info.get('filename') or '')
    return attachments

def make_video_with_forced_subtitle(path, file_duration):
    """Legacy function - creates video with single forced subtitle for backward compatibility"""
    subtitle_content = """1
00:00:01,000 --> 00:00:03,000
First subtitle entry

2
00:00:05,000 --> 00:00:07,000
Second forced subtitle

3
00:00:08,500 --> 00:00:09,500
Final entry
"""

    subtitle_configs = [{
        'content': subtitle_content,
        'language': 'en',
        'disposition': 'forced+default',
        'temp_file': 'tmp_test_subtitle_forced.srt'
    }]

    return make_video_with_subtitles(path, file_duration, subtitle_configs)

def test_mkv_with_video_and_audio_passthru():
    file_duration = 30

    final_input = 'video_and_two_audio.mkv'
    make_video_and_audio_mkv(final_input, file_duration)

    output_path = test_mkv_with_video_and_audio_passthru.__name__ + '.mkv'

    source_container = MediaContainer(final_input)

    passthru_settings = AudioExportSettings(codec='passthru')
    export_info = AudioExportInfo(output_tracks=[passthru_settings, passthru_settings])
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)

    assert len(result_container.audio_tracks) == 2
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[1])

    export_info = AudioExportInfo(output_tracks=[None, passthru_settings])
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[0])

    export_info = AudioExportInfo(output_tracks=[passthru_settings, None])
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)
    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])

def test_mkv_with_video_and_audio_mix():
    file_duration = 30

    final_input = 'video_and_two_audio.mkv'
    make_video_and_audio_mkv(final_input, file_duration)

    output_path = test_mkv_with_video_and_audio_mix.__name__ + '.mkv'

    source_container = MediaContainer(final_input)

    mix = MixInfo([1., 0.])
    mix_export_settings = AudioExportSettings(codec='aac', channels='mono', bitrate=92_000, sample_rate=44_100)
    export_info = AudioExportInfo(mix, mix_export_settings)
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)

    result_container = MediaContainer(output_path)

    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])

    mix = MixInfo([0., 1.])
    export_info = AudioExportInfo(mix, mix_export_settings)
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[0])

    mix = MixInfo([0.5, 0.5])
    export_info = AudioExportInfo(mix, mix_export_settings)
    run_smartcut_test(final_input, output_path, n_cuts=5, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    assert len(result_container.audio_tracks) == 1

    reference_mix_path = 'reference_mix.ogg'
    generate_double_sine_wave(file_duration, reference_mix_path, 440, 630)
    reference_container = MediaContainer(reference_mix_path)
    compare_tracks(reference_container.audio_tracks[0], result_container.audio_tracks[0], rms_threshold=0.7)

def test_mix_with_rate_conversion():
    in_1 = '48k.ogg'
    freq_1 = 440
    in_2 = '26k.ogg'
    freq_2 = 600

    out_sr = 44_100

    file_duration = 30
    generate_sine_wave(file_duration, in_1, frequency=freq_1, sample_rate=48_000)
    generate_sine_wave(file_duration, in_2, frequency=freq_2, sample_rate=26_000)

    output_path = test_mix_with_rate_conversion.__name__ + '.ogg'

    n_cuts = 10
    source_container = MediaContainer(in_1)
    source_container.add_audio_file(in_2)

    cutpoints = np.arange(file_duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([0.5, 0.5])
    settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=out_sr)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    reference_mix_path = test_mix_with_rate_conversion.__name__ + '_reference_mix.ogg'
    generate_double_sine_wave(file_duration, reference_mix_path, freq_1, freq_2, sample_rate=out_sr)
    ref_container = MediaContainer(reference_mix_path)

    output_container = MediaContainer(output_path)
    compare_tracks(ref_container.audio_tracks[0], output_container.audio_tracks[0], rms_threshold=0.12)

def test_denoiser():
    out_sr = 48_000
    test_sample_rates = [8_000, 16_000, 24_000, 36_000, 44_100, 48_000]
    if platform.system() == 'Windows':
        test_sample_rates = [36_000, 44_100, 48_000]
    for sr in test_sample_rates:
        in_file = f'denoiser_in_{sr}.ogg'
        file_duration = 3
        generate_sine_wave(file_duration, in_file, frequency=440, sample_rate=sr)

        output_path = test_denoiser.__name__ + f'_{sr}.ogg'

        n_cuts = 3
        source_container = MediaContainer(in_file)

        cutpoints = np.arange(file_duration*1000)[1:-1]
        cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

        segments = list(zip(cutpoints[:-1], cutpoints[1:]))

        mix = MixInfo([1.])
        settings = AudioExportSettings(codec='libvorbis', channels = 'mono', bitrate=64000, sample_rate=out_sr, denoise=1)
        export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

        smart_cut(source_container, segments, output_path, audio_export_info=export_info)

        output_container = MediaContainer(output_path)
        assert len(output_container.audio_tracks) == 1

    file_duration = 10

    audio_file_440 = 'denoise_in_440.ogg'
    generate_sine_wave(file_duration, audio_file_440, frequency=440, sample_rate=48_000)
    audio_file_630 = 'denoise_in_630.ogg'
    generate_sine_wave(file_duration, audio_file_630, frequency=630, sample_rate=48_000)

    output_path = test_denoiser.__name__ + '.ogg'

    source_container = MediaContainer(audio_file_440)
    source_container.add_audio_file(audio_file_630)

    mix = MixInfo([1., 1.])

    segments = [(0, source_container.duration)]

    # output denoise
    mix_export_settings = AudioExportSettings(codec='libvorbis', channels='mono', bitrate=92_000, sample_rate=48_000, denoise=2)
    export_info = AudioExportInfo(mix, mix_export_settings)
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    assert_silence(result_container.audio_tracks[0])

    # input 0 denoise
    mix_export_settings = AudioExportSettings(codec='libvorbis', channels='mono', bitrate=92_000, sample_rate=48_000, denoise=0)
    export_info = AudioExportInfo(mix, mix_export_settings)
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[1], result_container.audio_tracks[0])

    # input 1 denoise
    mix_export_settings = AudioExportSettings(codec='libvorbis', channels='mono', bitrate=92_000, sample_rate=48_000, denoise=1)
    export_info = AudioExportInfo(mix, mix_export_settings)
    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    result_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], result_container.audio_tracks[0])

def test_vp9_smart_cut():
    filename = 'vp9.mkv'

    create_test_video(filename, 30, 'vp9', 'yuv420p', 30, (256, 144))
    output_path = test_vp9_smart_cut.__name__ + '.mkv'
    for c in [2, 6]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_vp9_profile_1():
    filename = 'vp9_p1_422.mkv'

    create_test_video(filename, 30, 'vp9', 'yuv422p', 30, (256, 144), profile='1')
    output_path = test_vp9_profile_1.__name__ + '.mkv'
    for c in [2, 6]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_av1_smart_cut():
    filename = 'av1.mkv'

    create_test_video(filename, 30, 'av1', 'yuv420p', 30, (256, 144))
    output_path = test_av1_smart_cut.__name__ + '.mkv'
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_avi_smart_cut():
    filename = 'mpeg4.avi'

    create_test_video(filename, 30, 'mpeg4', 'yuv420p', 30, (32, 18))
    output_path = test_avi_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'mjpeg.avi'
    create_test_video(filename, 30, 'mjpeg', 'yuvj420p', 30, (32, 18))
    output_path = test_avi_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'h263.avi'
    create_test_video(filename, 30, 'h263', 'yuv420p', 30, (128, 96))
    output_path = test_avi_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_avi_to_mkv_smart_cut():
    filename = 'h264_input.avi'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (128, 96))
    output_path = test_avi_to_mkv_smart_cut.__name__ + '.mkv'
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c, pixel_tolerance=50)

def test_flv_smart_cut():
    filename = 'flv.flv'
    create_test_video(filename, 30, 'flv', 'yuv420p', 30, (32, 16))
    output_path = test_avi_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_mov_smart_cut():
    filename = 'h264.mov'
    create_test_video(filename, 30, 'h264', 'yuv420p', 30, (32, 16))
    output_path = test_mov_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'prores.mov'
    create_test_video(filename, 30, 'prores', 'yuv422p10le', 30, (32, 16))
    output_path = test_mov_smart_cut.__name__ + filename
    for c in [2, 5]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_wmv_smart_cut():
    filename = 'mpeg4.wmv'
    create_test_video(filename, 30, 'mpeg4', 'yuv420p', 30, (32, 16))
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 5, 10]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'wmv1.wmv'
    create_test_video(filename, 30, 'wmv1', 'yuv420p', 30, (32, 16))
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 5]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = 'wmv2.wmv'
    create_test_video(filename, 30, 'wmv2', 'yuv420p', 30, (32, 16))
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 5]:
        run_smartcut_test(filename, output_path, n_cuts=c)

    filename = cached_download('https://filesamples.com/samples/video/wmv/sample_960x540.wmv', 'msmpeg.wmv')
    output_path = test_wmv_smart_cut.__name__ + filename
    for c in [2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c, pixel_tolerance=50)

def test_night_sky():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
    output_path = test_night_sky.__name__ + '.mp4'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_night_sky_to_mkv():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
    output_path = test_night_sky_to_mkv.__name__ + '.mkv'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_sunset():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.curated("pexels/time-lapse-video-of-sunset-by-the-sea-854400.mp4")
    output_path = test_sunset.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings)

def test_seeking():
    in_file = "seek_in.mkv"
    ref_file = "seek_ref.mkv"
    create_test_video(in_file, 600, 'h264', 'yuv420p', 30, (32, 18))
    create_test_video(ref_file, 10, 'h264', 'yuv420p', 30, (32, 18))
    output_path = test_seeking.__name__ + '.mkv'

    source = MediaContainer(in_file)
    segments = [(590, 600)]

    smart_cut(source, segments, output_path, log_level='warning')

    ref_container = MediaContainer(ref_file)
    result_container = MediaContainer(output_path)

    check_videos_equal(ref_container, result_container)

# This tests cutting of interlaced video and it fails. I don't see a way to make it work.
# Therefore interlacing is unsupported, probably forever. Leaving the test here for future reference.
def test_fate_interlaced_crop():
    os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = av_datasets.fate("h264/interlaced_crop.mp4")
    output_path = test_fate_interlaced_crop.__name__ + '.mp4'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)

def test_broken_ref_vid():
    # os.environ["PYAV_TESTDATA_DIR"] = 'pyav_datasets'
    filename = '../ref_videos/remove_mistakes_short.mkv'
    output_path = test_broken_ref_vid.__name__ + '.mkv'
    for c in [1, 2, 3]:
        run_smartcut_test(filename, output_path, n_cuts=c)


def test_manual():
    source_container = MediaContainer('../../out.flac')

    output_path = test_manual.__name__ + '.ogg'
    n_cuts = 3
    cutpoints = np.arange(source_container.duration*1000)[1:-1]
    cutpoints = [0] + [Fraction(x, 1000) for x in np.sort(np.random.choice(cutpoints, n_cuts, replace=False))] + [source_container.duration]

    segments = list(zip(cutpoints[:-1], cutpoints[1:]))

    mix = MixInfo([1.])
    settings = AudioExportSettings(codec='libopus', channels = 'mono', bitrate=64000, sample_rate=48000)
    export_info = AudioExportInfo(mix_info=mix, mix_export_settings=settings)

    smart_cut(source_container, segments, output_path, audio_export_info=export_info)

    output_container = MediaContainer(output_path)
    compare_tracks(source_container.audio_tracks[0], output_container.audio_tracks[0])


# Real-world video tests using publicly available videos

def test_google_bigbuckbunny():
    """Test with Google's hosted Big Buck Bunny H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4', 'google_bigbuckbunny.mp4')
    output_base = test_google_bigbuckbunny.__name__
    test_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=2, audio_export_info='auto')

def test_google_elephantsdream():
    """Test with Google's hosted Elephant's Dream H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4', 'google_elephantsdream.mp4')
    output_base = test_google_elephantsdream.__name__
    test_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=2, audio_export_info='auto')

def test_google_forbiggerblaze():
    """Test with Google's hosted ForBiggerBlazes H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4', 'google_forbiggerblaze.mp4')
    output_base = test_google_forbiggerblaze.__name__
    test_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=2, audio_export_info='auto')

def test_google_forbiggeresc():
    """Test with Google's hosted ForBiggerEscapes H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4', 'google_forbiggeresc.mp4')
    output_base = test_google_forbiggeresc.__name__
    test_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=2, audio_export_info='auto')

def test_google_subaru():
    """Test with Google's hosted Subaru Outback commercial H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4', 'google_subaru.mp4')
    output_base = test_google_subaru.__name__
    test_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=2, audio_export_info='auto')

def test_google_tears_of_steel():
    """Test with Google's hosted Tears of Steel H.264 video (using partial segments)"""
    filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4', 'google_tears_of_steel.mp4')
    output_base = test_google_tears_of_steel.__name__
    test_partial_smart_cut(filename, output_base, segment_duration=15, n_segments=2, audio_export_info='auto')

def get_tears_of_steel_annexb():
    """
    Get Tears of Steel in Annex B format (TS container) for testing H.264 NAL parsing.
    Converts from MP4 to TS to force Annex B NAL format.
    """
    mp4_filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4', 'google_tears_of_steel.mp4')
    ts_filename = 'google_tears_of_steel_annexb.ts'

    if not os.path.exists(ts_filename):
        # Convert first ~200 seconds to TS format (covers our problematic non-IDR keyframes)
        import subprocess
        result = subprocess.run([
            'ffmpeg', '-i', mp4_filename,
            '-t', '200',  # First 200 seconds (covers 18.5s, 143.4s, 183.3s non-IDR frames)
            '-c', 'copy',  # Stream copy to preserve exact H.264 stream
            '-f', 'mpegts',  # Force MPEG-TS output (uses Annex B)
            '-y', ts_filename
        ], check=False, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert to TS format: {result.stderr}")

    return ts_filename


def get_testvideos_jellyfish_h265_ts():
    """Fetch Jellyfish HEVC sample and convert it to Annex B in an MPEG-TS container."""
    mp4_filename = cached_download(
        'https://test-videos.co.uk/vids/jellyfish/mp4/h265/360/Jellyfish_360_10s_1MB.mp4',
        'testvideos_jellyfish_h265.mp4'
    )
    ts_filename = 'testvideos_jellyfish_h265.ts'

    if not os.path.exists(ts_filename):
        import subprocess
        result = subprocess.run([
            'ffmpeg', '-i', mp4_filename,
            '-c', 'copy',
            '-f', 'mpegts',
            '-y', ts_filename
        ], check=False, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert HEVC sample to TS format: {result.stderr}")

    return ts_filename

def test_h264_non_idr_keyframes():
    """
    Test that H.264 non-IDR I-frame issues are properly handled.

    Cuts Google Tears of Steel video at specific points that would force segments
    to start with non-IDR I-frames (NAL type 1). Without proper NAL filtering,
    this causes '[h264] no frame!' errors during playback validation.

    This test should PASS after implementing H.264 NAL unit filtering.
    """
    original_filename = cached_download('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4', 'google_tears_of_steel.mp4')

    source = MediaContainer(original_filename)

    # Cut just before the first 3 non-IDR keyframes (same as comprehensive test)
    segments = [
        (Fraction(0), Fraction(18400, 1000)),           # 0 to 18.4s (before 18.5s non-IDR)
        (Fraction(18400, 1000), Fraction(143280, 1000)), # 18.4s to 143.28s (before 143.375s non-IDR)
        (Fraction(143280, 1000), Fraction(183230, 1000)), # 143.28s to 183.23s (before 183.33s non-IDR)
        (Fraction(183230, 1000), source.duration)     # 183.23s to end
    ]

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)

    output_filename = 'h264_non_idr_minimal_test.mp4'

    smart_cut(source, segments, output_filename,
                audio_export_info=audio_export_info,
                video_settings=video_settings,
                log_level='info')

    result = MediaContainer(output_filename)

    # Test video playback - this should work after NAL filtering is implemented
    check_videos_equal_segment(source, result, 16.5, 4, pixel_tolerance=20)

def test_ts_h264_to_mp4_cut_on_keyframes():
    """
    Verify TS -> MP4 remux works when cutting on keyframes only.

    This exercises the pure passthrough path (no recoding) to ensure
    Annex B H.264 from TS containers can be muxed into MP4 correctly.
    """
    ts_input = get_tears_of_steel_annexb()
    source = MediaContainer(ts_input)

    # Pick GOP-aligned segments so the cut stays in passthrough mode
    gop_times = source.gop_start_times_pts_s
    assert len(gop_times) > 14, "Expected Tears of Steel sample to provide enough GOPs"

    segments = [
        (Fraction(gop_times[5]), Fraction(gop_times[7])),
        (Fraction(gop_times[12]), Fraction(gop_times[14])),
    ]

    cut_segments = make_cut_segments(source, segments, keyframe_mode=True)
    assert all(not segment.require_recode for segment in cut_segments), "Keyframe cuts should not require recoding"

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))

    output_filename = test_ts_h264_to_mp4_cut_on_keyframes.__name__ + ".mp4"

    smart_cut(
        source,
        segments,
        output_filename,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.KEYFRAMES, VideoExportQuality.HIGH),
        log_level='warning'
    )

    with av.open(output_filename) as container:
        assert 'mp4' in container.format.name, f"Expected MP4 container, got {container.format.name}"
        video_streams = [stream for stream in container.streams if stream.type == 'video']
        assert len(video_streams) == 1, f"Expected 1 video stream, found {len(video_streams)}"
        assert video_streams[0].codec_context.name == 'h264', \
            f"Expected H.264 video codec, got {video_streams[0].codec_context.name}"

    result = MediaContainer(output_filename)
    assert len(result.video_frame_times) > 0, "Remuxed output should contain video frames"
    assert result.video_stream.codec_context.codec_tag == 'avc1', "MP4 remux should use avc1 codec tag"


def test_ts_h264_to_mp4_smart_cut():
    """
    Verify converting H.264 in MPEG-TS to MP4 works.

    Uses a known-good H.264 stream repackaged into TS (Annex B) as input,
    then performs smart cut to MP4 and compares against a full recode MP4
    of the same segments to validate correctness. This exercises the
    remux path and container conversion logic from TS -> MP4.
    """
    # Prepare H.264-in-TS input
    ts_input = get_tears_of_steel_annexb()

    source = MediaContainer(ts_input)

    # Pick two short non-overlapping segments well inside the file
    # to keep the test quick and stable.
    # Tears of Steel is long enough for these constants.
    segments = [
        (Fraction(10, 1), Fraction(18, 1)),
        (Fraction(30, 1), Fraction(38, 1)),
    ]

    # Passthrough all audio tracks by default
    s = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[s] * len(source.audio_tracks))

    # Output targets (force MP4 container)
    smartcut_output = test_ts_h264_to_mp4_smart_cut.__name__ + "_smartcut.mp4"
    recode_output = test_ts_h264_to_mp4_smart_cut.__name__ + "_recode.mp4"

    # Smart cut MP4
    smart_cut(
        source,
        segments,
        smartcut_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH),
        log_level='warning'
    )

    # Full recode MP4 for comparison
    smart_cut(
        source,
        segments,
        recode_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH),
        log_level='warning'
    )

    # Validate MP4 container and codec
    with av.open(smartcut_output) as c:
        assert 'mp4' in c.format.name, f"Expected MP4 container, got {c.format.name}"
        vstreams = [s for s in c.streams if s.type == 'video']
        assert len(vstreams) == 1, f"Expected 1 video stream, found {len(vstreams)}"
        assert vstreams[0].codec_context.name in ['h264', 'libx264'], \
            f"Expected H.264 video codec, got {vstreams[0].codec_context.name}"

    # Compare smart cut vs full recode outputs for equivalence
    smartcut_container = MediaContainer(smartcut_output)
    recode_container = MediaContainer(recode_output)

    # Allow a slightly looser tolerance for real-world content
    check_videos_equal(smartcut_container, recode_container, pixel_tolerance=40)


def test_ts_h265_to_mp4_smart_cut():
    """Verify converting H.265 in MPEG-TS to MP4 works through the smart-cut path."""
    ts_input = get_testvideos_jellyfish_h265_ts()

    source = MediaContainer(ts_input)

    segments = [
        (Fraction(0, 1), Fraction(4, 1)),
        (Fraction(5, 1), Fraction(9, 1)),
    ]

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))

    smartcut_output = test_ts_h265_to_mp4_smart_cut.__name__ + '_smartcut.mp4'
    recode_output = test_ts_h265_to_mp4_smart_cut.__name__ + '_recode.mp4'

    smart_cut(
        source,
        segments,
        smartcut_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH),
        log_level='warning'
    )

    smart_cut(
        source,
        segments,
        recode_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH),
        log_level='warning'
    )

    with av.open(smartcut_output) as container:
        assert 'mp4' in container.format.name, f"Expected MP4 container, got {container.format.name}"
        video_streams = [stream for stream in container.streams if stream.type == 'video']
        assert len(video_streams) == 1, f"Expected 1 video stream, found {len(video_streams)}"
        assert video_streams[0].codec_context.name in ['hevc', 'libx265'], \
            f"Expected H.265 video codec, got {video_streams[0].codec_context.name}"

    smartcut_container = MediaContainer(smartcut_output)
    recode_container = MediaContainer(recode_output)

    check_videos_equal(smartcut_container, recode_container, pixel_tolerance=80, allow_failed_frames=2)


def test_ts_h264_to_mkv_smart_cut():
    """
    Verify converting H.264 in MPEG-TS to MKV works using the smart-cut pipeline.

    Mirrors the MP4 smart cut test but targets MKV to ensure remux + recode
    coverage without interfering codec-tag logic.
    """
    ts_input = get_tears_of_steel_annexb()
    source = MediaContainer(ts_input)

    segments = [
        (Fraction(10, 1), Fraction(18, 1)),
        (Fraction(30, 1), Fraction(38, 1)),
    ]

    audio_passthru = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_passthru] * len(source.audio_tracks))

    smartcut_output = test_ts_h264_to_mkv_smart_cut.__name__ + "_smartcut.mkv"
    recode_output = test_ts_h264_to_mkv_smart_cut.__name__ + "_recode.mkv"

    smart_cut(
        source,
        segments,
        smartcut_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH),
        log_level='warning'
    )

    smart_cut(
        source,
        segments,
        recode_output,
        audio_export_info=audio_export_info,
        video_settings=VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH),
        log_level='warning'
    )

    with av.open(smartcut_output) as container:
        assert 'matroska' in container.format.name, f"Expected MKV container, got {container.format.name}"
        video_streams = [s for s in container.streams if s.type == 'video']
        assert len(video_streams) == 1, f"Expected 1 video stream, found {len(video_streams)}"
        assert video_streams[0].codec_context.name in ['h264', 'libx264'], \
            f"Expected H.264 video codec, got {video_streams[0].codec_context.name}"

    smartcut_container = MediaContainer(smartcut_output)
    recode_container = MediaContainer(recode_output)
    check_videos_equal(smartcut_container, recode_container, pixel_tolerance=40)

def test_h264_non_idr_keyframes_annexb():
    """
    Test that H.264 non-IDR I-frame issues are properly handled in Annex B format.

    Uses Tears of Steel converted to TS format (which uses Annex B NAL format instead
    of length-prefixed). This verifies our NAL parsing works for both MP4 (ISOBMFF)
    and TS (Annex B) formats.

    This test should PASS after implementing H.264 NAL unit filtering.
    """
    ts_filename = get_tears_of_steel_annexb()

    source = MediaContainer(ts_filename)

    # Use the same cut points that expose non-IDR keyframe issues
    # These times should still be valid in the converted TS file
    segments = [
        (Fraction(0), Fraction(18400, 1000)),           # 0 to 18.4s (before 18.5s non-IDR)
        (Fraction(18400, 1000), Fraction(143280, 1000)), # 18.4s to 143.28s (before 143.375s non-IDR)
        (Fraction(143280, 1000), Fraction(183230, 1000)), # 143.28s to 183.23s (before 183.33s non-IDR)
        (Fraction(183230, 1000), source.duration)     # 183.23s to end
    ]

    audio_settings = AudioExportSettings(codec='passthru')
    audio_export_info = AudioExportInfo(output_tracks=[audio_settings] * len(source.audio_tracks))
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)

    output_filename = 'h264_non_idr_annexb_test.ts'

    smart_cut(source, segments, output_filename,
                audio_export_info=audio_export_info,
                video_settings=video_settings,
                log_level='info')

    result = MediaContainer(output_filename)

    # Test video playback - this should work after NAL filtering is implemented
    check_videos_equal_segment(source, result, 16.5, 4, pixel_tolerance=20)

def test_testvideos_bigbuckbunny_h264():
    """Test with test-videos.co.uk Big Buck Bunny H.264"""
    filename = cached_download('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4', 'testvideos_bbb_h264.mp4')
    output_path = test_testvideos_bigbuckbunny_h264.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_bigbuckbunny_h265():
    """Test with test-videos.co.uk Big Buck Bunny H.265/HEVC"""
    filename = cached_download('https://test-videos.co.uk/vids/bigbuckbunny/mp4/h265/360/Big_Buck_Bunny_360_10s_1MB.mp4', 'testvideos_bbb_h265.mp4')
    output_path = test_testvideos_bigbuckbunny_h265.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_bigbuckbunny_vp9():
    """Test with test-videos.co.uk Big Buck Bunny VP9"""
    filename = cached_download('https://test-videos.co.uk/vids/bigbuckbunny/webm/vp9/360/Big_Buck_Bunny_360_10s_1MB.webm', 'testvideos_bbb_vp9.webm')
    output_path = test_testvideos_bigbuckbunny_vp9.__name__ + '.webm'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_jellyfish_h264():
    """Test with test-videos.co.uk Jellyfish H.264"""
    filename = cached_download('https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4', 'testvideos_jellyfish_h264.mp4')
    output_path = test_testvideos_jellyfish_h264.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=60)

def test_testvideos_jellyfish_h265():
    """Test with test-videos.co.uk Jellyfish H.265"""
    filename = cached_download('https://test-videos.co.uk/vids/jellyfish/mp4/h265/360/Jellyfish_360_10s_1MB.mp4', 'testvideos_jellyfish_h265.mp4')
    output_path = test_testvideos_jellyfish_h265.__name__ + '.mp4'
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    for c in [1, 2]:
        run_smartcut_test(filename, output_path, n_cuts=c, video_settings=video_settings, pixel_tolerance=80, allow_failed_frames=10)


def test_partial_smart_cut(input_path: str, output_base_name: str, segment_duration=15, n_segments=2, audio_export_info=None, video_settings=None, pixel_tolerance=20):
    """
    Test smart cutting on short segments from random positions in long videos.

    Selects multiple random segments and merges them into a single output file,
    then compares smart cut output against complete recode for quality validation.
    This tests both the cutting algorithm and segment merging logic while being
    much faster than testing the entire video.

    Args:
        input_path: Path to input video file
        output_base_name: Base name for output files (will be suffixed)
        segment_duration: Duration of each test segment in seconds
        n_segments: Number of random segments to merge into one file
        audio_export_info: Audio export settings (defaults to passthrough)
        video_settings: Video export settings for smart cut
        pixel_tolerance: Pixel difference tolerance for comparison
    """
    # Handle audio-only files
    if os.path.splitext(input_path)[1] in ['.mp3', '.ogg']:
        return run_audiofile_smartcut(input_path, output_base_name + '.ogg', 2)

    source = MediaContainer(input_path)
    total_duration = float(source.duration)

    # Skip if video is too short for meaningful testing
    if total_duration < segment_duration * n_segments * 2:
        print(f"Video too short ({total_duration:.1f}s) for partial testing, using regular test")
        return run_smartcut_test(input_path, output_base_name + os.path.splitext(input_path)[1], 2,
                             audio_export_info, video_settings, pixel_tolerance)

    # Calculate safe range for random segment selection (avoid first/last 30 seconds)
    safe_start = 30
    safe_end = total_duration - segment_duration - 30

    if safe_end <= safe_start:
        safe_start = segment_duration
        safe_end = total_duration - segment_duration

    # print(f"Testing {n_segments} segments of {segment_duration}s each merged into one file from {input_path}")

    # Set up default audio export if not specified
    if audio_export_info == 'auto':
        s = AudioExportSettings(codec='passthru')
        audio_export_info = AudioExportInfo(output_tracks=[s] * len(source.audio_tracks))

    # Select multiple random non-overlapping segments
    segments = []

    # Generate all possible non-overlapping segments
    max_segments = int((safe_end - safe_start) // (segment_duration + 1))  # +1 for minimum gap
    if max_segments < n_segments:
        print(f"  Warning: Can only fit {max_segments} non-overlapping segments, reducing from {n_segments}")
        n_segments = max_segments

    # Use a more systematic approach to ensure non-overlapping segments
    attempts = 0
    while len(segments) < n_segments and attempts < 100:
        start_time = safe_start + np.random.random() * (safe_end - safe_start)
        end_time = start_time + segment_duration

        # Check if this segment overlaps with any existing segment
        overlaps = False
        for existing_start, existing_end in segments:
            # Two segments overlap if: start < existing_end AND end > existing_start
            if start_time < existing_end and end_time > existing_start:
                overlaps = True
                break

        if not overlaps and end_time <= safe_end:
            segments.append((start_time, end_time))

        attempts += 1

    if len(segments) < n_segments:
        print(f"  Warning: Could only find {len(segments)} non-overlapping segments out of {n_segments} requested")

    # Sort segments by start time (smart_cut expects them in chronological order)
    segments.sort(key=lambda x: x[0])

    # Print selected segments in chronological order
    # for i, (start, end) in enumerate(segments):
    #     print(f"  Segment {i+1}: {start:.1f}s to {end:.1f}s")

    # Verify segments are non-overlapping after sorting
    for i in range(len(segments) - 1):
        current_end = segments[i][1]
        next_start = segments[i + 1][0]
        if current_end > next_start:
            raise ValueError(f"Segments overlap after sorting: segment {i+1} ends at {current_end:.1f}s but segment {i+2} starts at {next_start:.1f}s")

    # Output paths
    smartcut_output = output_base_name + "_smartcut" + os.path.splitext(input_path)[1]
    recode_output = output_base_name + "_recode" + os.path.splitext(input_path)[1]

    # Test 1: Smart cut (default mode) - merges all segments into one file
    smart_cut_settings = video_settings or VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.HIGH)
    smart_cut(source, segments, smartcut_output,
             audio_export_info=audio_export_info,
             video_settings=smart_cut_settings,
             log_level='warning')

    # Test 2: Complete recode for comparison - merges all segments into one file
    recode_settings = VideoSettings(VideoExportMode.RECODE, VideoExportQuality.HIGH)
    smart_cut(source, segments, recode_output,
             audio_export_info=audio_export_info,
             video_settings=recode_settings,
             log_level='warning')

    # Compare results - both should produce equivalent output
    smartcut_container = MediaContainer(smartcut_output)
    recode_container = MediaContainer(recode_output)

    check_videos_equal(smartcut_container, recode_container, pixel_tolerance=pixel_tolerance)

    smartcut_container.close()
    recode_container.close()

    source.close()

def check_stream_dispositions(source_path, output_path):
    """Helper function to verify that stream dispositions are preserved"""
    with av.open(source_path) as source_container, av.open(output_path) as output_container:
        # Check video stream disposition if it exists
        if source_container.streams.video:
            src_video = source_container.streams.video[0]
            out_video = output_container.streams.video[0]
            assert src_video.disposition.value == out_video.disposition.value, \
                    f"Video disposition mismatch: {src_video.disposition} vs {out_video.disposition}"

        # Check audio stream dispositions
        assert len(source_container.streams.audio) == len(output_container.streams.audio), \
                "Audio stream count mismatch"
        for i, (src_audio, out_audio) in enumerate(zip(source_container.streams.audio, output_container.streams.audio)):
            assert src_audio.disposition.value == out_audio.disposition.value, \
                    f"Audio stream {i} disposition mismatch: {src_audio.disposition} vs {out_audio.disposition}"

        # Check subtitle stream dispositions - this is the main focus
        assert len(source_container.streams.subtitles) == len(output_container.streams.subtitles), \
                "Subtitle stream count mismatch"
        for i, (src_sub, out_sub) in enumerate(zip(source_container.streams.subtitles, output_container.streams.subtitles)):
            assert src_sub.disposition.value == out_sub.disposition.value, \
                    f"Subtitle stream {i} disposition mismatch: {src_sub.disposition} vs {out_sub.disposition}"

            # Specifically check for forced flag preservation
            src_forced = av.stream.Disposition.forced in src_sub.disposition
            out_forced = av.stream.Disposition.forced in out_sub.disposition
            assert src_forced == out_forced, \
                    f"Subtitle stream {i} forced flag mismatch: source={src_forced}, output={out_forced}"

def test_subtitle_disposition_preservation():
    """Test that subtitle disposition flags (especially forced) are preserved during smart_cut"""
    file_duration = 10
    input_path = make_video_with_forced_subtitle('test_subtitle_forced.mkv', file_duration)
    output_path = test_subtitle_disposition_preservation.__name__ + '.mkv'

    # Load source container and verify it has forced subtitles
    source = MediaContainer(input_path)

    # Verify the source has the expected subtitle with forced flag
    with av.open(input_path) as container:
        assert len(container.streams.subtitles) == 1, "Source should have exactly 1 subtitle stream"
        sub_stream = container.streams.subtitles[0]
        assert av.stream.Disposition.forced in sub_stream.disposition, \
            f"Source subtitle should have forced flag, got: {sub_stream.disposition}"

    # Run smart_cut with a simple segment
    segments = [(Fraction(2), Fraction(8))]  # Cut from 2s to 8s
    smart_cut(source, segments, output_path, log_level='warning')

    # Verify disposition preservation
    check_stream_dispositions(input_path, output_path)

def test_multiple_language_subtitles():
    """Test that multiple non-forced subtitle tracks with different languages are preserved"""
    file_duration = 10

    # Define English subtitle content
    en_subtitle_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:05,000 --> 00:00:07,000
This is English

3
00:00:08,500 --> 00:00:09,500
The end
"""

    # Define Finnish subtitle content
    fi_subtitle_content = """1
00:00:01,000 --> 00:00:03,000
Hei maailma

2
00:00:05,000 --> 00:00:07,000
Tämä on suomea

3
00:00:08,500 --> 00:00:09,500
Loppu
"""

    # Configure both subtitle tracks
    subtitle_configs = [
        {
            'content': en_subtitle_content,
            'language': 'en',
            'disposition': 'default',  # English is default
            'temp_file': 'tmp_test_subtitle_en.srt'
        },
        {
            'content': fi_subtitle_content,
            'language': 'fi',
            'disposition': '',  # Finnish has no special disposition
            'temp_file': 'tmp_test_subtitle_fi.srt'
        }
    ]

    # Create test video with both subtitles
    input_path = make_video_with_subtitles('test_multi_subtitle.mkv', file_duration, subtitle_configs)
    output_path = test_multiple_language_subtitles.__name__ + '.mkv'

    # Verify source has both subtitle tracks
    with av.open(input_path) as container:
        assert len(container.streams.subtitles) == 2, f"Source should have 2 subtitle streams, got {len(container.streams.subtitles)}"

        # Check English subtitle (first stream)
        en_sub = container.streams.subtitles[0]
        assert av.stream.Disposition.default in en_sub.disposition, \
            f"English subtitle should have default flag, got: {en_sub.disposition}"

        # Check Finnish subtitle (second stream)
        fi_sub = container.streams.subtitles[1]
        assert fi_sub.disposition.value == 0, \
            f"Finnish subtitle should have no disposition flags, got: {fi_sub.disposition}"

    # Run smart_cut
    source = MediaContainer(input_path)
    segments = [(Fraction(1), Fraction(9))]  # Cut from 1s to 9s to include all subtitles
    smart_cut(source, segments, output_path, log_level='warning')

    # Verify both subtitle tracks are preserved with correct dispositions
    with av.open(output_path) as container:
        assert len(container.streams.subtitles) == 2, f"Output should have 2 subtitle streams, got {len(container.streams.subtitles)}"

        en_sub_out = container.streams.subtitles[0]
        fi_sub_out = container.streams.subtitles[1]

        # Check that dispositions are preserved
        assert av.stream.Disposition.default in en_sub_out.disposition, \
            f"English subtitle disposition not preserved: {en_sub_out.disposition}"
        assert fi_sub_out.disposition.value == 0, \
            f"Finnish subtitle disposition not preserved: {fi_sub_out.disposition}"

    # Run comprehensive disposition check
    check_stream_dispositions(input_path, output_path)


def test_mkv_attachment_preservation():
    """Verify that attachment streams are preserved during smart_cut operations."""
    input_path = make_video_with_attachment('test_mkv_with_attachment.mkv')
    output_path = test_mkv_attachment_preservation.__name__ + '.mkv'

    source = MediaContainer(input_path)
    segments = [(Fraction(0), source.duration)]
    smart_cut(source, segments, output_path, log_level='warning')
    source.close()

    input_attachments = get_attachment_stream_metadata(input_path)
    output_attachments = get_attachment_stream_metadata(output_path)

    assert input_attachments, "Source file is expected to carry at least one attachment"
    assert input_attachments == output_attachments, "Attachment streams metadata mismatch after smart_cut"


def get_test_categories():
    """
    Returns a dictionary of test categories.
    """
    test_categories = {
        # Core synthetic test categories (generated test videos)
        'basic': [
            test_h264_cut_on_keyframes,
            test_h264_smart_cut,
            test_mp4_cut_on_keyframe,
            test_mp4_smart_cut,
        ],

        'h264': [
            test_h264_multiple_cuts,
            test_h264_profile_baseline,
            test_h264_profile_main,
            test_h264_profile_high,
            test_h264_profile_high10,
            test_h264_profile_high422,
            test_h264_profile_high444,
            test_h264_non_idr_keyframes,
            test_h264_non_idr_keyframes_annexb,
            test_mp4_to_mkv_smart_cut,
            test_mkv_to_mp4_smart_cut,
            test_ts_h264_to_mp4_cut_on_keyframes,
            test_ts_h264_to_mkv_smart_cut,
        ],

        'h265': [
            test_h265_cut_on_keyframes,
            test_h265_smart_cut,
            test_mp4_h265_smart_cut,
            test_peaks_mkv_memory_usage,
            test_ts_h265_to_mp4_smart_cut,
        ],

        'codecs': [
            test_vp9_smart_cut,
            test_vp9_profile_1,
            test_av1_smart_cut,
            test_mpg_cut_on_keyframes,
            test_mpg_smart_cut,
            test_m2ts_mpeg2_smart_cut,
            # test_m2ts_h264_smart_cut,
            test_ts_smart_cut,
        ],

        'containers': [
            test_avi_smart_cut,
            test_avi_to_mkv_smart_cut,
            test_flv_smart_cut,
            test_mov_smart_cut,
            test_wmv_smart_cut,
            test_ts_h264_to_mp4_cut_on_keyframes,
            test_ts_h264_to_mp4_smart_cut,
            test_ts_h264_to_mkv_smart_cut,
            test_mkv_attachment_preservation,
        ],

        'audio': [
            test_vorbis_passthru,
            test_mp3_passthru,
        ],

        'mixed': [
            test_mkv_with_video_and_audio_passthru,
            test_subtitle_disposition_preservation,
            test_multiple_language_subtitles,
        ],

        'transforms': [
            # test_vertical_transform, # removed - video transform feature removed
            test_video_recode_codec_override,
        ],

        'long': [
            test_h264_24_fps_long,
            test_h264_1080p,
            test_h265_smart_cut_large,
        ],

        'external': [
            test_night_sky,
            test_night_sky_to_mkv,
            test_sunset,
            test_seeking,
        ],

        # Real-world test categories (using actual videos from the wild)
        'real_world_h264': [
            # Google Common Data Storage videos (H.264)
            test_google_bigbuckbunny,
            test_google_elephantsdream,
            test_google_forbiggerblaze,
            test_google_forbiggeresc,
            test_google_subaru,
            test_google_tears_of_steel,
            # test-videos.co.uk H.264 samples
            test_testvideos_bigbuckbunny_h264,
            test_testvideos_jellyfish_h264,
        ],

        'real_world_h265': [
            # test-videos.co.uk H.265/HEVC samples
            test_testvideos_bigbuckbunny_h265,
            test_testvideos_jellyfish_h265,
        ],

        'real_world_vp9': [
            # test-videos.co.uk VP9 samples
            test_testvideos_bigbuckbunny_vp9,
        ],

        # Meta-categories that combine multiple subcategories
        'real_world': [
            # This will be populated dynamically by combining all real_world_* categories
        ],
    }

    # Populate meta-categories dynamically
    real_world_categories = [key for key in test_categories if key.startswith('real_world_')]
    for category in real_world_categories:
        test_categories['real_world'].extend(test_categories[category])

    # SMC-specific tests (require additional dependencies)
    smc_tests = {
        'smc_audio': [
            test_vorbis_encode_mix,
            test_flac_conversions,
            test_wav_conversions,
            test_vorbis_track_cut,
            test_mp3_track_cut,
        ],
        'smc_mixed': [
            test_mkv_with_video_and_audio_mix,
            test_mix_with_rate_conversion,
            test_denoiser,
        ],
    }

    try:
        # Audio mixing, etc, is omitted from the CLI version, because Librosa and some other libs add a lot of bloat to the binary
        from smc.audio_handling import MixAudioCutter, RecodeTrackAudioCutter

        # Add SMC tests to existing categories
        for category, tests in smc_tests.items():
            test_categories[category] = tests

        print("Including smc tests")
    except ImportError:
        print("Skipping smc tests")
        pass

    return test_categories

def run_tests(category=None, single_test=None, flaky_runs=None, base_seed=None):
    """
    Runs tests from specified category, single test, or all tests.
    """
    test_categories = get_test_categories()

    if single_test:
        # Find the specific test function
        all_tests = []
        for cat_tests in test_categories.values():
            all_tests.extend(cat_tests)

        # Remove duplicates
        seen = set()
        unique_tests = []
        for test in all_tests:
            if test not in seen:
                seen.add(test)
                unique_tests.append(test)

        # Find the test by name
        target_test = None
        for test in unique_tests:
            if test.__name__ == single_test:
                target_test = test
                break

        if target_test is None:
            print(f"Test function '{single_test}' not found.")
            print("Available test functions:")
            for test in sorted(unique_tests, key=lambda t: t.__name__):
                print(f"  {test.__name__}")
            return

        tests_to_run = [target_test]
        print(f"Running single test: {single_test}")

    elif category and category != 'all':
        if category not in test_categories:
            print(f"Unknown category: {category}")
            print("Available categories:", list(test_categories.keys()))
            return

        tests_to_run = test_categories[category]
        print(f"Running {category} tests ({len(tests_to_run)} tests)")
    else:
        # Run all tests
        tests_to_run = []
        for cat_tests in test_categories.values():
            tests_to_run.extend(cat_tests)

        # Remove duplicates while preserving order
        seen = set()
        unique_tests = []
        for test in tests_to_run:
            if test not in seen:
                seen.add(test)
                unique_tests.append(test)
        tests_to_run = unique_tests

        print(f"Running all tests ({len(tests_to_run)} tests)")

    if flaky_runs is not None and flaky_runs < 1:
        print("--flaky requires a positive integer")
        return

    perf_timer = time()
    passed = 0
    failed = 0

    total_runs = flaky_runs if flaky_runs else 1
    seed_value = base_seed if base_seed is not None else DEFAULT_SEED

    if not flaky_runs:
        seed_all(seed_value)

    for test in tests_to_run:
        test_name = test.__name__

        for run_index in range(total_runs):
            run_seed = seed_value if not flaky_runs else (seed_value + run_index) & 0x7FFFFFFF
            seed_all(run_seed)

            try:
                # print(f"Running {test_name}")
                test()
                if flaky_runs:
                    print(f"{test_name} [{run_index + 1}/{total_runs}] seed={run_seed}: ✅ PASS")
                else:
                    print(f"{test_name}: ✅ PASS")
                passed += 1
            except Exception as e:
                if flaky_runs:
                    print(f"{test_name} [{run_index + 1}/{total_runs}] seed={run_seed}: ❌ FAIL")
                else:
                    print(f"{test_name}: ❌ FAIL:")
                traceback.print_exc()
                failed += 1
                # Continue running remaining iterations to gather full flake info

    elapsed = time() - perf_timer
    print(f'\nResults: {passed} passed, {failed} failed')
    print(f'Tests ran in {elapsed:0.1f}s')

if __name__ == "__main__":
    if manual_input is None:
        # Parse arguments for category or single test selection
        args = parse_args()
        base_seed = resolve_base_seed(args)
        run_tests(args.category, args.single, args.flaky, base_seed)
    else:
        # Legacy mode - test specific files
        seed_all(DEFAULT_SEED)
        for file in manual_input:
            try:
                print(f"Testing {file}")
                run_smartcut_test(file, os.path.split(file)[1], 2, audio_export_info='auto', pixel_tolerance=50)
                print(f"Done: {file}")
            except Exception as e:
                print(f"Fail: {file}")
                traceback.print_exc()
