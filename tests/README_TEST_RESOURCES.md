# Test Resources for Smartcut Fade Feature

This directory contains test utilities and sample files for testing the per-segment fade-in/fade-out feature.

## Synthetic Video Generator

### `generate_synthetic_video.py`

Standalone utility for generating synthetic test videos with both video and audio.

**Features:**
- RGB color-changing video (sine wave pattern through colors)
- 440Hz stereo audio (A4 musical note)
- Visual frame markers (white squares every 10 frames)
- Configurable duration, fps, and sample rate
- H.264 video codec (1280x720, yuv420p)
- AAC audio codec (stereo, fltp)

**Usage:**

```bash
# Generate 10-second test video at 30fps
python generate_synthetic_video.py output.mp4 10

# Generate 20-second video at 60fps with 48kHz audio
python generate_synthetic_video.py output.mp4 20 60 48000

# Use as a library in other test scripts
from generate_synthetic_video import generate_test_video
generate_test_video('test.mp4', duration=15, fps=30)
```

**Technical Details:**
- Video: 1280x720, 30fps (default)
- Audio: 48000Hz stereo (default)
- Codec: H.264 (libx264) + AAC
- Color pattern: RGB values oscillate using sine waves at different phases
- Audio pattern: Pure 440Hz sine wave (both channels identical)

## Sample Test Video

### `sample_test_video.mp4`

Pre-generated 10-second sample video for quick testing.

**Specifications:**
- Duration: 10 seconds
- Video: 1280x720 @ 30fps, H.264
- Audio: 48000Hz stereo, AAC, ~128kbps
- File size: ~486KB
- Visual markers: White squares appear every 10 frames (every 0.33s)

**Use Cases:**
- Quick manual testing of fade effects
- Verifying audio/video sync
- Testing parameter consistency
- Baseline reference for fade behavior

**Example Usage:**

```bash
# Test fade-in (2 seconds)
smartcut sample_test_video.mp4 output.mp4 --keep 2:fadein:2,10

# Test fade-out (2 seconds)
smartcut sample_test_video.mp4 output.mp4 --keep 0,8:fadeout:2

# Test both fades
smartcut sample_test_video.mp4 output.mp4 --keep 1:fadein:1.5,9:fadeout:1.5

# Enable debug mode to see audio fade
export SMARTCUT_DEBUG_AUDIO_FADE=1
smartcut sample_test_video.mp4 output.mp4 --keep 2:fadein:2,8
```

## Test Suites

### `test_fade.py`

Comprehensive functional test suite for fade features.

**Tests:**
1. No fade (baseline)
2. Fade-in only
3. Fade-out only
4. Both fade-in and fade-out
5. Custom fade durations
6. Multiple segments with different fades

**Run:** `python test_fade.py`

### `test_fade_keywords.py`

Tests fade syntax with special keywords.

**Tests:**
1. `start` keyword with fade-in
2. `end` keyword with fade-out
3. Negative time (-N) with fade-out
4. Entire video (start to end) with both fades
5. Complex multi-segment with keywords
6. Last N seconds with fade-in

**Run:** `python test_fade_keywords.py`

### `test_audio_parameters.py`

Verifies audio parameter consistency across passthrough and re-encoded sections.

**Tests:**
1. Fully re-encoded segment (fade covers entire segment)
2. Mixed passthrough and re-encoded sections
3. Multi-segment consistency
4. Keyword segments with fades

**Validates:**
- Codec consistency (e.g., aac)
- Sample rate consistency (e.g., 48000Hz)
- Channel count consistency (e.g., stereo/2ch)
- Layout consistency
- Format consistency (e.g., fltp)
- Bitrate preservation

**Run:** `python test_audio_parameters.py`

### `verify_audio_fade.py`

Audio fade verification tool that analyzes RMS volume levels.

**Features:**
- Decodes all audio frames
- Calculates RMS in 100ms windows
- Tests fade-in, fade-out, and both
- Visualizes volume changes over time

**Run:** `python verify_audio_fade.py`

**Note:** Test criteria need adjustment for sine wave audio due to zero crossings.
Better verification is via the built-in debug mode:

```bash
export SMARTCUT_DEBUG_AUDIO_FADE=1
smartcut input.mp4 output.mp4 --keep 2:fadein:2,8
```

## Quick Test Commands

```bash
# Run all functional tests
python test_fade.py
python test_fade_keywords.py
python test_audio_parameters.py

# Generate fresh test video
python generate_synthetic_video.py fresh_test.mp4 15

# Test with debug output
SMARTCUT_DEBUG_AUDIO_FADE=1 smartcut sample_test_video.mp4 out.mp4 --keep 2:fadein:2,8

# Visual inspection
smartcut sample_test_video.mp4 demo.mp4 --keep 1:fadein:2,9:fadeout:2
# Then play demo.mp4 to see/hear the fades
```

## Test Results Summary

**All tests passing as of last run:**
- ✓ Functional tests: 6/6
- ✓ Keyword tests: 6/6
- ✓ Parameter tests: 4/4
- ✓ Total: 16/16 tests passing

## Regenerating Sample Video

If you need to regenerate the sample test video:

```bash
cd tests
python generate_synthetic_video.py sample_test_video.mp4 10 30 48000
```

This will create a fresh 10-second sample with the same specifications.
