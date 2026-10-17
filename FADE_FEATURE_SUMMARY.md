# Per-Segment Fade-In/Fade-Out Feature

## Overview
Optional fade-in and fade-out effects at the start and/or end of individual kept segments, with independent
control of video and audio. By default no fading is applied.

This fork (`sataz-ehl/smartcut`) adds the feature on top of upstream `skeskinen/smartcut`. It was ported to
upstream's 2.0 module layout in Oct 2026 (see "Implementation" below); the original implementation lived in
the old `cut_video.py`.

## CLI Syntax
The `--keep` option supports fade parameters with independent control over video and audio fading:

- **No fade** (default):
  ```bash
  --keep 10,20
  ```

- **Basic fade-in/out** (applies to both video and audio, 1 second default):
  ```bash
  --keep 30:fadein,40          # Fade-in only
  --keep 50,60:fadeout         # Fade-out only
  --keep 70:fadein:1.5,80:fadeout:2.0  # Custom durations
  ```

- **Video-only fades** (audio is copied):
  ```bash
  --keep 10:videofadein:2,30:videofadeout:3
  ```

- **Audio-only fades**:
  ```bash
  --keep 10:audiofadein:2,30:audiofadeout:3
  ```

- **Mixed fade effects** (independent video/audio control):
  ```bash
  --keep 10:videofadein:2,30:audiofadeout:3
  ```

- **Multiple segments** with different fade configurations:
  ```bash
  --keep 30,40 --keep 60:videofadein,70 --keep 80:audiofadein,90:fadeout
  ```

**Available fade keywords**:
- `fadein` / `fadeout` - Apply to both video and audio (backward compatible)
- `videofadein` / `videofadeout` - Apply to video only
- `audiofadein` / `audiofadeout` - Apply to audio only

**Default fade duration**: 1 second if not specified.

Segments are validated (start must be before end) and sorted by start time. Fades can't be combined with
`--frames` mode, and `--cut` mode doesn't support fades.

## Python API (`smartcut/__init__.py`)
High-level API for programmatic use (Gradio, web apps):
- `process()`: one segment with optional independent video/audio fades
- `process_segments()`: several segments, each with its own fades (dicts or `CutSegmentConfig`)
- `CutSegmentConfig`: segment configuration; `end=None` means the end of the video

Both keep every audio track, like the CLI (with audio fades each one is re-encoded as a whole). They validate that
start < end (raising `ValueError` with "Invalid segment ..."), sort segments by start time,
and report progress as `progress_callback(current, total)`: thousandths of the estimated work (total is 1000).
Re-encoding a GOP can take a thousand times longer than copying one, so `smart_cut()` weighs re-encoded frames
by measured per-codec costs, copied frames and re-encoded audio far less, and counts re-encoded frames as the
encoder outputs them (`_WorkProgress` in `smart_cut.py`). They honour `cancel_check()`: it is polled before every
GOP and after every re-encoded frame (also without a progress callback), processing stops, the partial output is
removed and the call returns `False`. Once it returned True it isn't polled again, so True once is enough, and the
encoder's buffered frames are dropped rather than encoded into the removed file. The input file is always closed
afterwards.

```python
import smartcut

# Single segment with video-only fade
smartcut.process("in.mp4", "out.mp4", start=10, end=20, video_fadein=2, video_fadeout=3)

# Multiple segments with mixed fades
smartcut.process_segments("in.mp4", "out.mp4", [
    {"start": 10, "end": 20, "video_fadein": 2},
    {"start": 30, "end": 40, "audio_fadeout": 2},
], progress_callback=lambda cur, tot: print(f"{cur}/{tot}"), cancel_check=lambda: False)
```

## Implementation
| File | Fade-related code |
|---|---|
| `misc_data.py` | `FadeInfo` (video/audio fade-in/out durations), `SegmentWithFade`; `CutSegment.fade_info`, `orig_segment_start/end` |
| `smart_cut.py` | `smart_cut()` accepts `SegmentWithFade` lists; attaches fade info to the cut pieces and marks the GOPs that overlap a video fade for re-encoding; picks `RecodeAudioCutter` when any segment has an audio fade |
| `video_cutter.py` | `VideoCutter._apply_fade_to_frame()` / `_fade_yuv_frame()`: per-frame video fade in the re-encode path; faster VP9/AV1 encoder settings |
| `track_cutters.py` | `RecodeAudioCutter` + `create_recode_audio_output_stream()`: audio fades |
| `__main__.py` | CLI parsing of the fade keywords, multiple `--keep` |
| `__init__.py` | High-level API above |

## How It Works

### Video
- Only GOPs that overlap a video fade region are re-encoded, in addition to the partial GOPs at cut points that
  every smart cut re-encodes. GOPs between the fades are copied untouched.
- The fade is computed per frame from the original segment bounds, so fade timing is frame-exact even when a
  segment spans several GOPs: `alpha = t / fadein` at the start, `alpha = (length - t) / fadeout` at the end.
- Frames are faded directly in YUV: luma towards the black level and chroma towards neutral, through one lookup
  table each (8/10/12-bit planar, limited or full range). The result goes into a new frame, because the decoder
  may still use the decoded frame as a reference. Other pixel formats (e.g. NV12) fall back to fading in RGB24.
  At 1080p this takes ~14 ms per frame, against ~39 ms for the earlier RGB round trip.

### Audio
- If any segment has an audio fade, each audio track is decoded and re-encoded **as a whole** by one continuous
  encoder; otherwise audio is copied. Mixing re-encoded packets (faded segments) with copied packets (others) in
  one track left the encoder's buffered samples misplaced at the joins (an audible tick) and flushed at the end.
- The output stream is created from the encoder rather than copied from the source, so the stream header matches
  the packets: AAC and HE-AAC sources become AAC-LC (FFmpeg can't encode HE-AAC) at the source bitrate but at
  least 128 kbps. Other codecs are re-encoded with their own encoder when FFmpeg has one, else AAC.
- `.webm` outputs: WebM holds only Opus and Vorbis audio, so other tracks (e.g. AAC, HE-AAC) are always
  re-encoded to Opus as a whole track (48 kHz, at least 128 kbps), fades or not. Opus/Vorbis tracks are copied
  when there are no audio fades. Other containers are unchanged.
- Audio is cut sample-accurately to each cut piece and kept locked to the video timeline (silence is inserted
  where the source has no audio). After a jump the decoder restarts two packets early, so the first kept samples
  decode cleanly.
- The gain is a per-sample linear ramp (vectorised with numpy).
- Cost: roughly 1.3 s per minute of kept audio on a desktop CPU with FFmpeg's default AAC coder. The `fast` AAC
  coder was evaluated (about 1.8x faster) and rejected: it left audible holes in the treble.

### VP9 / AV1
- Re-encoded GOPs use `cpu-used=5` (VP9) and `cpu-used=8` (libaom-av1), plus `row-mt`. On one CPU core this made
  AV1 smart cuts ~16x and VP9 ~2x faster, for ~1.7 dB / ~0.7 dB lower PSNR on the re-encoded GOPs only.
- Re-encoded GOPs use a 10-frame encoder lookahead: x264 `rc-lookahead`, libvpx/libaom `lag-in-frames`; libaom only
  from 720p up, since below that it was ~35% slower. The defaults hold 19-40 frames of several MB each. Full
  re-encodes (`VideoExportMode.RECODE`) keep the encoders' default lookahead.
- Full re-encodes use the same faster VP9/AV1 speed settings (`cpu-used` 5 / 8), on purpose: on a 1-CPU server AV1
  at the default speed is ~16x slower.
  - Peak memory on one core, 1080p with fades: x264 +429 → +239 MB, libaom +545 → +331 MB.
  - AV1 1080p also got faster: 209 → ~135 s.
  - Cost: 0.4-0.9 dB PSNR on the re-encoded frames only (at 47-52 dB).
- AV1 output streams keep the source's pixel format and colour signalling (10-bit sources stay 10-bit).
- 10/12-bit VP9 (profile 2/3) can't be re-encoded and fails.

## Debugging Audio Fades
```bash
export SMARTCUT_DEBUG_AUDIO_FADE=1
smartcut input.mp4 output.mp4 --keep 2:fadein:2,8
```
Prints one line per faded cut piece to stderr:
```
[AUDIO_FADE] t=0.000..1.999s | gain 0.000..1.000 | peak 0.250000 → 0.249850 (0.1% reduction)
```
Alternative verification tool: `python smartcut/tests/verify_audio_fade.py`

## Testing
Run from the repo root (tests live in `smartcut/tests/`):

| Script | Checks |
|---|---|
| `test_fade.py` | No fade, fade-in, fade-out, both, custom durations, multiple segments (CLI) |
| `test_fade_keywords.py` | Fades with `start`/`end`/negative times |
| `test_api.py` | `process()` / `process_segments()` with video-only, audio-only, legacy, mixed and multi-segment fades |
| `test_validation.py` | Reversed segments rejected, out-of-order segments sorted (CLI and API) |
| `test_audio_parameters.py` | Re-encoded audio keeps sample rate / channels; mixed and multi-segment cases |
| `verify_audio_fade.py` | Measured audio level ramps for fade-in, fade-out, both |
| `test_cancel.py` | Cancelling (also midway through re-encoding, with and without progress), steady progress, input closed |
| `test_last_frame.py` | Cuts keep their last frame, also when they end with re-encoded frames (H.264, VP9, AV1, fades) |
| `test_multitrack.py` | Every audio track is kept: copied without audio fades, each re-encoded as a whole with them |
| `test_webm.py` | `.webm` outputs: AAC re-encoded to Opus (with and without fades), Opus copied |
| `regression/` | Measurement-based regression harness, test media generator and diagnostic tools (see its README) |

## Usage Examples

### Basic Examples
```bash
# Keep segment from 10s to 20s with no fades
smartcut input.mp4 output.mp4 --keep 10,20

# Keep segment from 30s to 40s with 1-second fade-in (both video and audio)
smartcut input.mp4 output.mp4 --keep 30:fadein,40

# Keep segment from 50s to 60s with 1-second fade-out (both video and audio)
smartcut input.mp4 output.mp4 --keep 50,60:fadeout

# Keep segment from 70s to 80s with both fades (1.5s in, 2.0s out)
smartcut input.mp4 output.mp4 --keep 70:fadein:1.5,80:fadeout:2.0
```

### Independent Video/Audio Fade Examples
```bash
# Video-only fade (audio copied)
smartcut input.mp4 output.mp4 --keep 10:videofadein:2,30:videofadeout:3

# Audio-only fade
smartcut input.mp4 output.mp4 --keep 10:audiofadein:2,30:audiofadeout:3

# Mixed fade effects: video fades in, audio fades out
smartcut input.mp4 output.mp4 --keep 10:videofadein:2,30:audiofadeout:3

# Mixed fade effects: both fade in, only video fades out
smartcut input.mp4 output.mp4 --keep 10:fadein:1.5,30:videofadeout:2

# Complex mix: video fades both ways, audio only fades out
smartcut input.mp4 output.mp4 --keep 10:videofadein:2,30:videofadeout:2:audiofadeout:3
```

### Using Keywords and Special Time Formats
```bash
# Using "start" keyword - fade in from beginning of video
smartcut input.mp4 output.mp4 --keep start:fadein:2,30

# Using "end" keyword - fade out to end of video
smartcut input.mp4 output.mp4 --keep 100,end:fadeout:3

# Using negative time (-10 = last 10 seconds) - fade out at the end
smartcut input.mp4 output.mp4 --keep 60,-10:fadeout:2

# Keep entire video with fade-in at start and fade-out at end
smartcut input.mp4 output.mp4 --keep start:fadein:3,end:fadeout:3

# Keep last 30 seconds of video with fade-in
smartcut input.mp4 output.mp4 --keep -30:fadein:2,end

# Using HH:MM:SS time format with fades
smartcut input.mp4 output.mp4 --keep 00:01:00:fadein:1.5,00:02:30:fadeout:2
```

### Advanced Examples
```bash
# Multiple segments with different fade configurations (any order: they are sorted)
smartcut input.mp4 output.mp4 \
  --keep 50:fadein:2,60:fadeout:2 \
  --keep 10,20 \
  --keep 30:videofadein,40

# Multiple segments with independent video/audio fades
smartcut input.mp4 output.mp4 \
  --keep 10:videofadein:2,20 \
  --keep 30:audiofadein:1.5,40:audiofadeout:1.5 \
  --keep 50:fadein:1,60:fadeout:1

# Combine keywords and numeric times with video/audio fades
smartcut input.mp4 output.mp4 \
  --keep start:videofadein:3,30 \
  --keep 60,-10:audiofadeout:3 \
  --keep -5,end
```

## Windows Support
- `requirements.txt`: numpy, av==16.0.1, tqdm
- `setup_windows.bat`: creates a virtual environment and installs the requirements

## Future Enhancements (Not Implemented)
- Fade curve customization (linear, exponential, etc.)
- Cross-fade between segments
- Fade effects in `--cut` mode
- Splicing re-encoded AAC-LC audio around fades into copied audio (faster than re-encoding the whole track, but
  impossible for HE-AAC and risky at the splice points)
