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

Both keep every audio track, like the CLI: copied when the copy is exact, otherwise re-encoded as a whole (see
"Audio" below). They validate that
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
| `misc_data.py` | `FadeInfo` (video/audio fade-in/out durations), `SegmentWithFade`; `CutSegment.fade_info`, `orig_segment_start/end`, `join_before/join_after` |
| `smart_cut.py` | `smart_cut()` accepts `SegmentWithFade` lists; attaches fade info and segment bounds to the cut pieces, marks the GOPs that overlap a video fade for re-encoding, finds each output's joins and picks `RecodeAudioCutter` for the audio tracks a copy wouldn't be exact for; sets the MP4 movie timescale |
| `video_cutter.py` | `VideoCutter._apply_fade_to_frame()` / `_fade_yuv_frame()`: per-frame video fade in the re-encode path; faster VP9/AV1 encoder settings |
| `track_cutters.py` | `audio_needs_recode()`: copy or re-encode; `RecodeAudioCutter` + `create_recode_audio_output_stream()`: audio fades, join fades, lead-in, bitrate (`_recode_audio_bitrate()`); `PrerollPassthruAudioCutter` / `OpusPassthruAudioCutter`: exact copies into MP4 |
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
**Copy only when the copy is exact; otherwise re-encode the whole track** (`audio_needs_recode()` in
`track_cutters.py`).

- A copy keeps the source's packets, which are whole frames. It is used only when it is exact: no audio fade in that
  output, and
  - **no join**: the output's segments are contiguous in the source (a single segment, or segments that touch);
  - **a start that a copy can match**: with the source's audio, or mid-stream into an MP4-family output
    (.mp4/.mov/.m4a) for AAC, AC-3/E-AC-3 and Opus. These copies include the packets just before the cut
    (pre-roll: 50 ms for AAC/AC-3, 80 ms for Opus), at times before the output start. The MP4 edit list skips them,
    and for Opus also the pre-skip, which is set to their length (`PrerollPassthruAudioCutter`,
    `OpusPassthruAudioCutter`). Before PR #27 a mid-stream copy began at the first whole frame after the cut (up to
    a frame after the video), and its first ~20 ms were low. A pre-roll that reaches the stream's first packet is
    exact too.
  - in WebM, an Opus or Vorbis track (the only audio WebM holds).
- Otherwise every audio track of that output is re-encoded as one continuous stream (`RecodeAudioCutter`): decoded,
  cut sample-accurately to each cut piece, faded, and fed to one encoder. That covers any codec when:
  - the output has a join or an audio fade;
  - it starts mid-stream into Matroska/WebM (no edit list);
  - it starts mid-stream with a codec that has no pre-roll copy (MP3, FLAC, PCM, ...). FLAC and PCM re-encode
    losslessly.
  - it is a WebM output and the track's codec isn't Opus or Vorbis (re-encoded to Opus).

  Mixing re-encoded and copied packets in one track instead left the encoder's buffered samples misplaced at the
  joins (an audible tick).
- **Why joins aren't copied.** Six segments, five joins, no fades, MP4, measured with
  `tests/regression/join_sync.py` (level in 5 ms windows against the source, outside the 3 ms join fades):

  | | AAC-LC 44.1 kHz copied (before) | re-encoded (now) | HE-AAC 48 kHz copied (before) | re-encoded (now) |
  |---|---|---|---|---|
  | Largest A/V offset of a segment, as players play the audio | 15.5 ms | 0.4 ms | 27.1 ms | 0.4 ms |
  | Samples missing / overlapping by pts | 918 / 263 | 0 / 0 | 1344 / 32 | 0 / 0 |
  | Level at the joins | x0.48-x1.97 | x0.91-x1.02 | x0.13-x9.93 | x0.96-x1.04 |
  | Level inside the segments (re-encoding alone) | x1.00 | x0.82-x1.22 | x1.00 | x0.88-x1.14 |

  - Copied joins drift because a copy can only keep whole frames: each piece starts up to a frame after its cut and
    its last frame runs up to a frame past its end. MP4 stores the gaps and overlaps in the timestamps, but players
    play the decoded samples one after another, so the offset takes a random walk.
  - AAC and AC-3 frames overlap their neighbours (MDCT), so the frame at a join decodes as a mix of both sides.
  - Copied MP3 had ~50 ms of silence after every join: a frame can take bits from earlier frames (the bit reservoir).
- **Fades at joins.** At every join of a re-encoded track, the earlier segment fades out over its last 3 ms and the
  later one fades in over its first 3 ms (`RecodeAudioCutter.JOIN_FADE`).
  - Not at the very start or end of the output, and not between segments that touch.
  - On a side with a user audio fade, the longer of the two applies.
  - Edits with fades get them too: before, their re-encoded joins were hard cuts.
  - Measured on a steady 440 Hz tone, largest sample step near a join ÷ the tone's own largest step: hard cut
    5.0-11.0x (an audible click), with the fades 1.0x (copied joins: 1.3-1.4x).
- **Start of a re-encoded track.** When the output starts mid-stream, MP4-family outputs feed the encoder 80 ms of
  the source audio before the cut (`RecodeAudioCutter.LEAD_IN`).
  - It goes in at negative timestamps, and the edit list skips it, the same mechanism as the copy's pre-roll. Its
    `media_time` is the lead-in plus the encoder delay, e.g. 3528 + 1024 samples at 44.1 kHz.
  - The encoder then starts with the signal already running. On a 440 Hz tone the first 20 ms came out at x1.00 of
    the source (AAC, AC-3, MP3), A/V +0.05 ms. From a cold start the first 10 ms were x1.23 (encoder onset).
  - The lead-in takes the first segment's gain, so it is silent before a fade-in.
  - Real music re-encoded from a cold start was x0.94-1.09 in the first 40 ms, within the re-encoding's own range.
  - Exceptions, which start cold at the cut:
    - **Opus in MP4.** Its stream header's pre-skip (PreSkip in `dOps`), which some players may use instead of the
      edit list, is written by the encoder when it opens, before any audio, and PyAV can't change it afterwards.
      With a lead-in FFmpeg was exact (it follows the edit list), but a player that follows the pre-skip would play
      80 ms of audio from before the cut, 80 ms late. Cold start, first 20 ms: x0.93-0.96 on a tone (a 2.7x step at
      the onset), x0.91-0.96 on noise. The app never writes Opus into MP4.
    - **Matroska / WebM.** There's no edit list, and players would play a lead-in at its negative time. Cold start,
      first 20 ms in 2.5 ms windows: AAC into MKV x0.94-1.28 on noise (x0.94-1.61 on a tone); Opus into WebM
      x0.76-1.10 on noise (x0.90-0.96 on a tone). Before, both were silent for the first 15-20 ms: copies started a
      frame late, and Opus copies shifted all streams by 73 ms.
- **Matroska / WebM timing.** FFmpeg 8 (PyAV 16) writes the encoder's delay as the track's CodecDelay (AAC 1024
  samples, Opus 312) and keeps those samples at negative timestamps, so nothing is shifted.
  - Re-encoded AAC into MKV: container start -23 ms, the video exactly where it belongs, audio content +0.5 ms.
  - Matroska stores timestamps in milliseconds, so A/V there is exact to about ±1 ms.
- **End.**
  - A copy's last packet runs up to a frame past the end of the output. In MP4 its duration is shortened to end at
    the cut, so the track and its edit list end there, as FFmpeg's own encoders' last packets do.
  - Players that honour the edit list's end (QuickTime/AVFoundation) stop exactly there.
  - FFmpeg ignores it (PyAV 16 / FFmpeg 8 and the ffmpeg 7.1 CLI) and decodes the whole last frame. Copied audio
    then runs on past the last picture: 8 ms in a 9.7 s AAC-LC cut, 28 ms in a 10 s HE-AAC one, at most one frame.
    Re-encoded tracks run on by the encoder's padding (e.g. 17 ms).
  - Making FFmpeg stop exactly would need the last frame re-encoded. Left as is (decided).
- **MP4 movie timescale.** MP4-family outputs set the muxer's `movie_timescale` to the video stream's timescale
  (the first audio track's sample rate without video).
  - The edit lists that delay the first video frame to its place after the cut, and end the audio, are written in
    the movie timescale. With FFmpeg's default of 1000, rounded down, the video started up to 1 ms early: 0.02897 s
    instead of 0.02995 s, so the audio played ~1 ms late against the picture.
- **Output stream.** It is created from the encoder rather than copied from the source, so the stream header
  matches the packets.
  - Encoder: the source codec's own when FFmpeg has one (AAC and HE-AAC → FFmpeg's AAC, which writes AAC-LC since
    it can't encode HE-AAC; MP3 → LAME; Opus; Vorbis; AC-3; FLAC, ...), else AAC. In WebM, anything but Opus or
    Vorbis → Opus.
  - The sample rate and channels are kept, unless the encoder can't do the rate (Opus: 48 kHz; AC-3: 32/44.1/48).
- **Bitrate of re-encoded audio** (`_recode_audio_bitrate()` in `track_cutters.py`, for each track separately; the
  lead-in and the join fades go through the same encoder):
  1. **The source's average bitrate.**
     - As its stream reports it. MP4 always does, with the average, so VBR sources get their average.
     - Otherwise it is computed from the track's packets: total size × 8 ÷ the time those packets span. Matroska and
       WebM usually report none, e.g. for Opus.
     - Lossless sources (FLAC, ALAC, PCM) count as unknown: their rate says nothing about a lossy one's. A PCM source
       cut into .webm asked libopus for 1536 kb/s, which it refuses, and the cut failed.
  2. **At least a floor.**
     - 64 kb/s per channel, mono counted as two: 128 kb/s for mono and stereo, 384 kb/s for 5.1. It applies to AAC,
       Opus, Vorbis, AC-3/E-AC-3 and others. Typical AC-3 sources (192 kb/s stereo, 384-448 kb/s 5.1) keep their own
       rate.
     - MP3: 256 kb/s whatever the channels. Below that LAME scales the input down (0.95 up to 160 kb/s): -0.45 dB per
       re-encode.
     - HE-AAC sources thus become AAC-LC at 128 kb/s or more: 48 → 128 kb/s, outputs ~11 % bigger.
     - Mono at 64 kb/s fared worse. A 64 kb/s mono version of a music clip, 6 segments, re-encoded at 64 kb/s: level
       x0.64-1.20 inside the segments (99.8 % of 5 ms windows) and x0.90-1.10 at the joins. At 128 kb/s: x0.64-1.04
       and x0.92-1.02. On a steady tone at 70 kb/s, the level jumped up to x1.51 for ~20 ms after a join fade.
  3. **At most what the encoder accepts.** The encoder is opened with the rate (1.4 ms per try).
     - If it refuses, the highest rate it accepts below that is used. libopus takes at most 256 kb/s per channel;
       libvorbis about 89-249 kb/s per channel, depending on the sample rate.
     - The rate it reports after opening is used: aac lowers a too-high rate to 6144 bits per channel per frame
       (e.g. 576 kb/s for stereo at 48 kHz), ac3 to 640 kb/s.
     - MP3 is capped at 320 kb/s: LAME takes any rate.
     - Examples: an AC-3 640 kb/s source cut into .webm → Opus at 512 kb/s; AAC 5.1 at 192 kb/s → 384 kb/s.
  4. Lossless targets (FLAC, ALAC, PCM) get no bitrate: they re-encode losslessly.

  What comes out:
  - AAC at a 128 kb/s target came out at 128.2-134.7 kb/s (FFmpeg's AAC runs up to 5 % over).
  - Opus is VBR and averages below its target on music: an 85 kb/s Opus source at the 128 kb/s floor came out at
    115 kb/s.
  - An AAC 256 kb/s and an Opus 192 kb/s source in MKV/WebM came out at about their own rate.
- **WebM video.** `.webm` outputs need VP8, VP9 or AV1 video, and smartcut doesn't re-encode the whole video. Other
  video (e.g. H.264) fails before anything is written, with "WebM output needs VP8, VP9 or AV1 video; this video is
  H.264: use .mp4 or .mkv" (`OutputFormatError`; `process_segments` returns False, the CLI exits with 1).
- **Timing.** Audio is kept locked to the video timeline: silence is inserted where the source has no audio.
  - After a jump the decoder restarts four packets early, so the first kept samples decode cleanly (Opus needs
    80 ms, MP3 the frames a frame takes bits from).
  - Decoded frames count as contiguous when their timestamps differ by up to one tick of the time base. In Matroska
    (1 ms ticks) AAC frames of 21.33 ms arrive at 0, 21, 43 ms, and each "gap" or "overlap" of 16-32 samples
    became inserted or dropped samples: clicks of 6.6-11.5x a tone's largest step, also with fades.
- **Gain.** A per-sample linear ramp (vectorised with numpy).
- **Cost.** The whole-track re-encode with FFmpeg's default AAC coder (decoding included) took 0.029-0.030 s per
  second of kept audio on one desktop core. The `fast` AAC coder was evaluated (about 1.8x faster) and rejected: it
  left audible holes in the treble.

  Typical app edits without fades, one core, minimum of 3 runs, before (audio copied) and now (re-encoded):

  | Source (audio) | Segments (audio kept) | Time | Peak memory over the process before | Output size |
  |---|---|---|---|---|
  | real_h264_720p (AAC-LC 44.1 kHz) | 2 (30 s) | 5.21 → 5.71 s | +120 → +128 MB | 3.24 → 3.26 MB |
  | real_h264_720p | 3 (41.8 s) | 8.13 → 9.20 s | +122 → +132 MB | 4.54 → 4.56 MB |
  | fb_sd_h264 (HE-AAC 48 kHz) | 2 (30 s) | 3.00 → 3.89 s | +45 → +61 MB | 2.62 → 2.91 MB |
  | fb_sd_h264 | 3 (41.8 s) | 6.58 → 7.92 s | +46 → +61 MB | 3.42 → 3.83 MB |

  That's 0.5-1.3 s more per edit, 0.017-0.032 s per second of kept audio. HE-AAC outputs grow by ~11 %: they become
  AAC-LC at 128 kbps.

### Re-encoded GOPs (VP9, AV1, HEVC)
- Re-encoded GOPs use `cpu-used=5` (VP9) and `cpu-used=8` (libaom-av1), plus `row-mt`. On one CPU core this made
  AV1 smart cuts ~16x and VP9 ~2x faster, for ~1.7 dB / ~0.7 dB lower PSNR on the re-encoded GOPs only.
- Full re-encodes use the same faster VP9/AV1 speed settings (`cpu-used` 5 / 8), on purpose: on a 1-CPU server AV1
  at the default speed is ~16x slower.
- Re-encoded GOPs use a 10-frame encoder lookahead: x264 and x265 `rc-lookahead`, libvpx/libaom `lag-in-frames`;
  libaom only from 720p up, since below that it was ~35% slower. The defaults hold 19-40 frames of several MB each.
  Full re-encodes (`VideoExportMode.RECODE`) keep the encoders' default lookahead.
  - Peak memory on one core, 1080p with fades: x264 +429 → +239 MB, libaom +545 → +331 MB.
  - AV1 1080p also got faster: 209 → ~135 s.
  - Cost: 0.4-0.9 dB PSNR on the re-encoded frames only (at 47-52 dB).
  - x265 (default 20): a source's own lookahead, reused with its x265 options, is kept when shorter. x265 refuses a
    lookahead that isn't longer than its B-frame runs, so a source with 10 or more B-frames gets bframes + 1. One
    core, 2 segments with fades, x265 defaults (camera-like sources):

    | | Peak memory over the process | PSNR, re-encoded frames | Output |
    |---|---|---|---|
    | 1080p | +516 → +433 MB | 51.49 → 51.19 dB | 3.78 → 3.73 MB |
    | 720p | +470 → +428 MB | 50.30 → 49.91 dB | 2.40 → 2.36 MB |
    | 360p | +151 → +136 MB | 47.38 → 46.98 dB | 1.36 → 1.34 MB |

    With the source's options (`-preset fast`, lookahead 15): 720p +448 → +428 MB, 1080p +464 → +426 MB, −0.61 /
    −0.43 dB. No slower: without a thread pool x265 took 0.985-0.995x the cycles.
- HEVC open GOPs: a copied GOP after a jump re-encodes its RASL leading pictures, decoding the GOP before it to
  prime the decoder. Only the GOP's own frames are kept now: all the decoded frames were, a whole GOP (720p: 152
  frames, 200 MB, for 4 leading frames), and that piece set the edit's peak. 720p, 2 segments with fades:
  +470 → +297 MB, same output.
- HEVC sources with x265's options in their headers: smartcut reuses them (they make the re-encoded GOPs match), but
  not the threads of the machine that encoded the source (`frame-threads`, `numa-pools`, `lookahead-threads`):
  from a 12-thread PC they gave a 1-CPU server a 12-thread pool and 3 frame threads. Without them, one core,
  2 segments with fades: 1080p +590 → +426 MB, 720p +388 → +252 MB, same PSNR (±0.02 dB).
- Measuring x265 on one core: x265 sets its pool threads' affinity itself and sizes its pool from all the machine's
  CPUs. `run_cases.pin_to_one_core()` (the harness's `--one-core`) holds its threads on one core with a job object
  on Windows and gives it a 1-CPU machine's threads. Before, "one-core" x265 times were on ~4.8 cores. With x265's
  1-thread pool, times on one Windows core vary up to 2x from run to run (without a pool 1-2 %).
- AV1 output streams keep the source's pixel format and colour signalling (10-bit sources stay 10-bit).
- 10/12-bit VP9 (profile 2/3) can't be re-encoded and fails.

## Debugging Audio Fades
```bash
export SMARTCUT_DEBUG_AUDIO_FADE=1
smartcut input.mp4 output.mp4 --keep 2:fadein:2,8
```
Prints one line per cut piece that fades (also the 3 ms fades at joins) to stderr:
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
| `test_multitrack.py` | Every audio track is kept: copied when exact, each re-encoded as a whole with a join or audio fades |
| `test_webm.py` | `.webm` outputs: AAC re-encoded to Opus (with and without fades), Opus copied from the start, re-encoded from mid-stream |
| `test_audio_bitrate.py` | Bitrate of re-encoded audio: from the packets when the source reports none, floors per channel, MP3, lossless sources, encoder limits; cuts at about the source's rate |
| `test_audio_joins.py` | Copied or re-encoded as the rule says; A/V sync within 1 ms at the start and every join; level and clicks at the joins (AAC, AC-3, MP3, Opus; MP4, MKV, WebM; HE-AAC with a media dir) |
| `regression/` | Measurement-based regression harness, test media generator and diagnostic tools, e.g. `join_sync.py` for A/V sync at every join (see its README) |

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
- Splicing re-encoded AAC-LC audio around fades and joins into copied audio (faster than re-encoding the whole
  track, but impossible for HE-AAC: FFmpeg can't encode it, and a splice moves the mismatch to the splice points)
