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
encoder outputs them, also one by one in its flush (`_WorkProgress` in `smart_cut.py`). A re-encoded frame's cost
follows how busy the source is around it (its packets' bytes per frame over a second, against the whole video's)
and whether its video fades: x264 took 14 ms per frame on a static scene and 32 ms on a busy one in the same 720p
clip, and about twice as long for a faded frame. The costs are a desktop core's (`progress_costs.py`). A machine's
own, measured there by `tests/regression/calibrate_progress.py`, come from the `progress_costs.json` that
`SMARTCUT_PROGRESS_COSTS` names, else `~/.config/smartcut/` or `%APPDATA%\smartcut\`. A broken file falls back to the
built-in costs with a warning. They honour `cancel_check()`: it is polled before every
GOP and after every re-encoded frame, also each frame an encoder flush outputs and the last flush at the end of the
output (also without a progress callback). Processing stops within a frame, the partial output is removed and the
call returns `False`. Once it returned True it isn't polled again, so True once is enough, and the encoder's
buffered frames are dropped rather than encoded into the removed file. The input file is always closed afterwards.

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
| `misc_data.py` | `FadeInfo` (video/audio fade-in/out durations), `SegmentWithFade`; `CutSegment.fade_info`, `orig_segment_start/end`, `join_start/join_end` |
| `progress_costs.py` | The progress estimate's costs: built-in, or a machine's own `progress_costs.json` (validated, with a fallback) |
| `smart_cut.py` | `smart_cut()` accepts `SegmentWithFade` lists; attaches fade info and segment bounds to the cut pieces, marks the GOPs that overlap a video fade for re-encoding, finds each output's joins and where its segments' pieces start and end there, picks `RecodeAudioCutter` for the audio tracks to re-encode (or warns and copies a track that can't be re-encoded faithfully) |
| `video_cutter.py` | `VideoCutter._apply_fade_to_frame()` / `_fade_yuv_frame()`: per-frame video fade in the re-encode path; faster VP9/AV1 encoder settings |
| `track_cutters.py` | `audio_needs_recode()`: copy or re-encode; `recode_setup()`: faithful re-encodes only, else a copy; `RecodeAudioCutter` + `create_recode_audio_output_stream()`: audio fades, join fades, bitrate (`_recode_audio_bitrate()`); `PrerollPassthruAudioCutter` / `OpusPassthruAudioCutter`: exact copies into MP4 |
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
**Copy the track, unless it joins segments, fades, or is lossless and starts mid-stream; then re-encode it as a
whole, if that can be done faithfully** (decided; `audio_needs_recode()` and `recode_setup()` in `track_cutters.py`).

- **Re-encoded** as one continuous stream (`RecodeAudioCutter`: decoded, cut sample-accurately to each cut piece,
  faded, fed to one encoder), every audio track of an output that
  - has a **join**: segments that aren't contiguous in the source (segments that touch aren't a join);
  - has an **audio fade**;
  - starts **mid-stream with a lossless track** (FLAC, ALAC, PCM, ...): the re-encode is exact, while a copy would
    start at the first whole frame after the cut (FLAC: up to ~100 ms late);
  - is a **WebM** output and the track isn't Opus or Vorbis (re-encoded to Opus, the only audio WebM holds besides
    Vorbis).

  ...but only if the re-encode is faithful (see "Faithful re-encodes only" below); otherwise the track is copied, with
  a warning.
- **Copied** (the source's packets, whole frames), every other track:
  - one segment, or segments that touch, **from the source's start** or **mid-stream**:
    - into an MP4-family output (.mp4/.mov/.m4a), AAC, AC-3/E-AC-3 and Opus start exactly at the cut. These copies
      include the packets just before the cut (pre-roll: 50 ms for AAC/AC-3, 80 ms for Opus), at times before the
      output start. The MP4 edit list skips them, and for Opus also the pre-skip, which is set to their length
      (`PrerollPassthruAudioCutter`, `OpusPassthruAudioCutter`). A pre-roll that reaches the stream's first packet
      is exact too.
    - Other lossy tracks, and every lossy track into Matroska / WebM (no edit list), start at the first whole frame
      after the cut (decided, as main): up to a frame late, with a softened first frame, then exact. Measured: AAC
      into MKV silent for up to a frame; Opus into WebM silent for 10 ms, then x0.53-0.97 for 40 ms (the decoder
      settling), in sync (+1.0 ms, Matroska's ms); MP3's first frame may decode silent (its bit reservoir). Main's
      Opus pre-roll copy into WebM shifted every stream by 72 ms; the branch had re-encoded these.
  - in **KEYFRAMES mode** (`VideoExportMode.KEYFRAMES`, decided: it promises a lossless, fast cut, as on main),
    every track except for audio fades, WebM's Opus, and MP3 at a join (copied MP3 joins decode ~50 ms of silence).
    Before, its joins were re-encoded, and their join fades, timed from the requested bounds inside the whole GOPs
    kept, muted 1.2 s of a 6 s output.

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
  - They are timed from what the output keeps at the join: the segment's first piece's start and its last piece's
    end (`CutSegment.join_start` / `join_end`, decided). Those are the requested bounds, except in KEYFRAMES mode,
    which keeps whole GOPs; timed from the requested bounds, a re-encoded KEYFRAMES join was silent while the video
    went on (1.2 s of a 6 s output). The user's audio fades stay on the requested bounds.
  - Not at the very start or end of the output, and not between segments that touch.
  - On a side with a user audio fade, the longer of the two applies.
  - Edits with fades get them too: before, their re-encoded joins were hard cuts.
  - Measured on a steady 440 Hz tone, largest sample step near a join ÷ the tone's own largest step: hard cut
    5.0-11.0x (an audible click), with the fades 1.0x (copied joins: 1.3-1.4x).
- **Start of a re-encoded track.** The encoder starts cold at the cut, in every container (decided: an 80 ms
  lead-in at negative timestamps, skipped by the MP4 edit list, was tried and dropped).
  - On a 440 Hz tone the first 10 ms came out x1.23 of the source in MP4 (the encoder's onset). Real music was
    x0.94-1.09 in the first 40 ms, within the re-encoding's own range.
  - First 20 ms in 2.5 ms windows: AAC into MKV x0.94-1.28 on noise (x0.94-1.61 on a tone); Opus into WebM
    x0.76-1.10 on noise (x0.90-0.96 on a tone); Opus into MP4 x0.91-0.96 on noise (x0.93-0.96 on a tone).
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
- **MP4 movie timescale: FFmpeg's default, 1000** (decided). The edit lists that delay the first video frame to its
  place after the cut, and end the audio, are written in it, rounded down: the video can start up to 1 ms early
  (0.02897 s instead of 0.02995 s in one cut), so the audio plays up to ~1 ms late against the picture. Setting the
  video's timescale instead was tried and dropped.
- **Faithful re-encodes only** (decided; `recode_setup()` in `track_cutters.py`). A track is re-encoded only if the
  result keeps the codec, the sample format (bit depth), the channel layout and the sample rate, and the encoder
  opens with them. Otherwise it is copied as it is, and the logger `smartcut` warns, e.g. `smartcut: audio track 1
  (dca, stereo) is copied, not re-encoded: no dca encoder in this FFmpeg. So A/V may shift by up to a frame at each
  join.` (or "its audio fades are left out").
  - Encoders: the source codec's own (AAC → FFmpeg's AAC; MP3 → LAME; Opus; Vorbis; AC-3; E-AC-3; MP2; FLAC; ALAC;
    PCM; ...).
  - Exceptions (decided): HE-AAC (v1, v2) → AAC-LC, since FFmpeg can't encode HE-AAC; and in WebM anything but Opus
    or Vorbis → Opus (resampled to 48 kHz if needed), which WebM needs. If such a track can't be re-encoded to Opus,
    the cut fails before writing anything: "WebM output needs Opus or Vorbis audio, and audio track 1 (aac) can't be
    re-encoded to Opus: its channel layout (6 channels) has no channel order; use .mkv".
  - Copied instead: codecs without an encoder in PyAV's FFmpeg, or with an experimental one only (DTS, TrueHD, MLP,
    APE, ...: they became AAC before); AAC profiles other than LC and HE-AAC; an encoder that doesn't take the
    decoder's sample format; 64-bit integer samples; 32-bit FLAC or ALAC (FFmpeg's encoders write 24 of the bits);
    a lossy track whose layout has no channel order (AAC written from 5.1(side) into MP4 reads as "6 channels":
    FFmpeg's AAC encoder refused it and the cut failed).
  - Lossless tracks are worked on in their own sample format, through 64-bit floats, so every sample outside the
    fades is bit-identical to the source: FLAC 16/24-bit, ALAC 16/24-bit, PCM s16/s24/s32/f32/f64/u8. Before, 24-bit
    FLAC came out as 16-bit, 16-bit ALAC as 24-bit, and PCM went through 32-bit float. They keep a layout without a
    channel order (PCM in Matroska has none: "2 channels"), since lossless encoders store the channels as they are.
  - The stream is created from the encoder rather than copied from the source, so its header matches the packets.
    The source's tags are kept, except its encoder's name.
- **Bitrate of re-encoded audio** (`_recode_audio_bitrate()` in `track_cutters.py`, for each track separately; the
  join fades go through the same encoder):
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
  - After a jump the decoder restarts 160 ms, and at least four packets, before the first kept sample
    (`RecodeAudioCutter.PREROLL`), so the kept samples decode as in a continuous decode. Opus decoders settle by
    time, whatever their packet size: the first 20 ms kept differed from a continuous decode by 12 % with 40 ms (four
    10 ms packets, the old rule), 2.5-3.8 % with 80 ms (RFC 7845's minimum), 0.07-0.23 % with 160 ms. MP3 needs the
    frames a frame takes bits from (with 2 packets a cut MP3 started with 13-25 ms of silence).
  - A copy's pre-roll packets are chosen by their timestamps, not their durations, which can be missing (walking
    back by durations of 0 took the whole stream).
  - Decoded frames count as contiguous when their timestamps are less than one tick of the time base from where the
    previous frame ended. In Matroska (1 ms ticks) AAC frames of 21.33 ms arrive at 0, 21, 43 ms, and each "gap" or
    "overlap" of 16-32 samples became inserted or dropped samples: clicks of 6.6-11.5x a tone's largest step, also
    with fades. Strictly less: a gap of exactly one tick, where a tick is a whole packet, is a missing packet (PCM in
    1 ms packets: the audio after it came 1 ms early).
  - This is judged on the decoded frames, before resampling (44.1 kHz into a WebM's Opus). A new resampler's first
    output is short by its delay, and judged after it, the next frame looked like a gap: 16-17 samples of silence
    after every restart (a step x5.6 a tone's own, and every segment 0.35 ms late). Now in sync with the source
    resampled continuously (0.000 ms).
  - The decoder is drained after the source's last packet: frame-threaded decoders (FLAC, ALAC, ...) hold frames
    back, ~1 s of FLAC on 12 threads, which a cut ending near the source's end lost.
  - Matroska stores timestamps in ms: after a jump a re-encoded segment is placed from the first decoded packet's
    rounded timestamp, up to 0.5 ms off (PCM in 33.3 ms packets: 16 samples; in MOV: 0).
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
| `test_cancel.py` | Cancelling (also midway through re-encoding, with and without progress, and in the encoder's last flush), steady progress, input closed (also after a cancel in the last flush) |
| `test_last_frame.py` | Cuts keep their last frame, also when they end with re-encoded frames (H.264, VP9, AV1, fades) |
| `test_multitrack.py` | Every audio track is kept: copied when exact, each re-encoded as a whole with a join or audio fades |
| `test_webm.py` | `.webm` outputs: AAC re-encoded to Opus (with and without fades), Opus copied from the start, re-encoded from mid-stream |
| `test_audio_bitrate.py` | Bitrate of re-encoded audio: from the packets when the source reports none, floors per channel, MP3, lossless sources, encoder limits; cuts at about the source's rate |
| `test_audio_joins.py` | Copied or re-encoded as the rule says; A/V sync within 1.5 ms at the start and every join; level and clicks at the joins (AAC, AC-3, MP3, Opus; mono, stereo, 5.1; MP4, MKV, WebM; HE-AAC with a media dir); copies into MP4 end at the cut; KEYFRAMES mode |
| `test_audio_decode.py` | Decoding for re-encodes: Opus (10 and 20 ms packets) after a jump as in a continuous decode; a copy's pre-roll without packet durations; a one-tick gap; resampled re-encodes in sync |
| `test_audio_faithful.py` | Re-encodes keep the codec, sample format, bit depth and layout: lossless ones bit-identical outside the join fades (FLAC, ALAC, PCM); otherwise copied with a warning (AAC "6 channels"; DTS, TrueHD and 32-bit FLAC with the ffmpeg command); WebM's clear error |
| `test_progress_costs.py` | A machine's `progress_costs.json`: used when valid (an edit weighs its work with its costs); missing, malformed, other versions, out of range fall back to the built-in costs; paths, also without a home folder; logged once; edits with a broken file |
| `test_work_progress.py` | `_WorkProgress`, the progress estimate: frame costs, frames counted as the encoder outputs them, beyond or short of the estimate, the cap, cancelling |
| `test_x265_options.py` | x265 options of HEVC re-encodes: the source's without its machine's threads, the lookahead cap, sources without options |
| `test_calibrate_progress.py` | `calibrate_progress.py`: the adoption rule, held-out edits, CPU checks, file writes, memory release, `--cores`; on a synthetic clip a `--quick` run that must not adopt, `main()` with the verdict forced, and its refusal of changed or missing clips (needs ffmpeg) |
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
