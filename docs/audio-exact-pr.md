> The full description of the `audio-exact` pull request (too long for GitHub's PR body). Paths under `Q:\` refer to the maintainer's local scratch files.

# Audio exact: copy only exact copies, re-encode the rest in sync, with fades at joins; leaner HEVC re-encodes; even progress

## Summary

Audio is copied unless the output joins segments, has an audio fade or starts mid-stream with a lossless track; then
each audio track is re-encoded as a whole, in sync at every join, and only faithfully (main ef68601 → branch
`audio-exact`):
- **Joins stay in sync.** Copied audio drifted at every join (a random walk: AAC-LC −15.5 ms, HE-AAC −27.1 ms after
  5 joins) and its level jumped for a few ms (x0.13-x9.93). Outputs with a join are now re-encoded: within 0.4 ms
  at every join, at the source's level.
- **No clicks at joins.** Re-encoded joins get a 3 ms fade-out / fade-in, timed from the pieces kept at the join. A
  hard cut made a steady tone's largest sample step 5-11x its own; now 1.0x. This also fixes edits with fades, whose
  re-encoded joins were hard cuts.
- **Faithful re-encodes only** (decided, audio review): the same codec, sample format (bit depth), channel layout and
  sample rate, with an encoder that opens; else the track is copied and the logger `smartcut` warns. Exceptions:
  HE-AAC → AAC-LC, and `.webm` → Opus. Lossless tracks come out bit-identical outside the fades (24-bit FLAC was cut
  to 16 bits); DTS/TrueHD are copied (they became AAC); AAC "6 channels" is copied (the cut failed).
- **Copies where a copy is close enough** (decided): single segments starting mid-stream into Matroska / WebM, and
  MP3 starting mid-stream, are copied from the first whole frame after the cut, as on main. `KEYFRAMES` mode copies
  the audio again (its re-encoded joins muted 1.2 s of a 6 s output), except MP3 and lossless tracks at its joins
  (decided: a lossless re-encode is lossless, and copied FLAC drifted +40 ms at every join).
- **Re-encoded starts are cold** in every container (decided: the 80 ms lead-in was dropped), and MP4 uses FFmpeg's
  default movie timescale again (decided: `movie_timescale` dropped; the video can start up to 1 ms early).
- **Copies into MP4 end at the cut in the edit list**, for players that honour it.
- **Re-encoded audio decodes as continuously after a jump**: 160 ms of pre-roll (10 ms Opus packets had 40 ms: 12 %
  off), resampled re-encodes no longer insert 16-17 samples of silence after every restart (0.35 ms late), a
  one-tick gap counts as a gap, and the decoder is drained at the source's end (a FLAC cut lost its last 0.3 s on 12
  threads).
- **Clicks in re-encodes from Matroska/WebM AAC sources fixed** (pre-existing): 1 ms timestamp rounding was taken
  for gaps.
- **x265 quiet** when re-encoding camera-style HEVC (it printed its info log for every run).
- **Re-encoded audio bitrate** (decided):
  - The source's average, from its packets when the stream reports none (Matroska/WebM, Opus).
  - At least 64 kb/s per channel, mono counted as two (128k mono/stereo, 384k 5.1); MP3 256k.
  - At most what the encoder accepts.
  - Fixes a pre-existing failure: a PCM source cut into .webm asked libopus for 1536 kb/s, which it refuses.
- **HEVC verified frame by frame** (open and closed GOPs).
- **Matroska / WebM sources** (pre-existing on main):
  - An HEVC cut from inside the first GOP into the next failed ("pts < dts", `process_segments` returned False).
    Fixed.
  - The reported "one frame too many, first frame 12-26 ms off" is not a defect but the time origin. Segment times
    count from the container start, and in Matroska the audio encoder's delay is part of the timeline (decided:
    kept, documented).
  - Checked frame by frame on 300 cuts (H.264, HEVC, VP9, AV1; MKV, WebM, MP4).
- **HEVC re-encodes use less memory** (one core, 2 segments with fades):
  - a lookahead of at most 10 frames, as x264, libvpx and libaom have: 1080p +516 → +433 MB, 720p +470 → +428 MB,
    for 0.3-0.6 dB PSNR on the re-encoded frames (at 47-51 dB), no slower;
  - open-GOP pieces that re-encode their leading pictures no longer hold a whole decoded GOP: 720p +470 → +297 MB,
    byte-identical output;
  - the x265 options reused from a source no longer bring the encoding machine's threads (they gave a 1-CPU server
    12 pool threads and 3 frame threads): 1080p with x265 options +590 → +426 MB, same PSNR (commit e6adedc,
    decided: kept).
- **Progress keeps an even pace:** 26 typical edits, one core, largest distance from an even pace, on average
  9.5-9.7 % → 4.4-4.7 % (main 10.1-10.7 %), measured from the first update. Measured from the cut's real start
  (the first update comes only after the first re-encoded frames, with AV1 after a 2-4 s keyframe), and remeasured
  under load: main 9.3 % → this PR 5.1 % (from the first update in the same runs: 10.3 → 5.0 %).
  The most common app edit (2 segments without fades, H.264 720p) went from 16.8-17.1 % to 4.4-6.8 % (main
  19.0-19.6 %).
  - A re-encoded frame's estimated cost now follows the content and its fade. Encoders take 2-3x longer on busy
    scenes than on static ones (x264 at 720p: 14 vs 32 ms per frame in one clip). The estimate uses the source's
    bytes per frame over the surrounding second, against the whole video's.
  - Faded frames cost x264 and x265 about twice as much.
  - An encoder's flush now counts its frames one by one: before, progress paused for the whole flush, up to 3.5 s.
- **A cancel during the encoder's last flush is honoured** (fork code). In an output that ends with re-encoded frames,
  a cancel during the last flush was ignored: every frame was still encoded, then the file removed (a cancel at the
  3rd of 18 flushed frames: 15 more). It now stops at the next frame, like a cancel anywhere else.
- **One-core measurements now hold x265.**
  - x265 sets its threads' affinity itself and escaped the harness's pinning (4.8 cores). `--one-core` now uses a job
    object on Windows and gives x265 a 1-CPU machine's threads.
  - On Windows the cutting thread now also runs at x265's pool priority. Otherwise every x265 encoder close stalled
    0.5-10 s on one core: that, not encoding, was x265's "2-3x slower, 2x noise" with its 1-thread pool.
  - The progress calibration counted HEVC leading-picture pieces as copies; fixed.
  - The copy cost is confirmed on an idle PC: 33-35 µs per frame (model 35).
- **A machine's own progress costs** (`smartcut/progress_costs.py`, `tests/regression/calibrate_progress.py`).
  - The costs' ratios depend on the machine: more cores make the encoders faster, while copying and audio don't.
  - One command, run on the target machine, measures them and writes a `progress_costs.json`. smartcut reads it from
    `SMARTCUT_PROGRESS_COSTS`, else `~/.config/smartcut/` or `%APPDATA%\smartcut\`. The file stays outside the smartcut
    folder, which deploys overwrite.
  - A broken or out-of-range file logs a warning and falls back to the built-in costs. It never stops an edit.
  - The new costs are adopted only if progress is more even with them by at least 0.5 points (decided), both on edits
    they weren't fitted on and in fresh runs. Never from `--quick`, with other load on the CPU, when the pinning
    didn't hold, or without the Facebook clip (decided: the only footage other than the source clip's).
  - The test clips are made once on the dev PC and copied to the server with a manifest that the calibration checks.
  - The files are written to a temporary file and renamed over the old one, so smartcut never reads half a file. On
    Windows, where a file another process has open can't be replaced (nor opened while it is), the rename and
    smartcut's read are retried briefly (second review: 160 of 300 writes failed with a reader in a loop, and 282
    reads fell back to the built-in costs for the process's life; now none).
  - A calibration that leaves out one of the app's codecs (`--codecs h264`) adopts nothing, and no edit may get worse
    by more than 1.5 points (second review, decided).
- **pyright:** 8 → 0 errors in the core modules (smart_cut.py's 6 union types; video_cutter.py's 2 in the fade
  code, without a behaviour change: fade outputs byte-identical).

Upstream code touched:
- `smart_cut.py`: the segment loop now iterates each file's pieces; the `has_joins` / recode decision is replaced. The
  generators' `finish()` loop catches a cancel during the video cutter's last flush.
- `video_cutter.py`:
  - the x265 `log_level` default;
  - `hybrid_recode_cra_segment` keeps only its GOP's decoded frames;
  - the x265 options reused from the source: their threading options are dropped (decided: kept), and the
    lookahead is capped;
  - `flush_encoder` takes the packets one by one (`encode_lazy`, the same packets) for the progress hook;
  - `fetch_frame` counts a packet without a DTS at the last DTS seen, and the first ones at where decoding starts
    (was: the start of the GOP being cut). This changes nothing when every packet has a DTS (MP4).
- `media_container.py`: keeps each video packet's size (`video_frame_sizes`, sorted like `video_frame_times_pts`).

New fork modules: `smartcut/progress_costs.py` (the built-in costs, moved from smart_cut.py, and the file loader);
`smart_cut._WorkProgress` now takes a cost-free `_WorkPlan` and the costs, so a recorded edit can be replayed.

The rest is fork code; upstream's `PassthruAudioCutter` is unchanged.

## The rule (`track_cutters.audio_needs_recode`, `track_cutters.recode_setup`)

Every audio track of an output is re-encoded with one continuous encoder (`RecodeAudioCutter`) when the output
- has a join (segments that aren't contiguous in the source; segments that touch aren't a join);
- has an audio fade;
- starts mid-stream with a lossless track (FLAC, ALAC, PCM, ...): exact, where a copy starts at the next frame;
- is a WebM and the track isn't Opus or Vorbis (re-encoded to Opus).

...and only if `recode_setup` finds a faithful re-encode: the same codec, sample format, layout and rate, and the
encoder opens (exceptions: HE-AAC → AAC-LC; WebM → Opus). Otherwise the track is copied and `smartcut` logs a warning.

Every other track is copied, as whole frames:
- one segment (or touching ones) into MP4/MOV/M4A: AAC, AC-3/E-AC-3 and Opus start exactly at the cut (pre-roll
  packets the edit list or the pre-skip skips, since PR #27) and end there (the last sample shortened);
- other codecs, and Matroska / WebM outputs: from the first whole frame after the cut (decided, as main);
- `KEYFRAMES` mode: every track except for audio fades, WebM's Opus, and at a join MP3 (decided, as main) and lossless
  tracks (decided, second review: re-encoded losslessly). Its fades stay on the requested bounds, so with an
  audio-only fade-in the kept GOP's audio before the requested start is silent, as with video fades (decided: kept).

MPEG audio is re-encoded as the layer its frames say (`mpeg_audio_layer`): FFmpeg reads MP2 in MP4 with its MP3
decoder, and it was re-encoded as MP3. Layer II re-encodes with FFmpeg's mp2 encoder when that is faithful (it takes
s16 only: MP2 in MKV, read by the mp2 decoder, re-encodes), else it is copied with a warning (MP2 in MP4); never MP3.

The MP3-only rule (`_JOIN_UNSAFE_AUDIO_DECODERS`) is folded into this one.

## Measurements

One desktop core where timed; joins and sync measured with the new `tests/regression/join_sync.py` (level in 5 ms
windows against the source, outside the 3 ms join fades).

### Joins: 6 segments, 5 joins, no fades, MP4

| | AAC-LC 44.1 kHz (real_h264_720p) main | branch | HE-AAC 48 kHz (fb_sd_h264) main | branch |
|---|---|---|---|---|
| Largest A/V offset of a segment, by sample count (as players play) | 15.5 ms | 0.4 ms | 27.1 ms | 0.4 ms |
| Samples missing / overlapping by pts | 918 / 263 | 0 / 0 | 1344 / 32 | 0 / 0 |
| Level at the joins | x0.48-x1.97 | x0.91-x1.02 | x0.13-x9.93 | x0.96-x1.04 |
| Level inside the segments (re-encoding alone) | x1.00 (copy) | x0.82-x1.22 | x1.00 (copy) | x0.88-x1.14 |

### Clicks at joins: steady 440 Hz tone, largest sample step within 10 ms of a join ÷ the tone's own

| Copied (main) | Re-encoded, hard cut | Re-encoded with 3 ms join fades (branch) |
|---|---|---|
| 1.3-1.4x | 5.0-11.0x | 0.90-1.00x (MP4 and MKV, 44.1 / 48 kHz) |

### Start of a re-encoded track (2 segments, mid-stream start): first 20 ms in 2.5 ms windows ÷ the source

The encoder starts cold at the cut in every container (decided: the 80 ms lead-in was dropped).

| tone / noise | main (copied) | with the lead-in (dropped) | now: cold start |
|---|---|---|---|
| AAC in MP4 | 1.00 / 1.00 | 1.00 / 0.99-1.02 | 1.00-1.64 / 0.97-1.16 |
| AC-3 in MP4 | 1.00 / 1.00 | 1.00 / 0.98-1.01 | 1.00 / 0.99-1.00 |
| MP3 in MP4 | 1.00 (1.82x step) | 1.00 | 1.00 / 1.00 |
| Opus in MP4 | 0.86-0.93 / 0.81-0.91 | (cold) 0.93-0.96 / 0.91-0.96 | 0.99-1.00 / 0.98-1.01 |
| AAC in MKV | 0.00 for 15 ms (copy a frame late) | (cold) 0.94-1.61 / 0.94-1.28 | (cold, not re-measured) |
| Opus in WebM | 0.00 for 20 ms, all streams shifted 73 ms | (cold) 0.90-0.96 / 0.76-1.10 | (cold, not re-measured) |

Sync in MP4: +1.02 ms with FFmpeg's movie timescale (main and now), +0.05 ms with the video's (dropped). Real music
from a cold start: x0.94-1.09 in the first 40 ms. A single segment starting mid-stream into Matroska / WebM is now
copied (below), so these cold starts apply there only with a join or a fade.

### Matroska / WebM timing

FFmpeg 8 (PyAV 16) writes the encoder delay as CodecDelay (ffprobe `initial_padding` 1024 for AAC, 312 for Opus)
and keeps it at negative timestamps, so nothing is shifted:

| Mid-stream cut 30.3-40 s, re-encoded (now copied as a single segment; a re-encode needs a join or a fade) | container start | first video frame (expected) | audio content offset |
|---|---|---|---|
| AAC → MKV (re-encoded) | −23 ms | 0.033 s (0.033) | +0.54 ms |
| HE-AAC → MKV (re-encoded to AAC-LC) | −21 ms | 0.030 s (0.030) | +0.66 ms |
| Opus → WebM / MKV (re-encoded) | −7 ms | 0.027 s (0.027) | +0.50 ms |

The PyAV and ffmpeg 7.1 CLI sample counts are identical. Matroska stores ms, so A/V there is exact to about ±1 ms.

### MP4 first video frame: FFmpeg's movie timescale (decided: `movie_timescale` dropped)

| Cut at 1.37 s, first kept frame 1.4 s | main and now | with the video's timescale (dropped) |
|---|---|---|
| First video frame in the output (expected 0.03000 s) | 0.02897 s | 0.02995 s |
| Audio against the picture | +1.02 ms | +0.05 ms |

### End of a copy (single segment, MP4): decided to leave FFmpeg's behaviour

| | edit list end, main → branch | decoded audio end − last picture (PyAV = ffmpeg CLI) |
|---|---|---|
| AAC-LC 9.7 s cut | 9.708 → 9.700 s | +8.3 ms (unchanged) |
| HE-AAC 10 s cut | 10.028 → 10.000 s | +28.0 ms (unchanged) |

FFmpeg ignores the edit list's end and decodes the whole last frame (re-encoded tracks: their padding, e.g. +17 ms).
QuickTime/AVFoundation stop at the edit list's end. The file is right for AC-3 and MP3 too (stts' last delta 1408 /
61 samples), though PyAV's demuxer re-derives their packet durations from the frames.

### Audio review round (Oct 19): findings and decisions

| Finding | Before (c022263) | Now |
|---|---|---|
| P1-1 KEYFRAMES mode, 2 segments (kf.py) | audio re-encoded; 0.00 RMS for 1.2 s of a 6 s output (the join fades at the requested bounds inside whole GOPs) | AAC copied, 0.06 throughout; MP3 (re-encoded at its join) 0.08 throughout |
| P1-2 AAC "6 channels" (5.1(side) into MP4), a join or a mid-stream start into MKV | cut failed: `avcodec_open2(aac)` EINVAL | copied, warning "its channel layout (6 channels) has no channel order" |
| P2-3 DTS / TrueHD with a join | AAC-LC, with the source's `ENCODER=... dca` tag | copied (dts / truehd), warning "no dca encoder in this FFmpeg" |
| P2-4 24-bit FLAC with joins | FLAC s16, 16 bits | FLAC s32, 24 bits, bit-identical outside the join fades (also ALAC 16/24, PCM s16-f64) |
| P3-6 44.1 kHz AAC into WebM (Opus 48 kHz) | 16-17 samples of silence after every restart: a step x5.64 a tone's own at 22.8 ms; every segment 0.33-0.35 ms late | no step; 0.000 ms against the source resampled continuously |
| P3-7 Opus, 10 / 20 ms packets, first 20 ms after a jump vs a continuous decode | 12-24 % / 4-9 % (4 packets) | ≤ 1 % (160 ms; measured 0.07-0.23 %) |
| P3-7 copy pre-roll with packet durations of 0 | took the whole stream | 80 ms, by timestamps |
| P3-7 a gap of exactly one tick (PCM in 1 ms Matroska packets) | taken for rounding: audio after it 1 ms early | silence, in sync |
| Found while testing: frame-threaded decoders (FLAC) near the source's end | the last 0.3 s of a 3-segment cut lost on 12 threads (main too) | drained: 4.37 s as expected |

Opus copied into WebM from mid-stream (decision 2; real Opus, cut at 30.3 s): in sync (+1.0 ms, Matroska's ms),
silent for the first 10 ms, then x0.53-0.97 for 40 ms. Main's Opus pre-roll copy there: clean, but every stream
shifted by 72 ms (first video frame at 0.099 s instead of 0.027 s). The previous round's re-encode: x0.85 in the
first 5 ms.

App path (MP4, 6 segments, `join_sync.py`, one core):

| | real_h264_720p (AAC-LC) c022263 → now | fb_sd_h264 (HE-AAC) c022263 → now |
|---|---|---|
| Largest A/V offset of a segment | 0.1 → 0.4 ms | 0.1 → 0.4 ms |
| Level at the joins | x0.92-1.02 → x0.91-1.02 | x0.92-1.06 → x0.96-1.04 |
| Level inside the segments (99.8 % of windows) | x0.84-1.07 (0.86-1.06) → x0.82-1.22 (0.84-1.08) | x0.89-1.14 (0.91-1.09) → x0.88-1.14 (0.91-1.09) |
| First video frame | 0.033285 → 0.033040 s | 0.010260 → 0.010010 s |

- The offsets of 0.3-0.4 ms are the movie timescale (decision 5): the video starts 0.25 ms earlier.
- The level inside changed with the lead-in's removal only (the encoder starts cold, so every frame encodes a little
  differently): the same values as before the lead-in. Every later commit leaves the app outputs byte-identical (the
  6-segment output; an edit with fades and a 2 ms fade at a join).

### Second review round (Oct 20): findings and fixes

No P1/P2. Each fix has a test that fails without it (shown on the code before the fix, or with the fix undone).

| Finding | Before | Now | Test, without the fix |
|---|---|---|---|
| P3-1 audio timestamps jumping back by 3 AAC frames (64 ms; the reviewer's `overlap_mock.py`) | the overlap's first frame dropped, the rest kept: everything after it 42.7 ms late | the whole overlap dropped, the audio after it where its timestamps say (+64.0 ms of source = 0 off) | test_audio_decode: "+42.7 ms late" |
| P3-2 the resampler's flush at a gap (44.1 kHz AAC into WebM) | untested | each side of the gap as the source resampled on its own, silence between (a sample off at most) | test_audio_decode: "differ by up to 0.199" |
| P3-2 the decoder drain | tested on many-CPU machines only | FLAC with 4 frame threads set on the decoder, on any number of CPUs | test_audio_decode, also with the process on one CPU: "1.220 s of 1.500 s" |
| P3-2 the source's ENCODER tag on re-encoded streams | untested | dropped, other tags kept | test_audio_faithful: the tag kept |
| P3-3 segment mode, a track that can't be re-encoded faithfully | one warning per track: file 2's "its audio fades are left out" unsaid | one per track and warning | test_audio_faithful (32-bit FLAC): one warning of two |
| P3-5 MP2 in MP4 (FFmpeg reads it with mp3float) at a join or a fade | re-encoded with LAME: MP3 out (also into MKV, and in KEYFRAMES mode at a join) | copied with a warning ("mp2 doesn't take fltp samples (MPEG audio Layer II read by FFmpeg's mp3float decoder)"); MP2 in MKV re-encodes as MP2 | test_audio_faithful: "0/184 packets copied" |
| Decided: KEYFRAMES, lossless tracks at joins (`kf_lossless.py`, 3 joins) | copied: FLAC +40 / +80 / +120 ms, ALAC -22 / -43 / -65 ms | re-encoded losslessly: within 0.3 ms, bit-identical outside the 3 ms join fades | test_audio_joins: FLAC +11.2 ms, ALAC -21.7 ms late |
| Decided: KEYFRAMES with an audio-only fade-in | the kept GOP's audio before the requested start silent | the same, documented (as video fades) | - |
| P3-4 docs | the MP2 example, README's KEYFRAMES line, FORMAT_SUPPORT's PCM line | fixed; WebM → Opus, HE-AAC, MPEG layers, KEYFRAMES in FORMAT_SUPPORT | - |

### HEVC

- `make_test_media.py` makes `synth_hevc.mp4` (x265 defaults: open GOPs, CRA keyframes with RASL leading pictures),
  `synth_hevc_closed.mp4`, and with `--real` `real_hevc*.mp4` without x265 options, like camera footage.
- Cases E1-E6: cuts, fades, a short cut, closed GOPs, real 360p/720p, all PASS.
- Frame by frame: every output frame is the expected source frame (E1 270, E2 300, E3 30, E4 330 frames); copied
  frames are bit-exact.
- No wrong frame found; the HEVC changes below are about memory and measurement.

Lookahead of re-encoded GOPs: at most 10 frames, as x264, libvpx and libaom have (x265's default 20; a source's own,
reused with its x265 options, kept when shorter; at least bframes + 1, which x265 requires: a 12-B-frame source gets
13). One core, 2 segments with fades (TWO_SEGMENTS_REAL), x265 defaults (camera-like sources):

| | Peak memory over the process | PSNR, re-encoded frames | Output |
|---|---|---|---|
| 1080p | +516 → +433 MB | 51.49 → 51.19 dB | 3.78 → 3.73 MB |
| 720p | +470 → +428 MB | 50.30 → 49.91 dB | 2.40 → 2.36 MB |
| 360p | +151 → +136 MB | 47.38 → 46.98 dB | 1.36 → 1.34 MB |
| 720p / 1080p, x265 options of the source (lookahead 15) | +448 → +428 / +464 → +426 MB | −0.61 / −0.43 dB | |
| For comparison, H.264 1080p (x264 40 → 10) | +421 → +237 MB | 54.20 → 53.99 dB | 6.65 → 6.61 MB |

- PSNR is luma, against what the encoder was given (the source frame, with smartcut's own fade where the segment
  fades); copied frames are bit-exact in every output. `psnr_reencoded.py` now checks the copies against smartcut's
  own plan of the cut and exits 1 on any copy that isn't bit-exact. Checked on R1, R2, E1 and E5: every copy is
  bit-exact; an output re-encoded whole by ffmpeg is reported, 365 of 365 copies.
- No slower: without a thread pool x265 took 0.985-0.995x the cycles. With its 1-thread pool, times on one Windows
  core vary up to 2x from run to run (the encoder-close stalls, see below), and the medians were no higher (360p:
  27.0 → 19.7 s, 8 interleaved runs).
- The outputs are byte-identical to the ones measured with the option injected by the harness.

Other HEVC memory (one core, same edit):
- **Hybrid CRA pieces** (a copied GOP after a jump that re-encodes its RASL leading pictures) decoded the GOP before
  it and kept every frame: real_hevc_720p held 152 frames (57.04-63.50 s, 200 MB) to re-encode 4, and that piece set
  the peak. They now keep only their GOP's frames: +470 → +297 MB, byte-identical outputs (E1-E6, 1080p, 720p with
  x265 options). With the lookahead cap too: +255 MB.
- **The source's x265 threads (commit e6adedc, decided: kept):** the x265 options smartcut reuses from a source
  carry the encoding machine's `frame-threads=3 numa-pools=12` (a 12-thread PC). x265 honours them, so a 1-CPU
  server re-encoded with a 12-thread pool and 3 frame threads. Dropped (with `lookahead-threads`): 1080p
  +590 → +426 MB, 720p +388 → +252 MB, E1 / E2 / E4 +207 / +195 / +173 → +113 / +114 / +111 MB, PSNR ±0.02 dB;
  1080p ~9 % faster. (`cpuid=…` is in the string too; x265 has no such option.)

x265 and one-core measurements:
- x265 sets its pool threads' affinity to the whole NUMA node itself, so it escaped `pin_to_one_core()`: an encode
  used 4.8 cores (CPU time / wall time). It also sizes its pool from all the machine's CPUs (12 pool threads, 3 frame
  threads here; a 1-CPU server gets 1 and 1). `--one-core` now uses a job object on Windows (0.97 cores) and gives
  x265 a 1-CPU machine's threads unless the options set them. x264, libvpx and libaom don't set affinity: their
  earlier one-core numbers stand.
- **With its 1-thread pool, every x265 encoder close stalled 0.5-10 s on one Windows core.** This is a harness
  artifact (Windows scheduling), not encoding.
  - Mechanism, from x265's `threadpool.cpp`: pool threads run at `THREAD_PRIORITY_BELOW_NORMAL` (POSIX: `nice(10)`).
    `ThreadPool::stopWorkers()` waits for them in a loop of `GIVE_UP_TIME()`, which is `Sleep(0)` on Windows and
    `usleep(0)` on POSIX. On one core `Sleep(0)` only lets threads of the same or higher priority run, so the
    starved pool thread ran only when Windows' anti-starvation boost came.
  - Measured (100 frames of 360p, 6 runs each): feeding 0.82-0.84 s and the flush 0.11 s either way. The close took
    0.5-10 s (median 3.3) with the cutting thread at normal priority, and 0.01-0.04 s at below-normal priority.
  - `pin_to_one_core()` now puts the calling thread at the pool's priority on Windows. Whole edits: E5 20.6 → 7.4 s,
    E6 42.2 → 26.2 s, RE10 67.1 → 49.1 s, the same as without a pool (7.0 / 24.7-24.8 / 49.8-54.6 s).
  - This explains the "2-3x slower, up to 2x run-to-run noise" reported before.
  - On Linux `usleep(0)` sleeps for the timer slack (~50 µs), during which the pool thread runs. So the VPS very
    likely doesn't stall (not measured there).
- Per re-encoded HEVC frame:

| | 360p | 720p | 1080p |
|---|---|---|---|
| Before, ~4.8 cores (the old model: 0.005 + 0.020 s/MP) | 9.4-9.9 ms | 23.9-24.2 ms | 46.2-46.7 ms |
| One core, no thread pool | 19.5 ms | 70.5-70.8 ms | 163-179 ms |
| One core, 1-thread pool, stalling closes (reported before) | 43-58 ms | 108-116 ms | 181-199 ms |
| One core, 1-thread pool, no stalls (now) | 19.5-19.8 ms | 73-75 ms | 158-165 ms |
| Model now, a frame of average busyness | 17.5 ms | 60 ms | 130 ms |

### Matroska / WebM sources (branch `mkv-first-frame`)

**The report** ("a Matroska source cut into MKV / WebM with a re-encoded first piece gets one frame too many, its
first frame 12-26 ms off, and the re-encoded H.264's first dts out of order") is the time origin, not a defect.
Decided: kept.
- smartcut counts segment times from the container start, the earliest first timestamp of any stream. Upstream does
  the same: `make_adjusted_segment_times`, and the CLI's frame mode.
- Matroska has no edit list, so it keeps the audio encoder's delay in the timeline. The repro's clip has AAC from
  -21 ms and video from 0; the Opus WebM has audio from -7 ms. An MP4 with its audio encoded into it hides the delay
  in its edit list.
- The repro's checker counted from video pts 0. That explains every number:
  - AAC MKV: 1.37 s is pts 1.349, so frame 1.367 is kept (79 frames, not 78) and lands 18 ms after the cut, not 30
    (12 ms).
  - Opus WebM: the first frame lands 4 ms after the cut, not 30 (26 ms).
- FFmpeg agrees: `ffmpeg -ss 1.37 -to 2.5` keeps 34 frames from both files, as smartcut does; counting from video pts
  0 gives 33.
- The dts were in order. vstart.py printed the packets sorted by pts, which shows normal B-frame reordering.
  Matroska stores no dts, so FFmpeg gives the first 2 packets (the reorder delay) none, in the source too. In file
  order the rest strictly increase (ffprobe).
- main, c022263 and this branch write the same video for the repro.

**Found and fixed: HEVC in Matroska.**
- The failing cut: open GOPs, from inside the first GOP into the next one (that GOP is copied, so its RASL leading
  pictures are re-encoded). It failed with `[matroska] pts (-1349) < dts (419)`, into MKV and MP4, with or without
  fades. main ef68601 fails the same way.
- The first GOP's start dts is unknown in Matroska (the sentinel -100,000,000), so priming the decoder from it
  decodes from the file start.
- `fetch_frame` counted packets without a dts at the start of the GOP being cut. It therefore collected the file's
  first packets as that GOP's own and copied them after the cut.
- Now they count at the last dts seen, and the first ones at where decoding starts.

**Frame by frame.** 12 sources:
- frame-counter clips H.264, HEVC, VP9 and AV1 in MKV (remuxed: video from +21 ms; and audio at negative pts);
- AV1 and VP9 in WebM with Opus (audio from -7 ms);
- H.264 in MKV without audio;
- the 4 MP4 clips.

Each cut 10 ways (off and on keyframes, ending on a keyframe, from the file start, starting in the first GOP; with
and without fades), into MKV / WebM / MP4. For every output frame, the nearest source frame at full resolution and
its time against where it belongs:

| | 6e1d6dc | this branch |
|---|---|---|
| Right frames, each within 1.5 ms of its place (MP4 → MKV ±0.3 ms; MP4 outputs -1.0 ms, the movie timescale) | 292 of 300 | 300 of 300 |
| Cut failed (HEVC MKV from the first GOP into the next, both layouts, into MKV and MP4) | 8 | 0 |
| Outputs identical to 6e1d6dc (MP4 byte for byte, MKV / WebM by packets) | | 292 (116 MP4); the 8 others failed before |

A cut from the file start begins with the first video frame, as before. An MP4 source cut from 0 into MKV keeps the
copied AAC's priming packet, so the Matroska muxer moves the timeline by +21 ms. A/V is unchanged, the same layout
`ffmpeg -c copy` gives.

**run_cases.**
- New cases `K1`-`K7` cut Matroska / WebM sources into MKV / WebM / MP4 (and an MP4 source into MKV).
  `make_test_media.py` makes `synth_h264.mkv`, `synth_hevc.mkv` and `synth_av1_opus.webm`.
- The harness took the output's first frame as the cut. That frame comes up to a frame after the cut (18 ms in
  `K1`), so joins and fades were measured that much early, and 5 K cases failed on correct outputs.
- Output time 0 is now the cut, and source times include the container start.
- On the same outputs, 47 cases measure as before. W5, W6, X_fb_sd_h264 and J1-J5 keep their verdicts and measure
  closer: J5 `aFlat` 0.13/8.14 → 1.00/1.00, J3 0.47/2.33 → 0.92/1.15.

### Progress

`_WorkProgress` estimates the work up front and reports the share done. With one cost per re-encoded frame and
codec, the bar ran ahead or behind by up to 15-20 % of the time in common edits. The cost of a frame varied 2-3x,
and the model didn't know why. Measured on one core:
- **Content.** In one 720p clip x264 took 14 ms per frame on a static scene (20-24 s) and 32 ms on a busy one
  (60-62 s). The source's packets there had 1.4 and 5.6 kB per frame, against the clip's average of 3.8 kB. libaom
  took 118 and 198 ms. libvpx's time doesn't follow the content.
- **Fades.** Faded frames cost x264 and x265 about twice as much: a fade changes every block, so none is skipped as
  unchanged.
- **Flushes.** The frames still in the encoder came out in one flush, counted at its end. Progress paused there for
  up to 3.5 s (x265, 1080p).

The fixes:
- A re-encoded frame costs `(fixed + MP x per_mp) x (1 - busy + busy x busyness) + MP x fade` if its video fades.
  - Busyness is the source's bytes per frame within 0.5 s of the frame, against the whole video's. The packet sizes
    come from the index MediaContainer already builds, 8 bytes per frame.
  - `busy` is the share of the cost that follows busyness: x264 0.55, x265 0.75, libaom 0.85, libvpx 0.05.
  - The fade term: x264 +0.030 s/MP, x265 +0.034. libvpx and libaom have none: runs with a fade-in were faster there,
    the frames after the fade too, which a per-frame term can't place. Fitted per run, the term came out negative
    and made progress less even (RV7_vp9_720p 3.3 → 10.7-10.9 %).
  - Fitted by `progress_timeline.py --calibrate` over the encoder runs of 29 real-clip edits (360x360-1080p) on one
    core: least squares of each run's time. Errors per run, median: x264 8 %, x265 12 %, libvpx 10 %, libaom 9 %.
  - Cross-check: H.264 constants fitted on one source only cut the other's error. Fitted on real_h264*, the Facebook
    cases' mean went 7.1 → 4.5 %; fitted on Facebook, real_h264*'s went 9.7 → 5.5 % (replayed timings).
- Frames are counted in the order they were fed, as the encoder outputs them. A flush outputs them one by one
  (`encode_lazy`, the same packets).
- The copy (35 µs per frame) and audio (0.027 s per second) costs are unchanged: 33-35 µs and 0.027-0.029 s were
  measured.

Largest distance between the progress shown and the share of the time elapsed, at the updates: one core, idle PC,
the same harness for all (2 interleaved runs of main and the branch, 3 of this PR):

| Edit | Source | main | branch | this PR | longest pause, branch → PR |
|---|---|---|---|---|---|
| P2_h264_720p (2 segments) | H.264 720p | 19.0-19.6 | 16.8-17.1 | 4.4-6.8 | 0.3 s → 0.1 s |
| P2F_h264_720p (2 segments, 2 s fades) | H.264 720p | 7.5-8.7 | 7.8-8.2 | 3.3-3.4 | 0.2 s → 0.1 s |
| P3_h264_720p (3 segments) | H.264 720p | 5.4-6.7 | 6.0-6.4 | 2.5-3.5 | 0.3 s → 0.2 s |
| P3F_h264_720p (3 segments, 2 s fades) | H.264 720p | 9.6-9.7 | 8.5-9.4 | 2.8-3.2 | 0.4 s → 0.2 s |
| RH7_h264_720p (2 segments, fades on the first) | H.264 720p | 8.8-11.4 | 9.1-9.4 | 1.7-1.8 | 0.3 s → 0.1 s |
| R2_real_nofade (2 segments) | H.264 360p | 11.2-13.7 | 9.4-9.5 | 2.9-3.1 | 0.2 s → 0.2 s |
| J1_real_6seg (6 segments) | H.264 720p | 6.7-7.1 | 6.2-7.1 | 2.3-2.5 | 0.3 s → 0.2 s |
| J2_fb_6seg (6 segments) | Facebook H.264 360p, HE-AAC | 6.9-7.0 | 5.7-5.8 | 5.2-6.2 | 0.2 s → 0.2 s |
| X_fb_sd_h264 (2 short segments, fades on the first) | Facebook H.264 360p, HE-AAC | 6.6-7.0 | 6.6-7.7 | 3.5-4.3 | 0.2 s → 0.2 s |
| FB2_h264 (2 segments) | Facebook H.264 360p, HE-AAC | 6.8-7.2 | 3.8-4.0 | 2.7-5.9 | 0.2 s → 0.2 s |
| FB2F_h264 (2 segments, 2 s fades) | Facebook H.264 360p, HE-AAC | 5.5-5.7 | 4.5-5.1 | 3.9-5.4 | 0.1 s → 0.1-0.2 s |
| FB3_h264 (3 segments) | Facebook H.264 360p, HE-AAC | 5.5-7.2 | 6.1-6.6 | 2.1-5.2 | 0.2 s → 0.2 s |
| P2_h264_1080p (2 segments) | H.264 1080p | 15.4-16.0 | 13.9-14.8 | 4.3-5.7 | 0.6 s → 0.2 s |
| RH10_h264_1080p (2 segments, fades on the first) | H.264 1080p | 7.0-7.3 | 7.2-7.4 | 2.6-3.3 | 0.6 s → 0.2 s |
| RV_vp9_real (2 segments, fades on the first) | VP9 360p | 2.3-3.7 | 4.3-4.9 | 3.2-4.3 | 0.2 s → 0.2 s |
| RV7_vp9_720p (2 segments, fades on the first) | VP9 720p | 2.8-3.6 | 4.2-5.4 | 3.7-7.7 | 0.5-0.8 s → 0.5 s |
| P2_vp9_720p (2 segments) | VP9 720p | 4.9-5.8 | 5.1-8.8 | 8.1-8.8 | 0.5 s → 0.6 s |
| FB_vp9_heaac (2 segments, fades on the first) | Facebook VP9 360x360, HE-AAC | 3.8-4.0 | 3.8-4.4 | 3.8-6.1 | 0.1 s → 0.1 s |
| RA_av1_real (2 segments, fades on the first) | AV1 360p | 16.9-17.0 | 17.9-20.6 | 10.1-10.2 | 1.2-1.7 s → 0.8 s |
| RA7_av1_720p (2 segments, fades on the first) | AV1 720p | 19.1-20.0 | 18.5-18.6 | 3.3-4.6 | 2.1 s → 2.1-2.2 s |
| P2_av1_720p (2 segments) | AV1 720p | 15.1-15.2 | 14.6 | 4.0-5.0 | 2.1 s → 2.0-2.2 s |
| E5_hevc_real (2 segments, fades on the first) | HEVC 360p | 13.4-14.2 | 10.3-10.8 | 5.6-5.8 | 0.5 s → 0.3 s |
| E6_hevc_720p (2 segments, fades on the first) | HEVC 720p | 11.9-14.1 | 11.5-11.6 | 4.0-4.4 | 1.7 s → 0.4 s |
| RE10_hevc_1080p (2 segments, fades on the first) | HEVC 1080p | 13.5 | 12.3-12.4 | 3.6 | 3.5-3.8 s → 0.9 s |
| P2_hevc_360p (2 segments) | HEVC 360p | 18.2-19.5 | 11.7-13.5 | 6.3-6.8 | 0.5 s → 0.2 s |
| P2_hevc_720p (2 segments) | HEVC 720p | 16.0-16.5 | 14.3-14.5 | 5.3-5.4 | 1.7 s → 0.4 s |
| Mean over the 26 edits | | 10.1-10.7 | 9.5-9.7 | 4.4-4.7 | |

- P2_h264_720p's 6.8 % was a run that took 10 % longer than the others (6.3 s against 5.7 s); the others gave 4.4-4.5 %.
- The longest pause also drops for H.264 at 1080p (0.6 → 0.2 s) and HEVC (720p 1.7 → 0.4 s, 1080p 3.5-3.8 → 0.9 s).
  AV1 still pauses 2-4 s at the start of each re-encoded run: known and accepted (decided), see "AV1's pause" below.
- Not better with this PR:
  - VP9, at parity with the branch: smartcut caps its VP9 re-encodes at the source's bitrate, and how libvpx's time
    relates to the source's bytes differs from source to source (see "VP9 and its bitrate cap" below). P2_vp9_720p's first run took 69 ms
    per frame, its second 56.
  - main is more even on VP9 because it copies the audio, so re-encoding is nearly all the work.
  - Facebook H.264 (360p, HE-AAC) varies more between runs (FB2 2.7-5.9 %). HE-AAC audio costs ~0.032 s per second
    against AAC-LC's 0.027, but modelling that changed FB2 by 0.4 points (replayed).
- Also measured, not adopted:
  - the decoding of frames before a cut (from the GOP's keyframe, up to a whole 5 s GOP for one frame in FB2);
  - encoder start-up per run (x265: ~1 s at 720p).
  - Fitted, they made H.264 no more even (replayed 3.8 → 4.0 %). HEVC improved on average (5.9 → 4.9 %) but not in
    every case.

**Evenness measured two ways** (decided, Oct 19: both are reported, changes are judged from the cut's real start).
The table above measures from each edit's first update, as `progress_timeline.py` always did. The first update comes
only once the first re-encoded frames are out, so the wait before it is left out: with AV1 that is the first run's
keyframe, 2-4 s. `progress_timeline.py` now also measures from the cut's real start (its first piece; before it the
input is opened and indexed, which the estimate can't see): `real off %`. The calibration's adoption rule is unchanged
(it compares two sets of costs on the same timings). Remeasured, one core, 2 interleaved runs of main and this PR (at
20675f6: its code is 6e1d6dc's), under other agents' load:

| Edit | main, real start | main, first update | this PR, real start | this PR, first update | longest pause (from the first update), main → PR |
|---|---|---|---|---|---|
| P2_h264_720p | 17.3-18.0 | 18.8-19.4 | 3.5-11.2 | 4.8-9.5 | 0.3-0.5 s → 0.2-0.3 s |
| P2F_h264_720p | 8.8-9.5 | 9.7-10.5 | 3.9-8.9 | 3.7-6.1 | 0.3-0.6 s → 0.2-0.5 s |
| P3_h264_720p | 5.1 | 4.2-6.3 | 2.3-4.6 | 1.6-4.1 | 0.3-0.7 s → 0.2-0.3 s |
| P3F_h264_720p | 8.2-12.4 | 8.8-12.9 | 2.1-2.8 | 2.5-3.1 | 0.4-0.7 s → 0.2 s |
| RH7_h264_720p | 7.2-7.5 | 8.2-8.5 | 2.1-3.4 | 1.6-4.0 | 0.3-0.4 s → 0.2 s |
| R2_real_nofade | 12.5-12.7 | 13.3-13.5 | 2.3-3.6 | 2.9-3.3 | 0.1 s → 0.2 s |
| J1_real_6seg | 6.0-6.2 | 6.8-7.0 | 2.1-7.0 | 2.7-7.1 | 0.3-0.4 s → 0.3-0.4 s |
| J2_fb_6seg | 5.5-7.2 | 5.9-7.6 | 3.1-5.6 | 3.4-5.9 | 0.1 s → 0.3 s |
| X_fb_sd_h264 | 7.2-7.7 | 6.4-6.9 | 3.2-4.6 | 4.3-4.7 | 0.2-0.3 s → 0.3 s |
| FB2_h264 | 5.4-6.2 | 6.8-7.7 | 5.4-6.2 | 4.8-5.6 | 0.1-0.2 s → 0.2 s |
| FB2F_h264 | 4.4-7.9 | 4.1-7.6 | 4.1 | 3.9-4.0 | 0.2 s → 0.2 s |
| FB3_h264 | 6.7-6.9 | 7.2-7.4 | 2.6-6.6 | 2.9-6.7 | 0.1 s → 0.3-0.5 s |
| P2_h264_1080p | 14.3-15.2 | 16.5-17.3 | 2.6-4.6 | 4.3-5.2 | 0.6-0.7 s → 0.2-0.3 s |
| RH10_h264_1080p | 6.5-8.4 | 6.0-8.0 | 4.1-5.0 | 3.1-4.0 | 0.7 s → 0.3-0.5 s |
| RV_vp9_real | 2.8-3.0 | 3.5-3.7 | 3.0-3.4 | 3.8-4.1 | 0.2 s → 0.2-0.3 s |
| RV7_vp9_720p | 2.2-4.1 | 2.7-4.9 | 3.4 | 3.4-3.9 | 0.5-0.7 s → 0.6-0.7 s |
| P2_vp9_720p | 4.9-5.2 | 4.2-4.4 | 7.4-8.0 | 6.8-7.3 | 0.5-1.1 s → 0.7-0.9 s |
| FB_vp9_heaac | 2.8-5.9 | 2.9-6.5 | 2.6-3.6 | 2.3-4.3 | 0.2-0.3 s → 0.2 s |
| RA_av1_real | 11.3-15.2 | 13.2-17.6 | 10.8-11.4 | 9.9-10.3 | 1.4-3.3 s → 1.0-1.1 s |
| RA7_av1_720p | 11.0-19.5 | 12.8-20.5 | 3.6-4.3 | 4.6-5.3 | 2.2-5.0 s → 2.4-2.8 s |
| P2_av1_720p | 10.2-14.7 | 14.9-18.7 | 7.3-9.0 | 4.9-5.4 | 2.1-2.8 s → 2.4-2.8 s |
| E5_hevc_real | 9.6-10.6 | 11.2-12.1 | 5.3-10.5 | 5.5-9.1 | 0.9 s → 0.3 s |
| E6_hevc_720p | 10.4-10.6 | 11.8-12.0 | 4.5-8.7 | 3.1-9.5 | 2.5-2.6 s → 0.6 s |
| RE10_hevc_1080p | 10.4-13.4 | 12.6-15.3 | 3.8-8.2 | 3.9-6.7 | 5.6-5.9 s → 1.1 s |
| P2_hevc_360p | 12.7-14.9 | 15.1-16.9 | 6.3-7.2 | 6.7-7.6 | 0.7-0.8 s → 0.3 s |
| P2_hevc_720p | 14.3-16.7 | 16.1-18.4 | 5.0-5.2 | 5.6-5.8 | 1.9-4.4 s → 0.5-0.6 s |
| Mean over the 26 edits | 9.3 | 10.3 | 5.1 | 5.0 | |

- Means per codec, from the real start / from the first update, main → this PR: H.264 (14 edits) 8.8 / 9.4 →
  4.3 / 4.3 %; VP9 (4) 3.9 / 4.1 → 4.3 / 4.5 %; AV1 (3) 13.7 / 16.3 → 7.7 / 6.7 %; HEVC (5) 12.4 / 14.2 → 6.5 / 6.3 %.
  Longest pause: H.264 0.7 → 0.5 s, VP9 1.1 → 0.9 s, AV1 5.0 → 2.8 s, HEVC 5.9 → 1.1 s.
- The two measures differ by more than the noise only for AV1: this PR looks 1 point more even from the first update
  (6.7 %) than from the real start (7.7 %), P2_av1_720p 4.9-5.4 against 7.3-9.0 %. On main it was the other way
  round (16.3 from the first update, 13.7 % from the real start).
- Under load the same edit varied up to 3x between the two runs (P2_h264_720p 3.5-11.2 %); the idle-PC table above
  compares main and this PR better, this one shows how the measures differ.

**AV1's pause at each run's keyframe: known and accepted** (decided, Oct 19; documented in `smart_cut.py`).
- libaom outputs nothing until its lookahead is full (10 frames from 720p up, 35 below), then encodes the run's
  keyframe in one `encode()` call. One desktop core, idle: 2.1-2.4 s at 720p, 3.8-4.2 s at 1080p; 0.5-0.7 s at 360p,
  then up to 0.9 s in the next call. That is the work of 9-23 average frames (median 14); after a fade-in from black
  0.3 s (720p) and 0.65 s (1080p). x264 and libvpx encode theirs in 0.05-0.14 s at 720p.
- It is the keyframe's own intra encode: turning off libaom's keyframe filtering or its TPL model saved 0.1-0.6 s of
  2.9 s. (With the source's bitrate as the cap, as smartcut sets it; without the cap the call took 1.3 s.)
- A cut to another scene inside a run (two segments' re-encoded pieces back to back) makes libaom insert a keyframe
  there too: 1.2 s at 720p, 2.1 s at 1080p.
- Progress only moves between encoder calls, so the bar stands still for that long, and as the keyframe counts as one
  frame it then runs behind for a while.
- Measured, not adopted (the maintainer: no AV1-specific accounting):
  - a keyframe cost per encoder run, fitted from the timing (14 frames), with the feed share below: AV1 replayed
    8.7 → 7.1 % on the calibration clip, 12.4 → 9.1 % on other footage (9.5 % without the feed share), a fresh
    P2_av1_720p 7.8 → 3.7 % (from the real start, one core); the pause stays;
  - a feed share (a frame counts part of its cost when fed: x264 0.10 on one core and 0.26 on four, libaom 0.08,
    libvpx 0.04, measured from the lookahead's fill): H.264 4.4 → 4.0 % on one core and 5.0 → 3.7 % on four, where
    x264's frame threads delay its output;
  - a thread that moves the bar on during long encoder calls (PyAV releases the GIL in `encode()`), at the pace so far
    and never past the work in flight: AV1's longest pause 2.9 → 0.6 s on one core, 1.3 → 0.4 s on four (simulated on
    the recordings, with the keyframe cost).

**VP9 and its bitrate cap** (decided, Oct 19: no model change; documented in `progress_timeline.fit_recode_cost`).
- smartcut re-encodes VP9 at crf 22 with the source's average bitrate as the cap (constrained quality). On the
  calibration clip the cap binds on busy scenes and raises the quantizer, so libvpx gets faster where the source is
  busier: 720p, the same 120 frames, one idle core, a static scene 73-88 ms per frame and a busy one 63-67 ms, both
  at the cap's 2.2 kB per frame; without the cap about the same (78 / 76 and 94 / 93 ms, at 2.4 and 4.9 kB). That is
  P2_vp9_720p's 8-10 %.
- How libvpx's time relates to the source's bytes differs by source: on the Facebook VP9 clip the cap binds too
  (1.2-1.6 kB per frame, 1.5-4.5 kB without it), but its scenes of busyness 0.35-1.28 take 14-20 ms per frame with no
  trend.
- So a negative `busy` (the cost falling with busyness) fitted on one source didn't hold on another. Replayed, one
  core, from the real start, each footage left out of the fit in turn:

| Fitted on | signed `busy` | Replayed on | fitted with `busy` >= 0 (as now) | signed |
|---|---|---|---|---|
| the two YouTube clips | -0.15 | Facebook VP9, 18 edits | 5.7 % | 7.1 % |
| Facebook + one YouTube clip | -0.05 | the other YouTube clip, 6-7 edits | 6.1 % | 5.8 % (either way round) |
| the calibration clip alone | -0.10 | its own 6 edits / a fresh P2_vp9_720p | 5.6 / 10.1 % | 5.2 / 8.4 % |
| the calibration clip alone | -0.10 | the other YouTube clip, 6 edits | 6.0 % | 5.6 % |
| the calibration clip alone | -0.10 | Facebook VP9, 18 edits | 4.8 % | 5.7 % (16 of 18 worse) |

- The calibration's VP9 clips are all made from its one YouTube clip, and its Facebook clip is H.264, so its
  held-out check can't see this (noted in the regression README).

**HEVC cost on a true single core.** The previous round found that x265's measured one-core cost (0.0 + 0.08 s/MP)
made progress 0.2-3 points less even than the old 4.8-core cost, so the old cost was kept. Two causes:
- The "1-thread pool" per-frame costs were mostly the close stalls above.
- With one cost per frame, the bar's error comes from how the cost is spread over the frames (content, fades), not
  from its level. A lower video cost only shrinks re-encoding's share of the work, and with it the error. Replayed
  over the five HEVC edits, without stalls:
  - old 0.005 + 0.020: 12.2 %;
  - accurate single cost (0.075 s/MP, fitted): 14.0 %;
  - this model at the measured level: 5.7 %, at half of it 5.7 %, at twice 6.6 %.
- The measured cost is now both accurate and the most even (real runs: 3.6-6.8 %).

Re-encoded audio tracks are counted per piece, for each output file.

### A machine's own progress costs (`progress_costs.json`, `calibrate_progress.py`)

The costs above are one desktop core's. On another machine their ratios change. Measured on 4 cores of this PC
(`--cores 4`), the encoders took 2-3x less per frame, while copying and audio didn't change:

| Per frame, 720p, average busyness | 1 core (built-in) | 4 cores, measured |
|---|---|---|
| x264 | 24.6 ms | 9.0 ms |
| libvpx (VP9) | 53 ms | 28 ms |
| libaom (AV1) | 129 ms | 77 ms |
| A copied frame / a second of re-encoded audio | 35 µs / 0.027 s | 33 µs / 0.027 s |

With the built-in costs on 4 cores, progress was 6.9 % off an even pace; with the measured ones 4.5 % (paired real
runs, below).

**The file.**
- smartcut reads it from `SMARTCUT_PROGRESS_COSTS`, else `~/.config/smartcut/progress_costs.json` (Linux, or
  `$XDG_CONFIG_HOME`) or `%APPDATA%\smartcut\progress_costs.json` (Windows). It lives outside the smartcut folder,
  which deploys overwrite.
- It is read once per process at its first edit, and the costs in use are logged then (logger `smartcut`, INFO), e.g.
  `smartcut progress costs: /opt/gradio_apps/progress_costs.json (measured: date ..., host ..., cores 1)`.
- Validated: format and cost-model versions, four numbers per codec within wide ranges, the copy and audio costs. A
  malformed, other-version or out-of-range file logs a WARNING and the built-in costs are used. An edit never fails
  because of it.
- Codecs the file doesn't list keep their built-in costs.
- With the built-in costs, the progress values an edit emits are identical to before, value for value (H.264 with
  fades, Facebook HE-AAC, VP9, HEVC; one core).

**The harness** (`smartcut/tests/regression/calibrate_progress.py`; flow and options in its docstring and the
regression README):
1. The test clips, once, on the dev PC: `--make-media <dir> <clip.mp4> --facebook <fb.mp4>` (without the Facebook
   clip a calibration adopts nothing, decided).
   - From a real H.264 + AAC clip's first 90 s: H.264, VP9, AV1 (optionally HEVC) at 360/720/1080p, with
     make_test_media's `--real` settings and no upscaling. HEVC without x265 options; the Facebook clip copied.
   - 6 min on this 12-thread PC (~20 min on one core), 74 MB.
   - A `manifest.json` records the recipe and every clip's size, SHA-256, codec and size. The calibration refuses
     missing, changed or differently made clips.
2. The calibration, on the target machine: `--media <dir>`.
   - `--cores N` simulates N cores: one CPU per physical core, from the CPU topology (P-cores first on hybrid CPUs).
   - `--quick` (~2 min) is a smoke check that never adopts.
   - 22 edits: P2 and P2F for each codec at 360/720/1080p, P3/P3F at 720p, FB2/FB2F.
   - Then real runs of the 7 app edits with the costs in use and the new ones.
   - The edits record their progress calls, so they can be replayed under other costs through `_WorkProgress`.
3. The checks:
   - CPU time ÷ wall time per encoder run and per real run, against the cores expected. Above it, the pinning didn't
     hold. On one core, below 0.85: other load (a loud warning).
   - The fit per codec, the copy and the audio costs. The edits run in a process that releases memory after each
     one, as the app does (`gc.collect()`, glibc's `malloc_trim(0)`), so a 957 MB server doesn't swap during later
     edits; the memory left is printed.
   - **Held-out check:** costs fitted without the held-out edits (every edit at 720p, and the Facebook clip's: 10 of
     22), so on the 360p and 1080p ones, replayed on the held-out edits' calibration runs.
   - **Real runs:** fresh runs of the 7 app edits, each replayed under the costs in use and the new ones (fitted on
     all the edits). Same timings for both: the same short edit varies 2.4-6.7 % from one live run to the next.
4. Adopted (the costs fitted on all the edits) only if both checks are more even by at least 0.5 points, no single
   edit is worse by more than 1.5 points in either check (decided, second review), H.264, VP9 and AV1 were all
   calibrated (second review: `--codecs h264` wrote VP9 and AV1 costs nothing had checked), and not `--quick`, the
   pinning held, the CPU check didn't warn, the media has the Facebook clip (decided), and a held-out check could be
   made (two resolutions). Reported either way, with every reason that applies. The previous file is kept as
   `progress_costs.previous.json`; a rejected candidate goes to `progress_costs.candidate.json`.

**The adoption rule** (decided: the reviewer showed that the earlier rule's replay of the calibration runs favoured
the costs fitted on them). On one core it improved in-sample (5.46 → 5.33 % over the 22 edits) while fresh runs got
less even (4.74 → 5.05 %). So both checks are now out of sample, and need a margin.

The margin, 0.5 points, from the noise measured:
- Each app edit of the two calibrations ran twice: its calibration run and a fresh real run.
- The paired difference (new minus in use) of one edit varied between its two runs by sigma 0.26-0.48 points.
- The mean over a check's 7-10 edits varied by sigma 0.08-0.18 points (over the 7 app edits 0.10-0.18, over the 10
  held-out ones 0.08-0.15). The largest swing of one edit was 1.55 points.
- 0.5 is about 3 sigma, and a tenth of a typical 4.5-5 % evenness: a smaller gain isn't visible, and the built-in
  costs are the better tested ones.

The per-edit guard, 1.5 points (decided, second review): a better mean can hide one edit made clearly worse. The
1-core calibration's P2_h264_720p, the app's most common edit, went 3.68 → 5.38 % in the held-out check (+1.70) and
3.45 → 6.65 % in its last real run (+3.20; over both recorded runs +2.31). 1.5 is past one edit's noise (sigma
0.26-0.48, at most 1.55).

On this PC (largest distance from an even pace at the updates, mean; app edits P2/P2F/P3/P3F H.264 720p, VP9 and AV1
P2 at 720p, FB2). The verdicts are the new rule's, replayed on the recorded runs (no encoding):

| Run | Fitted per 720p frame (x264 / VP9 / AV1) | Held-out check, in use → new | Real runs, paired | Worst edit, held-out / real (last run) | Verdict | Time |
|---|---|---|---|---|---|---|
| 1 core (`--cores 1`, the built-in costs' setup) | 24.3 / 57.3 / 150.2 ms | 4.69 → 4.76 % | 4.2 → 4.4 % (the last run 4.74 → 5.05) | P2_h264_720p +1.70 / +3.20 | not adopted, correctly: no better, and P2_h264_720p worse in both checks | 14.2 min |
| 4 cores (`--cores 4`) | 9.0 / 27.9 / 77.0 ms | 6.03 → 4.52 % | 6.9 → 4.5 % (the last run 7.00 → 4.52; FB2 12.9 → 3.9) | P3F_h264_720p +0.49 / P2_vp9_720p +0.20 | adopted | 7.0 min |
| `--quick`, 1 core | - | none (one resolution) | 5.74 → 5.70 % | | not adopted (the earlier rule adopted it and wrote the file) | 2 min |

- The verdicts are the same with any margin from 0 to 1.5 points. The 1-core calibration replayed under the current
  rule (`Q:\work\r2fix\replay_rule2.py`): `NOT ADOPTED: not more even by 0.5 points (held-out edits 4.7 -> 4.8 %,
  real runs 4.7 -> 5.0 %); P2_h264_720p worse by 1.70 points in the held-out check (3.7 -> 5.4 %); P2_h264_720p worse
  by 3.20 points in the real runs (3.5 -> 6.7 %)`. The 4-core one: `ADOPTED`.
- **The Facebook clip is required** (decided). The 720p held-out edits are the same scenes as the edits fitted on;
  the Facebook clip is the only other footage. Replayed without its edits, the 4-core calibration's held-out gain fell
  from -1.51 to -0.53 points. Media without it still calibrates and reports, but writes only the candidate. Replayed,
  the 4-core calibration without its Facebook edits says: `NOT ADOPTED: no Facebook clip: no held-out footage (make
  the media with --facebook). <file> unchanged; the candidate is in <candidate>`.
- A `--quick` run with `SMARTCUT_PROGRESS_COSTS` naming a file (under the other agent's test load) printed `NOT
  ADOPTED: --quick is a smoke check, not a calibration; the CPU check warned of other load; no held-out check: ...`
  and left the file byte-identical (SHA-256).
- Cores used, CPU / wall per encoder run (median): one core 0.99 for all three codecs; four cores x264 3.58, VP9 2.79,
  AV1 2.77.
- Median error of a run's fitted time: x264 7.5-8.4 %, VP9 2.1-9.7 %, AV1 10.5-13.1 %.

### Bitrate of re-encoded audio (`track_cutters._recode_audio_bitrate`)

| Source | Before | Now |
|---|---|---|
| AAC 256k in MKV (no reported rate) | 128k (floor) | 256k (from its packets) |
| Opus 192k in WebM (no reported rate) | 128k | its average from packets |
| Opus 64k in WebM | 128k | 128k (floor) |
| AAC 5.1 at 192k | 192k | 384k (64k × 6) |
| AAC mono at 48k / HE-AAC stereo at 48k | 128k / 128k | 128k / 128k (mono counted as two) |
| AC-3 640k into .webm (Opus) | 640k: libopus refuses, cut fails | 512k (libopus's limit) |
| PCM 1536k into .webm | 1536k: libopus refuses, cut fails | 128k |
| MP3 128k / 320k | 256k / 320k | 256k / 320k |
| AC-3 5.1 at 448k | 448k | 448k |

Mono counted as two, because a 64 kb/s mono music source re-encoded at 64k measured worse than at 128k:
- inside the segments (99.8 % of 5 ms windows): x0.64-1.20, against x0.64-1.04;
- at the joins: x0.90-1.10, against x0.92-1.02.

Encoder limits are taken from the encoder itself: it is opened with the rate, and if it refuses, the highest
accepted rate below that is used (1.4 ms per open). aac/ac3 lower too-high rates themselves, and MP3 is capped at 320k.

### Cost of typical app edits (no fades, one core, minimum of 3 runs)

| Source (audio) | Segments (audio kept) | Time | Peak memory over the process before | Output size |
|---|---|---|---|---|
| real_h264_720p (AAC-LC 44.1 kHz) | 2 (30 s) | 5.21 → 5.71 s | +120 → +128 MB | 3.24 → 3.26 MB |
| real_h264_720p | 3 (41.8 s) | 8.13 → 9.20 s | +122 → +132 MB | 4.54 → 4.56 MB |
| fb_sd_h264 (HE-AAC 48 kHz) | 2 (30 s) | 3.00 → 3.89 s | +45 → +61 MB | 2.62 → 2.91 MB |
| fb_sd_h264 | 3 (41.8 s) | 6.58 → 7.92 s | +46 → +61 MB | 3.42 → 3.83 MB |

That's 0.5-1.3 s more per edit, 0.017-0.032 s per second of kept audio. HE-AAC outputs grow by ~11 %: they become
AAC-LC at 128 kbps.

## Test results

Second review round, `audio-exact` at 9fe4424 on 1b6c1d8 (Oct 20; another agent's runs on the same PC, so no timings
are compared):

| Suite | Result |
|---|---|
| smartcut unit scripts (20) | all pass: test_fade, test_fade_keywords, test_api 6/6, test_validation 5/5, test_audio_parameters, verify_audio_fade 3/3, test_cancel 9/9, test_yuv_fade 3/3, test_last_frame 31/31, test_multitrack 3/3, test_webm 7/7, test_audio_bitrate 15/15, test_progress_costs 8/8 (+1), test_work_progress 7/7, test_x265_options 5/5, test_calibrate_progress 16/16 (+3), test_audio_faithful 22/22 (+3; DTS / TrueHD / 32-bit FLAC with `SMARTCUT_TEST_FFMPEG`), test_audio_decode 8/8 (+3), test_matroska_source 34/34, test_audio_joins 37/37 (+2; with `Q:\work\media`) |
| The new checks with each fix undone (`Q:\work\r2fix\mutate.py`) | all 16 caught (the drain one also with the process on one CPU): the old overlap code; the reviewers' surviving mutants (encoder tag kept, no resampler flush, no drain: also with the process on one CPU); warned once per track; track_cutters before the MPEG-layer fix; KEYFRAMES copying lossless tracks; the reviewer's `heldfit_in_sample` and `real_check_live`; the guard not given to / not applied by `decide()`; the codecs not calibrated not given / ignored; the work folder left; no replace retry; no read retry |
| `regression/run_cases.py`, all media (62 cases) | 61 PASS, 1 XFAIL (`X_vp9_10bit`), the same as 1b6c1d8. Every output identical to 1b6c1d8's: the 48 MP4 byte for byte, the 13 Matroska / WebM by packets. No intended change reaches them (no overlaps, MP2, KEYFRAMES or lossless joins among the cases) |
| Reviewers' repros | `overlap_mock.py` +21.3 → +64.0 ms (the timestamps' own); MP2 in MP4 with joins: Layer III → Layer II copied, warned; `kf_lossless.py` FLAC / ALAC / PCM within 0.3 ms at every join; `atomic_open.py` 160 / 282 → 0 / 0 (a file held open throughout still can't be replaced: raised after the retries) |
| Calibration replays (`Q:\work\r2fix\replay_rule2.py`, no encoding) | 1 core not adopted (the margin; P2_h264_720p +1.70 / +3.20), 4 cores adopted (worst +0.49 / +0.20) |
| pyright (changed files and core modules) / ruff (changed files) | 0 errors / the same 2 findings as 1b6c1d8 (PLR0917 in `smart_cut` and `PassthruAudioCutter`) |
| Not run | the upstream suite and the app's end-to-end tests (the app repo is another agent's; its MP4 path is byte-identical) |

Progress round for AV1 and VP9, branch `progress-av1-vp9` at 20675f6 on 6e1d6dc (Oct 19). The library's code is
unchanged (`smart_cut.py` and `progress_costs.py`: comments only, their ASTs identical to 6e1d6dc's); the changes are
`progress_timeline.py`'s second evenness measure, a test and docs. Other agents' runs were on the same PC: the tests
ran on CPUs 4-11, the timed comparison alone on CPU 0. `python -m smartcut` (the CLI tests) imported the main clone,
whose code is the same.

| Suite | Result |
|---|---|
| smartcut unit scripts (19) | all pass: test_fade, test_fade_keywords, test_api 6/6, test_validation 5/5, test_audio_parameters, verify_audio_fade 3/3, test_cancel 9/9, test_yuv_fade 3/3, test_last_frame 31/31, test_multitrack 3/3, test_webm 7/7, test_audio_bitrate 15/15, test_progress_costs 7/7, test_work_progress 7/7, test_x265_options 5/5, test_calibrate_progress 13/13 (+1: evenness from the cut's real start), test_audio_faithful 19/19, test_audio_decode 5/5, test_audio_joins 35/35 (with `Q:\work\media`) |
| The new test on the previous `progress_timeline.py` | fails ("progress_timeline can't measure from the cut's real start") |
| `regression/run_cases.py`, a sample of 23 cases (T1, T2, T6, T8, R1, R2, H1, M1, M5, W1, W5, V2, A2, RV, RA, RH7, FB_vp9_heaac, X_vp9_10bit, X_fb_sd_h264, J1, J4, E1, E5) | 22 PASS, 1 XFAIL (`X_vp9_10bit`) |
| ruff, changed files | the same findings as 6e1d6dc's |
| Not run | the upstream suite and the app's end-to-end tests: no library code changed |

Matroska round, branch `mkv-first-frame` at 487ed15 on 6e1d6dc (Oct 19). Other agents' runs were on the same PC,
so no timings are compared. The "before" runs used 6e1d6dc exported with `git archive`.

| Suite | Result |
|---|---|
| smartcut unit scripts (20) | all pass: test_fade, test_fade_keywords, test_api 6/6, test_validation 5/5, test_audio_parameters, verify_audio_fade 3/3, test_cancel 9/9, test_yuv_fade 3/3, test_last_frame 31/31, test_multitrack 3/3, test_webm 7/7, test_audio_bitrate 15/15, test_progress_costs 7/7, test_work_progress 7/7, test_x265_options 5/5, test_calibrate_progress 12/12, test_audio_faithful 19/19, test_audio_decode 5/5, test_audio_joins 35/35 (with `Q:\work\media`), test_matroska_source 34/34 (new) |
| test_matroska_source on 6e1d6dc | 26/34: the 8 HEVC cuts from the first GOP into the next fail (`process_segments` returned False) |
| `regression/run_cases.py`, all media (62 cases: the 55 + `K1`-`K7`) | 61 PASS, 1 XFAIL (`X_vp9_10bit`). On 6e1d6dc (the same harness): `K3` / `K4` FAIL (returned False), the rest the same |
| Outputs against 6e1d6dc's | the 46 MP4 outputs of the 55 earlier cases byte-identical (47 with `K2`); the 12 Matroska / WebM outputs identical by packets; only `K3` / `K4` differ (they failed before) |
| Matrix of 300 cuts (see Matroska / WebM sources) | 300 right; on 6e1d6dc 292, and those 292 outputs identical |
| Upstream `smartcut_tests.py`, the same 8 categories | 34 passed, 16 failed: the same 16 (downloads) |
| pyright, core modules + the new test / ruff, changed files | 0 errors / no new findings (the same as 6e1d6dc's) |
| App end to end | not run: the app imports the main clone. Its MP4 path is byte-identical (above) |

Audio review round, branch at aa97285 (Oct 19; another agent's runs on the same PC, so no timings are compared):

| Suite | Result |
|---|---|
| smartcut unit scripts (17) | all pass: test_fade, test_fade_keywords, test_api 6/6, test_validation 5/5, test_audio_parameters, verify_audio_fade 3/3, test_cancel 8/8, test_yuv_fade 3/3, test_last_frame 31/31, test_multitrack 3/3, test_webm 7/7, test_audio_joins 35/35 (with `Q:\work\media`; was 29), test_audio_bitrate 15/15, test_progress_costs 6/6, test_calibrate_progress 4/4, test_audio_faithful 19/19 (new; DTS / TrueHD / 32-bit FLAC with `SMARTCUT_TEST_FFMPEG`), test_audio_decode 5/5 (new) |
| The new checks on the previous code (c022263, or the commit before each fix) | test_audio_faithful 12 of 18 fail; test_audio_decode 5 of 5; test_audio_joins: the lead-in (9 cases), the movie timescale (9), the Matroska / MP3 / Opus copies and KEYFRAMES (5), KEYFRAMES MP3 at its join (level 0.00); the copy-end check with `finish()`'s truncation removed (8) |
| `regression/run_cases.py`, all media (55 cases) | 54 PASS, 1 XFAIL (`X_vp9_10bit`), the same verdicts as c022263. Sync: J1/J2/J5 0.05-0.12 → 0.30-0.37 ms (movie timescale), J3 (now a copy) 0.55 → 0.30 ms |
| Upstream `smartcut_tests.py`, the same 8 categories | 34 passed, 16 failed: the same 16 (downloads) |
| App end to end: `tests/app_smart_edit.py Q:\work\media` | 83/83 (the app's test file has 83 checks since 15:23) |
| pyright, core modules / ruff, changed files | 0 errors / no new findings (42 before and after; the new tests clean) |

Branch at c022263 (Oct 19, idle PC):

| Suite | branch | main |
|---|---|---|
| smartcut unit scripts (15) | all pass: test_fade 6/6, test_fade_keywords 6/6, test_api 6/6, test_validation 5/5, test_audio_parameters 4/4, verify_audio_fade 3/3, test_cancel 8/8, test_yuv_fade 3/3, test_last_frame 31/31, test_multitrack 3/3, test_webm 7/7, test_audio_joins 29/29 (with `Q:\work\media`), test_audio_bitrate 15/15, test_progress_costs 6/6 (new), test_calibrate_progress 4/4 (new; needs the ffmpeg command) | test_webm on main fails the 2 new cases (PCM → Opus; Opus mid-stream re-encode) |
| `regression/run_cases.py`, all media (55 cases, this branch's harness) | 54 PASS, 1 XFAIL (`X_vp9_10bit`); outputs unchanged by the progress commits: the 46 MP4 outputs byte-identical to d396a8d's (at 210bfe4), the 8 Matroska/WebM ones with identical packets (their segment UID is random) | 50 PASS, 4 FAIL (A/V sync 15.3-27.1 ms: R2, J1, J2, J4), 1 XFAIL. Main's own harness: 43 PASS, 1 XFAIL of 44 |
| Upstream `smartcut_tests.py`, categories basic, h264, h265, codecs, containers, audio, mixed, transforms | 34 passed, 16 failed (the same tests as in the earlier rounds) | 33 passed, 17 failed |
| App end to end: `tests/app_smart_edit.py Q:\work\media` | 73/73 | - |
| pyright, core modules | 0 errors | 8 |
| ruff, changed files | no new findings | - |

Progress and calibration fixes (branch `audio-exact-progress` on c022263, Oct 19). They ran under another agent's
test load, so no timing conclusions are drawn from them. The CLI tests ran with an interpreter whose
`python -m smartcut` imports the branch.

| Suite | Result |
|---|---|
| smartcut unit scripts (17) | all pass: test_fade 6/6, test_fade_keywords 6/6, test_api 6/6, test_validation 5/5, test_audio_parameters 4/4, verify_audio_fade 3/3, test_cancel 9/9 (+1: a cancel in the last flush), test_yuv_fade 3/3, test_last_frame 31/31, test_multitrack 3/3, test_webm 7/7, test_audio_bitrate 15/15, test_progress_costs 7/7 (+1), test_work_progress 7/7 (new), test_x265_options 5/5 (new), test_calibrate_progress 12/12 (+8), test_audio_joins 29/29 |
| Mutation checks of the tests the review called weak, and of the new ones | 40 mutants of the code under test, all caught: the adoption rule and its wiring 15, the costs' loading and use 5 (with the reviewer's mutant), `--cores`' topology 3, `_WorkProgress` 9, the x265 options 7, the last-flush cancel 1 (the old code) |
| `regression/run_cases.py`, a sample of 21 cases (T1, T2, T6, T8, R1, R2, H1, M1, M5, W1, W5, V2, A2, RV, RH7, X_vp9_10bit, X_fb_sd_h264, J1, J4, E1, E5) | 20 PASS, 1 XFAIL; every output identical to c022263's: 17 MP4 byte for byte, 3 Matroska/WebM by packets |
| Upstream `smartcut_tests.py`, the same 8 categories | 34 passed, 16 failed: the same 16 (downloads) |
| `calibrate_progress.py --quick`, `SMARTCUT_PROGRESS_COSTS` naming a file | `NOT ADOPTED`; the file byte-identical |
| pyright, core modules / ruff, changed files | 0 errors / no new findings |

Upstream suite details:
- All 16 branch failures are downloads that can't work on this workstation:
  - HTTP 403 from Google's sample bucket (Tears of Steel);
  - TLS interception by the antivirus for filesamples.com, test-videos.co.uk and the `media-test-data` git clone.
- The same 16 fail on main.
- Main also fails `test_mp3_passthru`. It cuts an MP3 into segments that touch, and main's MP3 rule re-encoded that
  (+695 samples). The branch copies it, because touching segments aren't a join.

## Verification

- PowerPoint (Media Foundation) played an edited sample with the then movie timescale and lead-in correctly (maintainer, 2026-10-19; both since dropped).
- PowerPoint (Media Foundation) plays the app's single-segment edit cleanly: copied AAC with pre-roll, the last sample
  shortened from 1024 to 45 samples, the audio's edit ending at 9.709 s; full level from the start, no sound after the
  last picture (real_h264_720p 30.3-40.009 s; maintainer, 2026-10-19).
- No behaviour change where none was intended, byte for byte: the pyright fixes (fade outputs for H.264, VP9,
  AV1 10-bit, HEVC) and the hybrid CRA memory fix (E1-E6, a 1080p clip, a 720p clip with x265 options).
- The lookahead cap's outputs are byte-identical to the ones measured for it (option injected by the harness).
- A source with 12 B-frames cuts with a lookahead of 13; one with its own lookahead of 8 keeps 8.
- Not verified on the VPS: x265's speed with its 1-thread pool on Linux (see "open"). HEVC isn't offered in the app.
- The flush taken packet by packet gives the same packets: PyAV's `encode()` is `encode_lazy()` collected
  (`av/codec/context.pyx`, PyAV 16). The run_cases verdicts are unchanged (see the tests).

## Decisions taken (maintainer)

1. Re-encode the whole track at joins (any faithful codec); copy otherwise, except lossless tracks starting mid-stream
   (audio review, 2026-10-19: single mid-stream segments into Matroska / WebM and MP3 are copied, as main; so is
   `KEYFRAMES` mode).
2. 3 ms fades at joins, also with fades; timed from the pieces kept at the join (audio review).
3. Re-encoded tracks start cold everywhere (audio review: the 80 ms lead-in in MP4 was dropped).
4. FFmpeg's default movie timescale (audio review: `movie_timescale` = the video's timescale was dropped; MP4 sync
   tests allow 1.5 ms). PowerPoint had played a sample with it correctly (`Q:\fixes\ppt_sample.mp4`).
5. Copy end: the edit list ends at the cut; FFmpeg's whole last frame left as is.
6. Audio bitrate: source's average (from packets when unknown; lossless counts as unknown), floor 64k × max(channels, 2),
   MP3 256k, clamped to the encoder's limits; HE-AAC → 128k AAC-LC and AC-3 at its source rate kept.
7. HEVC lookahead: measure, then cap like the others if it behaves like them. It did (numbers above): capped at 10.
8. Copy cost: re-measure on an idle PC, change it only if confirmed different. 32-34 µs measured: 35 µs kept.
9. pyright: fix the 2 errors left in the fade code without a behaviour change. Done.
10. The source's x265 threading options (`frame-threads` / `numa-pools` / `lookahead-threads`) are dropped from the
    x265 options smartcut reuses (commit e6adedc): kept.
11. Faithful-only re-encodes (audio review): the same codec, sample format (bit depth), layout and rate, and the encoder
    opens; otherwise copy with a warning. Exceptions: HE-AAC → AAC-LC; `.webm` → Opus. Unknown layout → copy (lossy
    codecs; lossless encoders keep such a layout as it is, e.g. PCM in Matroska: my reading, reported).
12. Pre-roll by time, durations of 0, a strict one-tick tolerance, the resampler's delay (audio review P3-6, P3-7).
13. Segment times count from the container start (the earliest first timestamp of any stream), as upstream and
    FFmpeg's `-ss` do, also for Matroska sources whose audio encoder delay puts it before the first video frame
    (Oct 19: kept and documented; not changed to the first video frame).
14. Progress, AV1 (Oct 19): no AV1-specific accounting (keyframe cost per run, feed share) and no thread moving the bar
    during long encoder calls; the pause at each run's keyframe is documented as accepted.
15. Progress, VP9 (Oct 19): no model change; `busy` stays >= 0 (a negative one fitted on one source made Facebook VP9
    less even).
16. Evenness is reported from the first update and from the cut's real start; changes are judged on the latter.
17. KEYFRAMES mode re-encodes lossless tracks at joins too, losslessly (second review, Oct 20): `if keyframes: return
    has_joins and (MP3 or lossless)`, with fades and WebM as before.
18. KEYFRAMES with an audio-only fade-in leaves the kept GOP's audio before the requested start silent, as video fades
    do (second review: kept, documented).
19. The calibration's adoption also needs that no single edit gets worse by more than 1.5 points in either check
    (second review).
20. MPEG audio Layer II re-encodes with FFmpeg's mp2 encoder when that is faithful, else it is copied with a warning;
    never as MP3 (second review).

Small choices:
- Segments that touch are not a join.
- In segment mode the audio-fade check is per file.
- A pre-roll that reaches the stream's first packet counts as exact.
- A copy warning is given once per track and warning, so segment mode names every file's reason (second review).
- MP2 read by a float decoder (MP2 in MP4: mp3float) is copied: `libtwolame` (a Layer II encoder that takes float)
  is in PyAV's FFmpeg but wasn't used, per decision 20 (mp2).
- An audio track whose timestamps jump back by more than a frame: the whole overlap is dropped, as on main (second
  review; this branch dropped only the first frame's overlap, and the audio after it came late).
- `run_cases`' "A-V ms" now compares where the audio and the pictures end.

## Not changed / open

Recommendations (not done):

1. **Calibrate once on the VPS** with `calibrate_progress.py`, using clips made on the dev PC (steps in the handover /
   app CLAUDE.md). It adopts its costs only if they are more even there. Unverified on Linux:
   - the paths (`/root` for the service without `$HOME`);
   - `--cores` (sched_setaffinity, CPU topology);
   - process CPU time over all threads;
   - the VPS's speed (the estimate is for a desktop core).
2. **x265's 1-thread pool on the VPS:** the Windows slowdown was the close stall (Windows scheduling), so
   `pools=none` isn't needed for it. If an HEVC edit on the VPS takes clearly longer than its frames' encoding, look
   at the close there (unverified).
3. **Progress, what is left:**
   - VP9's runs vary ±15 % with the scene where the bitrate cap binds, differently by source (see "VP9 and its
     bitrate cap"); not modelled (decided).
   - AV1 pauses 2-4 s at each run's keyframe: known and accepted (decided). Measured options are in "AV1's pause".
   - Not modelled: the decoding before cuts, and x265's ~1 s start-up per run. Measured as no gain for H.264 and a
     mixed one for HEVC.
   - The calibration's adoption rule still measures from each edit's first update. Both sets of costs are replayed on
     the same timings, so it compares them fairly; switching it to the real start would change its printed numbers
     (not done: no calibration output change).
4. **Check the calibration's memory release on the VPS** (unverified on Linux). The calibration prints "memory after
   each edit, released: A-B MB", which should stay well below the 957 MB. malloc_trim exists only in glibc; with
   another C library only the garbage collector runs.
5. `--cores N` on a hybrid Intel CPU (P-cores with hyperthreads, E-cores) is tested with synthetic topologies only;
   this PC's 6 cores with 2 threads each give the same CPUs as before.

Known limits, documented:
- With AV1 the progress stands still while libaom encodes each re-encoded run's keyframe (2-4 s at 720p-1080p on one
  core; decided: accepted).
- An adopted progress_costs.json gives HEVC the costs in use, scaled by the calibrated codecs' speed, unchecked,
  unless `--codecs` includes `hevc` (the app doesn't offer HEVC; the rule requires H.264, VP9 and AV1 only, as
  decided).
- MP2 read by a float decoder (MP2 in MP4) is copied at joins and fades, with a warning: FFmpeg's mp2 encoder takes
  s16 only (decision 20). PyAV's FFmpeg also has `libtwolame`, which takes float and would make it faithful.
- After a gap in resampled audio (e.g. 44.1 kHz into a WebM's Opus) the audio after it can be placed a sample (21 µs)
  off: the resampler's output length is rounded.
- Windows: a costs file another process keeps open for more than ~1.8 s can't be replaced; the calibration then stops
  with the PermissionError, the file unchanged.
- FFmpeg ignores the MP4 edit list's end and decodes a copy's whole last frame (decided).
- Matroska timestamps are in ms (A/V ±1 ms there).
- Re-encodes start cold in every container (decided).
- Copies into Matroska / WebM, and of MP3, start at the first whole frame after the cut (decided).
- Matroska: after a jump a re-encoded segment is placed from the first decoded packet's 1 ms timestamp, up to 0.5 ms
  off (PCM in 33.3 ms packets: 16 samples; in MOV 0).
- The same content can cut differently from Matroska and from MP4: Matroska keeps the audio encoder's delay in the
  timeline that segment times count from (decision 13). One frame more, and the first frame up to the delay earlier
  (AAC 21 ms, Opus 6.5 ms). The earlier "Found, not fixed" note about it was this; see Matroska / WebM sources.
- run_cases' `aFlat` column is coarse on real audio: 50 ms windows on a fixed grid. Informational only; `sync ms` and
  `join_sync.py` measure properly. Cases cut between frames were also measured up to a frame off, fixed (J5
  0.13/8.14 → 1.00/1.00).
- An output file with no cut pieces raises IndexError (upstream, unchanged).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
