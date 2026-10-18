# Regression harness for cuts and fades

Measurement-based checks that catch what pass/fail unit tests miss: wrong frames at cut points, fades at the
wrong time or with the wrong shape, audio clicks at joins, A/V drift, and speed. Built while merging upstream 2.0
into this fork (Oct 2026); every bug fixed then shows up as a FAIL here on the pre-merge code.

## Quick start (from the repo root)

```bash
# 1. Test media (synthetic clips; --real derives VP9/AV1/720p variants from any H.264 + AAC clip you have)
python smartcut/tests/regression/make_test_media.py /tmp/sc_media --real my_clip.mp4

# 2. Run all cases whose media exists (exit code 1 if any case FAILs)
python smartcut/tests/regression/run_cases.py /tmp/sc_media /tmp/sc_out

# Only some cases, pinned to one CPU core (closer to a 1-CPU server, for timing)
python smartcut/tests/regression/run_cases.py /tmp/sc_media /tmp/sc_out T2_both RA_av1_real --one-core
python smartcut/tests/regression/run_cases.py --list
```

`make_test_media.py` also makes the HEVC clips (`synth_hevc*.mp4`; with `--real`, `real_hevc*.mp4`). For the
HEVC calibration case `RE10_hevc_1080p` in `progress_timeline.py`, transcode a 1080p clip with `--real`'s HEVC settings.

Media the generator can't make (add it to the media dir by hand to enable those cases):
`synth_heaac.mp4` and `synth_2audio_heaac.mp4` (HE-AAC audio; see make_test_media.py for the one-line ffmpeg
muxes), `real_fb_vp9.mp4`
(a Facebook VP9 + HE-AAC download), `fb_sd_h264.mp4` (a Facebook `sd` H.264 download). Real clips are
third-party content, so they are not committed.

## What a case measures

| Column | Meaning | PASS threshold |
|---|---|---|
| `dur/exp` | Output video duration (to the end of its last frame) vs the requested segments | - |
| `stream` | Video stream duration as players see it (in MP4 the edit list; `-` in Matroska) | ends within half a frame of the last frame |
| `frames/exp` | Frames decoded vs the source frames inside the segments. Decoding drops frames outside an MP4 edit list, as FFmpeg and players do (same count as `ffprobe -count_frames`) | equal |
| `gap` | Largest gap between consecutive frame timestamps, in frames | <= 1.5, no backwards steps |
| `same` | Consecutive frames that decode identically (informational: near-black fade frames are often identical) | - |
| `A-V ms` | Where the decoded audio ends minus where the last picture ends (FFmpeg decodes a whole last audio frame, up to 46 ms for HE-AAC) | within 60 ms |
| `sync ms` | Largest A/V offset of a segment, by cross-correlation with the source in the middle of each segment (real audio only; `join_sync.py` measures the joins) | within 1.5 ms |
| `vFadeErr` | Largest difference between measured brightness (vs the matching source frame) and the ideal fade ramp | 0.08 synthetic, 0.15 real |
| `aFadeErr` | Same for audio RMS (vs the matching source audio) | 0.08 synthetic, 0.15 real |
| `aFlat min/max` | Audio gain outside fades (should be ~1.0) | - |
| notes / dips | Audio dips below 0.9 in untouched parts (clicks, gaps, a ramp at the start); tone sources only | none |

The audio columns are the first track's. Every other audio track gets the same checks (`track N:` in the notes
gives its `A-V ms` and `aFadeErr`), and the output must have as many audio tracks as the source.

The `W*` cases write `.webm` (AAC / HE-AAC audio re-encoded to Opus, Opus audio copied when exact). The `J*` cases
have many joins (6 segments: audio re-encoded with 3 ms fades at the joins, in sync) and mid-stream starts into
Matroska (`J3`, `J4` write `.mkv`) and MP4 (`J5`: a copy). The `E*` cases are HEVC: `synth_hevc.mp4` has x265's open
GOPs (CRA keyframes with RASL leading pictures, which smartcut re-encodes after a jump), `synth_hevc_closed.mp4`
closed ones, and `real_hevc*.mp4` carry no x265 options, like camera footage.

Results are also written to `<out>/results.json`, and per-case time series to `<out>/<case>.npz`
(`vt, v_alpha, v_ideal, at, a_gain, a_ideal`). Expected failures report XFAIL, or XPASS if they start working:
`X_vp9_10bit` (smartcut can't re-encode 10/12-bit VP9).

Synthetic clips show a frame counter and timestamp, so stutter, repeated or skipped frames and the frames at a
join are easy to check by eye too.

## Diagnostic tools

| Script | Use |
|---|---|
| `join_sync.py <source> <out dir> [--segments a-b,...] [--ext .mkv] [--measure OUTPUT]` | A/V sync per segment (by sample count, as players play audio, and by pts), level and clicks at every join and the start, missing / overlapping samples, where the streams start and the audio ends. Default: 6 segments of a real clip, e.g. `real_h264_720p.mp4` or `fb_sd_h264.mp4` (needs noise-like audio) |
| `inspect_case.py <out dir> <case> [audio\|video\|gaps]` | Where a case deviates: time ranges with fade errors, or the frame timing gaps |
| `match_frames.py <source> <output> <first> <last>` | Which source frame each output frame is: exact match = copied, otherwise nearest = re-encoded. Shows stray or missing frames at joins |
| `psnr_frames.py <source> <output> <out first> <count> <src first> [<output> ...]` | PSNR of re-encoded frames vs their source frames, e.g. to compare encoder settings |
| `exp_encoder.py <config.json> <media> <out> [cases] [--one-core]` | Run the cases with another encoder / encoder options for re-encoded GOPs, without changing smartcut |
| `bench_audio.py <repo> <video> <out dir>` | Cost of re-encoding the whole audio track (audio fades) and of FFmpeg's AAC coders |
| `progress_timeline.py <media> <out> [cases] [--one-core] [--timeline] [--calibrate]` | How steadily `progress_callback` moves: longest stretch without an update and the largest distance from an even pace. `--calibrate` measures the per-codec costs that `smart_cut.py`'s work estimate uses |
| `audio_coder_ab.py <out dir> <video>` | Listening files, "what was lost" difference files and spectrograms for AAC default vs fast coder |

## Unit tests next to this folder

These need no test media and run like the other test scripts (`python smartcut/tests/test_yuv_fade.py`):
- `smartcut/tests/test_yuv_fade.py`: YUV fade values.
- `smartcut/tests/test_cancel.py`: cancellation, also midway through a re-encoded piece (with and without a
  progress callback; deterministic on any number of cores); steady progress; input file closed.
- `smartcut/tests/test_last_frame.py`: cuts keep their last frame (H.264 with and without B-frames, VP9, AV1,
  fades, MP4 and MKV), counted as FFmpeg decodes them; inputs without audio.
- `smartcut/tests/test_multitrack.py`: every audio track is kept (AAC + AC-3), copied or re-encoded with fades.
- `smartcut/tests/test_webm.py`: `.webm` outputs (AAC re-encoded to Opus, Opus copied).
- `smartcut/tests/test_audio_bitrate.py`: the bitrate of re-encoded audio (source's, floors, encoder limits).
- `smartcut/tests/test_audio_joins.py`: copied or re-encoded as the rule says, A/V sync within 1 ms at the start and
  every join, level and clicks at the joins (AAC, AC-3, MP3, Opus; MP4, MKV, WebM; HE-AAC with a media dir).

## Windows notes

- PyAV fails to load its DLLs ("The filename or extension is too long") when Python lives under a very long
  path. Map the folder to a drive letter first: `subst Q: <long folder>` (remove with `subst Q: /d`).
- With an embeddable Python, `python -m smartcut` (used by the CLI tests) needs the repo on the path: add a
  `.pth` file containing the repo root to its `site-packages`.
