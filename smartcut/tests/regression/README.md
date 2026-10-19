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

`--one-core` (`run_cases.pin_to_one_core()`) also holds x265's threads, which set their own affinity: on Windows
through a job object (x265 then logs "unable to set thread affinity for NUMA node mask", expected). x265 sizes its
pool from all the machine's CPUs, so its re-encodes get a 1-CPU machine's threads (a 1-thread pool, 1 frame thread)
unless their options set them. On Windows the calling thread also runs below normal priority, as x265's pool
threads do: at normal priority, every x265 encoder close stalled 0.5-10 s on one core (x265 waits for its pool with
`Sleep(0)`, which doesn't let a lower-priority thread run), so HEVC edits seemed 2-3x slower and varied 2x from run
to run. Call `pin_to_one_core()` from the thread that cuts. `pin_to_cores(n)` does the same for n cores (one per
physical core, x265 sized for n; the priority only for one core).

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
| `psnr_reencoded.py <source> '<segments json>' <output> [<output> ...]` | PSNR of all of an edit's re-encoded frames against what the encoder was given (faded frames against the source faded by smartcut), copied frames checked bit-exact; several outputs compared frame by frame |
| `exp_encoder.py <config.json> <media> <out> [cases] [--one-core]` | Run the cases with another encoder / encoder options for re-encoded GOPs, without changing smartcut |
| `bench_audio.py <repo> <video> <out dir>` | Cost of re-encoding the whole audio track (audio fades) and of FFmpeg's AAC coders |
| `progress_timeline.py <media> <out> [cases] [--one-core] [--timeline] [--calibrate]` | How steadily `progress_callback` moves: longest stretch without an update and the largest distance from an even pace, at the updates and as shown (a pause counts). `--calibrate` measures the costs that `smart_cut.py`'s work estimate uses: per case, and `progress_costs.RECODE_COST` per codec fitted over the encoder runs of all the cases (the built-in costs; a machine's own: `calibrate_progress.py`). Besides run_cases' cases it has 1080p calibration cases and typical app edits (`P2*`, `P3*`, `FB*`: 2 or 3 segments, without and with fades; H.264 720p, and 2 segments for 1080p, the other codecs and Facebook's H.264 + HE-AAC), run by name |
| `calibrate_progress.py` | A machine's own progress costs, see below |
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
- `smartcut/tests/test_progress_costs.py`: progress_costs.json loading: valid, missing, malformed, other versions, out
  of range (fallback to the built-in costs), the paths, logged once, edits with a broken file.
- `smartcut/tests/test_calibrate_progress.py`: a `--quick` run of `calibrate_progress.py` on a synthetic clip, and its
  refusal of changed or missing clips (needs the ffmpeg command: PATH or `SMARTCUT_TEST_FFMPEG`; ~40 s).

## A machine's own progress costs (`calibrate_progress.py`)

The progress estimate weighs re-encoded frames, copied frames and re-encoded audio by costs (`progress_costs.py`). Only
their ratios matter, and those change with the machine: more cores make the encoders faster, while copying and audio
don't. The built-in costs are one desktop core's. `calibrate_progress.py` measures the machine it runs on and writes
its costs to the `progress_costs.json` smartcut reads:
- the file `SMARTCUT_PROGRESS_COSTS` names;
- else `~/.config/smartcut/progress_costs.json` (Linux) or `%APPDATA%\smartcut\progress_costs.json` (Windows).

Keep the file outside the smartcut folder: deploying by copying that folder over would overwrite it.

The test clips come from one real H.264 + AAC clip: a YouTube download, landscape, ideally 1080p and at least 95 s.
- Synthetic clips re-encode 2-3x cheaper, so they don't calibrate.
- VP9, AV1 and (with `--codecs ...,hevc`) HEVC are derived from its first 90 s with `make_test_media.py`'s `--real`
  settings. HEVC carries no x265 options, so nothing of the machine that made it.
- A Facebook download (H.264 + HE-AAC) is optional. It is copied, since FFmpeg can't make HE-AAC.

Making them needs CPU (~20 min on one desktop core, 6 min on a 6-core / 12-thread desktop; 74 MB), so make them
where that is cheap and copy them over. The media folder's `manifest.json` records the recipe and every clip's size and SHA-256. The calibration refuses
media that is missing a clip, changed, or made by another recipe.

```bash
# On the dev PC: the clips, once (kept; rerunning reuses what is already made)
python smartcut/tests/regression/calibrate_progress.py --make-media calib_media clip.mp4 --facebook fb_clip.mp4

# On the target machine, after copying calib_media there: the plan and its estimated time, then the calibration
python smartcut/tests/regression/calibrate_progress.py --media calib_media --estimate
python smartcut/tests/regression/calibrate_progress.py --media calib_media

# Or everything on one machine
python smartcut/tests/regression/calibrate_progress.py clip.mp4 --facebook fb_clip.mp4 --media calib_media
```

The calibration itself:
- Runs 2-segment edits with and without 2 s fades for every codec and resolution (default H.264, VP9, AV1 at
  360/720/1080p, none above the clip's), plus the app's typical edits. One desktop core: ~10 min, 14 min with the
  real runs below (4 cores: 7 min in all).
- Uses the cores the process gets, as the app does. `--cores N` measures N cores, as on an N-CPU machine: one CPU per
  physical core, x265 sized for N.
- Checks the cores used, as CPU time / wall time per encoder run. Above the cores expected, the pinning didn't hold
  (as x265's threads once escaped it), and nothing is adopted. Below 0.85 of one core, something else used the CPU:
  rerun at a quiet time.
- Fits the costs per codec, the copy cost per frame and the audio cost per second (`progress_timeline.fit_recode_cost`;
  `--quick` keeps the built-in proportions between frame sizes). Codecs left out get the costs in use, scaled by the
  calibrated codecs' median speed ratio.
- Compares the costs in use with the new ones on the app's typical edits (P2/P2F/P3/P3F H.264 at 720p, VP9 and AV1 P2,
  FB2), replaying each edit's recorded progress calls through `smart_cut._WorkProgress`:
  - on the calibration runs;
  - on fresh real runs, ~4 min of one desktop core per `--repeat`. Each fresh run is replayed under both sets, so
    both get the same timings. The runs' own evenness is printed too, but the same short edit varies a few % from
    run to run.
- Adopts the new costs only if both comparisons are more even, and reports both either way:
  - adopted: the file is written, the one before kept as `progress_costs.previous.json`;
  - not adopted: the candidate goes to `progress_costs.candidate.json` next to it, and the file is left alone.
- Restart the app after an adoption. smartcut reads the file at a process's first edit, and logs which costs it uses
  (logger `smartcut`, INFO).
- `--quick` takes ~2 min: the smallest resolution, 2-segment edits, a 45 s excerpt when it makes the media. It is
  for a quick check, not a calibration: single short edits are noisy.
- To go back to the built-in costs, delete the file, or rename `progress_costs.previous.json` back.

### When the server gets more (or fewer) cores

Rerun the calibration there, with the media folder you already have, as the app runs:
`calibrate_progress.py --media calib_media`, without `--cores`. Nothing else changes. The file's `measured.cores`
says how many cores it was made with. Until you rerun, the old costs give a less even bar, but edits work as before.

## Windows notes

- PyAV fails to load its DLLs ("The filename or extension is too long") when Python lives under a very long
  path. Map the folder to a drive letter first: `subst Q: <long folder>` (remove with `subst Q: /d`).
- With an embeddable Python, `python -m smartcut` (used by the CLI tests) needs the repo on the path: add a
  `.pth` file containing the repo root to its `site-packages`.
