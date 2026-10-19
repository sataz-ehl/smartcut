#!/usr/bin/env python3
"""
Re-derive smartcut's progress costs on the machine that runs the edits, into its progress_costs.json (see
smartcut/progress_costs.py). The costs' ratios change with the machine: more cores make the encoders faster, while
copying and audio don't speed up.

Usage:
    # 1. The test clips, once, on any machine (e.g. the dev PC); cached in <media dir> with a manifest.json:
    python calibrate_progress.py --make-media <media dir> <clip.mp4> [--facebook <fb clip.mp4>] [--codecs ...]
    # 2. The calibration, on the machine whose costs these are (copy <media dir> there first):
    python calibrate_progress.py --media <media dir> [--cores N] [--quick] [--codecs ...] [--resolutions ...]
    # Or both on one machine:
    python calibrate_progress.py <clip.mp4> [--facebook <fb clip.mp4>] [--media <media dir>] [...]
    # Only the plan and its estimated time:
    python calibrate_progress.py --media <media dir> --estimate

Input clips:
- <clip.mp4>: a real H.264 + AAC clip (a YouTube download, landscape, ideally 1080p and 95 s or longer).
  Synthetic clips re-encode 2-3x cheaper. Its first 90 s (--quick: 45 s) give:
  - H.264 at each resolution (a copy at the clip's own, else libx264 veryfast);
  - VP9, AV1 and, with --codecs ...,hevc, HEVC, with make_test_media.py's --real settings. HEVC carries no x265
    options in its headers, so nothing of the encoding machine.
  - Resolutions above the clip's own are left out.
- --facebook <fb clip.mp4> (optional): a Facebook download, H.264 + HE-AAC. FFmpeg can't make HE-AAC, so it is copied.

What it does:
1. Checks the media against its manifest (the recipe version, every file's size and SHA-256, its codec and size). It
   refuses media that is missing, changed or made otherwise.
2. Runs calibration edits in a fresh process, with the cores this process gets, as the app runs (--cores N: N cores,
   as on an N-CPU server, with x265 sized for N). The edits are 2 segments with and without 2 s fades for each codec
   and resolution, plus the app's typical edits.
   - It times every encoder run, piece and audio track (progress_timeline.install_timers).
   - It records every progress call, to replay it under other costs.
   - It releases memory after every edit, as the app does (release_memory), and prints what is left.
3. Checks the cores used: CPU time / wall time per encoder run (and per real run). Above the cores expected, the
   pinning didn't hold, as x265's threads once escaped it. On one core, below LOW_CPU (0.85): other load.
4. Fits the costs per codec (progress_timeline.fit_recode_cost; one resolution: a factor on the shape of the costs in
   use), the copy cost per frame and the audio cost per second. Codecs not calibrated get their costs in use, scaled
   by the calibrated codecs' median speed ratio.
5. Compares how evenly progress moves under the costs in use and new ones, replaying the recorded progress calls
   through smart_cut._WorkProgress (the same timings for both):
   - held-out check: costs fitted without the held-out edits (every edit at the app edits' height, 720p, and the
     Facebook clip's), on those edits' calibration runs;
   - real runs: fresh runs of the app's typical edits, interleaved (--repeat), under the costs fitted on all edits.
6. The adoption rule (the regression README, "The adoption rule"): adopts the costs fitted on all the edits only if
   both comparisons are more even by at least ADOPT_MARGIN (0.5) points, and not with --quick, when the pinning
   didn't hold, when the CPU check warned, or without a held-out check (one resolution). It writes them to the file
   smartcut reads (SMARTCUT_PROGRESS_COSTS, else ~/.config/smartcut/progress_costs.json or %APPDATA%\\smartcut\\, or
   --out), keeping the previous file as progress_costs.previous.json. Otherwise it writes
   progress_costs.candidate.json next to it and leaves the file alone. Either way it prints the comparison and
   "ADOPTED: ..." or "NOT ADOPTED: <every reason>. ...". Files are written to a temporary file next to them and
   renamed over them (os.replace).

Time, on one desktop core (the built-in costs' machine; a server core may be slower):
- --make-media: H.264, VP9, AV1 at 360/720/1080p, ~20 min (VP9 ~10, AV1 ~7, HEVC another ~12); 6 min on a
  6-core / 12-thread desktop. 74 MB for a 1080p clip and a Facebook one.
- calibration, default: ~10 min of edits, plus ~4 min of real runs per --repeat (14 min in all, measured). With 4
  cores, 7 min.
- --quick (360p, 45 s): ~2 min. A smoke check, not a calibration: it never adopts.
The plan's estimate is printed first.

Run it at a quiet time: other load on the CPU shows as a CPU / wall ratio below 1 and makes the timings noisy.
"""
import argparse
import contextlib
import ctypes
import gc
import hashlib
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smartcut.progress_costs import ProgressCosts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # the smartcut repo root (smartcut/tests/regression/ -> ...)

MEDIA_RECIPE = 1  # bump when the clips' settings change: media made otherwise is refused
EXCERPT, QUICK_EXCERPT = 90.0, 45.0  # seconds of the clip used
CODECS = ("h264", "vp9", "av1", "hevc")
DEFAULT_CODECS, DEFAULT_RESOLUTIONS = ("h264", "vp9", "av1"), (360, 720, 1080)
VIDEO_ARGS = {  # make_test_media.py's --real settings (H.264: its real_h264_720p)
    "h264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "150"],
    "vp9": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33", "-g", "150", "-cpu-used", "4", "-row-mt", "1"],
    "av1": ["-c:v", "libsvtav1", "-crf", "35", "-preset", "8", "-g", "150"],
    "hevc": ["-c:v", "libx265", "-preset", "fast", "-crf", "26", "-x265-params", "keyint=150:info=0:log-level=error"],
}
AV1_FALLBACK = ["-c:v", "libaom-av1", "-crf", "35", "-cpu-used", "6", "-row-mt", "1", "-g", "150"]  # FFmpeg without SVT-AV1
# Edits, as segments of a 90 s excerpt (scaled to the excerpt's length): the app's typical Smart Edit jobs
LAYOUTS = {
    "P2": [(20, 35), (60, 75)],
    "P3": [(10.3, 25), (40.77, 55), (77.1, 90)],
}
FADES = {"fadein": 2, "fadeout": 2}
# The adoption rule (the regression README, "The adoption rule"): the new costs must be more even than the costs in
# use by at least ADOPT_MARGIN points on average, in both checks. Between two timing samples of the same edits, the
# mean paired difference over the held-out edits varied by 0.08-0.18 points (sigma); a smaller gain isn't visible.
ADOPT_MARGIN = 0.5
LOW_CPU = 0.85  # on one core, CPU time / wall time below this: something else used the CPU
# For the estimates, measured on one desktop core (Oct 2026): ms per frame to make the clips at 360/720/1080p, and
# seconds per edit of a 90 s excerpt (FB2: Facebook's 360p H.264 + HE-AAC)
MAKE_MS_PER_FRAME = {"h264": (9, 19, 28), "vp9": (34, 89, 166), "av1": (30, 58, 104), "hevc": (43, 103, 181)}
EDIT_SECONDS = {  # the default media (a 1080p YouTube clip), one core, Oct 2026
    ("h264", 360): {"P2": 2.2, "P2F": 4.1}, ("h264", 720): {"P2": 6.1, "P2F": 13.9, "P3": 9.2, "P3F": 22.3},
    ("h264", 1080): {"P2": 8.3, "P2F": 28.2}, ("vp9", 360): {"P2": 6.7, "P2F": 8.8}, ("vp9", 720): {"P2": 18.8, "P2F": 25.2},
    ("vp9", 1080): {"P2": 34.4, "P2F": 46.2}, ("av1", 360): {"P2": 21.7, "P2F": 27.3}, ("av1", 720): {"P2": 43.4, "P2F": 67.1},
    ("av1", 1080): {"P2": 78.2, "P2F": 130.4}, ("hevc", 360): {"P2": 6.5, "P2F": 7.1}, ("hevc", 720): {"P2": 21.3, "P2F": 25.8},
    ("hevc", 1080): {"P2": 48.0, "P2F": 48.3}, ("fb", 360): {"P2": 3.9, "P2F": 7.5},
}


# --------------------------------------------------------------------------------------------- media and manifest

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> dict:
    """Codec family, size, frame rate, duration and audio of a clip; whether HEVC carries x265 options."""
    import av  # noqa: PLC0415
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        name = v.codec_context.name or ""
        info = {"codec": "av1" if "av1" in name else name, "width": v.codec_context.width, "height": v.codec_context.height,
                "fps": float(v.average_rate or v.guessed_rate or 0), "duration": float(c.duration or 0) / 1e6}
        a = c.streams.audio[0] if c.streams.audio else None
        info["audio"] = f"{a.codec_context.name} {a.codec_context.profile or ''}".strip() if a else None
        if info["codec"] == "hevc":
            first = next((bytes(p) for p in c.demux(v) if p.size), b"")
            extradata = bytes(v.codec_context.extradata or b"")
            info["x265_options"] = b"options: " in extradata or b"options: " in first
    return info


def make_media(clip: Path, out: Path, *, codecs: list[str], resolutions: list[int], facebook: Path | None, quick: bool,
               ffmpeg: str) -> None:
    """The test clips and manifest.json in out. Files already made from the same clip and recipe are kept."""
    out.mkdir(parents=True, exist_ok=True)
    src = probe(clip)
    if src["codec"] != "h264" or not (src["audio"] or "").startswith("aac"):
        print(f"WARNING: {clip} is {src['codec']} + {src['audio']}; the recipe expects a YouTube-like H.264 + AAC clip")
    length = QUICK_EXCERPT if quick else EXCERPT
    if src["duration"] < length + 1:
        if src["duration"] < 31:
            sys.exit(f"{clip} lasts {src['duration']:.0f} s: at least 31 s are needed")
        length = float(int(src["duration"] - 1))
        print(f"WARNING: {clip} lasts {src['duration']:.0f} s: using {length:.0f} s (the edits are scaled to it)")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout
    manifest_path = out / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source_hash = sha256(clip)
    reuse = old.get("recipe") == MEDIA_RECIPE and old.get("source", {}).get("sha256") == source_hash and old.get("excerpt") == length
    files = dict(old.get("files", {})) if reuse else {}
    jobs = []
    for codec in codecs:
        for res in resolutions:
            if res > src["height"]:
                print(f"skipping {codec} {res}p: the clip is {src['height']}p (no upscaling)")
                continue
            args = list(VIDEO_ARGS[codec])
            if codec == "av1" and "libsvtav1" not in encoders:
                args = list(AV1_FALLBACK)
            copy = codec == "h264" and res == src["height"] and src["codec"] == "h264"
            jobs.append((f"{codec}_{res}p.mp4", codec, res, ["-c:v", "copy"] if copy else ["-vf", f"scale=-2:{res}", *args]))
    if facebook is not None:
        jobs.append(("fb.mp4", "facebook", None, ["-c:v", "copy"]))
    est = sum(length * src["fps"] * MAKE_MS_PER_FRAME[c][(360, 720, 1080).index(r) if r in (360, 720, 1080) else 1] / 1000
              for _, c, r, a in jobs if c in MAKE_MS_PER_FRAME and a[1] != "copy")
    print(f"making {len(jobs)} clips of {length:.0f} s in {out}: ~{est / 60:.0f} min on one desktop core (less on more cores)")
    for name, codec, _height, vargs in jobs:
        entry = files.get(name)
        target = out / name
        if entry and target.exists() and target.stat().st_size == entry["size"] and entry.get("args") == vargs:
            print(f"  {name}: already made")
            continue
        source = facebook if codec == "facebook" else clip
        t = time.perf_counter()
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-t", f"{length}",
               "-map", "0:v:0", "-map", "0:a:0", *vargs, "-c:a", "copy", "-movflags", "+faststart", str(target)]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode:
            sys.exit(f"making {name} failed:\n{' '.join(cmd)}\n{r.stderr[-2000:]}")
        info = probe(target)
        if info.get("x265_options"):
            sys.exit(f"{name} carries x265 options in its headers (the encoding machine's): the recipe forbids it")
        files[name] = {"codec": codec if codec != "facebook" else info["codec"], "height": info["height"], "width": info["width"],
                       "args": vargs, "size": target.stat().st_size, "sha256": sha256(target), "audio": info["audio"],
                       "facebook": codec == "facebook"}
        print(f"  {name}: {info['codec']} {info['width']}x{info['height']} {info['audio']} in {time.perf_counter() - t:.0f} s")
        if codec == "facebook" and "HE-AAC" not in (info["audio"] or ""):
            print(f"  WARNING: {facebook} has {info['audio']} audio, not HE-AAC like Facebook's")
    ffmpeg_version = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, check=False).stdout.splitlines()[0]
    manifest = {"recipe": MEDIA_RECIPE, "excerpt": length, "made": datetime.now(UTC).isoformat(timespec="seconds"),
                "ffmpeg": ffmpeg_version, "source": {"name": clip.name, "sha256": source_hash, **src},
                "facebook": facebook.name if facebook else None, "files": files}
    write_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


def check_media(media: Path, codecs: list[str], resolutions: list[int]) -> dict:
    """The manifest, after checking that every clip the calibration needs is there and unchanged. Exits otherwise."""
    manifest_path = media / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"{media} has no manifest.json: make the clips with --make-media")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("recipe") != MEDIA_RECIPE:
        sys.exit(f"{media} was made with recipe {manifest.get('recipe')}, this script uses {MEDIA_RECIPE}: remake it")
    problems = []
    for name, entry in manifest["files"].items():
        path = media / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            problems.append(f"{name} changed since it was made (size or SHA-256)")
        else:
            info = probe(path)
            if info["codec"] != entry["codec"] or info["height"] != entry["height"]:
                problems.append(f"{name} is {info['codec']} {info['height']}p, the manifest says {entry['codec']} {entry['height']}p")
            if info.get("x265_options"):
                problems.append(f"{name} carries x265 options in its headers")
    available = {(e["codec"], e["height"]) for e in manifest["files"].values() if not e["facebook"]}
    src_height = manifest["source"]["height"]
    wanted = [(c, r) for c in codecs for r in resolutions if r <= src_height]
    missing = [f"{c}_{r}p.mp4" for c, r in wanted if (c, r) not in available]
    if missing:
        problems.append(f"not in the media: {', '.join(missing)} (make them with --make-media, or choose --codecs / --resolutions)")
    if problems:
        sys.exit("The media can't be used:\n  " + "\n  ".join(problems))
    return manifest


# ------------------------------------------------------------------------------------------------------- the plan

def segments(layout: str, fades: bool, length: float) -> list[dict]:
    scale = length / EXCERPT
    return [{"start": round(a * scale, 3), "end": round(b * scale, 3), **(FADES if fades else {})} for a, b in LAYOUTS[layout]]


def plan_cases(manifest: dict, codecs: list[str], resolutions: list[int], quick: bool) -> tuple[list[dict], list[str], list[str]]:
    """The calibration edits, the names of the app edits among them (also run fresh, as real runs), and of the held-out
    edits: every edit at the app edits' height (720p, or the largest below) and the Facebook clip's. The held-out
    check fits costs on the other edits and judges them on these."""
    length = manifest["excerpt"]
    files = {(e["codec"], e["height"]): name for name, e in manifest["files"].items() if not e["facebook"]}
    heights = sorted({h for c, h in files if c in codecs and h in resolutions})
    if quick:
        heights = heights[:1]
    app_height = max([h for h in heights if h <= 720], default=heights[0]) if heights else None
    cases, app, held = [], [], []

    def add(name: str, file: str, layout: str, fades: bool, *, is_app: bool, is_held: bool) -> None:
        cases.append({"name": name, "file": file, "segments": segments(layout, fades, length)})
        if is_app:
            app.append(name)
        if is_held:
            held.append(name)

    for codec in codecs:
        for h in heights:
            if (codec, h) not in files:
                continue
            for fades in (False, True):
                name = f"P2{'F' if fades else ''}_{codec}_{h}p"
                add(name, files[(codec, h)], "P2", fades, is_app=h == app_height and (codec == "h264" or not fades),
                    is_held=h == app_height)
        if codec == "h264" and not quick and (codec, app_height) in files:
            for fades in (False, True):
                add(f"P3{'F' if fades else ''}_h264_{app_height}p", files[(codec, app_height)], "P3", fades, is_app=True, is_held=True)
    fb = next((name for name, e in manifest["files"].items() if e["facebook"]), None)
    if fb:
        add("FB2", fb, "P2", False, is_app=True, is_held=True)
        add("FB2F", fb, "P2", True, is_app=False, is_held=True)
    return cases, app, held


def split_problem(cases: list[dict], held: list[str], manifest: dict) -> str | None:
    """Why the held-out check can't be made, or None: every codec held out needs edits of its own to fit on."""
    def codec(case: dict) -> str:
        return manifest["files"][case["file"]]["codec"]
    fit_codecs = {codec(c) for c in cases if c["name"] not in held}
    missing = sorted({codec(c) for c in cases if c["name"] in held} - fit_codecs)
    if not fit_codecs:
        return "all the edits are at one resolution, the check needs two (see --resolutions)"
    if missing:
        return f"{', '.join(missing)} has edits at one resolution only, the check needs two (see --resolutions)"
    return None


def estimate_seconds(case: dict, manifest: dict) -> float:
    """A rough time for an edit on one desktop core: EDIT_SECONDS at the nearest size, scaled by megapixels and by
    the excerpt's length."""
    entry = manifest["files"][case["file"]]
    codec, height = ("fb", 360) if entry["facebook"] else (entry["codec"], entry["height"])
    known = {h: v for (c, h), v in EDIT_SECONDS.items() if c == codec}
    near = min(known, key=lambda h: abs(h - height))
    layout = case["name"].split("_")[0]
    seconds = known[near].get(layout, known[near]["P2F" if layout.endswith("F") else "P2"])
    return seconds * (height / near) ** 2 * manifest["excerpt"] / EXCERPT


# -------------------------------------------------------------------------------------------- measuring (child)

def child_setup(cores: int | None) -> dict:
    """In a measuring process: import smartcut from this repo, pin to cores (None: as the app runs)."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(1, str(HERE))
    import run_cases  # noqa: PLC0415

    import smartcut  # noqa: F401, PLC0415 (this repo's, as sys.path now says)
    info = {"available_cpus": run_cases.available_cpus()}
    if cores:
        info["cpus"] = run_cases.pin_to_cores(cores)
    info["expected_cores"] = cores or info["available_cpus"]
    return info


def install_recorder(record: dict) -> None:
    """Record each edit's work plan and every call smart_cut makes on its _WorkProgress, with its time."""
    import importlib  # noqa: PLC0415
    sc = importlib.import_module("smartcut.smart_cut")
    work = sc._WorkProgress
    orig_plan = sc._work_plan

    def work_plan(*args, **kw):  # noqa: ANN002, ANN003, ANN202
        plan = orig_plan(*args, **kw)
        record["plan"] = {"family": plan.family, "megapixels": plan.megapixels,
                          "busyness": [b.round(5).tolist() for b in plan.busyness], "faded": [f.tolist() for f in plan.faded],
                          "copied_frames": plan.copied_frames, "audio_seconds": plan.audio_seconds}
        return plan

    sc._work_plan = work_plan
    for name in ("piece_started", "frames_encoded", "piece_done", "done"):
        orig = getattr(work, name)

        def wrapper(self, *args, _orig=orig, _name=name):  # noqa: ANN001, ANN002, ANN202
            result = _orig(self, *args)
            record["events"].append([time.perf_counter() - record["t0"], _name, list(args)])
            return result

        setattr(work, name, wrapper)


def release_memory() -> None:
    """Between edits, as the app does after every edit (video_downloader.release_memory): the garbage collector, then
    glibc's malloc_trim(0) hands the memory freed back to the OS. Without it freed codec buffers stay mapped, and a
    957 MB server can swap during the later edits, which slows their timings."""
    gc.collect()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):  # not glibc (e.g. musl)
            ctypes.CDLL("libc.so.6").malloc_trim(0)


def resident_mb() -> float | None:
    """This process's resident memory in MB (Linux: /proc; Windows: its working set); None where unknown."""
    try:
        if sys.platform.startswith("linux"):
            return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20  # pyright: ignore[reportAttributeAccessIssue]
        if sys.platform == "win32":
            from ctypes import wintypes  # noqa: PLC0415

            class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            *((name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"))]
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            counters = Counters(cb=ctypes.sizeof(Counters))
            if kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / 2**20
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    return None


def run_edits(cases: list[dict], run) -> dict:  # noqa: ANN001 (run: case -> its result)
    """Each case's result; after each edit the memory is released (release_memory) and what is left recorded."""
    edits = {}
    for case in cases:
        result = run(case)
        release_memory()
        result["rss_mb"] = resident_mb()
        edits[case["name"]] = result
    return edits


def run_edit(media: Path, case: dict, out_dir: Path, record: dict | None = None) -> dict:
    import smartcut  # noqa: PLC0415
    calls = []
    t0 = time.perf_counter()
    c0 = time.process_time()
    if record is not None:
        record.update(t0=t0, events=[], plan=None)
    ok = smartcut.process_segments(str(media / case["file"]), str(out_dir / f"{case['name']}.mp4"), segments=[dict(s) for s in case["segments"]],
                                   progress_callback=lambda c, n: calls.append((time.perf_counter() - t0, c / n)))
    end = time.perf_counter() - t0
    result = {"ok": ok, "end": end, "cpu": time.process_time() - c0, "calls": calls}
    if record is not None:
        result.update(events=record["events"], plan=record["plan"])
    return result


def child_measure(spec: dict) -> dict:
    """Calibration edits with timers and the recorder: the runs to fit, the copy and audio costs, the records."""
    info = child_setup(spec["cores"])
    import progress_timeline  # noqa: PLC0415
    stats, runs, record = defaultdict(float), [], {}
    progress_timeline.install_timers(stats, runs)
    install_recorder(record)
    media, out_dir = Path(spec["media"]), Path(spec["out_dir"])

    def run(case: dict) -> dict:
        first, before = len(runs), dict(stats)
        result = run_edit(media, case, out_dir, record)
        for r in runs[first:]:
            r["case"] = case["name"]
        result["stats"] = {k: stats[k] - before.get(k, 0.0) for k in ("copy_s", "copy_frames", "audio_s", "audio_media_s")}
        print(f"  {case['name']:<18} {result['end']:6.1f} s", flush=True)
        return result

    return {"info": info, "runs": runs, "stats": dict(stats), "edits": run_edits(spec["cases"], run)}


def child_real(spec: dict) -> dict:
    """The app edits, without timers (the costs come from SMARTCUT_PROGRESS_COSTS), recorded for replay."""
    info = child_setup(spec["cores"])
    record: dict = {}
    install_recorder(record)
    media, out_dir = Path(spec["media"]), Path(spec["out_dir"])
    return {"info": info, "edits": run_edits(spec["cases"], lambda case: run_edit(media, case, out_dir, record))}


def run_child(mode: str, spec: dict, costs_file: Path, work: Path) -> dict:
    spec_path, out_path = work / f"{mode}_spec.json", work / f"{mode}_result.json"
    spec_path.write_text(json.dumps(spec))
    env = dict(os.environ, SMARTCUT_PROGRESS_COSTS=str(costs_file))
    r = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), f"--_{mode}", str(spec_path), str(out_path)],
                       env=env, stdout=None, stderr=subprocess.PIPE, text=True, check=False)
    if r.returncode:
        sys.exit(f"the measuring process failed:\n{r.stderr[-3000:]}")
    return json.loads(out_path.read_text())


# ------------------------------------------------------------------------------------- fitting, replay, verdict

def evenness_of(calls: list, end: float) -> tuple[float, float]:
    import progress_timeline  # noqa: PLC0415
    if not calls:
        return 0.0, 0.0
    at, shown = progress_timeline.evenness([tuple(c) for c in calls], calls[0][0], end)
    return at * 100, shown * 100


def replay(edit: dict, costs) -> tuple[float, float]:  # noqa: ANN001
    """The edit's largest distances from an even pace (%, at the updates and shown) had it used these costs: its
    recorded progress calls, replayed through smart_cut._WorkProgress."""
    import importlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    import smartcut  # noqa: PLC0415
    sc = importlib.import_module("smartcut.smart_cut")  # the module: the package re-exports its function smart_cut
    p = edit["plan"]
    plan = sc._WorkPlan(p["family"], p["megapixels"], [np.array(b, float) for b in p["busyness"]],
                        [np.array(f, bool) for f in p["faded"]], p["copied_frames"], p["audio_seconds"])
    calls, now = [], [0.0]
    work = sc._WorkProgress(smartcut._ProgressAdapter(lambda c, n: calls.append((now[0], c / n))), None, plan, costs)
    for t, name, args in edit["events"]:
        now[0] = t
        getattr(work, name)(*args)
    return evenness_of(calls, edit["end"])


def _at720(cost: tuple) -> float:
    """A 720p frame of average busyness, in seconds."""
    return cost[0] + cost[1] * 0.9216


def edit_stats(measured: dict, names: list[str] | None) -> dict:
    """Copy and audio times over the named edits (None: all). Recordings without per-edit stats give the totals."""
    edits = measured["edits"]
    if names is None or not all("stats" in edits[n] for n in names):
        return measured["stats"]
    total: dict[str, float] = defaultdict(float)
    for n in names:
        for k, v in edits[n]["stats"].items():
            total[k] += v
    return total


def fit(measured: dict, old, names: list[str] | None = None) -> "tuple[ProgressCosts, dict]":  # noqa: ANN001
    """The costs fitted on the runs of the named edits (None: all of them), and notes on the fit."""
    import numpy as np  # noqa: PLC0415
    import progress_timeline  # noqa: PLC0415

    from smartcut import progress_costs  # noqa: PLC0415
    by_codec: dict[str, list] = {}
    for r in measured["runs"]:
        if names is None or r["case"] in names:
            by_codec.setdefault(r["codec"], []).append(r)
    recode, notes, ratios = {}, {"codecs": {}}, []
    for codec, runs in sorted(by_codec.items()):
        mps = sorted({round(r["mp"], 4) for r in runs})
        shape = None if len(mps) > 1 else old.recode_for(codec)[:2]
        cost, error = progress_timeline.fit_recode_cost(runs, shape)
        recode[codec] = tuple(round(v, 6) for v in cost)
        ratios.append(_at720(cost) / _at720(old.recode_for(codec)))
        notes["codecs"][codec] = {"runs": len(runs), "frames": sum(r["n"] for r in runs), "megapixels": mps,
                                  "shape kept": shape is not None, "median run error %": round(float(np.median(np.abs(error))) * 100, 1),
                                  "ms per 720p frame": round(_at720(cost) * 1000, 1), "was": round(_at720(old.recode_for(codec)) * 1000, 1)}
    speed = statistics.median(ratios) if ratios else 1.0
    scaled = []
    for codec, (fixed, per_mp, busy, fade) in old.recode.items():
        if codec not in recode:
            recode[codec] = (round(fixed * speed, 6), round(per_mp * speed, 6), busy, round(fade * speed, 6))
            scaled.append(codec)
    notes["scaled"] = {"codecs": scaled, "factor": round(speed, 3)}
    st = edit_stats(measured, names)
    copy = st["copy_s"] / st["copy_frames"] if st.get("copy_frames") else old.copy_per_frame
    audio = st["audio_s"] / st["audio_media_s"] if st.get("audio_media_s") else old.audio_per_second
    notes["copy us per frame"] = round(copy * 1e6, 1)
    notes["audio s per second"] = round(audio, 4)
    copy = min(max(copy, progress_costs.COPY_RANGE[0]), progress_costs.COPY_RANGE[1])
    audio = min(max(audio, progress_costs.AUDIO_RANGE[0]), progress_costs.AUDIO_RANGE[1])
    return progress_costs.ProgressCosts(recode, round(copy, 9), round(audio, 6), "candidate"), notes


def core_check(measured: dict) -> tuple[list[str], list[str], dict]:
    """Warnings that the pinning didn't hold, warnings of other load (one core), and the CPU / wall ratio per codec,
    from the encoder runs of 0.5 s or more."""
    expected = measured["info"]["expected_cores"]
    escaped, loaded, per_codec = [], [], {}
    for r in measured["runs"]:
        if r["s"] >= 0.5:
            per_codec.setdefault(r["codec"], []).append(r["cpu"] / r["s"])
    for codec, ratios in per_codec.items():
        worst, median = max(ratios), statistics.median(ratios)
        if worst > expected * 1.15 + 0.1:
            escaped.append(f"{codec}: an encoder run used {worst:.2f} cores, more than the {expected} this process may use: "
                           "its threads escaped the pinning (as x265's once did); these timings are not this machine's")
        if expected == 1 and median < LOW_CPU:
            loaded.append(f"{codec}: encoder runs got {median:.2f} of the core (median): something else used the CPU "
                          "(another process, or the VM's steal time); rerun at a quiet time")
    return escaped, loaded, {c: round(statistics.median(v), 2) for c, v in per_codec.items()}


def real_core_check(result: dict) -> tuple[list[str], list[str]]:
    """As core_check, for the real runs (no timers there: whole edits of 1 s or more)."""
    expected = result["info"]["expected_cores"]
    escaped, loaded = [], []
    for name, edit in result["edits"].items():
        if edit["end"] < 1.0:
            continue
        used = edit["cpu"] / edit["end"]
        if used > expected * 1.15 + 0.1:
            escaped.append(f"real run {name} used {used:.2f} cores, more than the {expected} expected: the pinning didn't hold")
        if expected == 1 and used < LOW_CPU:
            loaded.append(f"real run {name} got {used:.2f} of the core: something else used the CPU; rerun at a quiet time")
    return escaped, loaded


def decide(*, quick: bool, pinned: bool, quiet: bool, no_split: str | None, held_out: tuple[float, float],
           real: tuple[float, float]) -> tuple[bool, list[str]]:
    """The adoption rule: whether to adopt the new costs, and the reasons not to. held_out and real are the mean
    evenness (%, in use and new) of the held-out check and of the real runs; both must improve by ADOPT_MARGIN."""
    reasons = []
    if quick:
        reasons.append("--quick is a smoke check, not a calibration")
    if not pinned:
        reasons.append("the pinning didn't hold")
    if not quiet:
        reasons.append("the CPU check warned of other load")
    if no_split:
        reasons.append(f"no held-out check: {no_split}")
    elif not (held_out[1] <= held_out[0] - ADOPT_MARGIN and real[1] <= real[0] - ADOPT_MARGIN):
        reasons.append(f"not more even by {ADOPT_MARGIN} points (held-out edits {held_out[0]:.1f} -> {held_out[1]:.1f} %, "
                       f"real runs {real[0]:.1f} -> {real[1]:.1f} %)")
    return not reasons, reasons


def write_atomic(path: Path, text: str) -> None:
    """Write text to path through a temporary file in its folder and os.replace(): readers (smartcut at a process's
    first edit) see the old file or the new one, never part of it, also if this process stops midway."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def save_costs(target: Path, text: str, adopt: bool) -> Path:
    """Adopted: text to target, the file it replaces kept as progress_costs.previous.json. Otherwise text to
    progress_costs.candidate.json next to it, target unchanged. Returns the file written."""
    if adopt:
        if target.exists():
            write_atomic(target.with_name("progress_costs.previous.json"), target.read_text(encoding="utf-8"))
        write_atomic(target, text)
        return target
    candidate = target.with_name("progress_costs.candidate.json")
    write_atomic(candidate, text)
    return candidate


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def banner(lines: list[str]) -> None:
    print("\n" + "!" * 100)
    for line in lines:
        print(f"!!! {line}")
    print("!" * 100 + "\n")


def cpu_name() -> str:
    """The CPU model: /proc/cpuinfo's "model name" on Linux (platform.processor() is often empty there)."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=False).stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------------------------------------- main

def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] in ("--_measure", "--_real"):  # a measuring process
        spec = json.loads(Path(sys.argv[2]).read_text())
        result = (child_measure if sys.argv[1] == "--_measure" else child_real)(spec)
        Path(sys.argv[3]).write_text(json.dumps(result))
        return
    sys.stdout.reconfigure(line_buffering=True)  # in order with the measuring processes' output
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?", type=Path, help="a real H.264 + AAC clip to make the test clips from")
    ap.add_argument("--facebook", type=Path, help="a Facebook clip (H.264 + HE-AAC), copied as it is")
    ap.add_argument("--make-media", type=Path, metavar="DIR", help="only make the test clips into DIR")
    ap.add_argument("--media", type=Path, metavar="DIR", help="the test clips (with a clip: made there if needed)")
    ap.add_argument("--codecs", default=",".join(DEFAULT_CODECS), help=f"of {', '.join(CODECS)} (default: %(default)s)")
    ap.add_argument("--resolutions", default=",".join(map(str, DEFAULT_RESOLUTIONS)), help="heights (default: %(default)s)")
    ap.add_argument("--quick", action="store_true", help="a smoke check that never adopts: the smallest resolution, 2-segment edits; a 45 s excerpt when making media")
    ap.add_argument("--cores", type=int, help="measure on N cores, as on an N-CPU machine (default: the cores this process gets)")
    ap.add_argument("--repeat", type=int, default=1, help="real runs of each set of costs (default: %(default)s)")
    ap.add_argument("--out", type=Path, help="where to write (default: the file smartcut reads)")
    ap.add_argument("--work", type=Path, help="for the edits' outputs (default: a temporary folder)")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--estimate", action="store_true", help="only print the plan and its estimated time")
    args = ap.parse_args()
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    if not codecs or any(c not in CODECS for c in codecs):
        ap.error(f"--codecs: of {', '.join(CODECS)}")
    resolutions = sorted(int(r) for r in args.resolutions.split(","))

    if args.make_media:
        if not args.clip:
            ap.error("--make-media needs the clip")
        make_media(args.clip, args.make_media, codecs=codecs, resolutions=resolutions, facebook=args.facebook, quick=args.quick,
                   ffmpeg=args.ffmpeg)
        return
    if args.clip:
        media = args.media or Path(tempfile.gettempdir()) / "smartcut_calibration_media"
        make_media(args.clip, media, codecs=codecs, resolutions=resolutions, facebook=args.facebook, quick=args.quick, ffmpeg=args.ffmpeg)
    elif args.media:
        media = args.media
    else:
        ap.error("give the clip, or --media with clips made by --make-media")

    sys.path.insert(0, str(REPO))
    sys.path.insert(1, str(HERE))
    from smartcut import progress_costs  # noqa: PLC0415
    if args.quick:
        resolutions = resolutions[:1]  # the smallest asked for
    manifest = check_media(media, codecs, resolutions)
    cases, app, held = plan_cases(manifest, codecs, resolutions, args.quick)
    no_split = split_problem(cases, held, manifest)
    fit_on = [c["name"] for c in cases if c["name"] not in held]
    target = args.out or progress_costs.progress_costs_path()
    old, messages = progress_costs.load(target)
    for _, text in messages:
        print(f"costs in use: {text}")
    real_cases = [c for c in cases if c["name"] in app]
    edits_s = sum(estimate_seconds(c, manifest) for c in cases)
    real_s = sum(estimate_seconds(c, manifest) for c in real_cases) * 2 * args.repeat
    print(f"\nplan: {len(cases)} calibration edits ({', '.join(c['name'] for c in cases)})")
    print(f"      held out: {', '.join(held)}" + (f" (no held-out check: {no_split})" if no_split else f"; fitted without them on {', '.join(fit_on)}"))
    print(f"      then the app edits {', '.join(app)} with the costs in use and the new ones, {args.repeat}x each")
    print(f"      cores: {args.cores or 'as this process gets them'}; media {media} (excerpt {manifest['excerpt']:.0f} s)")
    if args.quick:
        print("      --quick: a smoke check; it writes the candidate only, never the file smartcut reads")
    print(f"estimated: ~{edits_s / 60:.1f} min of edits + ~{real_s / 60:.1f} min of real runs on one desktop core "
          "(one server core may be slower; more cores, faster)")
    if args.estimate:
        return

    work = args.work or Path(tempfile.mkdtemp(prefix="smartcut_calibration_"))
    work.mkdir(parents=True, exist_ok=True)
    old_file = work / "costs_in_use.json"
    old_file.write_text(progress_costs.to_json(old))
    spec = {"media": str(media), "out_dir": str(work), "cores": args.cores, "cases": cases}
    print("\ncalibration edits:")
    t = time.perf_counter()
    measured = run_child("measure", spec, old_file, work)
    print(f"  took {(time.perf_counter() - t) / 60:.1f} min")
    failed = [n for n, e in measured["edits"].items() if not e["ok"]]
    if failed:
        sys.exit(f"edits failed: {', '.join(failed)}")
    memory = [e["rss_mb"] for e in measured["edits"].values() if e.get("rss_mb") is not None]
    if memory:
        print(f"  memory after each edit, released: {min(memory):.0f}-{max(memory):.0f} MB")

    escaped, loaded, ratios = core_check(measured)
    print(f"\ncores: {measured['info']['expected_cores']} expected (this process may use {measured['info']['available_cpus']}"
          f"{', pinned to CPUs ' + str(measured['info']['cpus']) if 'cpus' in measured['info'] else ''}); "
          f"CPU / wall per encoder run, median: {ratios}")
    if escaped or loaded:
        banner(escaped + loaded)

    new, notes = fit(measured, old)
    held_fit = fit(measured, old, fit_on)[0] if not no_split else None
    print("\nfitted on all the edits (fixed s, per_mp s, busy, fade s/MP):")
    for codec, values in sorted(new.recode.items()):
        n = notes["codecs"].get(codec)
        what = (f"{n['runs']} runs, median error {n['median run error %']} %, a 720p frame {n['was']} -> {n['ms per 720p frame']} ms"
                + (" (one resolution: shape in use kept)" if n["shape kept"] else "")) if n else \
            f"not calibrated: costs in use x {notes['scaled']['factor']}"
        print(f"  {codec:<5} {values}  {what}")
    print(f"  copy  {notes['copy us per frame']} us per frame (in use {old.copy_per_frame * 1e6:.1f}); "
          f"audio {notes['audio s per second']} s per second (in use {old.audio_per_second})")
    if held_fit is not None:
        print("  without the held-out edits: " + ", ".join(f"{c} {held_fit.recode[c]}" for c in sorted(notes["codecs"])))

    print("\nevenness, largest distance from an even pace at the updates (%), in use -> new:")
    replayed = {}
    for name, edit in measured["edits"].items():
        recorded = evenness_of(edit["calls"], edit["end"])
        r_old = replay(edit, old)
        replayed[name] = {"recorded": round(recorded[0], 2), "in use": round(r_old[0], 2), "new": round(replay(edit, new)[0], 2)}
        if held_fit is not None and name in held:
            replayed[name]["held-out fit"] = round(replay(edit, held_fit)[0], 2)
        if abs(r_old[0] - recorded[0]) > 0.5:
            print(f"  NOTE {name}: the replay ({r_old[0]:.1f} %) doesn't match the run ({recorded[0]:.1f} %)")
    real = {name: {"in use": [], "new": []} for name in app}  # each fresh run as it went, with its set of costs
    paired = {name: {"in use": [], "new": []} for name in app}  # every fresh run replayed under both sets
    new_file = work / "costs_new.json"
    new_file.write_text(progress_costs.to_json(new))
    real_spec = {**spec, "cases": real_cases}
    for k in range(args.repeat):
        for label, costs_file in (("in use", old_file), ("new", new_file)):
            print(f"  real runs, {label} costs ({k + 1}/{args.repeat})", flush=True)
            result = run_child("real", real_spec, costs_file, work)
            for name, edit in result["edits"].items():
                if not edit["ok"]:
                    sys.exit(f"real run {name} failed")
                real[name][label].append(round(evenness_of(edit["calls"], edit["end"])[0], 2))
                paired[name]["in use"].append(round(replay(edit, old)[0], 2))
                paired[name]["new"].append(round(replay(edit, new)[0], 2))
            run_escaped, run_loaded = real_core_check(result)
            if run_escaped or run_loaded:
                banner(run_escaped + run_loaded)
            escaped += run_escaped
            loaded += run_loaded
    print(f"\n  {'edit':<18} {'held-out check':>15} {'real runs, paired':>19}   real runs as they went")
    for name in held if held_fit is not None else app:
        r = replayed[name]
        check = f"{r['in use']:7.1f} -> {r['held-out fit']:5.1f}" if "held-out fit" in r else f"{'':>16}"
        live = ""
        if name in app:
            pr, q = paired[name], real[name]
            live = f"  {mean(pr['in use']):7.1f} -> {mean(pr['new']):5.1f}     {'/'.join(map(str, q['in use']))} -> {'/'.join(map(str, q['new']))}"
        print(f"  {name:<18} {check}{live}")
    held_old = mean([replayed[n]["in use"] for n in held if "held-out fit" in replayed[n]])
    held_new = mean([replayed[n]["held-out fit"] for n in held if "held-out fit" in replayed[n]])
    real_old, real_new = mean([v for n in app for v in paired[n]["in use"]]), mean([v for n in app for v in paired[n]["new"]])
    live_old, live_new = mean([v for n in app for v in real[n]["in use"]]), mean([v for n in app for v in real[n]["new"]])
    held_text = f"{held_old:7.1f} -> {held_new:5.1f}" if held_fit is not None else f"{'(none)':>16}"
    print(f"  {'mean':<18} {held_text}  {real_old:7.1f} -> {real_new:5.1f}     {live_old:.1f} -> {live_new:.1f}")
    all_old = mean([v["in use"] for v in replayed.values()])
    all_new = mean([v["new"] for v in replayed.values()])
    print(f"  all {len(replayed)} calibration edits, replayed under the new costs (fitted on them): {all_old:.1f} -> {all_new:.1f}")
    print(f"  Held-out check: the held-out edits' calibration runs, replayed under the costs in use and costs fitted\n"
          f"  without them. Real runs, paired: fresh runs of the app edits, each replayed under the costs in use and the\n"
          f"  new ones (fitted on all the edits), the same timings for both. Adopted if both are more even by at least\n"
          f"  {ADOPT_MARGIN} points. As they went: each run with its own costs; noisy (the same short edit varies a few % from\n"
          f"  run to run).")

    adopt, reasons = decide(quick=args.quick, pinned=not escaped, quiet=not loaded, no_split=no_split,
                            held_out=(held_old, held_new), real=(real_old, real_new))
    import importlib  # noqa: PLC0415

    import av  # noqa: PLC0415
    smart_cut = importlib.import_module("smartcut.smart_cut")
    measured_info = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"), "host": socket.gethostname(),
        "platform": platform.platform(), "cpu": cpu_name(),
        "cores": measured["info"]["expected_cores"], "cores_simulated": bool(args.cores),
        "available_cpus": measured["info"]["available_cpus"], "cpu_per_wall": ratios,
        "python": platform.python_version(), "av": av.__version__,
        "smartcut": f"{smart_cut.__version__} {git_rev() or ''}".strip(),
        "media": {"source": manifest["source"]["name"], "sha256": manifest["source"]["sha256"][:16], "recipe": MEDIA_RECIPE,
                  "excerpt": manifest["excerpt"], "facebook": manifest.get("facebook")},
        "fit": notes, "quick": args.quick,
    }
    comparison = {
        "evenness": {"calibration runs, replayed": replayed, "real runs, paired": paired, "real runs, as they went": real,
                     "means": {"held-out check": [round(held_old, 2), round(held_new, 2)] if held_fit is not None else None,
                               "real runs, paired": [round(real_old, 2), round(real_new, 2)],
                               "real runs, as they went": [round(live_old, 2), round(live_new, 2)],
                               "calibration runs, all": [round(all_old, 2), round(all_new, 2)]}},
        "held_out": held, "fitted_without_them_on": fit_on, "held_out_fit": json.loads(progress_costs.to_json(held_fit))["recode_cost"]
        if held_fit is not None else None,
        "margin": ADOPT_MARGIN, "not_adopted_because": reasons,
        "costs_before": json.loads(progress_costs.to_json(old))}
    text = progress_costs.to_json(new, measured_info, {**comparison, "adopted": adopt})
    progress_costs.parse(json.loads(text))  # what smartcut will read must load
    written = save_costs(target, text, adopt)
    if adopt:
        print(f"\nADOPTED: more even by at least {ADOPT_MARGIN} points (held-out edits {held_old:.1f} -> {held_new:.1f} %, "
              f"real runs {real_old:.1f} -> {real_new:.1f} %). Wrote {written}")
        print("smartcut reads it at the first edit of a process: restart the app (e.g. sudo systemctl restart gradio-apps).")
    else:
        print(f"\nNOT ADOPTED: {'; '.join(reasons)}. {target} unchanged; the candidate is in {written}")
    print(f"smartcut reads {progress_costs.progress_costs_path()} "
          f"({'SMARTCUT_PROGRESS_COSTS' if os.environ.get(progress_costs.ENV_VAR) else 'the default for this user'})"
          + ("; --out wrote elsewhere: point SMARTCUT_PROGRESS_COSTS at it" if args.out and args.out != progress_costs.progress_costs_path() else ""))
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    main()
