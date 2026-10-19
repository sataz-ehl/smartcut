"""How steadily process_segments' progress moves, and calibration data for the work estimate in smart_cut.py.

    python progress_timeline.py <media dir> <out dir> [case ...] [--one-core] [--calibrate] [--timeline]

Runs run_cases.py's cases (those whose media exists) and prints per case: total time, number of progress
updates, the longest stretch without an update, and how far the progress shown is from an even pace (the share of
the time elapsed; 0 = perfectly even): at the updates ("max off %"), and over the whole run ("shown off %": also the
value still shown just before each update, so a pause counts). The time before the first update (opening and
indexing the input) is reported separately and not counted.

--calibrate also times every cut piece (the whole loop iteration, muxing and audio passthrough included). It
prints per case the re-encode cost per frame and the cost per copied frame. Per codec it fits the
progress_costs.RECODE_COST model (see fit_recode_cost) over the encoder runs of all the cases (the frames from the
encoder's start to the end of its flush, with the flush; busyness and fades as smart_cut._recoded_frames gives them):
run several resolutions (CALIBRATION_CASES adds 1080p) and content. It also prints the cost per copied frame and per
second of re-encoded audio over all cases, to compare with progress_costs.COPY_COST_PER_FRAME and
AUDIO_RECODE_COST_PER_SECOND. Calibrate with --one-core. (A machine's own costs: calibrate_progress.py.)

CALIBRATION_CASES also has typical Smart Edit jobs of the app (P2*, P3*: H.264 720p, 2 or 3 segments, without fades
and with 2 s fades on every segment), run by name.
"""
import argparse
import importlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cases import CASES, TWO_SEGMENTS_REAL, pin_to_one_core  # noqa: E402

# More resolutions for --calibrate (run by name; not in run_cases.py, they're slow). real_*_1080p.mp4: any 1080p
# H.264 clip and its VP9/AV1/HEVC transcodes (make_test_media.py's --real settings without the scaling)
CALIBRATION_CASES = {
    "RH10_h264_1080p": ("real_h264_1080p.mp4", TWO_SEGMENTS_REAL),
    "RV10_vp9_1080p":  ("real_vp9_1080p.mp4", TWO_SEGMENTS_REAL),
    "RA10_av1_1080p":  ("real_av1_1080p.mp4", TWO_SEGMENTS_REAL),
    "RE10_hevc_1080p": ("real_hevc_1080p.mp4", TWO_SEGMENTS_REAL),
}
# Typical Smart Edit jobs in the app (run by name), for how evenly progress moves: H.264 720p, 2 or 3 segments, without
# fades and with the help's "most common" 2 s video and audio fades on every segment; the same without fades for the
# other codecs, and for Facebook's H.264 + HE-AAC
_FADES = {"fadein": 2, "fadeout": 2}
_TWO = [{"start": 20, "end": 35}, {"start": 60, "end": 75}]
_THREE = [{"start": 10.3, "end": 25}, {"start": 40.77, "end": 55}, {"start": 77.1, "end": 90}]
CALIBRATION_CASES.update({
    "P2_h264_720p":  ("real_h264_720p.mp4", _TWO),
    "P2F_h264_720p": ("real_h264_720p.mp4", [{**seg, **_FADES} for seg in _TWO]),
    "P3_h264_720p":  ("real_h264_720p.mp4", _THREE),
    "P3F_h264_720p": ("real_h264_720p.mp4", [{**seg, **_FADES} for seg in _THREE]),
    "P2_h264_1080p": ("real_h264_1080p.mp4", _TWO),
    "P2_vp9_720p":   ("real_vp9_720p.mp4", _TWO),
    "P2_av1_720p":   ("real_av1_720p.mp4", _TWO),
    "P2_hevc_360p":  ("real_hevc.mp4", _TWO),
    "P2_hevc_720p":  ("real_hevc_720p.mp4", _TWO),
    "FB2_h264":      ("fb_sd_h264.mp4", _TWO),
    "FB2F_h264":     ("fb_sd_h264.mp4", [{**seg, **_FADES} for seg in _TWO]),
    "FB3_h264":      ("fb_sd_h264.mp4", _THREE),
})


def install_timers(stats: dict[str, float], runs: list[dict]) -> None:
    """Wrap the cutters to time every piece: the whole loop iteration in smart_cut (video, audio passthrough,
    subtitles, muxing), from the video cutter's start to _WorkProgress.piece_done. Encoder flushes (the frames
    still inside the encoder when a re-encoded run ends) are counted as re-encoding, not as the copy that triggers
    them, and re-encoded audio separately. A copied HEVC piece that re-encodes its leading pictures after a jump
    (VideoCutter.hybrid_recode_cra_segment: it decodes the GOP before, then encodes those few frames) counts as
    re-encoding those frames, as _WorkProgress counts it. Counted as copies, they made copied frames look 7x
    dearer (249 instead of 35 us per frame over E5, E6, RE10, J1, J2, R2, RH7; E6 alone 1338 us).

    runs gets one entry per encoder run (fed frames to the end of their flush): codec, megapixels, seconds, the
    process's CPU seconds over the same time (all its threads: CPU / wall time is the cores the run used), and the
    sums over its frames of 1, busyness and faded."""
    from smartcut import track_cutters, video_cutter

    sc = importlib.import_module("smartcut.smart_cut")
    work_progress = sc._WorkProgress
    vc = video_cutter.VideoCutter
    orig_segment, orig_flush, orig_piece_done, orig_plan = vc.segment, vc.flush_encoder, work_progress.piece_done, sc._work_plan
    piece = {}  # the piece in progress: start time, flush and audio time so far, kind and frame count
    run = {}  # the encoder run in progress: seconds and frame features so far
    features = {}  # id(cut piece) -> (busyness, faded) of the frames it re-encodes

    def work_plan(media_container, cut_segments, *args, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202 (as smart_cut._work_plan)
        plan = orig_plan(media_container, cut_segments, *args, **kw)
        features.clear()
        features.update({id(s): (b, f) for s, b, f in zip(cut_segments, plan.busyness, plan.faded)})
        return plan

    def end_run(cutter: video_cutter.VideoCutter) -> None:
        ctx = cutter.in_stream.codec_context
        runs.append({"codec": "av1" if "av1" in (ctx.name or "") else ctx.name, "mp": ctx.width * ctx.height / 1e6,
                     **{k: v for k, v in run.items() if k != "ending"}})
        run.clear()

    def flush_encoder(self):
        had = self.enc_codec is not None
        t, c = time.perf_counter(), time.process_time()
        packets = orig_flush(self)
        spent, cpu = time.perf_counter() - t, time.process_time() - c
        stats["flush_s"] += spent
        stats["flush_cpu"] += cpu
        stats["flush_packets"] += len(packets)
        if had and run:
            run["s"] += spent
            run["cpu"] += cpu
            run["ending"] = True  # ended by segment() below; a flush in finish() ends it here
            if not piece:
                end_run(self)
        return packets

    def segment(self, s):
        fed = [0]
        hook = self.on_frames_encoded

        def counting_hook(n_fed: int, n_out: int) -> None:
            fed[0] += n_fed
            if hook is not None:
                hook(n_fed, n_out)

        self.on_frames_encoded = counting_hook
        hybrid = not s.require_recode and self._should_hybrid_recode_cra(s)
        t, c = time.perf_counter(), time.process_time()
        piece.update(t=t, flush_s=stats["flush_s"], flush_cpu=stats["flush_cpu"], audio_s=stats["audio_s"])
        flushed_before = stats["flush_packets"]
        if s.require_recode or hybrid:
            busy, faded = features.get(id(s), (np.zeros(0), np.zeros(0, bool)))
            if not run:
                run.update(s=0.0, cpu=0.0, n=0, busy=0.0, faded=0)
            run.update(n=run["n"] + len(busy), busy=run["busy"] + float(busy.sum()), faded=run["faded"] + int(faded.sum()))
        try:
            packets = orig_segment(self, s)
        finally:
            self.on_frames_encoded = hook
        ctx = self.in_stream.codec_context
        stats["megapixels"] = ctx.width * ctx.height / 1e6
        stats["codec"] = "av1" if "av1" in (ctx.name or "") else ctx.name  # as in smart_cut._WorkProgress
        if s.require_recode or hybrid:
            piece.update(recode=True, frames=fed[0])
            run["s"] += time.perf_counter() - t - (stats["flush_s"] - piece["flush_s"])  # its flush is counted above
            run["cpu"] += time.process_time() - c - (stats["flush_cpu"] - piece["flush_cpu"])
        else:
            piece.update(recode=False, frames=len(packets) - (stats["flush_packets"] - flushed_before))
        if run.get("ending"):
            end_run(self)
        return packets

    def piece_done(self, index):
        if piece:
            spent = (time.perf_counter() - piece["t"] - (stats["flush_s"] - piece["flush_s"])
                     - (stats["audio_s"] - piece["audio_s"]))
            kind = "recode" if piece["recode"] else "copy"
            stats[f"{kind}_frames"] += piece["frames"]
            stats[f"{kind}_s"] += spent
            piece.clear()
        orig_piece_done(self, index)

    vc.segment, vc.flush_encoder, work_progress.piece_done, sc._work_plan = segment, flush_encoder, piece_done, work_plan

    ac = track_cutters.RecodeAudioCutter
    orig_audio = ac.segment

    def audio_segment(self, s):
        t = time.perf_counter()
        packets = orig_audio(self, s)
        stats["audio_s"] += time.perf_counter() - t
        stats["audio_media_s"] += sum(p.duration * p.time_base for p in packets if p.duration and p.time_base)
        return packets

    ac.segment = audio_segment


def fit_recode_cost(runs: list[dict], shape: tuple[float, float] | None = None
                    ) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """progress_costs.RECODE_COST for one codec: least squares of the runs' seconds = sum over their frames of (fixed +
    megapixels x per_mp) x (1 - busy + busy x busyness) + megapixels x fade x faded. busy (the share of the cost that
    follows busyness) on a grid from 0 to 1, the others 0 or more (a negative one is dropped and the rest refitted).
    Returns (fixed, per_mp, busy, fade) and the error of each run's predicted time.

    shape: (fixed, per_mp) to keep in proportion, fitting one factor for both. Runs at a single frame size can't
    separate them (calibrate_progress.py --quick).

    Not a negative fade: runs with a fade-in took libvpx and libaom less time, but not only for the faded frames (one
    core, 720p, the same 120 frames: VP9 faded 46 instead of 73 ms per frame, the unfaded ones after them 63 instead
    of 81 ms; AV1 104 / 136 and 65 / 111 ms). A per-frame term can't place that: fitted per run (VP9 -0.076 s per
    megapixel) it made faded frames nearly free and progress less even (RV7_vp9_720p 3.3 -> 10.7-10.9 % from an even
    pace, P2_vp9_720p 3.8 -> 7.1-7.2 %)."""
    y = np.array([r["s"] for r in runs], float)
    best = None
    for busy in np.linspace(0, 1, 21):
        frames = np.array([r["n"] * (1 - busy) + busy * r["busy"] for r in runs], float)  # in frames of average busyness
        mp = np.array([r["mp"] for r in runs], float)
        faded = np.array([r["faded"] for r in runs], float) * mp
        X = np.stack([frames * (shape[0] + shape[1] * mp), faded] if shape else [frames, frames * mp, faded], axis=1)
        use = list(range(X.shape[1]))
        while True:
            coef = np.zeros(X.shape[1])
            coef[use] = np.linalg.lstsq(X[:, use], y, rcond=None)[0]
            negative = [i for i in use if coef[i] < 0]
            if not negative:
                break
            use.remove(min(negative, key=lambda i: coef[i]))
        residual = float(((X @ coef - y) ** 2).sum())
        cost = (coef[0] * shape[0], coef[0] * shape[1], busy, coef[1]) if shape else (coef[0], coef[1], busy, coef[2])
        if best is None or residual < best[0]:
            best = (residual, tuple(float(v) for v in cost), (X @ coef - y) / y)
    return best[1], best[2]


def evenness(calls: list[tuple[float, float]], first: float, total: float) -> tuple[float, float]:
    """(at the updates, over the whole run): the largest distance between the progress shown and the share of the
    time elapsed since the first update."""
    work_time = total - first
    if work_time <= 0:
        return 0.0, 0.0
    at = max(abs(f - (t - first) / work_time) for t, f in calls)
    shown, prev = at, 0.0
    for t, f in calls:
        shown = max(shown, abs(prev - (t - first) / work_time))
        prev = f
    return at, max(shown, abs(prev - 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]), help="smartcut repo root")
    ap.add_argument("--one-core", action="store_true", help="pin to one CPU core")
    ap.add_argument("--calibrate", action="store_true", help="time every cut piece (see above)")
    ap.add_argument("--timeline", action="store_true", help="also print the progress at every 5%% of the time")
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    import smartcut
    if args.one_core:
        pin_to_one_core()

    stats, runs = defaultdict(float), []
    if args.calibrate:
        install_timers(stats, runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'case':<16} {'total s':>7} {'first s':>7} {'updates':>7} {'max gap s':>9} {'at %':>5} {'max off %':>9} {'shown off %':>11}"
          + ("  re-encode ms/frame at MP  copy us/frame" if args.calibrate else ""))
    rows = []
    for case in args.cases or CASES:
        src, segs = CASES[case] if case in CASES else CALIBRATION_CASES[case]
        if not (args.media_dir / src).exists():
            continue
        before = dict(stats)
        calls = []
        t0 = time.perf_counter()
        ok = smartcut.process_segments(str(args.media_dir / src), str(args.out_dir / f"{case}.mp4"),
                                       segments=[dict(s) for s in segs],
                                       progress_callback=lambda c, n, calls=calls, t0=t0: calls.append((time.perf_counter() - t0, c / n)))
        total = time.perf_counter() - t0
        if not ok or not calls:
            print(f"{case:<16} failed or no progress (ok={ok}, updates={len(calls)})")
            continue
        first = calls[0][0]
        times = [first] + [t for t, _ in calls] + [total]
        values = [0.0] + [f for _, f in calls] + [1.0]
        gap, at = max((times[i + 1] - times[i], values[i]) for i in range(len(times) - 1))
        off, shown = evenness(calls, first, total)
        rows.append((off, shown))
        frames = stats["recode_frames"] - before.get("recode_frames", 0.0)
        seconds = (stats["recode_s"] + stats["flush_s"]) - (before.get("recode_s", 0.0) + before.get("flush_s", 0.0))
        per_frame = f"  {seconds / frames * 1000:7.1f} at {stats['megapixels']:.2f}" if args.calibrate and frames else ""
        copy_frames = stats["copy_frames"] - before.get("copy_frames", 0.0)
        if args.calibrate and copy_frames:
            copy_us = (stats["copy_s"] - before.get("copy_s", 0.0)) / copy_frames * 1e6
            per_frame = f"{per_frame or ' ' * 18}  {copy_us:13.0f}"
        print(f"{case:<16} {total:7.1f} {first:7.1f} {len(calls):7d} {gap:9.2f} {at * 100:5.0f} {off * 100:9.1f} {shown * 100:11.1f}{per_frame}")
        if args.timeline:
            shown_at = [max([f for t, f in calls if t <= total * k / 20], default=0.0) for k in range(1, 21)]
            print("  time %:     " + " ".join(f"{k * 5:3d}" for k in range(1, 21)))
            print("  progress %: " + " ".join(f"{f * 100:3.0f}" for f in shown_at))
    if len(rows) > 1:
        print(f"{'mean':<16} {'':>7} {'':>7} {'':>7} {'':>9} {'':>5} {np.mean([r[0] for r in rows]) * 100:9.1f} "
              f"{np.mean([r[1] for r in rows]) * 100:11.1f}")

    if args.calibrate and runs:
        print("\nMeasured re-encode cost per frame, (fixed, per_mp, busy, fade) as in progress_costs.RECODE_COST, least squares "
              "over the encoder runs:")
        for codec in sorted({r["codec"] for r in runs}):
            mine = [r for r in runs if r["codec"] == codec]
            coef, error = fit_recode_cost(mine)
            print(f"  {codec:<5} ({coef[0]:.4f}, {coef[1]:.4f}, {coef[2]:.2f}, {coef[3]:.4f})  {len(mine)} runs, "
                  f"{sum(r['n'] for r in mine)} frames, {min(r['mp'] for r in mine):.2f}-{max(r['mp'] for r in mine):.2f} MP; "
                  f"error per run: median {np.median(np.abs(error)) * 100:.0f} %, largest {np.abs(error).max() * 100:.0f} %")
        if stats["copy_frames"]:
            print(f"  copy:      {stats['copy_s'] / stats['copy_frames']:.6f} s per frame ({stats['copy_frames']:.0f} frames)")
        if stats["audio_media_s"]:
            print(f"  audio:     {stats['audio_s'] / stats['audio_media_s']:.4f} s per second of audio "
                  f"({stats['audio_media_s']:.0f} s of audio)")


if __name__ == "__main__":
    main()
