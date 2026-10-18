"""How steadily process_segments' progress moves, and calibration data for the work estimate in smart_cut.py.

    python progress_timeline.py <media dir> <out dir> [case ...] [--one-core] [--calibrate]

Runs run_cases.py's cases (those whose media exists) and prints per case: total time, number of progress
updates, the longest stretch without an update, and the largest difference between the progress shown and
the share of the time elapsed (0 = perfectly even). The time before the first update (opening and indexing the
input) is reported separately.

--calibrate also times every cut piece (the whole loop iteration, muxing and audio passthrough included). It
prints the re-encode cost per frame of each case, and per codec a fit of fixed + per-megapixel cost over the cases
(run several resolutions: CALIBRATION_CASES adds 1080p). It also prints the cost per copied frame and per second
of re-encoded audio, to compare with the _*_COST_* constants in smart_cut.py. Calibrate with --one-core.
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


def install_timers(stats):
    """Wrap the cutters to time every piece: the whole loop iteration in smart_cut (video, audio passthrough,
    subtitles, muxing), from the video cutter's start to _WorkProgress.piece_done. Encoder flushes (the frames
    still inside the encoder when a re-encoded run ends) are counted as re-encoding, not as the copy that triggers
    them, and re-encoded audio separately."""
    from smartcut import track_cutters, video_cutter

    work_progress = importlib.import_module("smartcut.smart_cut")._WorkProgress
    vc = video_cutter.VideoCutter
    orig_segment, orig_flush, orig_piece_done = vc.segment, vc.flush_encoder, work_progress.piece_done
    piece = {}  # the piece in progress: start time, flush and audio time so far, kind and frame count

    def flush_encoder(self):
        t = time.perf_counter()
        packets = orig_flush(self)
        stats["flush_s"] += time.perf_counter() - t
        stats["flush_packets"] += len(packets)
        return packets

    def segment(self, s):
        frames = [0]  # frames fed to the encoder: one hook call each in a re-encoded piece
        hook = self.on_frames_encoded

        def counting_hook(count):
            frames[0] += 1
            if hook is not None:
                hook(count)

        self.on_frames_encoded = counting_hook
        piece.update(t=time.perf_counter(), flush_s=stats["flush_s"], audio_s=stats["audio_s"])
        flushed_before = stats["flush_packets"]
        try:
            packets = orig_segment(self, s)
        finally:
            self.on_frames_encoded = hook
        ctx = self.in_stream.codec_context
        stats["megapixels"] = ctx.width * ctx.height / 1e6
        stats["codec"] = "av1" if "av1" in (ctx.name or "") else ctx.name  # as in smart_cut._WorkProgress
        piece.update(recode=s.require_recode,
                     frames=frames[0] if s.require_recode else len(packets) - (stats["flush_packets"] - flushed_before))
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

    vc.segment, vc.flush_encoder, work_progress.piece_done = segment, flush_encoder, piece_done

    ac = track_cutters.RecodeAudioCutter
    orig_audio = ac.segment

    def audio_segment(self, s):
        t = time.perf_counter()
        packets = orig_audio(self, s)
        stats["audio_s"] += time.perf_counter() - t
        stats["audio_media_s"] += sum(p.duration * p.time_base for p in packets if p.duration and p.time_base)
        return packets

    ac.segment = audio_segment


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

    stats = defaultdict(float)
    if args.calibrate:
        install_timers(stats)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'case':<16} {'total s':>7} {'first s':>7} {'updates':>7} {'max gap s':>9} {'at %':>5} {'max off %':>9}"
          + ("  re-encode ms/frame at MP" if args.calibrate else ""))
    recode_points = defaultdict(list)  # codec -> (megapixels, seconds per re-encoded frame, frames) per case
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
        work_time = total - first
        off = max(abs(f - (t - first) / work_time) for t, f in calls) if work_time > 0 else 0.0
        frames = stats["recode_frames"] - before.get("recode_frames", 0.0)
        seconds = (stats["recode_s"] + stats["flush_s"]) - (before.get("recode_s", 0.0) + before.get("flush_s", 0.0))
        per_frame = f"  {seconds / frames * 1000:7.1f} at {stats['megapixels']:.2f}" if args.calibrate and frames else ""
        if args.calibrate and frames:
            recode_points[stats["codec"]].append((stats["megapixels"], seconds / frames, frames))
        print(f"{case:<16} {total:7.1f} {first:7.1f} {len(calls):7d} {gap:9.2f} {at * 100:5.0f} {off * 100:9.1f}{per_frame}")
        if args.timeline:
            shown = [max([f for t, f in calls if t <= total * k / 20], default=0.0) for k in range(1, 21)]
            print("  time %:     " + " ".join(f"{k * 5:3d}" for k in range(1, 21)))
            print("  progress %: " + " ".join(f"{f * 100:3.0f}" for f in shown))

    if args.calibrate and stats["recode_frames"]:
        print("\nMeasured re-encode cost per frame = fixed + per megapixel (least squares over the cases, by frames):")
        for codec, points in recode_points.items():
            mp, sec, w = (np.array(x, dtype=float) for x in zip(*points))
            if len(set(mp)) > 1:
                fixed, per_mp = np.linalg.lstsq(np.stack([np.ones_like(mp), mp], 1) * np.sqrt(w)[:, None], sec * np.sqrt(w), rcond=None)[0]
            else:  # one resolution: no split into fixed and per-megapixel cost
                fixed, per_mp = 0.0, float((sec * w).sum() / (mp * w).sum())
            print(f"  {codec:<5} {fixed:.4f} s + {per_mp:.4f} s per megapixel ({len(points)} cases, "
                  f"{mp.min():.2f}-{mp.max():.2f} MP, {w.sum():.0f} frames)")
        if stats["copy_frames"]:
            print(f"  copy:      {stats['copy_s'] / stats['copy_frames']:.6f} s per frame ({stats['copy_frames']:.0f} frames)")
        if stats["audio_media_s"]:
            print(f"  audio:     {stats['audio_s'] / stats['audio_media_s']:.4f} s per second of audio "
                  f"({stats['audio_media_s']:.0f} s of audio)")


if __name__ == "__main__":
    main()
