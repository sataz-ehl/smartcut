#!/usr/bin/env python3
"""
A/V sync and audio level at the start and at every join of a multi-segment cut, measured against the source.

Usage (from the repo root):
    python smartcut/tests/regression/join_sync.py <source> <out dir> [--segments a-b,c-d,...] [--ext .mp4]
                                                  [--repo PATH] [--one-core] [--measure OUTPUT]

Cuts the source with smartcut.process_segments() (no fades; 6 segments by default) into <out dir>/join_sync<ext>,
or with --measure only measures an existing output of those segments. Needs real, noise-like audio: on a steady
tone the correlation can't tell periods apart (use smartcut/tests/test_audio_joins.py's tone checks for clicks).

Per segment, where its audio sits against its video: the output audio around the place the video says it should
come from is cross-correlated with the source. offset > 0: the audio is late (it plays after the picture it belongs
to). "by count" places the decoded samples one after another, as players play audio; "by pts" places each decoded
frame at its timestamp. Copied audio drifted at every join by count (whole frames: AAC-LC after 5 joins -15.5 ms,
HE-AAC -27 ms), not by pts.

Per join, outside the short fade at joins of re-encoded audio (RecodeAudioCutter.JOIN_FADE, 3 ms each side): the
level against the source in 5 ms windows over the 40 ms on each side (copied AAC joins had x0.16-x33), and a click
measure: the largest 1 ms RMS of the sample-to-sample change within 3 ms of the join, against its median over
+-100 ms ("mid" is the same measure inside the segments, for comparison).

Also: audio samples missing or overlapping by pts, where the streams start, and where the audio ends against the
last picture.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import av
import numpy as np

SEGMENTS = [(10.3, 16.9), (25.05, 31.6), (40.77, 46.2), (58.4, 64.95), (77.1, 83.3), (96.6, 103.15)]
WIN, SEARCH = 0.5, 0.15  # correlation window and search range (s)
JOIN_FADE = 0.003  # RecodeAudioCutter.JOIN_FADE
LEVEL_SPAN, LEVEL_WIN = 0.04, 0.005


def pin_to_one_core() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_cases import pin_to_one_core as pin  # noqa: PLC0415
    pin()


def read(path: Path) -> dict:
    """Video frame times and frame duration; first audio track: mono samples by count and by pts, rate, first pts,
    overlapping and missing samples by pts; stream start times."""
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        starts = {s.type: (float(s.start_time * s.time_base) if s.start_time is not None else None) for s in c.streams}
        starts["container"] = c.start_time / av.time_base if c.start_time is not None else None
        rate_v = vs.average_rate
        vt = sorted(float(f.pts * f.time_base) for f in c.decode(video=0))
    chunks, pts = [], []
    with av.open(str(path)) as c:
        rate = c.streams.audio[0].sample_rate
        for f in c.decode(audio=0):
            chunks.append(f.to_ndarray().astype(np.float64).mean(axis=0))
            pts.append(float(f.pts * f.time_base))
    by_count = np.concatenate(chunks)
    a0 = pts[0]
    end = max(round((s - a0) * rate) + len(ch) for s, ch in zip(pts, chunks))
    by_pts, filled = np.zeros(end), np.zeros(end, bool)
    overlaps = 0
    for s, ch in zip(pts, chunks):
        i = round((s - a0) * rate)
        overlaps += int(filled[i:i + len(ch)].sum())
        by_pts[i:i + len(ch)], filled[i:i + len(ch)] = ch, True
    return {"vt": vt, "frame": float(1 / rate_v) if rate_v else 0.0, "count": by_count, "pts": by_pts, "rate": rate,
            "a0": a0, "overlaps": overlaps, "gaps": int((~filled).sum()), "starts": starts}


def find(src: np.ndarray, s_a0: float, rate: int, out: np.ndarray, o_a0: float, out_t: float,  # noqa: PLR0917
         src_t: float) -> tuple[float, float]:
    """Lag (s) of the output audio at out_t against the source at src_t (normalised cross-correlation) and r."""
    i = round((out_t - o_a0) * rate)
    w = out[i:i + int(WIN * rate)]
    j0 = round((src_t - s_a0 - SEARCH) * rate)
    s = src[max(j0, 0):j0 + len(w) + int(2 * SEARCH * rate)]
    if len(w) < WIN * rate * 0.9 or len(s) <= len(w):
        return float("nan"), 0.0
    n = 1 << int(np.ceil(np.log2(len(s) + len(w))))
    corr = np.fft.irfft(np.fft.rfft(s, n) * np.conj(np.fft.rfft(w, n)), n)[:len(s) - len(w) + 1]
    energy = np.convolve(s ** 2, np.ones(len(w)), "valid")
    r = corr / (np.linalg.norm(w) * np.sqrt(np.maximum(energy, 1e-18)))
    k = int(np.argmax(r))
    return (max(j0, 0) + k) / rate + s_a0 - src_t, float(r[k])


def level_at(src: np.ndarray, s_a0: float, out: np.ndarray, o_a0: float, rate: int, out_t: float,  # noqa: PLR0917
             src_t: float, sign: int) -> list[float]:
    """Output/source RMS in 5 ms windows from JOIN_FADE to LEVEL_SPAN after (sign +1) or before (-1) out_t."""
    ratios = []
    w = int(LEVEL_WIN * rate)
    for k in range(int((LEVEL_SPAN - JOIN_FADE) / LEVEL_WIN)):
        off = JOIN_FADE + k * LEVEL_WIN if sign > 0 else -(JOIN_FADE + (k + 1) * LEVEL_WIN)
        i, j = round((out_t + off - o_a0) * rate), round((src_t + off - s_a0) * rate)
        o, s = out[i:i + w], src[j:j + w]
        rs = np.sqrt((s ** 2).mean()) if len(s) == w else 0
        if len(o) == w and rs > 1e-3:
            ratios.append(float(np.sqrt((o ** 2).mean()) / rs))
    return ratios


def click(audio: np.ndarray, rate: int, a0: float, t: float) -> float:
    i, n, w = round((t - a0) * rate), int(0.1 * rate), max(1, rate // 1000)
    d = np.diff(audio[max(i - n, 0):i + n])
    r = np.sqrt((d[: len(d) // w * w].reshape(-1, w) ** 2).mean(axis=1))
    c = len(r) // 2
    return float(r[c - 3:c + 4].max() / (np.median(r) or 1e-9))


def measure(src_path: Path, out_path: Path, segments: list[tuple[float, float]]) -> None:
    s, o = read(src_path), read(out_path)
    if s["rate"] != o["rate"]:
        print(f"sample rate {o['rate']} vs the source's {s['rate']}: resampled, not measured")
        return
    rate = s["rate"]
    st = o["starts"]
    print(f"streams start: container {st.get('container')}, video {st.get('video')}, audio {st.get('audio')}; first decoded "
          f"audio {o['a0']:+.4f} s, first video frame {o['vt'][0]:+.4f} s")
    v_end = o["vt"][-1] + o["frame"]
    a_end = o["a0"] + len(o["count"]) / rate
    print(f"audio by pts: {o['overlaps']} overlapping, {o['gaps']} missing samples; decoded audio ends {(a_end - v_end) * 1000:+.1f} ms "
          f"after the last picture")
    n, joins, worst = 0, [], 0.0
    st0 = s["starts"]["container"] or 0.0  # segment times are relative to the source's start, as in smartcut
    for k, (a, b) in enumerate(segments):
        kept = [x for x in s["vt"] if st0 + a - 1e-9 <= x < st0 + b - 1e-9]
        S, V = kept[0], o["vt"][n]  # first kept frame: in the source, in the output
        n += len(kept)
        rows, offsets = [], {}
        for where, u in (("start", 0.15), ("end", kept[-1] - S - WIN - 0.1)):
            for name in ("count", "pts"):
                lag, r = find(s[name], s["a0"], rate, o[name], o["a0"], V + u, S + u)
                offsets[(where, name)] = -lag
                rows.append(f"{where}/{name} {-lag * 1000:+6.1f} ms (r={r:.3f})")
        worst = max(worst, *(abs(x) for x in offsets.values() if not math.isnan(x)))
        # The join: where the segment's audio starts, at its cut time (its first frame is up to a frame later)
        joins.append((V - (S - (st0 + a)), offsets[("start", "count")], offsets[("end", "count")]))
        print(f"  segment {k + 1} (src {a}-{b}, video at {V:7.3f} s): " + "  ".join(rows))
    for k in range(1, len(joins)):
        J, late, _ = joins[k]
        _, _, plate = joins[k - 1]
        # Output J + late + u holds source a + u of this segment (offset > 0: late), J + plate - u the previous one's b - u
        after = level_at(s["count"], s["a0"], o["count"], o["a0"], rate, J + late, st0 + segments[k][0], +1)
        before = level_at(s["count"], s["a0"], o["count"], o["a0"], rate, J + plate, st0 + segments[k - 1][1], -1)
        lv = before + after
        mid = max(click(o["count"], rate, o["a0"], J + u) for u in (1.0, 1.5, 2.0))
        print(f"  join {k} at {J:7.3f} s: level outside the join fades x{min(lv):.2f}..x{max(lv):.2f}, "
              f"click {click(o['count'], rate, o['a0'], J + late):.1f} (mid {mid:.1f})")
    # The same 5 ms level inside the segments (re-encoding alone moves it, e.g. x0.77-1.27 for 99.8 % of windows)
    inside = []
    for (J, late, _), (a, b) in zip(joins, segments):
        for u in np.arange(0.5, b - a - 0.5, 0.01):
            inside += level_at(s["count"], s["a0"], o["count"], o["a0"], rate, J + late + u - JOIN_FADE, st0 + a + u - JOIN_FADE, +1)[:1]
    if inside:
        q = np.percentile(inside, [0.1, 99.9])
        print(f"level inside the segments x{min(inside):.2f}..x{max(inside):.2f} (99.8 % of windows x{q[0]:.2f}..x{q[1]:.2f})")
    print(f"largest |offset|: {worst * 1000:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--segments", help="a-b,c-d,... in seconds (default: 6 segments from 10 s to 103 s)")
    ap.add_argument("--ext", default=".mp4", help="output extension (default .mp4)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]), help="smartcut repo root")
    ap.add_argument("--one-core", action="store_true", help="cut on one CPU core")
    ap.add_argument("--measure", type=Path, help="measure this existing output instead of cutting")
    args = ap.parse_args()
    segments = ([tuple(float(x) for x in s.split("-")) for s in args.segments.split(",")] if args.segments else SEGMENTS)
    out = args.measure
    if out is None:
        sys.path.insert(0, args.repo)
        import smartcut  # noqa: PLC0415
        if args.one_core:
            pin_to_one_core()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"join_sync{args.ext}"
        t = time.perf_counter()
        if not smartcut.process_segments(str(args.source), str(out), [{"start": a, "end": b} for a, b in segments]):
            sys.exit("process_segments failed")
        print(f"{out}: cut in {time.perf_counter() - t:.1f} s")
    measure(args.source, out, segments)


if __name__ == "__main__":
    main()
