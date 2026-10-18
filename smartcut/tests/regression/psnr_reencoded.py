"""PSNR (luma) of an edit's re-encoded frames, against what the encoder was given.

Usage: python psnr_reencoded.py <source> '<segments json>' <output> [<output> ...] [--repo PATH]

The segments are process_segments' (e.g. run_cases.TWO_SEGMENTS_REAL). Every frame the edit keeps (start <= time <
end) is compared with its source frame: copied frames must be bit-exact, the others count as re-encoded, and where
the segment fades the reference is the source frame faded by smartcut itself (video_cutter._fade_yuv_frame). Several
outputs of the same edit (e.g. encoder settings) are compared frame by frame against the first. 8-bit sources.
"""
import argparse
import json
import sys
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.frame import VideoFrame


def planes(f: VideoFrame) -> list[np.ndarray]:
    return [np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, :p.width] for p in f.planes]


def seconds(x: float) -> Fraction:
    return Fraction(x).limit_denominator(1000)


def alpha_of(seg: dict, t: Fraction) -> float:
    """VideoCutter._apply_fade_to_frame's alpha; t is the time from the segment start, in seconds."""
    fade_in = seg.get("video_fadein", seg.get("fadein"))
    fade_out = seg.get("video_fadeout", seg.get("fadeout"))
    alpha = 1.0
    if fade_in is not None and t < seconds(fade_in):
        alpha = min(1.0, float(t / seconds(fade_in)))
    if fade_out is not None:
        to_end = seconds(seg["end"]) - seconds(seg["start"]) - t
        if to_end < seconds(fade_out):
            alpha = min(alpha, float(to_end / seconds(fade_out)))
    return alpha


def kept_frames(src: str, segs: list[dict]) -> Iterator[tuple[np.ndarray, list[np.ndarray] | None]]:
    """Per frame the edit keeps, in output order: the reference luma, and the source planes if a copy must match them."""
    from smartcut.video_cutter import _fade_yuv_frame  # noqa: PLC0415

    with av.open(src, "r") as c:
        st = c.streams.video[0]
        assert st.time_base is not None
        start = Fraction(c.start_time or 0, 1_000_000)
        frames = c.decode(st)
        for seg in segs:
            s, e = seconds(seg["start"]) + start, seconds(seg["end"]) + start
            for f in frames:
                assert f.pts is not None
                t = f.pts * st.time_base
                if t < s:
                    continue
                if t >= e:
                    break
                alpha = alpha_of(seg, t - s)
                if alpha < 1.0:
                    faded = _fade_yuv_frame(f, alpha)
                    assert faded is not None, f"{f.format.name}: not a planar YUV format smartcut fades in YUV"
                    yield planes(faded)[0].astype(np.float64), None
                else:
                    yield planes(f)[0].astype(np.float64), [p.copy() for p in planes(f)]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10 * np.log10(255 ** 2 / max(float(np.mean((a - b) ** 2)), 1e-10))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("segments")
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]), help="smartcut repo root")
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    segs = json.loads(args.segments)
    outputs: list[str] = args.outputs

    reencoded: dict[str, list[tuple[int, bool, float]]] = {o: [] for o in outputs}  # (frame, faded, psnr)
    copied = dict.fromkeys(outputs, 0)
    containers = [av.open(o, "r") for o in outputs]
    decoders = [c.decode(c.streams.video[0]) for c in containers]
    n = 0
    for ref, exact in kept_frames(args.source, segs):
        for o, frames in zip(outputs, decoders):
            out = planes(next(frames))
            if exact is not None and all(np.array_equal(a, b) for a, b in zip(out, exact)):
                copied[o] += 1
            else:
                reencoded[o].append((n, exact is None, psnr(out[0].astype(np.float64), ref)))
        n += 1
    for c in containers:
        c.close()

    print(f"{n} frames kept")
    first = None
    for o in outputs:
        r = reencoded[o]
        all_db = np.array([x[2] for x in r])
        faded = [x[2] for x in r if x[1]]
        line = (f"  {Path(o).name:<28} copied (bit-exact) {copied[o]:4d}, re-encoded {len(r):4d} (faded {len(faded)}): "
                f"PSNR mean {all_db.mean():5.2f} dB, min {all_db.min():5.2f}")
        if first is None:
            first = {i: db for i, _, db in r}
        else:
            diff = [db - first[i] for i, _, db in r if i in first]
            line += f"; vs the first: {np.mean(diff):+.2f} dB mean over {len(diff)} frames"
        print(line, flush=True)


if __name__ == "__main__":
    main()
