"""PSNR (luma) of an edit's re-encoded frames, against what the encoder was given; copied frames must be bit-exact.

Usage: python psnr_reencoded.py <source> '<segments json>' <output> [<output> ...] [--repo PATH]

The segments are process_segments' (e.g. run_cases.TWO_SEGMENTS_REAL). Which source frames the edit re-encodes comes
from smartcut itself: the plan of the cut it would make (smart_cut._work_plan: the frames of re-encoded pieces, and the
leading pictures an HEVC CRA piece re-encodes after a jump). Every frame the edit keeps (start <= time < end) is
compared with its source frame:
- a copied frame must be bit-exact (all planes); one that isn't is reported, and the script exits with status 1;
- a re-encoded one gets its PSNR, against the source frame faded by smartcut itself where the segment fades
  (video_cutter._fade_yuv_frame).
Several outputs of the same edit (e.g. encoder settings) are compared frame by frame against the first. 8-bit sources.
"""
import argparse
import contextlib
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

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


class _Planned(Exception):
    pass


def recoded_pts(src: str, segs: list[dict]) -> set[int]:
    """The pts of the source frames smartcut re-encodes for these segments: its own plan of the cut (smart_cut stops
    once it has made it, before writing anything)."""
    import smartcut  # noqa: PLC0415
    from smartcut.media_container import MediaContainer  # noqa: PLC0415
    sc: Any = importlib.import_module("smartcut.smart_cut")  # the module: the package re-exports its function smart_cut
    planned = {}
    real_plan = sc._work_plan

    def work_plan(media_container, cut_segments, *args, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        planned.update(pieces=cut_segments, plan=real_plan(media_container, cut_segments, *args, **kw))
        raise _Planned()

    mc = MediaContainer(src)
    fd, out = tempfile.mkstemp(suffix=Path(src).suffix or ".mp4")
    os.close(fd)
    sc._work_plan = work_plan
    try:
        cuts = sorted((smartcut.CutSegmentConfig(**seg).to_segment_with_fade(mc.duration) for seg in segs), key=lambda s: s.start_time)
        with contextlib.suppress(_Planned):
            sc.smart_cut(mc, cuts, out, None, cancel_object=sc.CancelObject())
        assert mc.video_stream is not None and mc.video_stream.time_base is not None, f"{src}: no video"
        frame_pts, time_base = mc.video_frame_times_pts, mc.video_stream.time_base
        result = set()
        for piece, busyness in zip(planned["pieces"], planned["plan"].busyness):
            first = int(np.searchsorted(frame_pts, float(piece.start_time / time_base)))
            result.update(int(p) for p in frame_pts[first:first + len(busyness)])
        return result
    finally:
        sc._work_plan = real_plan
        mc.close()
        Path(out).unlink(missing_ok=True)


def kept_frames(src: str, segs: list[dict]) -> Iterator[tuple[int, np.ndarray, list[np.ndarray] | None]]:
    """Per frame the edit keeps, in output order: its source pts, the reference luma, and the source planes (None
    where the segment fades: such frames are re-encoded)."""
    from smartcut.video_cutter import _fade_yuv_frame  # noqa: PLC0415

    with av.open(src, "r") as c:
        st = c.streams.video[0]
        assert st.time_base is not None
        start = Fraction(c.start_time or 0, 1_000_000)
        frames = c.decode(st)
        pending = None  # a frame read past one segment's end: it may start the next one
        for seg in sorted(segs, key=lambda x: seconds(x["start"])):
            s, e = seconds(seg["start"]) + start, seconds(seg["end"]) + start
            while True:
                f, pending = (pending, None) if pending is not None else (next(frames, None), None)
                if f is None:
                    break
                assert f.pts is not None
                t = f.pts * st.time_base
                if t < s:
                    continue
                if t >= e:
                    pending = f
                    break
                alpha = alpha_of(seg, t - s)
                if alpha < 1.0:
                    faded = _fade_yuv_frame(f, alpha)
                    assert faded is not None, f"{f.format.name}: not a planar YUV format smartcut fades in YUV"
                    yield f.pts, planes(faded)[0].astype(np.float64), None
                else:
                    yield f.pts, planes(f)[0].astype(np.float64), [p.copy() for p in planes(f)]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10 * np.log10(255 ** 2 / max(float(np.mean((a - b) ** 2)), 1e-10))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("segments")
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]), help="smartcut repo root")
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    segs = json.loads(args.segments)
    outputs: list[str] = args.outputs

    recoded = recoded_pts(args.source, segs)
    reencoded: dict[str, list[tuple[int, bool, float]]] = {o: [] for o in outputs}  # (frame, faded, psnr)
    copied = dict.fromkeys(outputs, 0)
    not_exact: dict[str, list[int]] = {o: [] for o in outputs}  # copied frames that aren't bit-exact
    containers = [av.open(o, "r") for o in outputs]
    decoders = [c.decode(c.streams.video[0]) for c in containers]
    n = 0
    for pts, ref, source in kept_frames(args.source, segs):
        for o, frames in zip(outputs, decoders):
            out = planes(next(frames))
            if pts in recoded:
                reencoded[o].append((n, source is None, psnr(out[0].astype(np.float64), ref)))
            else:
                assert source is not None, f"frame {n} fades but smartcut's plan copies it"
                copied[o] += 1
                if not all(np.array_equal(a, b) for a, b in zip(out, source)):
                    not_exact[o].append(n)
        n += 1
    for c in containers:
        c.close()

    print(f"{n} frames kept, {len(recoded)} of them re-encoded by smartcut's plan")
    first = None
    for o in outputs:
        r = reencoded[o]
        all_db = np.array([x[2] for x in r]) if r else np.array([np.nan])
        faded = [x[2] for x in r if x[1]]
        bad = not_exact[o]
        line = (f"  {Path(o).name:<28} copied {copied[o]:4d} ({'all bit-exact' if not bad else f'{len(bad)} NOT BIT-EXACT, first frame {bad[0]}'}), "
                f"re-encoded {len(r):4d} (faded {len(faded)}): PSNR mean {np.nanmean(all_db):5.2f} dB, min {np.nanmin(all_db):5.2f}")
        if first is None:
            first = {i: db for i, _, db in r}
        else:
            diff = [db - first[i] for i, _, db in r if i in first]
            line += f"; vs the first: {np.mean(diff):+.2f} dB mean over {len(diff)} frames"
        print(line, flush=True)
    return 1 if any(not_exact.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
