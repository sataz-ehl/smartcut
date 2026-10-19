#!/usr/bin/env python3
"""
Unit tests of smart_cut._WorkProgress, the progress estimate (no encoding):

1. A frame's cost: (fixed + MP x per_mp) x (1 - busy + busy x busyness) + MP x fade if it fades; copied frames and
   re-encoded audio count when their piece is done
2. Frames count when the encoder outputs them, in the order they were fed (first in, first out); feeding alone
   doesn't move the progress
3. Frames fed beyond a piece's estimate cost the average frame, and the re-encoding never counts more than its
   estimate in all
4. Frames short of the estimate leave the progress below 1 until done() reports 1.0
5. An encoder that outputs more frames than were counted as fed doesn't break the count
6. Progress never goes back or above 1.0, and ends at 1.0
7. A set cancel object stops a frame fed or flushed, with or without a progress object

Usage (from the repo root):
    python smartcut/tests/test_work_progress.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

from smartcut import progress_costs
from smartcut.smart_cut import CancelObject, _Cancelled, _WorkPlan, _WorkProgress

# One unit per re-encoded frame at busyness 1 (megapixels 1, busy 1: the cost is the frame's busyness), a copied frame
# 0.01, a second of audio 0.1
COSTS = progress_costs.ProgressCosts({"h264": (0.0, 1.0, 1.0, 0.5)}, copy_per_frame=0.01, audio_per_second=0.1)


class Recorder:
    def __init__(self) -> None:
        self.values: list[float] = []

    def emit(self, value: int) -> None:
        pass

    def emit_fraction(self, fraction: float) -> None:
        self.values.append(fraction)


class Cancel(CancelObject):
    def __init__(self) -> None:
        self.cancelled = False


def work(busyness: list[list[float]], *, copied: list[int] | None = None, audio: list[float] | None = None,
         faded: list[list[bool]] | None = None, costs: progress_costs.ProgressCosts = COSTS,
         cancel: CancelObject | None = None, progress: bool = True) -> tuple[_WorkProgress, Recorder]:
    n = len(busyness)
    plan = _WorkPlan("h264", 1.0, [np.array(b, float) for b in busyness],
                        [np.array(f, bool) for f in (faded or [[False] * len(b) for b in busyness])],
                        copied or [0] * n, audio or [0.0] * n)
    recorder = Recorder()
    return _WorkProgress(recorder if progress else None, cancel, plan, costs), recorder


def close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def test_costs() -> None:
    costs = progress_costs.ProgressCosts({"h264": (0.01, 0.02, 0.5, 0.03)}, copy_per_frame=0.001, audio_per_second=0.2)
    plan = _WorkPlan("h264", 2.0, [np.array([1.0, 3.0])], [np.array([False, True])], [100], [5.0])
    w = _WorkProgress(None, None, plan, costs)
    base = 0.01 + 0.02 * 2.0
    assert np.allclose(w.frame_costs[0], [base * (0.5 + 0.5 * 1.0), base * (0.5 + 0.5 * 3.0) + 2.0 * 0.03]), w.frame_costs
    assert close(w.piece_weights[0], 100 * 0.001 + 5.0 * 0.2), w.piece_weights
    assert close(w.total, float(w.frame_costs[0].sum()) + w.piece_weights[0])


def test_fifo() -> None:
    w, r = work([[1.0, 2.0, 3.0]], copied=[0])
    total = w.total
    assert close(total, 6.0)
    w.piece_started(0)
    for _ in range(3):
        w.frames_encoded(1, 0)  # taken in, nothing out yet (lookahead)
    assert r.values == [], "feeding alone must not move the progress"
    w.frames_encoded(0, 1)
    assert close(r.values[-1], 1 / 6), f"the first frame fed comes out first: {r.values}"
    w.frames_encoded(0, 1)
    assert close(r.values[-1], 3 / 6), r.values
    w.frames_encoded(0, 1)
    assert close(r.values[-1], 1.0), r.values


def test_beyond_estimate() -> None:
    w, r = work([[1.0, 1.0], [1.0, 1.0]], copied=[0, 10])  # piece 1 copies 10 frames (0.1) besides
    w.piece_started(0)
    for _ in range(3):  # one frame more than estimated: the average frame's cost
        w.frames_encoded(1, 1)
    assert close(w.video_done, 3.0), w.video_done
    w.piece_done(0)
    w.piece_started(1)
    for _ in range(3):
        w.frames_encoded(1, 1)
    assert close(w.video_done, 6.0), w.video_done
    assert close(r.values[-1], 4.0 / 4.1), f"re-encoding counts its estimate at most: {r.values[-1]}"
    w.piece_done(1)
    assert close(r.values[-1], 1.0) and max(r.values) <= 1.0, r.values


def test_short_of_estimate() -> None:
    w, r = work([[1.0] * 4], copied=[0], audio=[10.0])  # 4 + 1 (audio)
    w.piece_started(0)
    for _ in range(2):
        w.frames_encoded(1, 1)
    w.piece_done(0)
    assert close(r.values[-1], 3.0 / 5.0), r.values
    w.done()
    assert r.values[-1] == 1.0


def test_more_out_than_fed() -> None:
    w, _ = work([[1.0, 1.0]])
    w.piece_started(0)
    w.frames_encoded(1, 0)
    w.frames_encoded(0, 5)  # an encoder can't, but the count must not fail
    assert close(w.video_done, 1.0) and not w.queued


def test_monotonic() -> None:
    rng = np.random.default_rng(1)
    busyness = [list(rng.uniform(0.3, 3.0, k)) for k in (5, 0, 12, 0, 7)]
    w, r = work(busyness, copied=[0, 300, 0, 150, 0], audio=[1.0, 2.0, 1.0, 2.0, 1.0])
    for i, b in enumerate(busyness):
        w.piece_started(i)
        for _ in range(len(b) + (1 if i == 2 else 0)):
            w.frames_encoded(1, int(rng.integers(0, 2)))
        w.frames_encoded(0, 1)
        w.piece_done(i)
    while w.queued:
        w.frames_encoded(0, 1)
    w.done()
    v = r.values
    assert all(b >= a for a, b in itertools.pairwise(v)), "progress went back"
    assert max(v) <= 1.0 and v[-1] == 1.0, v[-5:]


def test_cancel() -> None:
    for progress in (True, False):
        cancel = Cancel()
        w, _ = work([[1.0] * 3], cancel=cancel, progress=progress)
        w.piece_started(0)
        w.frames_encoded(1, 0)
        cancel.cancelled = True
        for fed, out in ((1, 0), (0, 1)):  # feeding, and a flush (also the last one, in finish())
            try:
                w.frames_encoded(fed, out)
                raise AssertionError(f"not cancelled at ({fed}, {out}), progress {progress}")
            except _Cancelled:
                pass


def main() -> int:
    tests = [("A frame's cost: busyness, fade; copies and audio per piece", test_costs),
             ("Frames count as output, first in first out; feeding alone doesn't", test_fifo),
             ("Frames beyond the estimate: the average cost, the total capped", test_beyond_estimate),
             ("Frames short of the estimate: below 1 until done()", test_short_of_estimate),
             ("More frames output than fed doesn't break the count", test_more_out_than_fed),
             ("Never back, never above 1.0, ends at 1.0", test_monotonic),
             ("A set cancel object stops a frame fed or flushed", test_cancel)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"✓ PASS - {name}")
        except Exception as e:  # a mutant can raise anything
            failed += 1
            print(f"✗ FAIL - {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
