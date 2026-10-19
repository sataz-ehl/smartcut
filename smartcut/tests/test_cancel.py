#!/usr/bin/env python3
"""
Tests for process_segments() cancellation, progress and resource handling.

1. cancel_check returning True stops processing: the call returns False and no output file is left
2. Without cancellation the call succeeds and cancel_check is polled (before every cut piece)
3. The input file is closed afterwards, also after an error (on Windows an open file can't be deleted)
4. Cancelling midway through a re-encoded piece stops it within a frame, not at the end of the piece, also
   without progress_callback; cancel_check isn't asked again once it returned True, so True once is enough
5. Cancelling during the encoder's last flush (at the end of the output) stops it within a frame too, and the
   input is closed
6. Progress is reported as thousandths of the work, rising steadily to 1000, also during re-encoding

Usage (from the repo root):
    python smartcut/tests/test_cancel.py
"""
import gc
import importlib
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut
from smartcut.media_container import MediaContainer
from smartcut.smart_cut import make_adjusted_segment_times, make_cut_segments
from smartcut.tests.generate_synthetic_video import generate_test_video
from smartcut.video_cutter import VideoCutter

SEGMENTS = [{"start": 2, "end": 8, "fadein": 1, "fadeout": 1}, {"start": 12, "end": 18}]
LAST_FLUSH_SEGMENTS = [{"start": 2, "end": 8}, {"start": 12, "end": 17.5}]  # ends mid-GOP: re-encoded to the end


def test_cancel_stops_and_removes_output(tmp: Path, src: Path):
    out = tmp / "cancelled.mp4"
    calls = {"n": 0}

    def cancel_after_3():
        calls["n"] += 1
        return calls["n"] > 3

    ok = smartcut.process_segments(str(src), str(out), SEGMENTS, cancel_check=cancel_after_3)
    assert ok is False, f"expected False when cancelled, got {ok}"
    assert not out.exists(), "partial output was not removed"


def test_no_cancel_polls_and_succeeds(tmp: Path, src: Path):
    out = tmp / "complete.mp4"
    calls = {"n": 0}

    def never():
        calls["n"] += 1
        return False

    ok = smartcut.process_segments(str(src), str(out), SEGMENTS, cancel_check=never)
    assert ok is True and out.exists(), "processing without cancellation failed"
    assert calls["n"] >= 2, f"cancel_check polled only {calls['n']} times"


def test_input_closed(tmp: Path, src: Path):
    copy = tmp / "input_copy.mp4"
    copy.write_bytes(src.read_bytes())
    assert smartcut.process_segments(str(copy), str(tmp / "from_copy.mp4"), SEGMENTS)
    os.remove(copy)  # raises on Windows if process_segments left the input open


def test_input_closed_after_error(tmp: Path, src: Path) -> None:
    # The video cutter opens the input a second time; an error must not leave that open either. PyAV containers
    # are freed by the garbage collector only, which is held off here so that it can't close the file by chance.
    copy = tmp / "input_copy_error.mp4"
    copy.write_bytes(src.read_bytes())
    real_segment = VideoCutter.segment

    def failing_segment(self, cut_segment):
        raise RuntimeError("test error while cutting")

    VideoCutter.segment = failing_segment
    gc.disable()
    try:
        assert smartcut.process_segments(str(copy), str(tmp / "failed.mp4"), SEGMENTS) is False
        os.remove(copy)  # raises on Windows if the input was left open
    finally:
        gc.enable()
        VideoCutter.segment = real_segment


def count_pieces(src: Path, segments: list[dict]) -> int:
    """Cut pieces smart_cut makes for these segments; it polls cancel_check before each of them."""
    mc = MediaContainer(str(src))
    try:
        times = [(Fraction(s["start"]), Fraction(s["end"])) for s in segments]
        return len(make_cut_segments(mc, make_adjusted_segment_times(times, mc)))
    finally:
        mc.close()


def cancel_midway(tmp: Path, src: Path, with_progress: bool, once: bool = False) -> None:
    """Only polls made midway through re-encoding can cancel: cancel_check answers True from (or, with once, only
    at) the first poll after more polls than smart_cut makes outside re-encoding (one per piece and a few per
    output file). Checks between pieces alone never cancel, whatever the number of cores; the clip re-encodes
    far more frames than that."""
    out = tmp / "cancelled_midway.mp4"
    threshold = count_pieces(src, SEGMENTS) + 10
    state = {"polls": 0, "polls_after": 0, "updates": 0, "updates_after": 0, "cancelled": False}

    def check() -> bool:
        if state["cancelled"]:
            state["polls_after"] += 1
        state["polls"] += 1
        state["cancelled"] = state["cancelled"] or state["polls"] > threshold
        return state["polls"] == threshold + 1 if once else state["cancelled"]

    def on_progress(current, total):
        state["updates"] += 1
        state["updates_after"] += state["cancelled"]

    kw = {"progress_callback": on_progress} if with_progress else {}
    ok = smartcut.process_segments(str(src), str(out), SEGMENTS, cancel_check=check, **kw)
    assert ok is False and not out.exists(), (f"not cancelled (ok={ok}) after {state['polls']} polls, more than "
                                              f"{threshold} needed, or the partial output was left")
    assert state["polls_after"] == 0, f"cancel_check was asked {state['polls_after']} more times after it returned True"
    assert state["updates_after"] <= 1, f"{state['updates_after']} progress updates after cancelling: the piece kept being re-encoded"


def test_cancel_midway_through_reencoding(tmp: Path, src: Path) -> None:
    cancel_midway(tmp, src, with_progress=True)


def test_cancel_midway_without_progress(tmp: Path, src: Path) -> None:
    cancel_midway(tmp, src, with_progress=False)


def test_cancel_true_only_once(tmp: Path, src: Path) -> None:
    cancel_midway(tmp, src, with_progress=True, once=True)


def run_watching_last_flush(src: Path, out: Path, cancel_at: int | None) -> tuple[bool, dict]:
    """process_segments, counting the frames the encoder's last flush outputs (VideoCutter.finish). With cancel_at,
    cancel_check answers True from that flushed frame on."""
    sc = importlib.import_module("smartcut.smart_cut")
    real_finish, real_encoded = VideoCutter.finish, sc._WorkProgress.frames_encoded
    state = {"finishing": False, "flushed": 0, "after_cancel": 0, "cancel": False}

    def finish(self):  # noqa: ANN001, ANN202
        state["finishing"] = True
        try:
            return real_finish(self)
        finally:
            state["finishing"] = False

    def frames_encoded(self, fed: int, out_frames: int) -> None:  # noqa: ANN001
        if state["finishing"] and fed == 0:
            state["after_cancel"] += state["cancel"]
            state["flushed"] += 1
            state["cancel"] = state["cancel"] or state["flushed"] == cancel_at
        real_encoded(self, fed, out_frames)

    VideoCutter.finish, sc._WorkProgress.frames_encoded = finish, frames_encoded
    try:
        ok = smartcut.process_segments(str(src), str(out), LAST_FLUSH_SEGMENTS, cancel_check=lambda: state["cancel"])
    finally:
        VideoCutter.finish, sc._WorkProgress.frames_encoded = real_finish, real_encoded
    return ok, state


def test_cancel_during_last_flush(tmp: Path, src: Path) -> None:
    # The output ends with a re-encoded piece, so the encoder's last frames come out in finish()'s flush. A cancel
    # there used to be ignored: every frame was still encoded, then the file removed.
    ok, state = run_watching_last_flush(src, tmp / "flushed.mp4", None)
    assert ok, "the edit failed"
    assert state["flushed"] >= 6, f"the last flush output only {state['flushed']} frames: the test needs more"
    out = tmp / "cancelled_in_flush.mp4"
    copy = tmp / "input_copy_flush.mp4"  # the cancelled finish() must still close the input
    copy.write_bytes(src.read_bytes())
    gc.disable()  # PyAV containers are freed by the garbage collector only: it mustn't close the file by chance
    try:
        ok, state = run_watching_last_flush(copy, out, 3)
        os.remove(copy)  # raises on Windows if the input was left open
    finally:
        gc.enable()
    assert ok is False and not out.exists(), f"not cancelled (ok={ok}), or the partial output was left"
    assert state["flushed"] == 3 and state["after_cancel"] == 0, \
        f"the flush went on after the cancel: {state['after_cancel']} more frames ({state['flushed']} in all)"


def test_progress_steady(tmp: Path, src: Path):
    # Video fades only (with audio fades, re-encoding this clip's audio is most of the work), covering the whole
    # 6 s segment: progress counts frames as the encoder outputs them, and on a many-core PC x264's frame threads
    # plus lookahead hold back ~60 frames, so a short re-encoded run would only report at its flush
    segments = [{"start": 2, "end": 8, "video_fadein": 3, "video_fadeout": 3}, {"start": 12, "end": 18}]
    updates = []
    assert smartcut.process_segments(str(src), str(tmp / "progress.mp4"), segments,
                                     progress_callback=lambda c, n: updates.append((c, n)))
    values = [c for c, _ in updates]
    assert all(n == 1000 for _, n in updates), "total should always be 1000"
    assert values == sorted(values) and values[-1] == 1000, f"not rising to 1000: {values[:5]} ... {values[-5:]}"
    assert len(values) > 30, f"only {len(values)} updates: no progress during re-encoding"


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        src = tmp / "input.mp4"
        generate_test_video(str(src), duration=20)
        tests = [("Cancel stops processing and removes the output", test_cancel_stops_and_removes_output),
                 ("Without cancellation: succeeds, cancel_check polled", test_no_cancel_polls_and_succeeds),
                 ("Input file is closed afterwards", test_input_closed),
                 ("Input file is closed after an error too", test_input_closed_after_error),
                 ("Cancel midway through a re-encoded piece stops within a frame", test_cancel_midway_through_reencoding),
                 ("Cancel midway also works without progress_callback", test_cancel_midway_without_progress),
                 ("A cancel_check that returns True only once still cancels", test_cancel_true_only_once),
                 ("Cancel during the encoder's last flush stops it within a frame", test_cancel_during_last_flush),
                 ("Progress rises steadily to 1000, also during re-encoding", test_progress_steady)]
        failed = 0
        for name, test in tests:
            try:
                test(tmp, src)
                print(f"✓ PASS - {name}")
            except (AssertionError, OSError) as e:
                failed += 1
                print(f"✗ FAIL - {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
