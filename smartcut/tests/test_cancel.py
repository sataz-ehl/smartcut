#!/usr/bin/env python3
"""
Tests for process_segments() cancellation and resource handling.

1. cancel_check returning True stops processing: the call returns False and no output file is left
2. Without cancellation the call succeeds and cancel_check is polled (before every cut piece)
3. The input file is closed afterwards (on Windows an open file can't be deleted)

Usage (from the repo root):
    python smartcut/tests/test_cancel.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut  # noqa: E402
from smartcut.tests.generate_synthetic_video import generate_test_video  # noqa: E402

SEGMENTS = [{"start": 2, "end": 8, "fadein": 1, "fadeout": 1}, {"start": 12, "end": 18}]


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


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        src = tmp / "input.mp4"
        generate_test_video(str(src), duration=20)
        tests = [("Cancel stops processing and removes the output", test_cancel_stops_and_removes_output),
                 ("Without cancellation: succeeds, cancel_check polled", test_no_cancel_polls_and_succeeds),
                 ("Input file is closed afterwards", test_input_closed)]
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
