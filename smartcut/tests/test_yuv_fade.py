#!/usr/bin/env python3
"""
Unit tests for the YUV video fade (smartcut.video_cutter._fade_yuv_frame). No media needed.

Checks the exact output values for 8-bit limited/full range, 4:4:4 and 10-bit frames (with line padding),
that pts/time_base are kept (also when time_base is None, as for decoded frames), that the source frame is
left unchanged, and that unsupported pixel formats return None (RGB fallback).

Usage (from the repo root):
    python smartcut/tests/test_yuv_fade.py
"""
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)
from av import VideoFrame  # noqa: E402

from smartcut.video_cutter import _fade_yuv_frame  # noqa: E402

W, H = 642, 362  # sizes that give the planes line padding


def make_frame(fmt, y_val, c_val, bits, with_timing=True):
    frame = VideoFrame(W, H, fmt)
    dtype = np.dtype(np.uint8) if bits == 8 else np.dtype("<u2")
    for i, plane in enumerate(frame.planes):
        plane.update(np.full((plane.height, plane.line_size // dtype.itemsize), y_val if i == 0 else c_val, dtype))
    if with_timing:
        frame.pts, frame.time_base = 1234, Fraction(1, 90000)
    return frame


def plane_ranges(frame, bits):
    dtype = np.dtype(np.uint8) if bits == 8 else np.dtype("<u2")
    ranges = []
    for plane in frame.planes:
        arr = np.frombuffer(plane, dtype).reshape(plane.height, plane.line_size // dtype.itemsize)[:, : plane.width]
        ranges.append((int(arr.min()), int(arr.max())))
    return ranges


def test_values():
    cases = [  # format, bits, Y, C, alpha, black level
        ("yuv420p", 8, 200, 60, 0.5, 16), ("yuvj420p", 8, 200, 60, 0.5, 0), ("yuv444p", 8, 235, 240, 0.25, 16),
        ("yuv420p10le", 10, 800, 300, 0.5, 64), ("yuv422p10le", 10, 940, 64, 0.0, 64),
    ]
    for fmt, bits, y, c, alpha, black in cases:
        src = make_frame(fmt, y, c, bits)
        before = plane_ranges(src, bits)
        out = _fade_yuv_frame(src, alpha)
        neutral = 1 << (bits - 1)
        exp_y, exp_c = round(black + (y - black) * alpha), round(neutral + (c - neutral) * alpha)
        got = plane_ranges(out, bits)
        assert got[0] == (exp_y, exp_y), f"{fmt}: luma {got[0]} != {exp_y}"
        assert all(g == (exp_c, exp_c) for g in got[1:]), f"{fmt}: chroma {got[1:]} != {exp_c}"
        assert out.pts == 1234 and out.time_base == Fraction(1, 90000), f"{fmt}: timing not kept"
        assert out.format.name == fmt, f"{fmt}: format changed to {out.format.name}"
        assert plane_ranges(src, bits) == before, f"{fmt}: source frame was modified"


def test_frame_without_time_base():
    out = _fade_yuv_frame(make_frame("yuv420p", 200, 60, 8, with_timing=False), 0.5)
    assert out is not None and out.time_base is None


def test_unsupported_format_falls_back():
    assert _fade_yuv_frame(VideoFrame(64, 64, "nv12"), 0.5) is None


def main():
    tests = [("Exact values for 8/10-bit, limited/full range, 4:2:0/4:2:2/4:4:4", test_values),
             ("Frame without time_base (decoded frames)", test_frame_without_time_base),
             ("Unsupported pixel format returns None (RGB fallback)", test_unsupported_format_falls_back)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"✓ PASS - {name}")
        except AssertionError as e:
            failed += 1
            print(f"✗ FAIL - {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
