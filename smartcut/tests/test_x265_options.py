#!/usr/bin/env python3
"""
Unit tests of the x265 options VideoCutter.init_encoder gives HEVC re-encodes (no encoding):

1. The source's x265 options (from its headers) are reused, without the encoding machine's threading options
   (frame-threads, numa-pools, pools, lookahead-threads); x265's own syntax is rewritten (a:b -> a,b, flag -> flag=1)
2. Smart cuts get a lookahead of at most 10 frames: the source's when shorter, and bframes + 1 for 10 or more
   B-frames (x265 refuses a lookahead that isn't longer than its B-frame runs)
3. Sources without x265 options (camera footage) get the cap too, and log level error
4. An option that isn't a number where one is expected leaves the lookahead alone (no error)
5. Full re-encodes (RECODE mode) keep the source's lookahead; lossless and the log level are passed on

Usage (from the repo root):
    python smartcut/tests/test_x265_options.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

from smartcut.media_utils import VideoExportMode, VideoExportQuality
from smartcut.video_cutter import VideoCutter, VideoSettings

# The start of a real source's options (x265 3.5 on a 6-core / 12-thread PC, -preset fast)
REAL = ("cpuid=1111039 frame-threads=3 numa-pools=12 wpp no-pmode log-level=0 bitdepth=8 input-res=640x360 open-gop "
        "keyint=150 bframes=4 b-pyramid rc-lookahead=15 lookahead-slices=0 scenecut=40 deblock=0:0 sao crf=26.0 "
        "lookahead-threads=2 pools=4")


def params(options: str | None, mode: VideoExportMode = VideoExportMode.SMARTCUT,
           quality: VideoExportQuality = VideoExportQuality.NORMAL, log_level: str | None = None) -> list[str]:
    """The x265-params init_encoder sets for a source with these options in its headers (None: no x265 info SEI)."""
    header = b"\x00\x00\x01\x4e\x01x265 (build 199) - 3.5"
    extradata = header + b"\x00" if options is None else header + b" - options: " + options.encode() + b"\x00"
    cutter = cast(Any, VideoCutter.__new__(VideoCutter))  # only what init_encoder reads
    cutter.codec_name = "hevc"
    cutter.out_stream = SimpleNamespace(codec_context=SimpleNamespace(profile=None))
    cutter.in_stream = SimpleNamespace(codec_context=SimpleNamespace(width=1280, height=720, extradata=extradata))
    cutter.video_settings = VideoSettings(mode, quality)
    cutter.log_level = log_level
    cutter.init_encoder()
    return cutter.encoding_options["x265-params"].split(":")


def values(ps: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in ps:
        name, _, value = p.partition("=")
        out.setdefault(name, []).append(value)
    return out


def test_reused_without_threads() -> None:
    v = values(params(REAL))
    for name in ("frame-threads", "numa-pools", "pools", "lookahead-threads"):
        assert name not in v, f"{name} is the encoding machine's: {v.get(name)}"
    assert v["bframes"] == ["4"] and v["keyint"] == ["150"] and v["crf"] == ["26.0"], v
    assert v["deblock"] == ["0,0"] and v["wpp"] == ["1"] and v["open-gop"] == ["1"], "x265's syntax rewritten"
    assert v["repeat-headers"] == ["1"] and v["info"] == ["0"] and v["log_level"] == ["error"], v


def test_lookahead() -> None:
    for options, expected in ((REAL.replace("rc-lookahead=15", "rc-lookahead=20"), "10"),
                              (REAL, "10"),
                              (REAL.replace("rc-lookahead=15", "rc-lookahead=8"), "8"),
                              (REAL.replace("bframes=4", "bframes=12").replace("rc-lookahead=15", "rc-lookahead=20"), "13"),
                              (REAL.replace(" rc-lookahead=15", ""), "10")):  # x265's default 20
        v = values(params(options))
        assert v["rc-lookahead"] == [expected], f"{options[60:140]}...: {v['rc-lookahead']}, expected {expected}"


def test_no_options() -> None:
    ps = params(None)
    assert ps == ["repeat-headers=1", "info=0", "log_level=error", "rc-lookahead=10"], ps


def test_not_a_number() -> None:
    v = values(params(REAL.replace("rc-lookahead=15", "rc-lookahead=x")))
    assert v["rc-lookahead"] == ["x"], v["rc-lookahead"]
    v = values(params(REAL.replace("bframes=4", "bframes=many")))
    assert v["rc-lookahead"] == ["15"], v["rc-lookahead"]


def test_recode_and_settings() -> None:
    v = values(params(REAL, mode=VideoExportMode.RECODE))
    assert v["rc-lookahead"] == ["15"], "a full re-encode keeps the source's lookahead"
    assert "frame-threads" not in v
    v = values(params(REAL, quality=VideoExportQuality.LOSSLESS, log_level="info"))
    assert v["lossless"] == ["1"] and v["log_level"] == ["info"], v


def main() -> int:
    tests = [("The source's options reused without its machine's threads", test_reused_without_threads),
             ("Lookahead at most 10, the source's when shorter, bframes + 1 when needed", test_lookahead),
             ("No x265 options in the source: the cap and log level error", test_no_options),
             ("Not a number: the lookahead is left alone", test_not_a_number),
             ("RECODE keeps the lookahead; lossless and log level passed on", test_recode_and_settings)]
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
