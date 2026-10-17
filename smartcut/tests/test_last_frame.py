#!/usr/bin/env python3
"""
Tests that a cut keeps its last frame, also when the output ends with re-encoded frames, and that an input without
audio can be cut.

Encoders give their packets no duration (x264, libvpx) or one of a single tick (libaom without a frame rate). When
no copied packet came first, the output's last packet kept that duration: the video stream ended one frame early,
and in MP4 with B-frames the edit list then ended before the last frame, so FFmpeg and players dropped it.

Frames are counted by decoding the whole output with PyAV, which drops the frames outside the MP4 edit list as
FFmpeg does (the same count as ffprobe -count_frames), and the stream duration must cover the last frame.

Usage (from the repo root):
    python smartcut/tests/test_last_frame.py
"""
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut

FPS = 30
H264 = {"g": "30", "keyint_min": "30", "sc_threshold": "0", "bf": "3"}
# (encoder, options, with audio): H.264 with and without B-frames, VP9, AV1; a keyframe every second
SOURCES = {
    "h264": ("libx264", H264, True),
    "h264_nob": ("libx264", {**H264, "bf": "0"}, True),
    "vp9": ("libvpx-vp9", {"g": "30", "cpu-used": "8", "deadline": "realtime"}, True),
    "av1": ("libaom-av1", {"g": "30", "cpu-used": "8", "usage": "realtime"}, True),
    "h264_no_audio": ("libx264", H264, False),
}
# (name, segments): a cut inside one GOP (re-encoded as a whole), fades re-encoding every GOP, a cut that ends
# with copied GOPs, and a cut that ends with re-encoded frames after copied ones
CUTS = [
    ("inside one GOP", [{"start": 0.3, "end": 0.7}]),
    ("short, from 1 s", [{"start": 1.0, "end": 1.5}]),
    ("fades on every GOP", [{"start": 0.5, "end": 2.5, "fadein": 1, "fadeout": 1}]),
    ("ends on a keyframe", [{"start": 0.5, "end": 3.0}]),
    ("ends mid-GOP", [{"start": 0.5, "end": 3.4}, {"start": 3.6, "end": 3.9}]),
]


def make_source(path: Path, codec: str, options: dict[str, str], with_audio: bool, seconds: int = 4) -> None:
    with av.open(str(path), "w") as out:
        stream = out.add_stream(codec, rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 192, 108, "yuv420p"
        stream.codec_context.options = dict(options)
        audio = out.add_stream("aac", rate=48000, layout="mono") if with_audio else None
        for i in range(seconds * FPS):
            img = np.zeros((108, 192, 3), np.uint8)
            img[:, : (i * 3) % 192 + 1] = (i * 7 % 256, 128, 255 - i * 5 % 256)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, FPS)
            out.mux(stream.encode(frame))
            if audio is not None:
                t = np.arange(i * 1600, (i + 1) * 1600) / 48000
                samples = av.AudioFrame.from_ndarray((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :],
                                                     format="fltp", layout="mono")
                samples.sample_rate, samples.pts = 48000, i * 1600
                out.mux(audio.encode(samples))
        out.mux(stream.encode())
        if audio is not None:
            out.mux(audio.encode())


def frame_times(path: Path) -> list[float]:
    with av.open(str(path)) as c:
        return [float(f.pts * f.time_base) for f in c.decode(video=0)]


def check_cut(src: Path, out: Path, segments: list[dict]) -> str | None:
    """None if the output has every frame of the segments and its stream covers the last one, else a message."""
    if not smartcut.process_segments(str(src), str(out), segments):
        return "process_segments failed"
    times = frame_times(src)
    expected = sum(sum(1 for t in times if s["start"] - 1e-9 <= t < s["end"] - 1e-9) for s in segments)
    got = len(frame_times(out))
    if got != expected:
        return f"{got} frames decoded, expected {expected}"
    with av.open(str(out)) as c:
        vs = c.streams.video[0]
        if vs.duration is None or vs.time_base is None:
            return None  # e.g. Matroska: no per-stream duration
        stream_s = float(vs.duration * vs.time_base)
    if abs(stream_s - expected / FPS) > 0.5 / FPS:
        return f"video stream lasts {stream_s:.4f} s, expected {expected / FPS:.4f} s for {expected} frames"
    return None


def main() -> int:
    failed = total = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name, (codec, options, with_audio) in SOURCES.items():
            src = tmp / f"src_{name}.mp4"
            make_source(src, codec, options, with_audio)
            for ext in ("mp4", "mkv") if name == "h264" else ("mp4",):
                for i, (cut, segments) in enumerate(CUTS if with_audio else CUTS[:1]):
                    total += 1
                    problem = check_cut(src, tmp / f"out_{name}_{i}.{ext}", segments)
                    label = f"{name} -> {ext}, {cut}"
                    if problem:
                        failed += 1
                        print(f"✗ FAIL - {label}: {problem}")
                    else:
                        print(f"✓ PASS - {label}")
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
