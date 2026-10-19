#!/usr/bin/env python3
"""
Tests for Matroska / WebM sources: cuts keep exactly the frames they should, where they belong, into MKV, WebM and MP4.

Matroska stores no DTS, so FFmpeg gives the first packets of a file (as many as the codec's reorder delay) none, and
smartcut doesn't know where the first GOP starts in decoding order. An HEVC source with open GOPs (CRA keyframes with
RASL leading pictures) cut from inside its first GOP into the next one failed: the next GOP's leading pictures are
re-encoded, with the decoder primed from the first GOP, so from the file start, and the file's first packets were
taken for that GOP's own and copied after the cut ("pts < dts", process_segments returned False).

Segment times count from the container's start, the earliest first timestamp of any stream, as FFmpeg's -ss does. A
Matroska file keeps its audio encoder's delay at negative timestamps (AAC 1024 samples, Opus 312: FFmpeg writes it as
CodecDelay), so its container starts before the first video frame, unlike the same content in MP4, whose edit list
hides the delay. The cuts here are checked against that rule: the frames with start <= t - container start < end,
each at its own distance from the cut. A segment from the file start (0) starts the output at the first video frame.

Frames are identified by their content (a bar that grows by 3 pixels a frame), against the source frames near the one
expected.

Usage (from the repo root):
    python smartcut/tests/test_matroska_source.py
"""
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
from av import AudioStream, VideoStream

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut
from smartcut.media_container import MediaContainer

FPS = 30
W, H = 192, 108
SECONDS = 6
# Open GOPs (x265's default): CRA keyframes. Fixed runs of 3 B-frames, so every keyframe, 30 frames apart, has B-frames
# before it that reference it: RASL leading pictures, also in the second GOP (with adaptive B-frames it had none)
X265 = "keyint=30:min-keyint=30:scenecut=0:bframes=3:b-adapt=0:log-level=error"
H264 = {"g": "30", "keyint_min": "30", "sc_threshold": "0", "bf": "3"}
# name: (container, video encoder, options, audio encoder or None)
SOURCES = {
    "hevc_mkv": ("mkv", "libx265", {"x265-params": X265}, "aac"),
    "hevc_mkv_no_audio": ("mkv", "libx265", {"x265-params": X265}, None),
    "h264_mkv": ("mkv", "libx264", H264, "aac"),
    "vp9_webm": ("webm", "libvpx-vp9", {"g": "30", "cpu-used": "8", "deadline": "realtime"}, "libopus"),
}
# (name, segments, sources): HEVC cuts from inside the first GOP into the next ones (its leading pictures re-encoded,
# primed from the first GOP), also with fades and with two segments; cuts with a re-encoded first piece (the repro:
# from 1.37 s), on a keyframe, from the file start (the first GOP copied: its first packets have no DTS)
CUTS = [
    ("first GOP into the next", [{"start": 0.5, "end": 2.5}], ("hevc_mkv", "hevc_mkv_no_audio")),
    ("first GOP into the next, fades", [{"start": 0.5, "end": 3.5, "fadein": 0.3, "fadeout": 0.7}], ("hevc_mkv",)),
    ("first GOP into the next, two segments", [{"start": 0.4, "end": 2.1}, {"start": 3.2, "end": 4.5}], ("hevc_mkv",)),
    ("re-encoded first piece", [{"start": 1.37, "end": 2.5}, {"start": 3.1, "end": 4.6}], ("hevc_mkv", "h264_mkv", "vp9_webm")),
    ("re-encoded first piece, fades", [{"start": 1.37, "end": 4.6, "fadein": 1, "fadeout": 1}], ("h264_mkv", "vp9_webm")),
    ("from a keyframe", "keyframe", ("hevc_mkv", "h264_mkv", "vp9_webm")),
    ("from the file start", [{"start": 0, "end": 2.6}, {"start": 3.1, "end": 4.6}], ("hevc_mkv", "h264_mkv", "vp9_webm")),
]


def make_source(path: Path, codec: str, options: dict[str, str], audio_codec: str | None) -> None:
    with av.open(str(path), "w") as out:
        stream = cast(VideoStream, out.add_stream(codec, rate=FPS))
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.codec_context.options = dict(options)
        audio = cast(AudioStream, out.add_stream(audio_codec, rate=48000, layout="stereo")) if audio_codec else None
        audio_pts = 0
        for i in range(SECONDS * FPS):
            img = np.zeros((H, W, 3), np.uint8)
            img[:, : (i * 3) % W + 1] = (i * 7 % 256, 128, 255 - i * 5 % 256)
            img[: H // 3, (i * 3) % W:] = (255 - i * 3 % 256, i * 11 % 256, 64)  # every frame differs (the bar wraps)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(stream.encode(frame))
            if audio is not None:
                while audio_pts < (i + 1) * 48000 // FPS:
                    t = np.arange(audio_pts, audio_pts + 960) / 48000
                    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                    samples = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="fltp", layout="stereo")
                    samples.sample_rate, samples.pts = 48000, audio_pts
                    out.mux(audio.encode(samples))
                    audio_pts += 960
        out.mux(stream.encode())
        if audio is not None:
            out.mux(audio.encode())


def decode(path: Path) -> tuple[float, list[float], list[np.ndarray]]:
    """Container start (s), video frame times (s) and luma planes, in display order."""
    c = av.open(str(path))
    try:
        start = (c.start_time or 0) / av.time_base
        frames = sorted(((f.time or 0.0, f.to_ndarray(format="gray").astype(np.int16)) for f in c.decode(video=0)),
                        key=lambda x: x[0])
    finally:
        c.close()
    return start, [t for t, _ in frames], [y for _, y in frames]


def faded(segment: dict, u: float) -> bool:
    """Whether the frame u seconds into the segment is inside a video fade."""
    fin, fout = segment.get("fadein", 0), segment.get("fadeout", 0)
    return u < fin or segment["end"] - segment["start"] - u < fout


def check_cut(src: Path, out: Path, segments: list[dict], source_frames: tuple[float, list[float], list[np.ndarray]]) -> str | None:
    """None if the output has exactly the frames of the segments (counted from the source's container start), each
    where it belongs (within 1.5 ms: Matroska's timestamps are milliseconds), else a message."""
    if not smartcut.process_segments(str(src), str(out), [dict(s) for s in segments]):
        return "process_segments failed"
    s_start, s_t, s_y = source_frames
    expected, positions, fade = [], [], []
    offset = 0.0
    for s in segments:
        kept = [i for i, t in enumerate(s_t) if s_start + s["start"] - 1e-9 <= t < s_start + s["end"] - 1e-9]
        # A segment from the file start starts where the video does: its first piece is the first GOP, from its first
        # frame
        start = s_t[0] if s["start"] <= 1e-6 else s_start + s["start"]
        expected += kept
        positions += [s_t[i] - start + offset for i in kept]
        fade += [faded(s, s_t[i] - (s_start + s["start"])) for i in kept]
        offset += s_start + s["end"] - start
    _, o_t, o_y = decode(out)
    if len(o_t) != len(expected):
        return f"{len(o_t)} frames decoded, expected {len(expected)}"
    for k, (i, y) in enumerate(zip(expected, o_y)):
        if abs(o_t[k] - positions[k]) > 0.0015:
            return f"output frame {k} at {o_t[k]:.4f} s, expected {positions[k]:.4f} s"
        if fade[k]:
            continue  # a faded frame is darker than every source frame
        near = range(max(0, i - 6), min(len(s_y), i + 7))
        j = near[int(np.argmin([np.abs(y - s_y[n]).mean() for n in near]))]
        if j != i:
            return f"output frame {k} is source frame {j}, expected {i}"
    return None


def first_keyframe_after(src: Path, t: float) -> float:
    """The first keyframe at or after t seconds after the container start, in the same terms."""
    c = MediaContainer(str(src))
    try:
        return float(next(k for k in c.gop_start_times_pts_s if k - c.start_time >= t) - c.start_time)
    finally:
        c.close()


def check_source(name: str, src: Path) -> str | None:
    """The sources must be what the tests are about: HEVC with RASL pictures, audio delay at negative timestamps."""
    c = MediaContainer(str(src))
    try:
        if name.startswith("hevc") and not (len(c.gop_has_rasl) > 1 and c.gop_has_rasl[1]):
            return "no RASL pictures in the second GOP of the HEVC source (open GOPs expected)"
        if SOURCES[name][3] is not None and not c.start_time < 0:
            return f"container start {float(c.start_time):.4f} s: the audio encoder delay should be at negative timestamps"
    finally:
        c.close()
    return None


def main() -> int:
    failed = total = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        sources = {}
        for name, (container, codec, options, audio_codec) in SOURCES.items():
            src = tmp / f"src_{name}.{container}"
            make_source(src, codec, options, audio_codec)
            total += 1
            problem = check_source(name, src)
            if problem:
                failed += 1
                print(f"✗ FAIL - source {name}: {problem}")
            sources[name] = (src, decode(src))
        for cut, segments, names in CUTS:
            for name in names:
                src, frames = sources[name]
                segs = segments
                if segs == "keyframe":
                    k = first_keyframe_after(src, 1.2)
                    segs = [{"start": k, "end": k + 1.4}, {"start": k + 1.6, "end": k + 2.9}]
                own = "webm" if name.endswith("webm") else "mkv"
                for ext in (own, "mp4"):
                    total += 1
                    problem = check_cut(src, tmp / f"out_{name}_{len(segs)}_{total}.{ext}", segs, frames)
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

