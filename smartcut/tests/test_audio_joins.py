#!/usr/bin/env python3
"""
Tests that cut audio has no gaps or ramps at the joins and at the start, for codecs whose copied packets don't
decode cleanly after a jump:
- MP3: a frame can take bits from earlier frames (the bit reservoir), so copied MP3 had ~50 ms of silence after
  every join. MP3 tracks are now re-encoded as a whole when the output has a join.
- Opus: decoders need 80 ms of pre-roll, so a copy that started mid-stream rose from ~40 % of the level over its
  first 80 ms. The copy now includes 80 ms of earlier packets, skipped through the stream's pre-skip.

The level is measured in 10 ms windows against the source's steady level. Sources (H.264 video + a 440 Hz tone)
are made with PyAV, no media needed.

Usage (from the repo root):
    python smartcut/tests/test_audio_joins.py
"""
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut

FPS, SECONDS, AMPLITUDE = 30, 6, 0.3
SEGMENTS = [{"start": 1, "end": 3}, {"start": 3.5, "end": 5}]  # a start mid-stream and a join at 2 s


def make_source(path: Path, video_codec: str, audio_codec: str, rate: int) -> None:
    with av.open(str(path), "w") as out:
        video = out.add_stream(video_codec, rate=FPS)
        video.width, video.height, video.pix_fmt = 160, 96, "yuv420p"
        video.codec_context.options = {"g": "30"} if video_codec == "libx264" else {"g": "30", "cpu-used": "8", "deadline": "realtime"}
        audio = out.add_stream(audio_codec, rate=rate, layout="mono")
        n = rate // FPS
        for i in range(SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(np.full((96, 160, 3), 40 + i % 150, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(video.encode(frame))
            t = np.arange(i * n, (i + 1) * n) / rate
            samples = av.AudioFrame.from_ndarray((AMPLITUDE * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :],
                                                 format="fltp", layout="mono")
            samples.sample_rate, samples.pts = rate, i * n
            out.mux(audio.encode(samples))
        out.mux(video.encode())
        out.mux(audio.encode())


def check(src: Path, out: Path, segments: list[dict], from_s: float) -> str | None:
    """None if every 10 ms window from from_s to 50 ms before the end is at the source's level, else a message."""
    if not smartcut.process_segments(str(src), str(out), segments):
        return "process_segments failed"
    with av.open(str(out)) as c:
        rate = c.streams.audio[0].sample_rate
        x = np.concatenate([f.to_ndarray().astype(np.float64).mean(axis=0) for f in c.decode(audio=0)])
    expected = sum(s["end"] - s["start"] for s in segments)
    if abs(len(x) / rate - expected) > 0.06:
        return f"{len(x) / rate:.3f} s of audio, expected {expected} s"
    win = rate // 100
    level = np.array([np.sqrt((x[i:i + win] ** 2).mean()) for i in range(0, len(x) - win, win)]) / (AMPLITUDE / np.sqrt(2))
    first, last = int(from_s * 100), len(level) - 5
    low = [(round(k / 100, 2), round(float(level[k]), 2)) for k in range(first, last) if level[k] < 0.85]
    return f"level below 0.85 at (s, level) {low[:6]}" if low else None


def main() -> int:
    cases = [  # name, video encoder, audio encoder, rate, container, check from (s)
        ("MP3 in MP4: start and join", "libx264", "libmp3lame", 44100, "mp4", 0.0),
        ("Opus in WebM: start", "libvpx-vp9", "libopus", 48000, "webm", 0.0),
        ("Opus in MP4: start", "libvpx-vp9", "libopus", 48000, "mp4", 0.0),
    ]
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        for i, (name, vcodec, acodec, rate, ext, from_s) in enumerate(cases):
            src = Path(d) / f"src{i}.{ext}"
            make_source(src, vcodec, acodec, rate)
            problem = check(src, Path(d) / f"out{i}.{ext}", SEGMENTS, from_s)
            if problem:
                failed += 1
                print(f"✗ FAIL - {name}: {problem}")
            else:
                print(f"✓ PASS - {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
