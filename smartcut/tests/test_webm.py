#!/usr/bin/env python3
"""
Tests .webm outputs. WebM holds only Opus and Vorbis audio, so other audio (e.g. AAC) is re-encoded to Opus as a
whole track, with or without audio fades; Opus audio is copied when there are no audio fades.

Checks: the output decodes completely, the audio track is Opus at 48 kHz and as long as the video, fades have the
right gain, and copied Opus packets are byte-identical to the source's. Sources (VP9 + AAC, VP9 + Opus) are made
with PyAV; HE-AAC sources are in the regression harness (cases W1-W6 in regression/run_cases.py).

Usage (from the repo root):
    python smartcut/tests/test_webm.py
"""
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut

FPS, SECONDS = 30, 6


def make_source(path: Path, audio_codec: str) -> None:
    """VP9 video and a 440 Hz tone, keyframe every second."""
    rate = 48000
    with av.open(str(path), "w") as out:
        video = out.add_stream("libvpx-vp9", rate=FPS)
        video.width, video.height, video.pix_fmt = 160, 96, "yuv420p"
        video.codec_context.options = {"g": "30", "cpu-used": "8", "deadline": "realtime"}
        audio = out.add_stream(audio_codec, rate=rate, layout="stereo")
        n = rate // FPS
        for i in range(SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(np.full((96, 160, 3), 40 + i % 150, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(video.encode(frame))
            t = np.arange(i * n, (i + 1) * n) / rate
            samples = av.AudioFrame.from_ndarray(np.tile((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), (2, 1)),
                                                 format="fltp", layout="stereo")
            samples.sample_rate, samples.pts = rate, i * n
            out.mux(audio.encode(samples))
        out.mux(video.encode())
        out.mux(audio.encode())


def audio_of(path: Path) -> tuple[str, int, np.ndarray, list[bytes]]:
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        name, rate = s.codec_context.name, s.sample_rate
        packets = [bytes(p) for p in c.demux(s) if p.size]
    with av.open(str(path)) as c:
        x = np.concatenate([f.to_ndarray().astype(np.float64).mean(axis=0) for f in c.decode(audio=0)])
    return name, rate, x, packets


def check(src: Path, out: Path, fades: bool, copied: bool) -> str | None:
    segment = {"start": 1, "end": 4, **({"audio_fadein": 1, "audio_fadeout": 1} if fades else {})}
    if not smartcut.process_segments(str(src), str(out), [segment]):
        return "process_segments failed"
    with av.open(str(out)) as c:
        frames = sum(1 for _ in c.decode(video=0))
    if frames != 3 * FPS:
        return f"{frames} video frames decoded, expected {3 * FPS}"
    name, rate, x, packets = audio_of(out)
    if name != "opus" or rate != 48000:
        return f"audio {name} at {rate} Hz, expected Opus at 48000 Hz"
    if abs(len(x) / rate - 3) > 0.06:
        return f"{len(x) / rate:.3f} s of audio for 3 s of video"
    level = float(np.sqrt((x[int(1.4 * rate):int(1.6 * rate)] ** 2).mean())) / (0.3 / np.sqrt(2))  # middle, unfaded
    if not 0.85 < level < 1.15:
        return f"level {level:.2f} of the source in the middle"
    if fades:
        got = float(np.sqrt((x[int(0.48 * rate):int(0.52 * rate)] ** 2).mean())) / (0.3 / np.sqrt(2))
        if abs(got - 0.5) > 0.08:
            return f"gain {got:.2f} halfway through the fade-in, expected 0.5"
    source_packets = set(audio_of(src)[3])
    same = sum(p in source_packets for p in packets)
    if copied and same != len(packets):
        return f"only {same}/{len(packets)} audio packets copied from the source"
    if not copied and same:
        return f"{same} audio packets copied, expected a re-encode"
    return None


def main() -> int:
    failed = total = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        sources = {"aac": tmp / "vp9_aac.mp4", "libopus": tmp / "vp9_opus.webm"}
        for codec, path in sources.items():
            make_source(path, codec)
        cases = [("AAC -> Opus, no fades", "aac", False, False), ("AAC -> Opus, audio fades", "aac", True, False),
                 ("Opus copied, no fades", "libopus", False, True), ("Opus re-encoded, audio fades", "libopus", True, False)]
        for i, (name, codec, fades, copied) in enumerate(cases):
            total += 1
            problem = check(sources[codec], tmp / f"out{i}.webm", fades, copied)
            if problem:
                failed += 1
                print(f"✗ FAIL - {name}: {problem}")
            else:
                print(f"✓ PASS - {name}")
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
