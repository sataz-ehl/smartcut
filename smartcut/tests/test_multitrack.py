#!/usr/bin/env python3
"""
Tests that process_segments() keeps every audio track: copied when the copy is exact (no fades, no join), each
re-encoded as a whole with audio fades or a join, and each one as long as the video and faded like the first.

Source: video + AAC 440 Hz (48 kHz stereo) + AC-3 880 Hz (44.1 kHz mono), made with PyAV (no media needed).
The regression harness has more (HE-AAC and MP3 second tracks): cases M1-M4 in regression/run_cases.py.

Usage (from the repo root):
    python smartcut/tests/test_multitrack.py
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
TRACKS = [("aac", 48000, "stereo", 440), ("ac3", 44100, "mono", 880)]  # codec, rate, layout, tone Hz


def make_source(path: Path) -> None:
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=FPS)
        video.width, video.height, video.pix_fmt = 192, 108, "yuv420p"
        video.codec_context.options = {"g": "30", "keyint_min": "30", "sc_threshold": "0"}
        audio = [out.add_stream(codec, rate=rate, layout=layout) for codec, rate, layout, _ in TRACKS]
        for i in range(SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(np.full((108, 192, 3), i % 200, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(video.encode(frame))
            for stream, (_, rate, layout, hz) in zip(audio, TRACKS):
                n = rate // FPS
                t = np.arange(i * n, (i + 1) * n) / rate
                channels = 2 if layout == "stereo" else 1
                samples = np.tile((0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32), (channels, 1))
                a = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
                a.sample_rate, a.pts = rate, i * n
                out.mux(stream.encode(a))
        out.mux(video.encode())
        for stream in audio:
            out.mux(stream.encode())


def read_tracks(path: Path) -> tuple[float, list[tuple[str, int, np.ndarray]]]:
    """Video duration and per audio track: codec, sample rate, mono samples."""
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        video_s = float(vs.duration * vs.time_base)
        tracks = []
        for j, s in enumerate(c.streams.audio):
            with av.open(str(path)) as c2:
                x = np.concatenate([f.to_ndarray().astype(np.float64).mean(axis=0) for f in c2.decode(c2.streams.audio[j])])
            tracks.append((s.codec_context.name, s.sample_rate, x))
    return video_s, tracks


def packets_of(path: Path) -> list[set[bytes]]:
    with av.open(str(path)) as c:
        found: list[set[bytes]] = [set() for _ in c.streams.audio]
        for p in c.demux(*c.streams.audio):
            if p.size:
                found[list(c.streams.audio).index(p.stream)].add(bytes(p))
    return found


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


def check(src: Path, out: Path, segments: list[dict], fades: bool, copied: bool) -> str | None:
    if not smartcut.process_segments(str(src), str(out), segments):
        return "process_segments failed"
    _, src_tracks = read_tracks(src)
    video_s, tracks = read_tracks(out)
    if len(tracks) != len(TRACKS):
        return f"{len(tracks)} audio tracks, expected {len(TRACKS)}"
    for k, (out_packets, src_packets) in enumerate(zip(packets_of(out), packets_of(src)), start=1):
        same = len(out_packets & src_packets)
        if copied and same < len(out_packets):
            return f"track {k}: {same}/{len(out_packets)} packets copied, expected a copy"
        if not copied and same > 2:
            return f"track {k}: {same}/{len(out_packets)} packets copied, expected a re-encode"
    for k, ((name, rate, x), (_, _, sx)) in enumerate(zip(tracks, src_tracks), start=1):
        if name != TRACKS[k - 1][0] or rate != TRACKS[k - 1][1]:
            return f"track {k}: {name} {rate} Hz, expected {TRACKS[k - 1][0]} {TRACKS[k - 1][1]} Hz"
        if abs(len(x) / rate - video_s) > 0.06:
            return f"track {k}: {len(x) / rate:.3f} s of audio for {video_s:.3f} s of video"
        level = rms(x[int(1.4 * rate):int(1.6 * rate)]) / rms(sx[int(2.4 * rate):int(2.6 * rate)])  # middle, unfaded
        if not 0.85 < level < 1.15:
            return f"track {k}: level {level:.2f} of the source in the middle"
        if fades:  # fade-in of 1 s: ~0.1 at 0.1 s, ~0.5 at 0.5 s
            for t, want in ((0.1, 0.1), (0.5, 0.5)):
                got = rms(x[int((t - 0.02) * rate):int((t + 0.02) * rate)]) / rms(sx[int(2.4 * rate):int(2.6 * rate)])
                if abs(got - want) > 0.08:
                    return f"track {k}: gain {got:.2f} at {t} s into the fade-in, expected {want}"
    return None


def main() -> int:
    cases = [("no fades, no join: tracks copied", [{"start": 1, "end": 4}], False, True),
             ("no fades, a join: each track re-encoded", [{"start": 1, "end": 3}, {"start": 3.5, "end": 5}], False, False),
             ("audio fades: each track re-encoded", [{"start": 1, "end": 4, "fadein": 1, "fadeout": 1}], True, False)]
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "two_tracks.mp4"
        make_source(src)
        for i, (name, segments, fades, copied) in enumerate(cases):
            problem = check(src, Path(d) / f"out{i}.mp4", segments, fades, copied)
            if problem:
                failed += 1
                print(f"✗ FAIL - {name}: {problem}")
            else:
                print(f"✓ PASS - {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
