#!/usr/bin/env python3
"""
Tests .webm outputs. WebM holds only Opus and Vorbis audio, so other audio (e.g. AAC) is re-encoded to Opus as a
whole track, with or without audio fades. Opus audio is copied without audio fades or a join, also from mid-stream
(decided: from the first whole packet after the cut, as WebM has no edit list).

Checks: the output decodes completely, the audio track is Opus at 48 kHz and as long as the video, fades have the
right gain, and copied Opus packets are byte-identical to the source's. Sources (VP9 + AAC, VP9 + Opus) are made
with PyAV; HE-AAC sources are in the regression harness (cases W1-W6 in regression/run_cases.py).

Usage (from the repo root):
    python smartcut/tests/test_webm.py
"""
import contextlib
import io
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut

FPS, SECONDS = 30, 6


def make_source(path: Path, audio_codec: str, video_codec: str = "libvpx-vp9") -> None:
    """VP9 (or other) video and a 440 Hz tone, keyframe every second."""
    rate = 48000
    with av.open(str(path), "w") as out:
        video = out.add_stream(video_codec, rate=FPS)
        video.width, video.height, video.pix_fmt = 160, 96, "yuv420p"
        video.codec_context.options = {"g": "30", "cpu-used": "8", "deadline": "realtime"} if "vpx" in video_codec else {"g": "30"}
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


def check(src: Path, out: Path, fades: bool, copied: bool, start: float = 1) -> str | None:
    segment = {"start": start, "end": start + 3, **({"audio_fadein": 1, "audio_fadeout": 1} if fades else {})}
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


def check_h264_rejected(src: Path, out: Path) -> str | None:
    """WebM can't hold H.264 and smartcut doesn't re-encode the whole video: a clear error, from the API and the
    CLI, and no output file."""
    expected = "WebM output needs VP8, VP9 or AV1 video; this video is H.264: use .mp4 or .mkv"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
        result = smartcut.process_segments(str(src), str(out), [{"start": 1, "end": 4}])
    if result is not False or expected not in printed.getvalue() or out.exists():
        return f"API: returned {result!r}, output left {out.exists()}, printed {printed.getvalue().strip()[:160]!r}"
    cli = subprocess.run([sys.executable, "-m", "smartcut", str(src), str(out), "--keep", "1,4"], capture_output=True,
                         text=True, cwd=Path(__file__).resolve().parents[2], check=False)
    if cli.returncode == 0 or expected not in cli.stderr or "Traceback" in cli.stderr or out.exists():
        return f"CLI: exit {cli.returncode}, stderr {cli.stderr.strip()[-200:]!r}"
    return None


def main() -> int:
    failed = total = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        sources = {"aac": tmp / "vp9_aac.mp4", "libopus": tmp / "vp9_opus.webm", "pcm_s16le": tmp / "vp9_pcm.mkv"}
        for codec, path in sources.items():
            make_source(path, codec)
        make_source(tmp / "h264_aac.mp4", "aac", "libx264")
        total += 1
        problem = check_h264_rejected(tmp / "h264_aac.mp4", tmp / "h264.webm")
        failed += problem is not None
        print(f"✗ FAIL - H.264 video into .webm: {problem}" if problem else "✓ PASS - H.264 video into .webm: clear error")
        cases = [("AAC -> Opus, no fades", "aac", False, False, 1), ("AAC -> Opus, audio fades", "aac", True, False, 1),
                 ("PCM (1536 kb/s) -> Opus at a lossy rate", "pcm_s16le", False, False, 1),
                 ("Opus copied from the start, no fades", "libopus", False, True, 0),
                 ("Opus copied from mid-stream, no fades", "libopus", False, True, 1),
                 ("Opus re-encoded, audio fades", "libopus", True, False, 1)]
        for i, (name, codec, fades, copied, start) in enumerate(cases):
            total += 1
            problem = check(sources[codec], tmp / f"out{i}.webm", fades, copied, start)
            if problem:
                failed += 1
                print(f"✗ FAIL - {name}: {problem}")
            else:
                print(f"✓ PASS - {name}")
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
