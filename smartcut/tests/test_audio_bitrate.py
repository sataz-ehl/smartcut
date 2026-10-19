#!/usr/bin/env python3
"""
Tests the bitrate of re-encoded audio: the source's, at least 64 kb/s per channel with mono counted as two (MP3:
256 kb/s), at most what the encoder accepts.

- The source's bitrate is what its stream reports, else the average of its packets (total size x 8 / the time they
  span): Matroska and WebM report none, e.g. for Opus. Lossless sources count as unknown: their rate says nothing
  about a lossy one's (PCM into WebM asked libopus for 1536 kb/s, which it refuses, and the cut failed).
- The encoder's limits: libopus refuses more than 256 kb/s per channel, libvorbis about 89-249 kb/s per channel
  depending on the sample rate; aac and ac3 lower a too-high rate themselves; MP3 tops out at 320 kb/s.

Checks the bitrate each re-encoded stream is set up with, for sources made with PyAV (no media needed), and that
a cut with a join comes out at about the computed source rate.

Usage (from the repo root):
    python smartcut/tests/test_audio_bitrate.py
"""
import io
import sys
import tempfile
from contextlib import closing  # with closing(av.open()): PyAV types a container's __exit__ as maybe swallowing errors
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
from av import AudioStream, VideoStream

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut
from smartcut.media_container import MediaContainer
from smartcut.track_cutters import _accepted_audio_bitrate, create_recode_audio_output_stream

FPS, SECONDS = 30, 6


def make_source(path: Path, codec: str, rate: int, layout: str, bit_rate: int, *, video: str = "libx264") -> None:
    """Video and a noise-like audio track (so VBR encoders spend their bits)."""
    rng = np.random.default_rng(2)
    channels = len(av.AudioLayout(layout).channels)
    x = np.stack([np.convolve(rng.standard_normal(SECONDS * rate), np.hanning(9), "same") for _ in range(channels)])
    x = (0.1 * x / np.abs(x).max()).astype(np.float32)
    with av.open(str(path), "w") as out:
        v = cast(VideoStream, out.add_stream(video, rate=FPS))
        v.width, v.height, v.pix_fmt = 96, 64, "yuv420p"
        v.codec_context.options = {"g": "30"} if video == "libx264" else {"g": "30", "cpu-used": "8", "deadline": "realtime"}
        a = cast(AudioStream, out.add_stream(codec, rate=rate))
        a.codec_context.layout = layout
        if bit_rate:
            a.codec_context.bit_rate = bit_rate
        n = rate // FPS
        for i in range(SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 96, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(v.encode(frame))
            samples = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, i * n:(i + 1) * n]), format="fltp", layout=layout)
            samples.sample_rate, samples.pts = rate, i * n
            out.mux(a.encode(samples))
        out.mux(v.encode())
        out.mux(a.encode())


def packet_bitrate(path: Path) -> tuple[int | None, float]:
    """The first audio stream's reported bitrate, and its packets' average (size x 8 / the time they span)."""
    with closing(av.open(str(path))) as c:
        s = c.streams.audio[0]
        packets = [p for p in c.demux(s) if p.size and p.pts is not None]
        span = float((max(cast(int, p.pts) + (p.duration or 0) for p in packets) - min(cast(int, p.pts) for p in packets))
                     * cast(Fraction, s.time_base))
        return s.bit_rate, sum(p.size for p in packets) * 8 / span


def configured(src: Path, out_format: str) -> tuple[str, int]:
    """Encoder and bitrate a re-encode of the source's first audio track into this format is set up with."""
    media = MediaContainer(str(src))
    try:
        with closing(av.open(io.BytesIO(), "w", format=out_format)) as out:
            stream = create_recode_audio_output_stream(media, out, 0)
            return stream.codec_context.name, cast(int, stream.codec_context.bit_rate)
    finally:
        media.close()


def main() -> int:
    failed = total = 0

    def report(name: str, problem: str | None) -> None:
        nonlocal failed, total
        total += 1
        failed += problem is not None
        print(f"✗ FAIL - {name}: {problem}" if problem else f"✓ PASS - {name}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # name, file, codec, rate, layout, source bit_rate, output format, expected encoder, expected rate
        # (None: the source's computed from its packets)
        cases = [
            ("AAC 256k in MKV, no reported rate: from its packets", "aac.mkv", "aac", 48000, "stereo", 256_000, "mp4", "aac", None),
            ("Opus 192k in WebM, no reported rate: from its packets", "opus.webm", "libopus", 48000, "stereo", 192_000, "webm", "libopus", None),
            ("Opus 64k in WebM: the floor, 128k stereo", "opus64.webm", "libopus", 48000, "stereo", 64_000, "webm", "libopus", 128_000),
            ("AAC 5.1 at 192k: the floor, 384k", "aac51.mp4", "aac", 48000, "5.1", 192_000, "mp4", "aac", 384_000),
            ("AAC mono at 48k: the floor, 128k (mono counted as two)", "aacmono.mp4", "aac", 48000, "mono", 48_000, "mp4", "aac", 128_000),
            ("AC-3 640k into WebM: libopus's limit, 512k stereo", "ac3hi.mp4", "ac3", 48000, "stereo", 640_000, "webm", "libopus", 512_000),
            ("PCM into WebM: lossless counts as unknown, 128k", "pcm.mkv", "pcm_s16le", 48000, "stereo", 0, "webm", "libopus", 128_000),
            ("MP3 mono 128k: the MP3 floor, 256k", "mp3mono.mp4", "libmp3lame", 44100, "mono", 128_000, "mp4", "libmp3lame", 256_000),
            ("MP3 320k: kept", "mp3hi.mp4", "libmp3lame", 44100, "stereo", 320_000, "mp4", "libmp3lame", 320_000),
            ("AC-3 5.1 at 448k: kept", "ac351.mp4", "ac3", 48000, "5.1", 448_000, "mp4", "ac3", 448_000),
        ]
        for name, file, codec, rate, layout, bit_rate, out_format, want_encoder, want in cases:
            src = tmp / file
            make_source(src, codec, rate, layout, bit_rate, video="libvpx-vp9" if out_format == "webm" else "libx264")
            reported, average = packet_bitrate(src)
            if want is None:
                if reported:
                    report(name, f"the source reports {reported} b/s: no case for the packets")
                    continue
                want = round(average)
            encoder, got = configured(src, out_format)
            ok = encoder == want_encoder and abs(got - want) <= max(1000, 0.002 * want)
            report(name, None if ok else f"{encoder} at {got} b/s, expected {want_encoder} at {want} b/s")

        # Encoder limits, asked directly
        for name, codec, rate, layout, wanted, most in (
                ("libvorbis 22.05 kHz stereo 300k: at most ~179k", "libvorbis", 22050, "stereo", 300_000, 180_000),
                ("libopus 5.1 3000k: at most 1536k", "libopus", 48000, "5.1", 3_000_000, 1_536_000),
                ("aac stereo 48 kHz 1000k: lowered to 576k", "aac", 48000, "stereo", 1_000_000, 576_000)):
            got = _accepted_audio_bitrate(codec, rate, av.AudioLayout(layout), wanted)
            report(name, None if most * 0.95 <= got <= most else f"{got} b/s")

        # End to end: a cut with a join (re-encoded) comes out at about the computed source rate
        for file, low in (("aac.mkv", 0.9), ("opus.webm", 0.75)):
            src, out = tmp / file, tmp / f"cut_{file}"
            ok = smartcut.process_segments(str(src), str(out), [{"start": 0.7, "end": 2.6}, {"start": 3.4, "end": 5.5}])
            _, src_avg = packet_bitrate(src)
            _, out_avg = packet_bitrate(out) if ok else (None, 0)
            report(f"{file} cut with a join: about the source's rate",
                   None if ok and low * src_avg <= out_avg <= 1.15 * src_avg else
                   f"{out_avg / 1000:.0f} kb/s for a {src_avg / 1000:.0f} kb/s source (ok={ok})")
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
