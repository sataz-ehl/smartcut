#!/usr/bin/env python3
"""
Tests that audio is re-encoded only faithfully (track_cutters.recode_setup), and copied with a warning otherwise.

A re-encode (for a join, an audio fade, or a lossless track starting mid-stream) has to keep the codec, the sample
format (bit depth), the channel layout and the sample rate, and the encoder has to open with them. Exceptions: HE-AAC
becomes AAC-LC (test_audio_joins.py with a media dir), and .webm outputs re-encode other codecs to Opus.

Cases, each cut into three segments (two joins) or one mid-stream segment:
- Lossless re-encodes keep the codec, the sample format and the bit depth, and every sample outside the 3 ms join
  fades is bit-identical to the source: FLAC 16 and 24 bits, ALAC 16 and 24 bits, PCM s16 / s24 / s32 / f32 / f64 /
  u8. Before, 24-bit FLAC came out as 16 bits, 16-bit ALAC as 24 bits, and PCM went through 32-bit float. PCM in
  Matroska has no channel layout ("2 channels"): kept, as lossless encoders store channels as they are.
- Lossy codecs with an encoder are re-encoded as themselves: MP2, and AAC 5.1 (keeping its layout).
- Copied, with a warning naming the track, and the cut succeeds:
  - AAC "6 channels" (5.1(side) written into MP4 has no channel order): FFmpeg's AAC encoder refused to open, and
    the cut failed;
  - DTS and TrueHD: no encoder in PyAV's FFmpeg (they became AAC), and 32-bit FLAC (FFmpeg's encoder writes 24 of
    the bits). Their sources need the ffmpeg command (on PATH, or SMARTCUT_TEST_FFMPEG); skipped without it.
- AAC "6 channels" into .webm, which needs Opus: a clear error, nothing written.

Usage (from the repo root):
    python smartcut/tests/test_audio_faithful.py
"""
import contextlib
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut

FPS, SECONDS, RATE = 30, 7, 48000
JOINS = [(0.9, 2.37), (2.9, 4.1), (4.6, 6.3)]
JOIN_FADE = 0.003  # RecodeAudioCutter.JOIN_FADE
DTYPES = {"u8": np.uint8, "s16": np.int16, "s32": np.int32, "flt": np.float32, "dbl": np.float64}


def noise(channels: int, n: int, sample_format: str, bits: int) -> np.ndarray:
    """Independent noise per channel in the format's own scale, bits deep (24-bit audio in s32: low 8 bits zero)."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-0.5, 0.5, (channels, n))
    kind = av.AudioFormat(sample_format).packed.name
    if kind in ("flt", "dbl"):
        return x.astype(DTYPES[kind])
    if kind == "u8":
        return np.rint(x * 255 + 128).astype(np.uint8)
    full = 8 * np.dtype(DTYPES[kind]).itemsize
    return (np.rint(x * 2 ** (bits - 1)).astype(np.int64) << (full - bits)).astype(DTYPES[kind])


def make_source(path: Path, codec: str, layout: str, sample_format: str, bits: int, bit_rate: int = 0) -> np.ndarray:  # noqa: PLR0917
    """H.264 video and one audio track; returns the audio fed to the encoder, channels first."""
    channels = len(av.AudioLayout(layout).channels)
    x = noise(channels, SECONDS * RATE, sample_format, bits)
    fmt = av.AudioFormat(sample_format)
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=FPS)
        video.width, video.height, video.pix_fmt = 96, 64, "yuv420p"
        video.codec_context.options = {"g": "30", "keyint_min": "30", "sc_threshold": "0"}
        audio = out.add_stream(codec, rate=RATE)
        audio.codec_context.layout, audio.codec_context.format = layout, sample_format
        if bit_rate:
            audio.codec_context.bit_rate = bit_rate
        n = RATE // FPS
        for i in range(SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), 40 + i % 150, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(video.encode(frame))
            chunk = x[:, i * n:(i + 1) * n]
            samples = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk if fmt.is_planar else chunk.T.reshape(1, -1)),
                                                 format=sample_format, layout=layout)
            samples.sample_rate, samples.pts = RATE, i * n
            out.mux(audio.encode(samples))
        out.mux(video.encode())
        out.mux(audio.encode())
    return x


def header_bits(ctx: av.AudioCodecContext) -> int | None:
    """Bits per sample in a FLAC STREAMINFO or an ALAC magic cookie (as FFmpeg keeps them in extradata)."""
    data = bytes(ctx.extradata or b"")
    if ctx.name == "flac":
        data = data[8:] if data[:4] == b"fLaC" else data
        return (((data[12] & 1) << 4) | (data[13] >> 4)) + 1
    if ctx.name == "alac":
        return data[17]
    return None


def decoded(path: Path) -> tuple[str, str, str, int | None, np.ndarray, set[bytes]]:
    """First audio track: codec, sample format, layout, bits (FLAC / ALAC), samples channels first, packets."""
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        name, fmt, layout = s.codec_context.name, s.codec_context.format.name, s.codec_context.layout.name
        bits = header_bits(s.codec_context)
        frames = [f.to_ndarray() if f.format.is_planar else f.to_ndarray().reshape(-1, len(f.layout.channels)).T
                  for f in c.decode(s)]
    with av.open(str(path)) as c:
        packets = {bytes(p) for p in c.demux(audio=0) if p.size}
    return name, fmt, layout, bits, np.concatenate(frames, axis=1), packets


class Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def cut(src: Path, out: Path, segments: list[tuple[float, float]]) -> tuple[bool, list[str]]:
    handler = Warnings()
    log = logging.getLogger("smartcut")
    log.addHandler(handler)
    try:
        ok = smartcut.process_segments(str(src), str(out), [{"start": a, "end": b} for a, b in segments])
    finally:
        log.removeHandler(handler)
    return ok, handler.messages


def check_lossless(tmp: Path, codec: str, ext: str, layout: str, sample_format: str, bits: int) -> str | None:  # noqa: PLR0917
    """A cut with two joins: the same codec, sample format and bits, samples bit-identical outside the join fades.

    In Matroska, whose timestamps are in ms, a segment's audio may be placed up to 0.5 ms off (PCM packets of 33.3 ms;
    FLAC's 96 ms frames fall on whole ms): there the samples must be bit-identical at some offset within 0.5 ms."""
    src, out = tmp / f"{codec}_{sample_format}_{bits}.{ext}", tmp / f"out_{codec}_{sample_format}_{bits}.{ext}"
    make_source(src, codec, layout, sample_format, bits)
    ok, warnings = cut(src, out, JOINS)
    if not ok:
        return "process_segments failed"
    s_name, s_fmt, s_layout, s_bits, sx, s_packets = decoded(src)  # PCM in Matroska has no layout: "2 channels"
    o_name, o_fmt, o_layout, o_bits, ox, o_packets = decoded(out)
    if (o_name, o_fmt, o_layout, o_bits) != (s_name, s_fmt, s_layout, s_bits):
        return f"{o_name} {o_fmt} {o_layout} {o_bits} bits, expected {s_name} {s_fmt} {s_layout} {s_bits} bits"
    if len(o_packets & s_packets) == len(o_packets):
        return "copied, expected a re-encode"
    if warnings:
        return f"warned: {warnings}"
    expected = round(sum(b - a for a, b in JOINS) * RATE)
    if abs(ox.shape[1] - expected) > RATE // 1000:
        return f"{ox.shape[1] / RATE:.3f} s of audio, expected {expected / RATE:.3f} s"
    t, m = 0.0, round(JOIN_FADE * RATE) + 2
    max_shift = RATE // 2000 if ext == "mkv" else 0
    for k, (a, b) in enumerate(JOINS):
        i, j, n = round(t * RATE), round(a * RATE), round((b - a) * RATE)
        lo, hi = (m if k else 0) + max_shift, n - (m if k < len(JOINS) - 1 else 2) - max_shift
        if not any(np.array_equal(ox[:, i + lo:i + hi], sx[:, j + lo + d:j + hi + d]) for d in range(-max_shift, max_shift + 1)):
            diff = np.argwhere(ox[:, i + lo:i + hi] != sx[:, j + lo:j + hi])
            return f"segment {k + 1}: {len(diff)} samples differ from the source, the first at {(lo + diff[0][1]) / RATE:.4f} s"
        t += b - a
    return None


def check_copied(src: Path, out: Path, segments: list[tuple[float, float]], why: str) -> str | None:
    """Copied (every output packet is a source packet), the cut succeeds, and a warning names the track and why."""
    ok, warnings = cut(src, out, segments)
    if not ok:
        return "process_segments failed"
    *_, s_packets = decoded(src)
    s_name, *_ = decoded(src)
    o_name, *_, o_packets = decoded(out)
    if o_name != s_name or len(o_packets & s_packets) < len(o_packets):
        return f"{o_name}, {len(o_packets & s_packets)}/{len(o_packets)} packets copied: expected a copy of {s_name}"
    if not any("audio track 1" in w and why in w for w in warnings):
        return f"no warning saying '{why}': {warnings}"
    return None


def ffmpeg_source(tmp: Path, name: str, codec_args: list[str]) -> Path | None:
    """A source made with the ffmpeg command (what PyAV's FFmpeg can't encode), or None without it."""
    ffmpeg = os.environ.get("SMARTCUT_TEST_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    pcm = tmp / f"{name.replace(' ', '_')}_pcm.mkv"
    make_source(pcm, "pcm_s32le", "stereo", "s32", 32)
    out = tmp / f"{name.replace(' ', '_')}.mkv"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(pcm), "-c:v", "copy", "-channel_layout", "stereo", *codec_args,
                    "-strict", "-2", str(out)], check=True)
    return out


def main() -> int:
    failed = total = 0

    def report(name: str, problem: str | None) -> None:
        nonlocal failed, total
        total += 1
        failed += problem is not None
        print(f"✗ FAIL - {name}: {problem}" if problem else f"✓ PASS - {name}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for codec, ext, layout, sample_format, bits in (
                ("flac", "mkv", "stereo", "s16", 16), ("flac", "mkv", "stereo", "s32", 24),
                ("alac", "mov", "stereo", "s16p", 16), ("alac", "mov", "stereo", "s32p", 24),
                ("pcm_s16le", "mov", "stereo", "s16", 16), ("pcm_s24le", "mov", "5.1", "s32", 24),
                ("pcm_s32le", "mov", "stereo", "s32", 32), ("pcm_f32le", "mov", "stereo", "flt", 32),
                ("pcm_f64le", "mov", "stereo", "dbl", 64), ("pcm_u8", "mov", "mono", "u8", 8),
                ("pcm_s32le", "mkv", "stereo", "s32", 32)):  # Matroska: no layout ("2 channels"), ms timestamps
            report(f"{codec} {sample_format} {bits}-bit {layout} in {ext.upper()}, joins: re-encoded bit-exact",
                   check_lossless(tmp, codec, ext, layout, sample_format, bits))

        # Lossy codecs with an encoder: re-encoded as themselves (AAC 5.1 keeps its layout)
        mp2 = tmp / "mp2.mkv"
        make_source(mp2, "mp2", "stereo", "s16", 16, 192_000)
        ok, warnings = cut(mp2, tmp / "out_mp2.mkv", JOINS)
        name, *_, packets = decoded(tmp / "out_mp2.mkv") if ok else ("",)
        report("MP2, joins: re-encoded as MP2", None if ok and name == "mp2" and not warnings and len(packets & decoded(mp2)[5]) <= 2
               else f"ok={ok} {name}, warnings {warnings}")
        src, out = tmp / "aac51.mp4", tmp / "out_aac51.mp4"
        make_source(src, "aac", "5.1", "fltp", 0, 384_000)
        ok, warnings = cut(src, out, JOINS)
        name, _, layout, _, _, packets = decoded(out) if ok else ("", "", "", None, None, set())
        report("AAC 5.1, joins: re-encoded as 5.1",
               None if ok and name == "aac" and layout == "5.1" and not warnings and len(packets & decoded(src)[5]) <= 2
               else f"ok={ok} {name} {layout}, warnings {warnings}")

        # Copied, with a warning
        unordered = tmp / "aac_6ch.mp4"
        make_source(unordered, "aac", "5.1(side)", "fltp", 0, 384_000)  # reads back as "6 channels"
        report("AAC '6 channels', joins: copied, warned",
               check_copied(unordered, tmp / "out_6ch_joins.mp4", JOINS, "no channel order"))
        for name, args, why in (("DTS", ["-c:a", "dca", "-b:a", "768k"], "no dca encoder"),
                                ("TrueHD", ["-c:a", "truehd"], "no truehd encoder"),
                                ("FLAC 32-bit", ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "32"],
                                 "24 of its 32 bits")):
            src = ffmpeg_source(tmp, name.lower(), args)
            if src is None:
                print(f"(no ffmpeg command: {name} skipped)")
                continue
            report(f"{name}, joins: copied, warned ({why})",
                   check_copied(src, tmp / f"out_{name.replace(' ', '_')}.mkv", JOINS, why))

        # WebM needs Opus: a clear error when the track can't be re-encoded to it
        webm = tmp / "out.webm"
        vp9 = tmp / "vp9_6ch.mkv"
        with av.open(str(unordered)) as c_in, av.open(str(vp9), "w") as c_out:  # VP9 video (WebM can't hold H.264)
            v = c_out.add_stream("libvpx-vp9", rate=FPS)
            v.width, v.height, v.pix_fmt = 96, 64, "yuv420p"
            v.codec_context.options = {"g": "30", "cpu-used": "8", "deadline": "realtime"}
            a = c_out.add_stream_from_template(c_in.streams.audio[0])
            for i in range(SECONDS * FPS):
                frame = av.VideoFrame.from_ndarray(np.zeros((64, 96, 3), np.uint8), format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, FPS)
                c_out.mux(v.encode(frame))
            c_out.mux(v.encode())
            for p in c_in.demux(c_in.streams.audio[0]):
                if p.size:
                    p.stream = a
                    c_out.mux(p)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
            ok = smartcut.process_segments(str(vp9), str(webm), [{"start": a, "end": b} for a, b in JOINS])
        report("AAC '6 channels' into .webm: a clear error, nothing written",
               None if not ok and "can't be re-encoded to Opus" in printed.getvalue() and not webm.exists()
               else f"returned {ok}, printed {printed.getvalue().strip()[-200:]!r}, output left {webm.exists()}")
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
