#!/usr/bin/env python3
"""
Tests how re-encoded audio is decoded (RecodeAudioCutter) and how a copy's pre-roll is chosen.

- After a jump the decoder restarts PREROLL (160 ms) and at least PREROLL_PACKETS before the first kept sample. Opus
  decoders settle by time, not by packets: with the old four packets, 10 ms Opus packets got 40 ms and the first
  kept audio differed from a continuous decode by ~12 %. Checked: within 1 % for 10 and 20 ms packets.
- A copy's pre-roll packets are chosen by their timestamps: with packet durations missing (0), walking back by them
  took the whole stream.
- A backwards timestamp jump longer than a frame (three AAC frames, 64 ms): only the first frame's overlap was
  dropped, and the audio after it came two frames (42.7 ms) late. Checked: the overlap dropped, the audio after it
  where its timestamps say.
- A gap of exactly one tick of the time base was taken for timestamp rounding, and the audio after it came a tick
  early: where a tick is a whole packet (here PCM in 1 ms Matroska packets). Checked: in sync after the gap, silence
  in it.
- Resampled re-encodes (44.1 kHz AAC into a WebM's 48 kHz Opus): the resampler's delay was taken for a gap after every
  restart, 16-17 samples of silence (a step x5.6 a tone's own; every segment 0.35 ms late). Checked: in sync with the
  source resampled continuously, within 0.05 ms.

Sources are made with PyAV, no media needed.

Usage (from the repo root):
    python smartcut/tests/test_audio_decode.py
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
from smartcut.track_cutters import RecodeAudioCutter, _preroll_start, create_recode_audio_output_stream

SECONDS = 7


def make_source(path: Path, video: str, audio: str, rate: int, fps: int, options: dict | None = None,  # noqa: PLR0917
                drop: float | None = None) -> None:
    """Video and mono low-passed noise; drop: the audio packet at this time is left out."""
    rng = np.random.default_rng(3)
    x = np.convolve(rng.standard_normal(SECONDS * rate), np.hanning(17), "same")
    x = (0.3 * x / np.sqrt((x ** 2).mean())).astype(np.float32)
    with av.open(str(path), "w") as out:
        v = cast(VideoStream, out.add_stream(video, rate=fps))
        v.width, v.height, v.pix_fmt = 96, 64, "yuv420p"
        v.codec_context.options = {"g": str(fps)} if video == "libx264" else {"g": str(fps), "cpu-used": "8", "deadline": "realtime"}
        a = cast(AudioStream, out.add_stream(audio, rate=rate))
        a.codec_context.layout = "mono"
        a.codec_context.options = options or {}
        n = rate // fps
        dropped = False
        for i in range(SECONDS * fps):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), i % 200, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            out.mux(v.encode(frame))
            samples = av.AudioFrame.from_ndarray(x[None, i * n:(i + 1) * n], format="fltp", layout="mono")
            samples.sample_rate, samples.pts = rate, i * n
            for p in a.encode(samples):
                if drop is not None and not dropped and p.pts is not None and float(p.pts * p.time_base) >= drop:
                    dropped = True
                    continue
                out.mux(p)
        out.mux(v.encode())
        out.mux(a.encode())


def by_pts(path: Path, rate: int | None = None) -> tuple[np.ndarray, int]:
    """The first audio track on the source's timeline: each decoded frame at its timestamp (silence where there's
    none). With rate: the whole track resampled continuously, from its first timestamp."""
    with closing(av.open(str(path))) as c:
        s = c.streams.audio[0]
        frames = list(c.decode(s))
    if rate:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        x = np.concatenate([g.to_ndarray()[0].astype(np.float64) for f in [*frames, None] for g in resampler.resample(f)])
        chunks = [(float(cast(int, frames[0].pts) * cast(Fraction, frames[0].time_base)), x)]
    else:
        rate = s.sample_rate
        chunks = [(float(cast(int, f.pts) * cast(Fraction, f.time_base)), f.to_ndarray().astype(np.float64).mean(axis=0))
                  for f in frames]
    out = np.zeros(round(max(t for t, _ in chunks) * rate) + sum(len(ch) for _, ch in chunks[-1:]) + rate)
    for t, ch in chunks:
        i = round(t * rate)
        ch = ch[max(-i, 0):]  # before 0 (encoder priming): left out
        out[max(i, 0):max(i, 0) + len(ch)] = ch
    return out, rate


def lag(ref: np.ndarray, out: np.ndarray, rate: int, out_t: float, ref_t: float, *, search: float = 0.04) -> float:
    """Seconds by which the output at out_t is late against the reference at ref_t (normalised correlation)."""
    w = out[round(out_t * rate):round((out_t + 0.25) * rate)]
    j0 = round((ref_t - search) * rate)
    s = ref[j0:j0 + len(w) + round(2 * search * rate)]
    c = np.correlate(s, w, "valid") / np.sqrt(np.convolve(s ** 2, np.ones(len(w)), "valid") + 1e-18)
    return search - int(np.argmax(c)) / rate


def by_count(path: Path) -> tuple[np.ndarray, int]:
    """The first audio track from the output's time 0, decoded frames one after another (as players play)."""
    with closing(av.open(str(path))) as c:
        s = c.streams.audio[0]
        frames = list(c.decode(s))
    x = np.concatenate([f.to_ndarray().astype(np.float64).mean(axis=0) for f in frames])
    lead = round(float(cast(int, frames[0].pts) * cast(Fraction, frames[0].time_base)) * s.sample_rate)
    return (x[-lead:] if lead < 0 else np.concatenate([np.zeros(lead), x])), s.sample_rate


def check_restart(src: Path) -> str | None:
    """RecodeAudioCutter's samples after a jump against a continuous decode: within 1 % (relative RMS)."""
    reference, rate = by_pts(src)
    media = MediaContainer(str(src))
    try:
        with closing(av.open(io.BytesIO(), "w", format="webm")) as out:
            cutter = RecodeAudioCutter(media, create_recode_audio_output_stream(media, out, 0), 0)
            worst = 0.0
            for t in (2.0, 3.517, 5.25):
                start = round(t * rate)
                cutter._restart_decoder(start)  # a jump, as at a join
                got = cutter._take(start, start + rate // 50)[0].astype(np.float64)  # the first 20 ms kept
                want = reference[start:start + len(got)]
                worst = max(worst, float(np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want ** 2).mean())))
    finally:
        media.close()
    return None if worst < 0.01 else f"the first 20 ms after a jump differ from a continuous decode by {worst:.1%}"


class Relabelled:
    """A decoder whose frames from `at` seconds on come with timestamps `shift` samples earlier: a backwards timestamp
    jump, as in a source whose timestamps restart or overlap."""

    def __init__(self, decoder: av.AudioCodecContext, at: float, shift: int) -> None:
        self.decoder, self.at, self.shift = decoder, at, shift

    def __getattr__(self, name: str) -> object:
        return getattr(self.decoder, name)

    def decode(self, packet: av.Packet | None = None) -> list[av.AudioFrame]:
        frames = self.decoder.decode(packet)
        for f in frames:
            if f.pts is not None and f.time_base is not None and f.pts * f.time_base >= self.at:
                f.pts -= round(Fraction(self.shift, f.sample_rate) / f.time_base)
        return frames


def check_overlap(src: Path) -> str | None:
    """A backwards timestamp jump of three AAC frames (64 ms) at 3 s: the overlap is dropped, and the audio after it
    is placed by its timestamps (it holds the source's audio from 64 ms later). Before, only the first frame of the
    overlap was dropped, and everything after it came two frames (42.7 ms) late."""
    reference, rate = by_pts(src)  # the source's audio at its real times
    shift = 3 * 1024
    media = MediaContainer(str(src))
    try:
        with closing(av.open(io.BytesIO(), "w", format="mp4")) as out:
            cutter = RecodeAudioCutter(media, create_recode_audio_output_stream(media, out, 0), 0)
            cutter.decoder = cast(av.AudioCodecContext, Relabelled(cutter.decoder, 3.0, shift))
            cutter._restart_decoder(2 * rate)
            got = cutter._take(2 * rate, 5 * rate)[0].astype(np.float64)  # by the timestamps, 2-5 s
    finally:
        media.close()
    for at in (2.5, 3.3, 4.0, 4.6):  # by the timestamps; from 3 s on they are 64 ms early
        late = lag(reference, got, rate, at - 2, at + (shift / rate if at >= 3 else 0), search=0.06)
        if abs(late) > 0.0001:
            return f"the audio timestamped {at} s is {late * 1000:+.1f} ms late"
    return None


def check_preroll_without_durations(src: Path) -> str | None:
    """The copy's pre-roll (80 ms for Opus) by timestamps, also with packet durations of 0."""
    media = MediaContainer(str(src))
    try:
        track = media.audio_tracks[0]
        start = len(track.packets) // 2
        want = _preroll_start(track, start, Fraction(80, 1000))
        for p in track.packets:
            p.duration = 0
        got = _preroll_start(track, start, Fraction(80, 1000))
    finally:
        media.close()
    if got != want or not Fraction(80, 1000) <= got[1] < Fraction(90, 1000):
        return f"pre-roll from packet {got[0]} ({float(got[1]) * 1000:.0f} ms) for packet {start}; with durations {want[0]}"
    return None


def check_one_tick_gap(tmp: Path) -> str | None:
    """A gap of exactly one tick of the time base: PCM in Matroska written in 1 ms packets, one of them left out at 2 s.
    Cut with a join (re-encoded): silent in the gap and in sync after it (taken for rounding, the audio after it came
    1 ms early)."""
    src, out = tmp / "gap.mkv", tmp / "gap_out.mkv"
    rate, fps = 48000, 25
    rng = np.random.default_rng(5)
    x = np.convolve(rng.standard_normal(SECONDS * rate), np.hanning(17), "same")
    x = (0.3 * x / np.sqrt((x ** 2).mean())).astype(np.float32)
    with av.open(str(src), "w") as c:
        v = c.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = 96, 64, "yuv420p"
        a = c.add_stream("pcm_s16le", rate=rate)
        a.codec_context.layout = "mono"
        for i in range(SECONDS * fps):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), i % 200, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            c.mux(v.encode(frame))
            for k in range(40):  # 1 ms packets
                j = (i * 40 + k) * 48
                if j == 2 * rate:
                    continue
                samples = av.AudioFrame.from_ndarray(np.rint(x[None, j:j + 48] * 32767).astype(np.int16), format="s16", layout="mono")
                samples.sample_rate, samples.pts = rate, j
                c.mux(a.encode(samples))
        c.mux(v.encode())
        c.mux(a.encode())
    if not smartcut.process_segments(str(src), str(out), [{"start": 0.9, "end": 3.5}, {"start": 4.6, "end": 6.3}]):
        return "process_segments failed"
    ref, _ = by_pts(src)
    o, _ = by_count(out)
    for out_t, ref_t in ((0.4, 1.3), (1.6, 2.5)):  # before and after the gap
        late = lag(ref, o, rate, out_t, ref_t)
        if abs(late) > 0.0004:
            return f"audio {late * 1000:+.2f} ms late at {ref_t} s of the source"
    gap = o[round(1.1 * rate) + 2:round(1.101 * rate) - 2]  # the output's time for the source's 2.000-2.001 s
    return None if np.abs(gap).max() < 0.01 else f"sound where the source has none (peak {np.abs(gap).max():.2f})"


def check_resampled(tmp: Path) -> str | None:
    """44.1 kHz AAC into .webm (Opus at 48 kHz), from the start and with joins: in sync with the source resampled
    continuously."""
    src = tmp / "aac441.mp4"
    make_source(src, "libvpx-vp9", "aac", 44100, 30)
    ref, rate = by_pts(src, 48000)
    for segments in ([(0, 3)], [(0.9, 2.37), (2.9, 4.1), (4.6, 6.3)]):
        out = tmp / f"resampled{len(segments)}.webm"
        if not smartcut.process_segments(str(src), str(out), [{"start": a, "end": b} for a, b in segments]):
            return "process_segments failed"
        o, _ = by_count(out)
        t = 0.0
        for a, b in segments:
            for u in (0.03, b - a - 0.3):
                late = lag(ref, o, rate, t + u, a + u, search=0.005)
                if abs(late) > 0.00005:
                    return f"segment {a}-{b}: audio {late * 1000:+.3f} ms late at {u:.2f} s into it"
            t += b - a
    return None


def main() -> int:
    failed = total = 0

    def report(name: str, problem: str | None) -> None:
        nonlocal failed, total
        total += 1
        failed += problem is not None
        print(f"✗ FAIL - {name}: {problem}" if problem else f"✓ PASS - {name}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for ms in (10, 20):
            src = tmp / f"opus{ms}.mp4"  # MP4: exact timestamps (Matroska's 1 ms ones would misplace the reference)
            make_source(src, "libvpx-vp9", "libopus", 48000, 30, {"frame_duration": str(ms)})
            report(f"Opus, {ms} ms packets: after a jump, decoded as continuously", check_restart(src))
            if ms == 10:
                report("Copy pre-roll by timestamps: packets without durations", check_preroll_without_durations(src))
        aac = tmp / "aac48.mp4"
        make_source(aac, "libx264", "aac", 48000, 30)
        report("A backwards timestamp jump of three frames: the overlap dropped, the rest in place", check_overlap(aac))
        report("A gap of one tick (PCM in 1 ms Matroska packets): silence there, in sync after it", check_one_tick_gap(tmp))
        report("Resampled re-encode (44.1 kHz AAC into WebM Opus): in sync, no silence after restarts", check_resampled(tmp))
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
