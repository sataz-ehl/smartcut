#!/usr/bin/env python3
"""
Tests that cut audio stays in sync with the video, at the start and at every join, without level changes or clicks.

Audio is copied only when the copy is exact (track_cutters.audio_needs_recode), otherwise re-encoded as a whole:
- Joins: copied whole frames shifted A/V by up to a frame at every join (a random walk: AAC-LC -15.5 ms after
  5 joins), frames that overlap their neighbours (AAC, AC-3) decoded as a mix of both sides, and copied MP3 had
  ~50 ms of silence after a join (bit reservoir). Outputs with a join are re-encoded, with a 3 ms fade-out and
  fade-in at each join (a hard cut clicks).
- Mid-stream starts: copied exactly only into MP4 (AAC, AC-3 and Opus, with pre-roll packets the edit list or the
  Opus pre-skip skips). Matroska / WebM outputs, and other codecs, are re-encoded.

Checks per case, against the source:
- copied or re-encoded, as the rule says
- A/V sync: per segment, the audio's offset from where its video says it belongs (cross-correlation on noise-like
  audio; a steady tone is ambiguous), within 1 ms (1.5 ms in Matroska / WebM, whose timestamps are rounded to 1 ms);
  the video starts where it should (nothing shifted)
- level: 10 ms RMS against the ideal output (the source with the expected fades), 0.85-1.2, from the start through
  every join (copied MDCT joins had bursts up to x33). Re-encodes that start mid-stream get a lead-in in MP4, so
  they start like a copy; Matroska / WebM have no edit list to skip one, and Opus in MP4 none either (its pre-skip),
  so there the encoder starts cold and its first 20 ms may reach x1.3 on a pure tone: not checked
- clicks, on a steady tone: the largest sample step within 10 ms of each join, against the tone's own largest step,
  at most 1.2 (a hard cut: 5-11x)

Also a source without audio across a join (its audio packets left out from 3 to 5 s): the output keeps the timing
with silence there, in sync on both sides.

Sources (H.264 / VP9 video, 1 s GOPs, and a noise-like signal or a 440 Hz tone) are made with PyAV, no media needed.
With a media dir argument, HE-AAC cases also run on its synth_heaac.mp4 (make_test_media.py; FFmpeg can't encode
HE-AAC).

Usage (from the repo root):
    python smartcut/tests/test_audio_joins.py [media dir]
"""
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut

FPS, SECONDS, AMPLITUDE = 30, 7, 0.3
JOIN_FADE = 0.003  # RecodeAudioCutter.JOIN_FADE
# Three segments, given out of order: a mid-stream start, two joins (all off the video's frame grid and its GOPs)
JOINS = [{"start": 2.9, "end": 4.1}, {"start": 0.9, "end": 2.37}, {"start": 4.6, "end": 6.3}]
START = [{"start": 1.37, "end": 4.6}]  # a single segment starting mid-stream
FROM_0 = [{"start": 0, "end": 3}]  # from the media start: copied in any container
TOUCHING = [{"start": 1.37, "end": 2.5}, {"start": 2.5, "end": 4.6}]  # contiguous in the source: no join
FADES = [{"start": 0.9, "end": 2.37, "audio_fadeout": 0.001}, {"start": 2.9, "end": 4.1, "audio_fadein": 0.5}]
GAP = (3.0, 5.0)  # source audio left out here
GAP_SEGMENTS = [{"start": 0.9, "end": 3.5}, {"start": 4.6, "end": 6.3}]  # the join falls inside the gap


def make_source(path: Path, video_codec: str, audio_codec: str, rate: int, signal: str, *,
                gap: tuple[float, float] | None = None) -> None:
    """signal: 'noise' (low-passed white noise, for correlation) or 'tone' (440 Hz, for clicks). gap: seconds whose
    audio packets are left out."""
    rng = np.random.default_rng(1)
    n_all = SECONDS * rate
    if signal == "tone":
        x = AMPLITUDE * np.sin(2 * np.pi * 440 * np.arange(n_all) / rate)
    else:
        white = rng.standard_normal(n_all)
        x = np.convolve(white, np.hanning(17), "same")  # below ~5 kHz: a second lossy generation keeps it closely
        x *= AMPLITUDE / np.sqrt((x ** 2).mean())
    with av.open(str(path), "w") as out:
        video = out.add_stream(video_codec, rate=FPS)
        video.width, video.height, video.pix_fmt = 160, 96, "yuv420p"
        video.codec_context.options = ({"g": "30", "keyint_min": "30", "sc_threshold": "0"} if video_codec == "libx264"
                                       else {"g": "30", "cpu-used": "8", "deadline": "realtime"})
        audio = out.add_stream(audio_codec, rate=rate, layout="mono")
        n = rate // FPS
        for i in range(SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(np.full((96, 160, 3), 40 + i % 150, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(video.encode(frame))
            samples = av.AudioFrame.from_ndarray(x[i * n:(i + 1) * n].astype(np.float32)[None, :], format="fltp", layout="mono")
            samples.sample_rate, samples.pts = rate, i * n
            for packet in audio.encode(samples):
                if gap is None or not gap[0] <= packet.pts / rate < gap[1]:
                    out.mux(packet)
        out.mux(video.encode())
        out.mux(audio.encode())


def read(path: Path) -> dict:
    with av.open(str(path)) as c:
        start = (c.start_time or 0) / av.time_base
        vt = sorted(float(f.pts * f.time_base) for f in c.decode(video=0))
    with av.open(str(path)) as c:
        rate, codec = c.streams.audio[0].sample_rate, c.streams.audio[0].codec_context.name
        frames = list(c.decode(audio=0))
    with av.open(str(path)) as c:
        packets = [bytes(p) for p in c.demux(audio=0) if p.size]
    x = np.concatenate([f.to_ndarray().astype(np.float64).mean(axis=0) for f in frames])
    return {"start": start, "vt": vt, "rate": rate, "codec": codec, "a0": float(frames[0].pts * frames[0].time_base), "x": x,
            "packets": packets}


def lag(src: dict, out: dict, out_t: float, src_t: float, *, win: float = 0.3, search: float = 0.03) -> float:
    """Seconds by which the output audio at out_t is late against the source audio at src_t."""
    rate = src["rate"]
    i = round((out_t - out["a0"]) * rate)
    w = out["x"][i:i + int(win * rate)]
    j0 = round((src_t - src["a0"] - search) * rate)
    s = src["x"][j0:j0 + len(w) + int(2 * search * rate)]
    c = np.correlate(s, w, mode="valid") / np.sqrt(np.convolve(s ** 2, np.ones(len(w)), "valid") + 1e-18)
    k = int(np.argmax(c))
    return -((j0 + k) / rate + src["a0"] - src_t)  # content from later than expected: the audio is early


def expected(segments: list[dict], src: dict) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Sorted (start, end) in the source's timestamps, (fade-in, fade-out) with the join fades, and each segment's
    start in the output. A segment from 0 starts with the first video frame, as smartcut's output does."""
    segs = sorted(segments, key=lambda s: s["start"])
    out, starts, t = [], [], 0.0
    for k, s in enumerate(segs):
        join_before = k > 0 and segs[k - 1]["end"] != s["start"]
        join_after = k < len(segs) - 1 and segs[k + 1]["start"] != s["end"]
        fin = max(s.get("audio_fadein") or 0, JOIN_FADE if join_before else 0)
        fout = max(s.get("audio_fadeout") or 0, JOIN_FADE if join_after else 0)
        a = src["vt"][0] if s["start"] == 0 else src["start"] + s["start"]
        out.append((a, src["start"] + s["end"], fin, fout))
        starts.append(t)
        t += out[-1][1] - a
    return out, starts


def check(src_path: Path, out_path: Path, segments: list[dict], copied: bool, signal: str) -> str | None:
    if not smartcut.process_segments(str(src_path), str(out_path), [dict(s) for s in segments]):
        return "process_segments failed"
    src, out = read(src_path), read(out_path)
    same = sum(p in set(src["packets"]) for p in out["packets"])
    if copied and same < len(out["packets"]):
        return f"only {same}/{len(out['packets'])} audio packets copied, expected a copy"
    if not copied and same > 2:  # a re-encode can repeat a source packet by chance (silence)
        return f"{same}/{len(out['packets'])} audio packets copied, expected a re-encode"
    if src["rate"] != out["rate"]:
        return f"audio at {out['rate']} Hz, the source at {src['rate']} Hz"
    rate = src["rate"]
    segs, seg_out = expected(segments, src)
    tolerance = 0.0015 if out_path.suffix in (".mkv", ".webm") else 0.001  # Matroska: timestamps in 1 ms

    # Video where it belongs, and A/V sync per segment (noise sources)
    n = 0
    for k, (a, b, _, _) in enumerate(segs):
        kept = [t for t in src["vt"] if a - 1e-6 <= t < b - 1e-6]
        S, V = kept[0], out["vt"][n]
        n += len(kept)
        if abs((V - seg_out[k]) - (S - a)) > tolerance:
            return f"segment {k + 1}'s video starts at {V:.4f} s, expected {seg_out[k] + S - a:.4f} s"
        if signal == "noise":
            for u in (0.05, kept[-1] - S - 0.35):
                late = lag(src, out, V + u, S + u)
                if abs(late) > tolerance:
                    return f"segment {k + 1}: audio {late * 1000:+.2f} ms late at {u:.2f} s into its video"

    # Level against the ideal output: the source per segment, faded as expected (10 ms windows)
    total = round(seg_out[-1] * rate + (segs[-1][1] - segs[-1][0]) * rate)
    ideal = np.zeros(total)
    for (a, b, fin, fout), o0 in zip(segs, seg_out):
        i0, m = round(o0 * rate), round((b - a) * rate)
        j0 = round((a - src["a0"]) * rate)
        u = np.arange(m) / rate
        gain = np.ones(m)
        if fin:
            gain = np.minimum(gain, np.clip(u / fin, 0, 1))
        if fout:
            gain = np.minimum(gain, np.clip(((b - a) - u) / fout, 0, 1))
        seg = src["x"][j0:j0 + m]
        ideal[i0:i0 + len(seg)] = seg * gain[:len(seg)]
    lead = round(out["a0"] * rate)  # output time 0 on (encoder priming at negative times in Matroska)
    o = out["x"][-lead:] if lead < 0 else np.concatenate([np.zeros(lead), out["x"]])
    win = rate // 100
    bad = []
    cold = not copied and segs[0][0] > src["vt"][0] and (out_path.suffix in (".mkv", ".webm") or "opus" in out["codec"])
    for i in range(int(0.02 * rate) if cold else 0, min(len(o), total) - win, win):
        ri = np.sqrt((ideal[i:i + win] ** 2).mean())
        if ri > 0.05 * AMPLITUDE:
            r = np.sqrt((o[i:i + win] ** 2).mean()) / ri
            if not 0.85 <= r <= 1.2:
                bad.append((round(i / rate, 2), round(float(r), 2)))
    if bad:
        return f"level against the ideal outside 0.85-1.2 at (s, ratio) {bad[:6]}"
    if abs(len(o) / rate - total / rate) > 0.03:
        return f"{len(o) / rate:.3f} s of audio from the start, expected {total / rate:.3f} s"

    # Clicks at the joins (tone sources)
    if signal == "tone" and len(segs) > 1:
        steady = np.abs(np.diff(src["x"][int(0.5 * rate):int(1.5 * rate)])).max()
        d = np.abs(np.diff(o))
        w = int(0.01 * rate)
        for k in range(1, len(segs)):
            j = round(seg_out[k] * rate)
            ratio = d[j - w:j + w].max() / steady
            if ratio > 1.2:
                return f"click at join {k}: largest step {ratio:.2f}x the tone's own"
    return None


def by_pts(path: Path) -> dict:
    """The first audio track with each decoded frame at its timestamp (silence where the source has none)."""
    with av.open(str(path)) as c:
        rate = c.streams.audio[0].sample_rate
        frames = [(float(f.pts * f.time_base), f.to_ndarray().astype(np.float64).mean(axis=0)) for f in c.decode(audio=0)]
    a0 = frames[0][0]
    x = np.zeros(round((frames[-1][0] - a0) * rate) + len(frames[-1][1]))
    for t, samples in frames:
        i = round((t - a0) * rate)
        x[i:i + len(samples)] = samples
    return {"rate": rate, "a0": a0, "x": x, "frames": [(t, t + len(s) / rate) for t, s in frames]}


def check_gap(src_path: Path, out_path: Path) -> str | None:
    """GAP_SEGMENTS of a source without audio in GAP: silence where the source has none, the rest at the source's
    level and in sync."""
    if not smartcut.process_segments(str(src_path), str(out_path), [dict(s) for s in GAP_SEGMENTS]):
        return "process_segments failed"
    src, out, src_info = by_pts(src_path), read(out_path), read(src_path)
    rate, st = src["rate"], src_info["start"]
    (a1, b1), (a2, b2) = ((st + s["start"], st + s["end"]) for s in GAP_SEGMENTS)  # in the source's timestamps
    j2 = b1 - a1  # the join, in the output
    gap = (max(e for s, e in src["frames"] if s < GAP[0]), min(s for s, _ in src["frames"] if s >= GAP[0]))
    lead = round(out["a0"] * rate)
    o = out["x"][-lead:] if lead < 0 else np.concatenate([np.zeros(lead), out["x"]])
    if abs(len(o) / rate - (j2 + b2 - a2)) > 0.03:
        return f"{len(o) / rate:.3f} s of audio, expected {j2 + b2 - a2:.3f} s (silence where the source has none)"
    rms = lambda x: float(np.sqrt((x ** 2).mean()))  # noqa: E731
    m = 0.025  # around the source's own edges of the gap (codec ringing)
    for lo, hi in ((gap[0] - a1 + m, j2 - m), (j2 + m, j2 + gap[1] - a2 - m)):
        if rms(o[round(lo * rate):round(hi * rate)]) > 0.01 * AMPLITUDE:
            return f"sound at {lo:.2f}-{hi:.2f} s, where the source has none"
    win = rate // 100
    for lo, hi, src_at in ((0.0, gap[0] - a1 - m, a1), (j2 + gap[1] - a2 + m, j2 + b2 - a2 - m, a2 - j2)):
        for i in range(round(lo * rate), round(hi * rate) - win, win):
            j = round((i / rate + src_at - src["a0"]) * rate)
            r = rms(o[i:i + win]) / max(rms(src["x"][j:j + win]), 1e-9)
            if not 0.85 <= r <= 1.2:
                return f"level {r:.2f} of the source at {i / rate:.2f} s"
    s_vt = src_info["vt"]
    first = [next(v for v in s_vt if v >= a - 1e-6) for a in (a1, a2)]  # each segment's first frame in the source
    n1 = sum(1 for v in s_vt if a1 - 1e-6 <= v < b1 - 1e-6)
    for (S, V, u) in ((first[0], out["vt"][0], 0.3), (first[1], out["vt"][n1], gap[1] - a2 + 0.2)):
        late = lag(src, out, V + u, S + u)
        if abs(late) > 0.001:
            return f"audio {late * 1000:+.2f} ms late at {S + u:.2f} s of the source"
    return None


def main() -> int:
    media = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    cases = [  # name, video encoder, audio encoder, rate, container, segments, copied
        ("AAC in MP4: joins", "libx264", "aac", 44100, "mp4", JOINS, False),
        ("AAC in MP4: mid-stream start, copied", "libx264", "aac", 44100, "mp4", START, True),
        ("AAC in MP4: touching segments, copied", "libx264", "aac", 44100, "mp4", TOUCHING, True),
        ("AAC in MP4: joins with short and long fades", "libx264", "aac", 44100, "mp4", FADES, False),
        ("AAC in MKV: joins", "libx264", "aac", 48000, "mkv", JOINS, False),
        ("AAC in MKV: mid-stream start", "libx264", "aac", 48000, "mkv", START, False),
        ("AAC in MKV: from the start, copied", "libx264", "aac", 48000, "mkv", FROM_0, True),
        ("AC-3 in MP4: joins", "libx264", "ac3", 48000, "mp4", JOINS, False),
        ("AC-3 in MP4: mid-stream start, copied", "libx264", "ac3", 48000, "mp4", START, True),
        ("MP3 in MP4: joins", "libx264", "libmp3lame", 44100, "mp4", JOINS, False),
        ("MP3 in MP4: mid-stream start", "libx264", "libmp3lame", 44100, "mp4", START, False),
        ("Opus in WebM: joins", "libvpx-vp9", "libopus", 48000, "webm", JOINS, False),
        ("Opus in WebM: mid-stream start", "libvpx-vp9", "libopus", 48000, "webm", START, False),
        ("Opus in WebM: from the start, copied", "libvpx-vp9", "libopus", 48000, "webm", FROM_0, True),
        ("Opus in MP4: joins", "libvpx-vp9", "libopus", 48000, "mp4", JOINS, False),
        ("Opus in MP4: mid-stream start, copied", "libvpx-vp9", "libopus", 48000, "mp4", START, True),
    ]
    failed = total = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        made = {}
        for name, vcodec, acodec, rate, ext, segments, copied in cases:
            for signal in ("noise", "tone"):
                if signal == "tone" and len(segments) == 1:
                    continue  # no join: nothing to click
                key = (vcodec, acodec, rate, ext, signal)
                if key not in made:
                    made[key] = tmp / f"src{len(made)}.{ext}"
                    make_source(made[key], vcodec, acodec, rate, signal)
                total += 1
                problem = check(made[key], tmp / f"out{total}.{ext}", segments, copied, signal)
                failed += problem is not None
                print(f"✗ FAIL - {name} ({signal}): {problem}" if problem else f"✓ PASS - {name} ({signal})")
        for ext in ("mp4", "mkv"):
            total += 1
            src = tmp / f"gap.{ext}"
            make_source(src, "libx264", "aac", 48000, "noise", gap=GAP)
            problem = check_gap(src, tmp / f"out{total}.{ext}")
            failed += problem is not None
            name = f"AAC in {ext.upper()}: no source audio across a join"
            print(f"✗ FAIL - {name}: {problem}" if problem else f"✓ PASS - {name}")
        heaac = media / "synth_heaac.mp4" if media else None
        if heaac is not None and heaac.exists():  # real (noise-like) HE-AAC audio
            for name, segments, copied in (("HE-AAC in MP4: joins", JOINS, False),
                                           ("HE-AAC in MP4: mid-stream start, copied", START, True),
                                           ("HE-AAC in MKV: mid-stream start", START, False)):
                total += 1
                ext = "mkv" if "MKV" in name else "mp4"
                problem = check(heaac, tmp / f"out{total}.{ext}", segments, copied, "noise")
                failed += problem is not None
                print(f"✗ FAIL - {name}: {problem}" if problem else f"✓ PASS - {name}")
        elif media is not None:
            print(f"(no {heaac}: HE-AAC cases skipped)")
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
