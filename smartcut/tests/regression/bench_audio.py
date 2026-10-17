"""How much does whole-track audio re-encoding cost?

Usage: python bench_audio.py <smartcut repo root> <video> <out dir>

1. process_segments() on the whole video with video fades only (audio copied) vs with audio fades too
   (whole audio track re-encoded).
2. Decode + AAC-encode the whole audio track at 128 kbps with FFmpeg's default coder ("twoloop") vs
   aac_coder=fast, to see what a faster encoder setting would save.
"""
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, sys.argv[1])
import smartcut  # noqa: E402

src, out = Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
with av.open(str(src)) as c:
    duration = c.duration / 1e6
print(f"source: {src.name}, {duration:.1f} s")

for label, seg in (("video fades only (audio copied)", {"start": 0, "end": None, "video_fadein": 2, "video_fadeout": 2}),
                   ("video + audio fades (audio re-encoded)", {"start": 0, "end": None, "fadein": 2, "fadeout": 2})):
    t = time.perf_counter()
    ok = smartcut.process_segments(str(src), str(out / "bench.mp4"), [seg])
    print(f"process_segments, {label:<40} {time.perf_counter() - t:6.2f} s  ok={ok}")


def transcode(coder):
    with av.open(str(src)) as c:
        a = c.streams.audio[0]
        enc = av.CodecContext.create("aac", "w")
        enc.sample_rate = a.codec_context.sample_rate
        enc.layout = a.codec_context.layout
        enc.format = "fltp"
        enc.bit_rate = 128_000
        enc.time_base = Fraction(1, enc.sample_rate)
        if coder:
            enc.options = {"aac_coder": coder}
        enc.open()
        n = 0
        t = time.perf_counter()
        for frame in c.decode(a):
            frame.pts = n
            frame.time_base = enc.time_base
            n += frame.samples
            enc.encode(frame)
        enc.encode(None)
        return time.perf_counter() - t, n / enc.sample_rate


for coder in (None, "fast"):
    secs, audio_s = transcode(coder)
    print(f"decode+encode whole audio, aac_coder={coder or 'twoloop (default)':<20} {secs:6.2f} s  ({audio_s / secs:5.0f}x real time)")
