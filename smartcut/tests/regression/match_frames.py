"""Find which source frame each output frame is (exact luma match = copied, nearest = re-encoded).

Usage: python match_frames.py <source.mp4> <output.mp4> <first output frame> <last output frame>
"""
import hashlib
import sys

import av
import numpy as np


def frames(path):
    out = []
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            p = f.planes[0]
            y = np.frombuffer(p, np.uint8).reshape(f.height, p.line_size)[:, :f.width]
            out.append((float(f.pts * f.time_base), hashlib.md5(y.tobytes()).hexdigest(), y[::8, ::8].astype(np.int16)))
    return out


src, dst = frames(sys.argv[1]), frames(sys.argv[2])
lo, hi = int(sys.argv[3]), int(sys.argv[4])
by_hash = {h: i for i, (_, h, _) in enumerate(src)}
for i in range(max(lo, 0), min(hi, len(dst) - 1) + 1):
    t, h, small = dst[i]
    if h in by_hash:
        j = by_hash[h]
        how = "copied"
    else:  # nearest source frame by mean absolute difference (re-encoded or faded frame)
        diffs = [np.abs(small - s[2]).mean() for s in src]
        j = int(np.argmin(diffs))
        how = f"re-encoded (closest, diff {diffs[j]:.1f})"
    print(f"  out #{i:<4} t={t:7.3f}s  <- source #{j:<5} t={src[j][0]:7.3f}s  {how}")
