"""PSNR (luma) of a run of output frames against the matching source frames.

Usage: python psnr_frames.py <source> <output> <output first frame> <count> <source first frame> [<output> ...]
Several outputs can be given (same frame ranges) to compare encoder settings.
"""
import sys
from pathlib import Path

import av
import numpy as np


def luma_frames(path: str, first: int, count: int) -> list[np.ndarray]:
    out = []
    with av.open(path) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i < first:
                continue
            if i >= first + count:
                break
            p = f.planes[0]
            out.append(np.frombuffer(p, np.uint8).reshape(f.height, p.line_size)[:, : f.width].astype(np.float64))
    return out


src_path, out_first, count, src_first = sys.argv[1], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
outputs = [sys.argv[2], *sys.argv[6:]]
src = luma_frames(src_path, src_first, count)
for path in outputs:
    frames = luma_frames(path, out_first, count)
    psnr = [10 * np.log10(255 ** 2 / max(np.mean((a - b) ** 2), 1e-10)) for a, b in zip(frames, src)]
    print(f"  {Path(path).parent.name + '/' + Path(path).name:<40} frames {out_first}-{out_first + len(psnr) - 1}: "
          f"PSNR mean {np.mean(psnr):5.1f} dB, min {np.min(psnr):5.1f} dB")
