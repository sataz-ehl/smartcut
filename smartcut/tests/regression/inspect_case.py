"""Print where a case's measured fade/timing deviates. Usage: python inspect_case.py <out dir> <case> [audio|video|gaps]"""
import sys
from pathlib import Path

import numpy as np

out, case = Path(sys.argv[1]), sys.argv[2]
what = sys.argv[3] if len(sys.argv) > 3 else "audio"
z = np.load(out / f"{case}.npz")
if what == "gaps":
    vt = z["vt"]
    d = np.diff(vt)
    nominal = np.median(d)
    for i in np.where((d > nominal * 1.5) | (d <= 0))[0]:
        print(f"  frame {i}->{i + 1}: t={vt[i]:.3f}->{vt[i + 1]:.3f}  gap={d[i] / nominal:.2f} frames")
    print(f"  frames={len(vt)} first={vt[0]:.3f} last={vt[-1]:.3f} nominal={nominal * 1000:.2f} ms")
    sys.exit()
t, g, ideal = (z["at"], z["a_gain"], z["a_ideal"]) if what == "audio" else (z["vt"], z["v_alpha"], z["v_ideal"])
err = np.abs(g - ideal)
bad = np.where(np.isfinite(err) & (err > 0.15))[0]
print(f"  {what}: {len(bad)} of {len(t)} samples deviate by >0.15")
# print contiguous runs of bad samples
if len(bad):
    runs = np.split(bad, np.where(np.diff(bad) > 1)[0] + 1)
    for r in runs[:12]:
        seg = slice(r[0], r[-1] + 1)
        print(f"  t={t[r[0]]:.2f}..{t[r[-1]]:.2f}s  measured {np.nanmin(g[seg]):.2f}..{np.nanmax(g[seg]):.2f}"
              f"  ideal {np.nanmin(ideal[seg]):.2f}..{np.nanmax(ideal[seg]):.2f}")
