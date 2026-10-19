#!/usr/bin/env python3
"""
Smoke test of tests/regression/calibrate_progress.py on a short synthetic clip (H.264 only, --quick):

1. --make-media makes the clips and a manifest
2. The calibration runs: it checks the cores, fits, compares, and writes a progress_costs.json (adopted) or a
   progress_costs.candidate.json (not); either loads with progress_costs.parse()
3. A changed clip, or one the calibration needs but the media lacks, is refused

Synthetic clips re-encode differently from real ones: this checks the harness works, not the costs. Needs the
ffmpeg command (on PATH, or SMARTCUT_TEST_FFMPEG); skipped without it.

Usage (from the repo root):
    python smartcut/tests/test_calibrate_progress.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from smartcut import progress_costs  # noqa: E402
from smartcut.tests.generate_synthetic_video import generate_test_video  # noqa: E402

HARNESS = ROOT / "smartcut" / "tests" / "regression" / "calibrate_progress.py"


def harness(*args: str) -> subprocess.CompletedProcess:
    ffmpeg = os.environ.get("SMARTCUT_TEST_FFMPEG") or shutil.which("ffmpeg")
    return subprocess.run([sys.executable, "-X", "utf8", str(HARNESS), *args, "--ffmpeg", str(ffmpeg)],
                          capture_output=True, text=True, timeout=1800, check=False)


def main() -> int:
    if not (os.environ.get("SMARTCUT_TEST_FFMPEG") or shutil.which("ffmpeg")):
        print("SKIPPED: no ffmpeg command (put it on PATH or set SMARTCUT_TEST_FFMPEG)")
        return 0
    failed, total = 0, 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failed, total
        total += 1
        failed += not ok
        print(f"{'✓ PASS' if ok else '✗ FAIL'} - {name}" + ("" if ok else f": {detail[-3000:]}"))

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        clip, media, out = tmp / "clip.mp4", tmp / "media", tmp / "costs" / "progress_costs.json"
        generate_test_video(str(clip), duration=35)
        r = harness("--make-media", str(media), str(clip), "--quick", "--codecs", "h264", "--resolutions", "360")
        manifest = media / "manifest.json"
        check("--make-media makes the clips and a manifest", r.returncode == 0 and manifest.exists()
              and (media / "h264_360p.mp4").exists(), r.stdout + r.stderr)

        r = harness("--media", str(media), "--quick", "--codecs", "h264", "--cores", "1", "--out", str(out))
        written = out if out.exists() else out.with_name("progress_costs.candidate.json")
        ok = r.returncode == 0 and written.exists() and "cores: 1 expected" in r.stdout and ("ADOPTED" in r.stdout)
        try:
            costs = progress_costs.parse(json.loads(written.read_text()))
            ok = ok and "h264" in costs.recode and json.loads(written.read_text())["measured"]["cores"] == 1
        except Exception as e:
            ok, r.stdout = False, f"{r.stdout}\n{e!r}"
        check("the calibration writes costs that load (adopted or as a candidate)", ok, r.stdout + r.stderr)

        r = harness("--media", str(media), "--quick", "--codecs", "h264,vp9", "--estimate")
        check("a clip the media lacks is refused", r.returncode != 0 and "vp9_360p.mp4" in r.stdout + r.stderr, r.stdout + r.stderr)

        with open(media / "h264_360p.mp4", "ab") as f:
            f.write(b"\0")
        r = harness("--media", str(media), "--quick", "--codecs", "h264", "--estimate")
        check("a changed clip is refused", r.returncode != 0 and "changed" in r.stdout + r.stderr, r.stdout + r.stderr)
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
