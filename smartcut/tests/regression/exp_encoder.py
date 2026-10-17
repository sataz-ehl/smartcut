#!/usr/bin/env python3
"""
Run run_cases.py with a different encoder or encoder options for smartcut's re-encoded GOPs (experiments only;
smartcut itself is not changed).

Usage:
    python exp_encoder.py <config.json> <media dir> <out dir> [case ...] [--one-core] [--repo PATH]

config.json maps smartcut's internal codec names (VideoCutter.codec_name: "h264", "hevc", "vp9", "av1") to an
optional encoder and options, e.g.:
    {"av1": {"encoder": "libsvtav1", "options": {"preset": "10"}},
     "vp9": {"options": {"cpu-used": "5", "row-mt": "1"}}}
Options are applied on top of smartcut's own (crf, profile, speed settings).
"""
import json
import runpy
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
run_cases_args = sys.argv[2:]
repo = Path(__file__).resolve().parents[3]
if "--repo" in run_cases_args:
    repo = Path(run_cases_args[run_cases_args.index("--repo") + 1])
sys.path.insert(0, str(repo))

import smartcut.video_cutter as vc  # noqa: E402

original_ensure = vc.VideoCutter._ensure_enc_codec


def patched_ensure(self):
    if self.enc_codec is not None:
        return
    cfg = config.get(self.codec_name)
    if not cfg:
        return original_ensure(self)
    internal_name = self.codec_name
    self.codec_name = cfg.get("encoder", internal_name)
    try:
        original_ensure(self)
    finally:
        self.codec_name = internal_name
    self.enc_codec.options.update(cfg.get("options", {}))


vc.VideoCutter._ensure_enc_codec = patched_ensure
print(f"encoder config: {config}", flush=True)
sys.argv = [str(Path(__file__).with_name("run_cases.py")), *run_cases_args]
runpy.run_path(sys.argv[0], run_name="__main__")
