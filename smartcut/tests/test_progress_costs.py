#!/usr/bin/env python3
"""
Tests for progress_costs.py: a machine's progress_costs.json, and the fallback to the built-in costs.

1. No file: the built-in costs, logged as such
2. A valid file: its costs; codecs it doesn't list keep the built-in ones; to_json() output loads back
3. Malformed JSON, another format or cost model, wrong types, values out of range: a warning and the built-in costs
4. SMARTCUT_PROGRESS_COSTS names the file, else the per-user default (%APPDATA% / $XDG_CONFIG_HOME / ~/.config)
5. The costs are loaded and logged once per process
6. An edit with a broken file still succeeds (and logs a warning); one with a valid file uses it

Usage (from the repo root):
    python smartcut/tests/test_progress_costs.py
"""
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests live in smartcut/tests/)

import smartcut
from smartcut import progress_costs
from smartcut.tests.generate_synthetic_video import generate_test_video

VALID = {
    "format": 1, "model": 1,
    "recode_cost": {"h264": [0.002, 0.03, 0.5, 0.02], "vp9": [0.01, 0.05, 0.0, 0.0]},
    "copy_cost_per_frame": 0.00004, "audio_recode_cost_per_second": 0.03,
    "measured": {"date": "2026-10-19", "host": "test", "cores": 1},
}


class Captured(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def write(tmp: Path, name: str, content: object) -> Path:
    path = tmp / name
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


def warned(messages: list[tuple[int, str]], *words: str) -> bool:
    return any(level == logging.WARNING and all(w in text for w in words) for level, text in messages)


def test_missing(tmp: Path) -> None:
    costs, messages = progress_costs.load(tmp / "absent.json")
    assert costs is progress_costs.BUILT_IN, "a missing file must give the built-in costs"
    assert messages and messages[0][0] == logging.INFO and "built-in" in messages[0][1], messages


def test_valid(tmp: Path) -> None:
    costs, messages = progress_costs.load(write(tmp, "valid.json", VALID))
    assert costs.recode["h264"] == (0.002, 0.03, 0.5, 0.02) and costs.recode["vp9"] == (0.01, 0.05, 0.0, 0.0)
    assert costs.recode["av1"] == progress_costs.RECODE_COST["av1"], "codecs the file doesn't list keep their built-in costs"
    assert costs.copy_per_frame == 0.00004 and costs.audio_per_second == 0.03
    assert not any(level >= logging.WARNING for level, _ in messages), messages
    assert "host test" in messages[0][1], messages
    again = progress_costs.parse(json.loads(progress_costs.to_json(costs)))
    assert again.recode == costs.recode and again.copy_per_frame == costs.copy_per_frame, "to_json() must load back"


def test_rejected(tmp: Path) -> None:
    cases = {
        "malformed JSON": ("{not json", ()),
        "another format": ({**VALID, "format": 2}, ("format",)),
        "another cost model": ({**VALID, "model": 99}, ("model",)),
        "no recode costs": ({**VALID, "recode_cost": {}}, ("recode_cost",)),
        "three numbers": ({**VALID, "recode_cost": {"h264": [0.002, 0.03, 0.5]}}, ("h264",)),
        "a string": ({**VALID, "recode_cost": {"h264": ["0.002", 0.03, 0.5, 0.0]}}, ("not a number",)),
        "a boolean": ({**VALID, "copy_cost_per_frame": True}, ("copy_cost_per_frame",)),
        "busy above 1": ({**VALID, "recode_cost": {"h264": [0.002, 0.03, 1.5, 0.0]}}, ("busy", "outside")),
        "a negative cost": ({**VALID, "recode_cost": {"h264": [-0.002, 0.03, 0.5, 0.0]}}, ("fixed", "outside")),
        "no cost at all": ({**VALID, "recode_cost": {"h264": [0.0, 0.0, 0.5, 0.0]}}, ("nothing",)),
        "copy too dear": ({**VALID, "copy_cost_per_frame": 0.5}, ("copy_cost_per_frame", "outside")),
        "audio missing": ({k: v for k, v in VALID.items() if k != "audio_recode_cost_per_second"}, ("audio_recode_cost_per_second",)),
        "NaN": ('{"format": 1, "model": 1, "recode_cost": {"h264": [NaN, 0.03, 0.5, 0.0]}, "copy_cost_per_frame": 0.00004, '
                '"audio_recode_cost_per_second": 0.03}', ("not a number",)),
        "not an object": ("[1, 2, 3]", ("object",)),
    }
    for name, (content, words) in cases.items():
        costs, messages = progress_costs.load(write(tmp, "bad.json", content))
        assert costs is progress_costs.BUILT_IN, f"{name}: the built-in costs expected"
        assert warned(messages, "not used", *words), f"{name}: no warning naming {words}: {messages}"


def test_paths(tmp: Path) -> None:
    saved = {k: os.environ.get(k) for k in (progress_costs.ENV_VAR, "APPDATA", "XDG_CONFIG_HOME")}
    platform = sys.platform
    try:
        os.environ[progress_costs.ENV_VAR] = str(tmp / "named.json")
        assert progress_costs.progress_costs_path() == tmp / "named.json", "SMARTCUT_PROGRESS_COSTS must win"
        del os.environ[progress_costs.ENV_VAR]
        sys.platform = "win32"
        os.environ["APPDATA"] = str(tmp / "roaming")
        assert progress_costs.progress_costs_path() == tmp / "roaming" / "smartcut" / "progress_costs.json"
        sys.platform = "linux"
        os.environ["XDG_CONFIG_HOME"] = str(tmp / "xdg")
        assert progress_costs.progress_costs_path() == tmp / "xdg" / "smartcut" / "progress_costs.json"
        del os.environ["XDG_CONFIG_HOME"]
        assert progress_costs.progress_costs_path() == Path.home() / ".config" / "smartcut" / "progress_costs.json"
    finally:
        sys.platform = platform
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def with_env(path: Path):  # noqa: ANN201 (a context manager)
    class Env:
        def __enter__(self) -> Captured:
            self.saved = os.environ.get(progress_costs.ENV_VAR)
            os.environ[progress_costs.ENV_VAR] = str(path)
            progress_costs.reset()
            self.handler = Captured()
            progress_costs.log.addHandler(self.handler)
            self.level = progress_costs.log.level
            progress_costs.log.setLevel(logging.INFO)
            return self.handler

        def __exit__(self, *exc: object) -> None:
            progress_costs.log.removeHandler(self.handler)
            progress_costs.log.setLevel(self.level)
            if self.saved is None:
                os.environ.pop(progress_costs.ENV_VAR, None)
            else:
                os.environ[progress_costs.ENV_VAR] = self.saved
            progress_costs.reset()
    return Env()


def test_logged_once(tmp: Path) -> None:
    with with_env(write(tmp, "once.json", VALID)) as log:
        first, second = progress_costs.current(), progress_costs.current()
        assert first is second and first.recode["h264"] == (0.002, 0.03, 0.5, 0.02)
        assert len(log.records) == 1 and "once.json" in log.records[0].getMessage(), [r.getMessage() for r in log.records]


def test_edit_with_files(tmp: Path) -> None:
    src = tmp / "input.mp4"
    generate_test_video(str(src), duration=6)
    segments = [{"start": 1, "end": 2.5}, {"start": 3.5, "end": 5}]
    updates: list[int] = []
    with with_env(write(tmp, "broken.json", "{broken")) as log:
        assert smartcut.process_segments(str(src), str(tmp / "out1.mp4"), segments, progress_callback=lambda c, n: updates.append(c))
        assert updates and updates[-1] == 1000, "progress must still reach 1000"
        assert any(r.levelno == logging.WARNING and "broken.json" in r.getMessage() for r in log.records), "no warning logged"
        assert progress_costs.current() is progress_costs.BUILT_IN
    with with_env(write(tmp, "used.json", VALID)):
        assert smartcut.process_segments(str(src), str(tmp / "out2.mp4"), segments, progress_callback=lambda c, n: None)
        assert progress_costs.current().source == str(tmp / "used.json")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tests = [("No file: the built-in costs", test_missing),
                 ("A valid file is used; unlisted codecs keep the built-in costs", test_valid),
                 ("Malformed, other versions, wrong types, out of range: warning and fallback", test_rejected),
                 ("SMARTCUT_PROGRESS_COSTS, else the per-user default path", test_paths),
                 ("Loaded and logged once per process", test_logged_once),
                 ("Edits work with a broken file, and use a valid one", test_edit_with_files)]
        failed = 0
        for name, test in tests:
            try:
                test(tmp)
                print(f"✓ PASS - {name}")
            except (AssertionError, OSError) as e:
                failed += 1
                print(f"✗ FAIL - {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
