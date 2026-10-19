#!/usr/bin/env python3
"""
Tests of tests/regression/calibrate_progress.py.

Unit tests (no encoding):
1. The adoption rule: adopted only if both checks are more even by ADOPT_MARGIN, never with --quick, when the
   pinning didn't hold, when the CPU check warned, without the Facebook clip or without a held-out check
2. The held-out edits (the app's height and the Facebook clip's, always), and when there is no held-out check
3. The costs fitted without the held-out edits use only the other edits' runs and copy / audio times
4. The CPU checks: an encoder run or a real run on more cores than expected, or less than LOW_CPU of one core
5. The files: adopted, the file is replaced through a temporary file in its folder and the one before kept as
   progress_costs.previous.json; not adopted, only progress_costs.candidate.json is written
6. The measuring processes release memory after every edit

Smoke test of the harness on a short synthetic clip (H.264 only, --quick; needs the ffmpeg command on PATH or in
SMARTCUT_TEST_FFMPEG, skipped without it; ~1 min):
7. --make-media makes the clips and a manifest
8. A --quick run checks the cores, fits, compares and says NOT ADOPTED: the file SMARTCUT_PROGRESS_COSTS names is
   left as it was, and the candidate it writes loads with progress_costs.parse()
9. A changed clip, or one the calibration needs but the media lacks, is refused

Synthetic clips re-encode differently from real ones: the smoke test checks the harness works, not the costs.

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
REGRESSION = ROOT / "smartcut" / "tests" / "regression"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(REGRESSION))

import calibrate_progress as cp  # noqa: E402  # pyright: ignore[reportMissingImports]
import progress_timeline  # noqa: E402  # pyright: ignore[reportMissingImports]

from smartcut import progress_costs  # noqa: E402
from smartcut.tests.generate_synthetic_video import generate_test_video  # noqa: E402

HARNESS = REGRESSION / "calibrate_progress.py"
M = cp.ADOPT_MARGIN


# ------------------------------------------------------------------------------------------------------ unit tests

def test_rule() -> None:
    ok = {"quick": False, "pinned": True, "quiet": True, "facebook": True, "no_split": None}
    better, worse = (6.0, 6.0 - M), (5.0, 5.0 - M + 0.01)
    assert cp.decide(**ok, held_out=better, real=better) == (True, []), "both checks better by the margin: adopted"
    for held, real in ((better, worse), (worse, better), (worse, worse), ((4.0, 4.5), better)):
        adopt, reasons = cp.decide(**ok, held_out=held, real=real)
        assert not adopt and len(reasons) == 1 and "not more even by" in reasons[0], (held, real, reasons)
    for change, words in (({"quick": True}, "--quick"), ({"pinned": False}, "pinning"), ({"quiet": False}, "CPU check"),
                          ({"facebook": False}, "no Facebook clip: no held-out footage"),
                          ({"no_split": "one resolution"}, "no held-out check: one resolution")):
        adopt, reasons = cp.decide(**{**ok, **change}, held_out=better, real=better)
        assert not adopt and reasons == [r for r in reasons if words in r] and reasons, (change, reasons)


def manifest_for(heights: list[int], facebook: bool) -> dict:
    files = {f"{c}_{h}p.mp4": {"codec": c, "height": h, "facebook": False} for c in ("h264", "vp9") for h in heights}
    if facebook:
        files["fb.mp4"] = {"codec": "h264", "height": 360, "facebook": True}
    return {"excerpt": 90.0, "files": files}


def test_held_out() -> None:
    manifest = manifest_for([360, 720, 1080], True)
    cases, app, held = cp.plan_cases(manifest, ["h264", "vp9"], [360, 720, 1080], False)
    assert held == ["P2_h264_720p", "P2F_h264_720p", "P3_h264_720p", "P3F_h264_720p", "P2_vp9_720p", "P2F_vp9_720p", "FB2", "FB2F"], held
    assert set(app) <= set(held), "the real runs must be of held-out edits"
    assert {"FB2", "FB2F"} <= set(held), "the Facebook clip's edits must be held out"
    assert cp.split_problem(cases, held, manifest) is None
    one = manifest_for([720], False)
    cases, app, held = cp.plan_cases(one, ["h264", "vp9"], [720], False)
    assert len(held) == len(cases) and "one resolution" in cp.split_problem(cases, held, one)
    cases, app, held = cp.plan_cases(manifest, ["h264", "vp9"], [360], True)  # --quick
    assert cp.split_problem(cases, held, manifest), "--quick has no held-out check"
    mixed = manifest_for([360, 720], False)  # vp9 at 720p only: nothing to fit it on
    mixed["files"].pop("vp9_360p.mp4")
    cases, app, held = cp.plan_cases(mixed, ["h264", "vp9"], [360, 720], False)
    assert "vp9" in (cp.split_problem(cases, held, mixed) or ""), cp.split_problem(cases, held, mixed)


def test_fit_without_held_out() -> None:
    def run(case: str, mp: float, s: float) -> dict:
        return {"codec": "h264", "mp": mp, "s": s, "cpu": s, "n": 100, "busy": 100.0, "faded": 0, "case": case}
    stats = {"A": {"copy_s": 0.1, "copy_frames": 1000, "audio_s": 1.0, "audio_media_s": 40.0},
             "B": {"copy_s": 0.1, "copy_frames": 1000, "audio_s": 1.0, "audio_media_s": 40.0},
             "C": {"copy_s": 5.0, "copy_frames": 1000, "audio_s": 30.0, "audio_media_s": 40.0}}
    measured = {"runs": [run("A", 0.23, 0.6), run("B", 2.07, 5.2), run("C", 0.92, 30.0)],
                "edits": {n: {"stats": v} for n, v in stats.items()},
                "stats": {k: sum(v[k] for v in stats.values()) for k in stats["A"]}}
    without_c = {"runs": measured["runs"][:2], "edits": {n: measured["edits"][n] for n in "AB"},
                 "stats": {k: stats["A"][k] + stats["B"][k] for k in stats["A"]}}
    held_fit = cp.fit(measured, progress_costs.BUILT_IN, ["A", "B"])[0]
    expected = cp.fit(without_c, progress_costs.BUILT_IN)[0]
    assert held_fit.recode["h264"] == expected.recode["h264"], (held_fit.recode["h264"], expected.recode["h264"])
    assert held_fit.copy_per_frame == expected.copy_per_frame and held_fit.audio_per_second == expected.audio_per_second
    assert cp.fit(measured, progress_costs.BUILT_IN)[0].recode["h264"] != held_fit.recode["h264"], "C must change the full fit"


def test_cpu_checks() -> None:
    def measured(cores: int, ratio: float) -> dict:
        return {"info": {"expected_cores": cores}, "runs": [{"codec": "h264", "s": 2.0, "cpu": 2.0 * ratio}] * 3}
    assert cp.core_check(measured(1, 0.99))[:2] == ([], []), "an idle core: no warning"
    escaped, loaded, _ = cp.core_check(measured(1, 1.6))
    assert escaped and not loaded, "1.6 cores on one: the pinning didn't hold"
    escaped, loaded, _ = cp.core_check(measured(1, 0.7))
    assert loaded and not escaped, "0.7 of one core: other load"
    assert cp.core_check(measured(4, 2.5))[:2] == ([], []), "four cores: no load check"
    assert cp.core_check(measured(4, 5.0))[0], "five cores on four"

    def real(cores: int, ratio: float, end: float = 3.0) -> dict:
        return {"info": {"expected_cores": cores}, "edits": {"P2": {"end": end, "cpu": end * ratio}}}
    assert cp.real_core_check(real(1, 0.98)) == ([], [])
    assert cp.real_core_check(real(1, 0.7))[1] and not cp.real_core_check(real(1, 0.7))[0]
    assert cp.real_core_check(real(1, 1.5))[0]
    assert cp.real_core_check(real(1, 0.5, end=0.5)) == ([], []), "too short to judge"


def test_files(tmp: Path) -> None:
    folder = tmp / "costs"
    folder.mkdir()
    target = folder / "progress_costs.json"
    target.write_text("old\n", encoding="utf-8")
    replaced = []
    real_replace = os.replace

    def recording_replace(src, dst):  # noqa: ANN001, ANN202
        replaced.append((Path(src), Path(dst)))
        real_replace(src, dst)

    os.replace = recording_replace
    try:
        written = cp.save_costs(target, "new\n", False)
        assert written == folder / "progress_costs.candidate.json" and written.read_text() == "new\n"
        assert target.read_text() == "old\n", "not adopted: the file must stay as it was"
        written = cp.save_costs(target, "new\n", True)
    finally:
        os.replace = real_replace
    assert written == target and target.read_text() == "new\n"
    assert (folder / "progress_costs.previous.json").read_text() == "old\n", "the file before must be kept"
    assert {dst for _, dst in replaced} == {target, folder / "progress_costs.previous.json", folder / "progress_costs.candidate.json"}, replaced
    assert all(src.parent == dst.parent and src != dst for src, dst in replaced), f"not through a temporary file next to it: {replaced}"
    assert sorted(p.name for p in folder.iterdir()) == ["progress_costs.candidate.json", "progress_costs.json",
                                                        "progress_costs.previous.json"], "temporary files left"
    real_replace = os.replace

    def failing_replace(src, dst):  # noqa: ANN001, ANN202
        raise OSError("disk full")

    os.replace = failing_replace
    try:
        cp.save_costs(target, "newer\n", True)
        raise AssertionError("the failure must be reported")
    except OSError:
        pass
    finally:
        os.replace = real_replace
    assert target.read_text() == "new\n" and len(list(folder.iterdir())) == 3, "a failed write must leave the file and no temporary"


def test_memory_released() -> None:
    events: list = []
    saved = (cp.child_setup, cp.install_recorder, cp.run_edit, cp.release_memory, progress_timeline.install_timers)
    cp.child_setup = lambda cores: {"available_cpus": 1, "expected_cores": 1}
    cp.install_recorder = lambda record: None
    progress_timeline.install_timers = lambda stats, runs: None
    cp.run_edit = lambda media, case, out_dir, record=None: events.append(case["name"]) or {"ok": True, "end": 1.0}
    cp.release_memory = lambda: events.append("released")
    try:
        spec = {"cores": None, "media": ".", "out_dir": ".", "cases": [{"name": "A"}, {"name": "B"}]}
        result = cp.child_measure(spec)
        assert events == ["A", "released", "B", "released"], f"calibration edits: {events}"
        assert all("rss_mb" in e and "stats" in e for e in result["edits"].values()), result["edits"]
        events.clear()
        cp.child_real(spec)
        assert events == ["A", "released", "B", "released"], f"real runs: {events}"
    finally:
        cp.child_setup, cp.install_recorder, cp.run_edit, cp.release_memory, progress_timeline.install_timers = saved
    cp.release_memory()  # the real one runs (malloc_trim on Linux)
    rss = cp.resident_mb()
    assert rss is None or 1 < rss < 100_000, rss


# ------------------------------------------------------------------------------------------------------ smoke test

def harness(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    ffmpeg = os.environ.get("SMARTCUT_TEST_FFMPEG") or shutil.which("ffmpeg")
    return subprocess.run([sys.executable, "-X", "utf8", str(HARNESS), *args, "--ffmpeg", str(ffmpeg)],
                          capture_output=True, text=True, timeout=1800, check=False, env=env)


def smoke(tmp: Path, check) -> None:  # noqa: ANN001
    clip, media = tmp / "clip.mp4", tmp / "media"
    generate_test_video(str(clip), duration=35)
    r = harness("--make-media", str(media), str(clip), "--quick", "--codecs", "h264", "--resolutions", "360")
    manifest = media / "manifest.json"
    check("--make-media makes the clips and a manifest", r.returncode == 0 and manifest.exists()
          and (media / "h264_360p.mp4").exists(), r.stdout + r.stderr)

    # The file smartcut reads, named as on the server; --quick must leave it alone
    target = tmp / "server" / "progress_costs.json"
    target.parent.mkdir()
    before = progress_costs.to_json(progress_costs.BUILT_IN, {"host": "before"})
    target.write_text(before, encoding="utf-8")
    r = harness("--media", str(media), "--quick", "--codecs", "h264", "--cores", "1",
                env={**os.environ, progress_costs.ENV_VAR: str(target)})
    candidate = target.with_name("progress_costs.candidate.json")
    verdicts = [line for line in r.stdout.splitlines() if line.startswith(("ADOPTED:", "NOT ADOPTED:"))]
    ok = (r.returncode == 0 and "cores: 1 expected" in r.stdout and len(verdicts) == 1
          and verdicts[0].startswith("NOT ADOPTED: --quick is a smoke check") and "no Facebook clip: no held-out footage" in verdicts[0]
          and target.read_text(encoding="utf-8") == before
          and not target.with_name("progress_costs.previous.json").exists())
    try:
        costs = progress_costs.parse(json.loads(candidate.read_text()))
        data = json.loads(candidate.read_text())
        ok = ok and "h264" in costs.recode and data["measured"]["cores"] == 1 and data["adopted"] is False
    except Exception as e:
        ok, r.stdout = False, f"{r.stdout}\n{e!r}"
    check("--quick never adopts: the file is unchanged, the candidate loads", ok, r.stdout + r.stderr)

    r = harness("--media", str(media), "--quick", "--codecs", "h264,vp9", "--estimate")
    check("a clip the media lacks is refused", r.returncode != 0 and "vp9_360p.mp4" in r.stdout + r.stderr, r.stdout + r.stderr)

    with open(media / "h264_360p.mp4", "ab") as f:
        f.write(b"\0")
    r = harness("--media", str(media), "--quick", "--codecs", "h264", "--estimate")
    check("a changed clip is refused", r.returncode != 0 and "changed" in r.stdout + r.stderr, r.stdout + r.stderr)


def main() -> int:
    failed, total = 0, 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failed, total
        total += 1
        failed += not ok
        print(f"{'✓ PASS' if ok else '✗ FAIL'} - {name}" + ("" if ok else f": {detail[-3000:]}"))

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name, test in (("The adoption rule: margin, --quick, pinning, CPU check, held-out check", test_rule),
                           ("The held-out edits, and when there is no held-out check", test_held_out),
                           ("The held-out fit uses only the other edits' runs and times", test_fit_without_held_out),
                           ("CPU checks: escaped pinning and other load, also in the real runs", test_cpu_checks),
                           ("Files: atomic replace, previous kept, candidate when not adopted", lambda: test_files(tmp)),
                           ("The measuring processes release memory after every edit", test_memory_released)):
            try:
                test()
                check(name, True)
            except (AssertionError, OSError) as e:
                check(name, False, repr(e))
        if os.environ.get("SMARTCUT_TEST_FFMPEG") or shutil.which("ffmpeg"):
            smoke(tmp, check)
        else:
            print("SKIPPED the smoke test: no ffmpeg command (put it on PATH or set SMARTCUT_TEST_FFMPEG)")
    print(f"\n{total - failed}/{total} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
