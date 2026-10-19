"""
The costs smart_cut's progress estimate weighs the work with (fork addition, see smart_cut._WorkProgress).

Only their ratios matter, and those depend on the machine: more cores make the encoders faster, while copying and
audio stay on one thread. So a machine can have its own costs in a progress_costs.json, written by
tests/regression/calibrate_progress.py. They are read from the file SMARTCUT_PROGRESS_COSTS names, else from
progress_costs_path()'s default: ~/.config/smartcut/progress_costs.json ($XDG_CONFIG_HOME if set), on Windows
%APPDATA%\\smartcut\\progress_costs.json. The file is machine-specific and lives outside the smartcut folder, which is
deployed by copying it over. Without a file, or with one that is malformed, from another cost model or out of range,
the built-in values below are used (with a warning for a bad file); a progress estimate never stops an edit.

The file is read once per process, at its first edit, and the costs in use are logged then (logger "smartcut",
INFO).
"""
import functools
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av

log = logging.getLogger("smartcut")

ENV_VAR = "SMARTCUT_PROGRESS_COSTS"
FORMAT = 1  # progress_costs.json's layout
MODEL = 1  # the cost model: recode costs (fixed, per_mp, busy, fade) per codec, copy per frame, audio per second

# Built-in costs, in seconds on one desktop core, measured with tests/regression/progress_timeline.py --one-core
# --calibrate (Oct 2026: 29 edits of real clips, 360x360 to 1080p; synthetic clips re-encode 2-3x cheaper).
# - A re-encoded frame costs (fixed + megapixels x per_mp) x (1 - busy + busy x busyness) + megapixels x fade if its
#   video fades, fitted over the encoder runs (each run's frames, from the encoder's start to the end of its flush):
#   - fixed + megapixels x per_mp: a frame of average busyness. 720p: x264 25 ms, x265 60 ms, libvpx 53 ms, libaom
#     129 ms.
#   - An encoder's time per frame follows the content. In one 720p clip x264 took 14 ms per frame on a static scene
#     and 32 ms on a busy one, whose source packets had 1.4 and 5.6 kB per frame (the clip's average: 3.8 kB); libaom
#     118 and 198 ms. Busyness is that ratio: the source's bytes per frame within BUSY_WINDOW of the frame, against
#     the whole video's. busy is the share of the cost that follows it: x264 0.55, x265 0.75, libaom 0.85; libvpx's
#     time hardly does (0.05).
#   - A faded frame costs x264 and x265 more: every block changes, so none is skipped as unchanged (x264 at 720p:
#     +28 ms, about 2x). libvpx and libaom no more (see progress_timeline.fit_recode_cost).
#   - With one cost per frame and codec, as before, progress was up to 15-20 % off an even pace in common edits
#     (2 segments without fades: H.264 720p 16.8-17.1 %, AV1 720p 14.6 %); now 3-7 % (one core, largest distance at
#     the updates from the first update; 26 cases on average 9.5-9.7 % -> 4.4-4.7 %).
#   - libvpx's runs vary +-15 % (P2_vp9_720p 8-9 %): smartcut caps its VP9 re-encodes at the source's average bitrate,
#     and where the cap binds libvpx is faster, by an amount that differs by source. A negative busy fitted on one
#     source made another less even, so libvpx keeps 0.05 (decided, Oct 2026; progress_timeline.fit_recode_cost).
#   - AV1 pauses at every run's keyframe: see smart_cut.py ("Known and accepted").
# - hevc: x265 with its default settings, as for camera footage (sources without x265's options in their headers;
#   with them smartcut reuses the source's settings), with the 1-thread pool and 1 frame thread it uses on a 1-CPU
#   machine. The earlier cost (0.005 + 0.020 per megapixel) was measured while x265 escaped the one-core pinning, and
#   the one-core costs with that pool (43-49 / 108-116 / 181 ms per frame at 360p / 720p / 1080p) were mostly encoder
#   closes stalling on Windows (see tests/regression/run_cases.pin_to_cores); without them x265 takes as long as
#   without a pool. An accurate single cost (0.075 per megapixel) made progress less even than the low one (replayed:
#   14.0 % against 12.2 %), as the errors come from how the cost is spread over the frames: a lower video cost only
#   shrinks re-encoding's share. With busyness and fades the measured level is the most even (5.7 %; half of it 5.7 %,
#   twice 6.6 %).
# - Not counted: decoding the frames before a cut (from the GOP's keyframe; an HEVC open GOP also decodes the GOP
#   before) and starting an encoder (x265: ~1 s per run at 720p). Fitted, they made H.264 progress no more even
#   (replayed: 3.8 % -> 4.0 %) and HEVC more even on average but not in every case (5.9 % -> 4.9 %).
# - A copied frame costs the whole loop: remuxing, audio passthrough and muxing: 33-35 us over the 29 edits (11 884
#   frames; timing the video cutter alone 14-19 us). HEVC pieces that re-encode their leading pictures count as
#   re-encoding (calibrated as copies, they made copied frames look 7x dearer).
# - Re-encoded audio: 0.027-0.029 s per second of audio over three calibrations (903 s each; HE-AAC sources ~0.032).
RECODE_COST = {  # (fixed, per_mp, busy, fade): s per frame, s per megapixel-frame, share, s per faded megapixel-frame
    'h264': (0.0025, 0.024, 0.55, 0.030),
    'hevc': (0.0035, 0.061, 0.75, 0.034),
    'vp9': (0.011, 0.046, 0.05, 0.0),
    'av1': (0.037, 0.10, 0.85, 0.0),
}
RECODE_COST_OTHER = (0.005, 0.1, 0.0, 0.0)  # other codecs (calibrate_progress.py doesn't measure them)
COPY_COST_PER_FRAME = 0.000035
AUDIO_RECODE_COST_PER_SECOND = 0.027
# Busyness: the source's bytes per frame within this many seconds either side of the frame, relative to the
# whole video's (part of the model, not calibrated)
BUSY_WINDOW = 0.5

# Accepted ranges for a file's values: far wider than any machine's, to catch a broken file, not to judge a slow one
RECODE_RANGES = (("fixed", 0.0, 1.0), ("per_mp", 0.0, 10.0), ("busy", 0.0, 1.0), ("fade", 0.0, 10.0))
COPY_RANGE = (1e-7, 1e-2)
AUDIO_RANGE = (1e-4, 1.0)


@dataclass(frozen=True)
class ProgressCosts:
    recode: dict[str, tuple[float, float, float, float]]
    copy_per_frame: float = COPY_COST_PER_FRAME
    audio_per_second: float = AUDIO_RECODE_COST_PER_SECOND
    source: str = "built-in"  # where they come from, for the log
    measured: dict[str, Any] = field(default_factory=dict)  # the file's "measured" record

    def recode_for(self, family: str) -> tuple[float, float, float, float]:
        """(fixed, per_mp, busy, fade) for a codec family (h264, hevc, vp9, av1, ...)."""
        return self.recode.get(family, RECODE_COST_OTHER)


BUILT_IN = ProgressCosts(dict(RECODE_COST))


class CostsFileError(ValueError):
    """A progress_costs.json that can't be used."""


def default_path() -> Path:
    """The per-user default: %APPDATA%\\smartcut\\ on Windows, else $XDG_CONFIG_HOME/smartcut/ or
    ~/.config/smartcut/ (for a service without $HOME, ~ is the user's home from the password database)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "smartcut" / "progress_costs.json"


def progress_costs_path() -> Path:
    """The file smartcut reads: SMARTCUT_PROGRESS_COSTS if set, else default_path()."""
    named = os.environ.get(ENV_VAR)
    return Path(named).expanduser() if named else default_path()


def _number(value: object, what: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CostsFileError(f"{what} is not a number: {value!r}")
    if not low <= value <= high:
        raise CostsFileError(f"{what} = {value} is outside {low}..{high}")
    return float(value)


def parse(data: object, source: str = "progress_costs.json") -> ProgressCosts:
    """The costs in a progress_costs.json's content (parsed JSON). Raises CostsFileError if it can't be used: another
    format or cost model, a missing or non-numeric value, or one outside the accepted ranges. Codecs the file
    doesn't list keep their built-in costs."""
    if not isinstance(data, dict):
        raise CostsFileError("not a JSON object")
    if data.get("format") != FORMAT:
        raise CostsFileError(f"format {data.get('format')!r}, this smartcut reads format {FORMAT}")
    if data.get("model") != MODEL:
        raise CostsFileError(f"made for cost model {data.get('model')!r}, this smartcut uses model {MODEL}")
    recode_in = data.get("recode_cost")
    if not isinstance(recode_in, dict) or not recode_in:
        raise CostsFileError("recode_cost is missing or empty")
    recode = dict(RECODE_COST)
    for family, values in recode_in.items():
        if not isinstance(values, list) or len(values) != len(RECODE_RANGES):
            raise CostsFileError(f"recode_cost.{family} is not a list of {len(RECODE_RANGES)} numbers")
        parsed = tuple(_number(v, f"recode_cost.{family}.{name}", low, high) for v, (name, low, high) in zip(values, RECODE_RANGES))
        if parsed[0] + parsed[1] <= 0:
            raise CostsFileError(f"recode_cost.{family}: a re-encoded frame would cost nothing")
        recode[str(family)] = parsed  # type: ignore[assignment]
    copy = _number(data.get("copy_cost_per_frame"), "copy_cost_per_frame", *COPY_RANGE)
    audio = _number(data.get("audio_recode_cost_per_second"), "audio_recode_cost_per_second", *AUDIO_RANGE)
    measured = data.get("measured")
    return ProgressCosts(recode, copy, audio, source, measured if isinstance(measured, dict) else {})


def load(path: str | os.PathLike | None = None) -> tuple[ProgressCosts, list[tuple[int, str]]]:
    """The costs to use, from path (default: progress_costs_path()), and the messages to log as (level, text).
    Falls back to BUILT_IN when there is no file or it can't be used; never raises."""
    try:
        path = Path(path) if path is not None else progress_costs_path()
    except Exception as e:  # no home folder for the default path (RuntimeError), ...
        return BUILT_IN, [(logging.WARNING, f"no path for progress_costs.json ({e!r}); using the built-in values")]
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return BUILT_IN, [(logging.INFO, f"built-in (no {path})")]
    except Exception as e:  # unreadable: permissions, a directory, ...
        return BUILT_IN, [(logging.WARNING, f"can't read {path} ({e}); using the built-in values")]
    try:
        costs = parse(json.loads(text), str(path))
    except Exception as e:  # malformed JSON (ValueError), CostsFileError, anything unexpected
        return BUILT_IN, [(logging.WARNING, f"{path} not used ({e}); using the built-in values")]
    m = costs.measured
    where = ", ".join(f"{k} {m[k]}" for k in ("date", "host", "cores") if k in m)
    messages = [(logging.INFO, f"{path}" + (f" (measured: {where})" if where else ""))]
    if m.get("av") and m["av"] != av.__version__:
        messages.append((logging.WARNING, f"{path} was measured with PyAV {m['av']}, this is {av.__version__}: "
                                          "encoders may be faster or slower now; re-run calibrate_progress.py"))
    return costs, messages


@functools.cache
def current() -> ProgressCosts:
    """The costs in use: loaded and logged once per process, at its first edit."""
    try:
        costs, messages = load()
    except Exception as e:  # load() doesn't raise; this only guards an edit against a bug there
        costs, messages = BUILT_IN, [(logging.WARNING, f"loading failed ({e!r}); using the built-in values")]
    for level, text in messages:
        log.log(level, "smartcut progress costs: %s", text)
    return costs


def reset() -> None:
    """Forget the costs in use, so that the next edit loads them again (tests)."""
    current.cache_clear()


def to_json(costs: ProgressCosts, measured: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> str:
    """progress_costs.json's content for these costs (the codecs in costs.recode; parse() accepts it)."""
    data: dict[str, Any] = {
        "format": FORMAT,
        "model": MODEL,
        "recode_cost": {family: [round(v, 6) for v in values] for family, values in costs.recode.items()},
        "copy_cost_per_frame": round(costs.copy_per_frame, 9),
        "audio_recode_cost_per_second": round(costs.audio_per_second, 6),
        "measured": measured if measured is not None else costs.measured,
    }
    data.update(extra or {})
    text = json.dumps(data, indent=2)
    # lists of plain values on one line, e.g. "h264": [0.0025, 0.024, 0.55, 0.03]
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text) + "\n"
