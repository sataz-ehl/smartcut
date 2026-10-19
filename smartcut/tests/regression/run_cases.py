#!/usr/bin/env python3
"""
Measurement-based regression cases for smartcut cuts and fades.

Usage:
    python run_cases.py <media dir> <out dir> [case ...] [--repo PATH] [--one-core] [--list]

Each case runs smartcut.process_segments() and measures the output against its source:
  - durations, frame counts, frame timing (largest gap in frames, non-monotonic timestamps, identical frames)
  - video fade curve: brightness relative to the matching source frame, vs the ideal linear ramp
  - audio fade curve: RMS relative to the matching source audio, vs the ideal ramp
  - audio dips (clicks / gaps) in parts that should be untouched (synthetic tone sources only)
  - A/V sync per segment (real audio only): the audio's offset from where its video says it belongs, by
    cross-correlation with the source (join_sync.py measures more at each join)
  - A/V length drift and processing time
and gets a PASS/FAIL verdict from thresholds that correct outputs meet. Time series are saved per case as .npz
and all numbers as results.json, so two runs (e.g. before/after a change) can be compared.

Cases whose media is missing are skipped; make the media with make_test_media.py. --one-core pins the process
to one CPU core (closer to a 1-CPU server) for speed comparisons.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import join_sync

SYNTH = "synth_h264.mp4"
REAL = "real_h264.mp4"
TWO_SEGMENTS_REAL = [{"start": 20, "end": 35, "fadein": 2, "fadeout": 2}, {"start": 60, "end": 75}]
# Six segments, five joins (join_sync.SEGMENTS): copied audio drifted at every join
SIX_SEGMENTS = [{"start": a, "end": b} for a, b in ((10.3, 16.9), (25.05, 31.6), (40.77, 46.2), (58.4, 64.95),
                                                    (77.1, 83.3), (96.6, 103.15))]
CASES = {
    # H.264 synthetic (2 s GOPs): cuts, every fade kind, segment at 0, unsorted input
    "T1_nofade":   (SYNTH, [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    "T2_both":     (SYNTH, [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "T3_video":    (SYNTH, [{"start": 3, "end": 13, "video_fadein": 2, "video_fadeout": 2}]),
    "T4_audio":    (SYNTH, [{"start": 3, "end": 13, "audio_fadein": 2, "audio_fadeout": 2}]),
    "T5_mixed":    (SYNTH, [{"start": 2, "end": 8, "video_fadein": 1.5, "audio_fadeout": 2},
                            {"start": 11, "end": 18, "fadein": 1, "fadeout": 1}]),
    "T6_start0":   (SYNTH, [{"start": 0, "end": 6, "fadein": 2}]),
    "T7_unsorted": (SYNTH, [{"start": 12, "end": 16}, {"start": 2, "end": 6, "fadeout": 1}]),
    # Outputs that end with re-encoded frames and no copied packet before them (they lost their last frame)
    "T8_short":    (SYNTH, [{"start": 1, "end": 2}]),
    "T9_recoded":  (SYNTH, [{"start": 3, "end": 6, "fadein": 1.5, "fadeout": 1.5}]),
    # Real content (H.264 + AAC-LC)
    "R1_real_fades":  (REAL, [{"start": 20, "end": 35, "fadein": 2, "fadeout": 2},
                              {"start": 60, "end": 75, "audio_fadein": 1.5, "video_fadeout": 2}]),
    "R2_real_nofade": (REAL, [{"start": 20, "end": 35}, {"start": 60, "end": 75}]),
    "R3_real_short":  (REAL, [{"start": 20, "end": 21.1}]),
    # HE-AAC audio (Facebook's format)
    "H1_heaac_fade":   ("synth_heaac.mp4", [{"start": 3, "end": 13, "audio_fadein": 2, "audio_fadeout": 2}]),
    "H2_heaac_nofade": ("synth_heaac.mp4", [{"start": 3, "end": 13}]),
    # Two audio tracks: AAC 440 Hz + AC-3 880 Hz at 44.1 kHz; AAC + HE-AAC (real audio); AAC + MP3 880 Hz 44.1 kHz
    "M1_2tracks":       ("synth_2audio.mp4", [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    "M2_2tracks_fades": ("synth_2audio.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "M3_heaac_2nd":     ("synth_2audio_heaac.mp4", [{"start": 2, "end": 8, "audio_fadein": 1.5},
                                                    {"start": 11, "end": 18, "fadein": 1, "fadeout": 1}]),
    "M4_mp3_fades":     ("synth_2audio_mp3.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "M5_mp3_cuts":      ("synth_2audio_mp3.mp4", [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    # .webm outputs (OUTPUT_EXT): AAC / HE-AAC audio is re-encoded to Opus, Opus audio is copied
    "W1_webm_aac":         ("synth_vp9.mp4", [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    "W2_webm_aac_fades":   ("synth_vp9.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "W3_webm_heaac":       ("real_fb_vp9.mp4", [{"start": 30, "end": 45}, {"start": 100, "end": 115}]),
    "W4_webm_heaac_fades": ("real_fb_vp9.mp4", [{"start": 30, "end": 45, "fadein": 2, "fadeout": 2}, {"start": 100, "end": 115}]),
    "W5_webm_opus":        ("synth_vp9_opus.webm", [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    "W6_webm_opus_fades":  ("synth_vp9_opus.webm", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    # VP9 / AV1
    "V1_vp9_cuts":   ("synth_vp9.mp4", [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    "V2_vp9_fades":  ("synth_vp9.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "A1_av1_cuts":   ("synth_av1.mp4", [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    "A2_av1_fades":  ("synth_av1.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "V3_vp9_short":  ("synth_vp9.mp4", [{"start": 1, "end": 2}]),
    "A3_av1_short":  ("synth_av1.mp4", [{"start": 1, "end": 2}]),
    "RV_vp9_real":   ("real_vp9.mp4", TWO_SEGMENTS_REAL),
    "RA_av1_real":   ("real_av1.mp4", TWO_SEGMENTS_REAL),
    "RH7_h264_720p": ("real_h264_720p.mp4", TWO_SEGMENTS_REAL),
    "RV7_vp9_720p":  ("real_vp9_720p.mp4", TWO_SEGMENTS_REAL),
    "RA7_av1_720p":  ("real_av1_720p.mp4", TWO_SEGMENTS_REAL),
    "FB_vp9_heaac":  ("real_fb_vp9.mp4", [{"start": 30, "end": 45, "fadein": 2, "fadeout": 2}, {"start": 100, "end": 115}]),
    # Variants YouTube / Facebook serve
    "X_vp9_10bit":     ("synth_vp9_10bit.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "X_av1_10bit":     ("synth_av1_10bit.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "X_h264_baseline": ("synth_h264_baseline.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "X_h264_bl_short": ("synth_h264_baseline.mp4", [{"start": 1, "end": 2}]),
    "X_h264_60fps":    ("synth_h264_60fps.mp4", [{"start": 3, "end": 9, "fadein": 1, "fadeout": 1}, {"start": 12, "end": 17}]),
    "X_portrait":      ("synth_portrait.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "X_fb_sd_h264":    ("fb_sd_h264.mp4", [{"start": 2, "end": 8, "fadein": 1, "fadeout": 1}, {"start": 10, "end": 14}]),
    # Many joins, no fades (audio re-encoded with 3 ms fades at the joins), and starts mid-stream into Matroska
    "J1_real_6seg":    ("real_h264_720p.mp4", SIX_SEGMENTS),
    "J2_fb_6seg":      ("fb_sd_h264.mp4", SIX_SEGMENTS),
    "J3_mkv_start":    ("real_h264_720p.mp4", [{"start": 30.3, "end": 40}]),
    "J4_mkv_heaac":    ("fb_sd_h264.mp4", [{"start": 30.3, "end": 40}, {"start": 50, "end": 55}]),
    "J5_mp4_start":    ("fb_sd_h264.mp4", [{"start": 30.3, "end": 40}]),  # copied: pre-roll and edit list
    # HEVC (x265: open GOPs with CRA keyframes and RASL leading pictures; and closed GOPs)
    "E1_hevc_cuts":    ("synth_hevc.mp4", [{"start": 5, "end": 9}, {"start": 12, "end": 17}]),
    "E2_hevc_fades":   ("synth_hevc.mp4", [{"start": 3, "end": 13, "fadein": 2, "fadeout": 2}]),
    "E3_hevc_short":   ("synth_hevc.mp4", [{"start": 1, "end": 2}]),
    "E4_hevc_closed":  ("synth_hevc_closed.mp4", [{"start": 3, "end": 9, "fadein": 1, "fadeout": 1}, {"start": 12, "end": 17}]),
    "E5_hevc_real":    ("real_hevc.mp4", TWO_SEGMENTS_REAL),
    "E6_hevc_720p":    ("real_hevc_720p.mp4", TWO_SEGMENTS_REAL),
}
EXPECTED_FAILURES = {"X_vp9_10bit"}  # smartcut can't re-encode 10/12-bit VP9 (profile 2/3)
OUTPUT_EXT = {case: ".webm" for case in CASES if case.startswith("W")}  # default .mp4
OUTPUT_EXT.update({"J3_mkv_start": ".mkv", "J4_mkv_heaac": ".mkv"})

# Verdict thresholds (correct outputs measured: fade errors <= 0.04, gaps <= 1.1 frames, |A-V| <= 20 ms)
MAX_FADE_ERR_SYNTH, MAX_FADE_ERR_REAL = 0.08, 0.15
MAX_GAP_FRAMES, MAX_AV_MS = 1.5, 60
MAX_SYNC_MS = 1.5  # per segment; re-encoded joins measured <= 0.4 ms (Matroska timestamps are rounded to 1 ms)
MAX_STREAM_END_FRAMES = 0.5  # the video stream must end where its last frame does


_PIN_JOB = []  # keeps the Windows job object open


def pick_cpus(n: int) -> list[int]:
    """n CPUs this process may run on, one per physical core where that is known: two hyperthreads of one core are not
    two cores. Linux: the affinity set, by /sys/devices/system/cpu/cpu*/topology (package, core). Windows: the process
    affinity mask; with psutil installed and twice as many logical CPUs as cores, the even ones (Windows numbers a
    core's hyperthreads next to each other). Hyperthreads are used only when there are fewer cores than n."""
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        seen, cores = set(), []
        for cpu in allowed:
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            try:
                key = ((topology / "physical_package_id").read_text().strip(), (topology / "core_id").read_text().strip())
            except OSError:
                key = (str(cpu), "")
            if key not in seen:
                seen.add(key)
                cores.append(cpu)
    else:
        allowed = _windows_affinity()
        cores = allowed
        try:
            import psutil  # noqa: PLC0415
            physical, logical = psutil.cpu_count(logical=False), psutil.cpu_count()
            if physical and logical == 2 * physical:
                cores = [c for c in allowed if c % 2 == 0]
        except ImportError:
            pass
    if n > len(allowed):
        raise ValueError(f"{n} cores asked, this process may use {len(allowed)} CPUs")
    return (cores + [c for c in allowed if c not in cores])[:n]


def _windows_affinity() -> list[int]:
    """The CPUs the process may use (its affinity mask; the first 64 CPUs)."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        return list(range(os.cpu_count() or 1))
    return [i for i in range(64) if process.value >> i & 1]


def available_cpus() -> int:
    """How many CPUs this process may run on (its affinity, as FFmpeg sizes its threads)."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else len(_windows_affinity())


def pin_to_cores(n: int) -> list[int]:
    """Run on n CPU cores (pick_cpus), like an n-CPU server. Returns the CPUs. Call it after importing smartcut, from
    the thread that cuts, before any encoding (see x265 below).

    FFmpeg sizes its codec thread pools from the CPUs the process may use. x265 doesn't:
    - On Windows it sets its pool threads' affinity to the whole NUMA node itself, which escaped
      SetProcessAffinityMask (x265 re-encodes ran on 4.8 cores). A job object with an affinity limit holds every
      thread of the process, and its child processes. x265 then logs "unable to set thread affinity for NUMA node
      mask" for each pool thread it starts: expected.
    - It sizes its thread pool and frame threads from all the machine's CPUs (Linux: all online CPUs, whatever the
      affinity; Windows: 12 pool threads and 3 frame threads on a 6-core / 12-thread desktop). x265_threads_for(n)
      gives smartcut's x265 re-encodes what x265 picks on an n-CPU machine.
    - Its pool threads run below normal priority (Linux: nice 10), and closing an encoder waits for them in a loop
      that gives up the CPU with Sleep(0) (Linux: usleep(0)). On one Windows core Sleep(0) only lets threads of the
      same or higher priority run, so each close stalled for 0.5-10 s, until Windows boosted the starved pool thread:
      that was why x265 with its 1-thread pool seemed 2-3x slower here and varied 2x from run to run. So on one
      Windows core this thread runs at the pool's priority too: the close takes 0.01-0.04 s and feeding and flushing
      take as long as before. With more cores the pool thread runs on another one. Linux's usleep(0) sleeps for the
      timer slack (~50 us), during which the pool thread runs (not measured on the VPS).
    """
    cpus = pick_cpus(n)
    if hasattr(os, "sched_setaffinity"):  # Linux: threads created later inherit it
        os.sched_setaffinity(0, set(cpus))
    elif sys.platform == "win32":
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class JobLimits(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Without these types ctypes truncates the handles and the calls fail silently
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        limits = JobLimits(LimitFlags=0x10, Affinity=sum(1 << c for c in cpus))  # JOB_OBJECT_LIMIT_AFFINITY
        if not (job and kernel32.SetInformationJobObject(job, 2, ctypes.byref(limits), ctypes.sizeof(limits))  # basic limits
                and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())):
            raise OSError(ctypes.get_last_error(), "pinning to CPUs through a job object failed")
        _PIN_JOB.append(job)
        if n == 1:
            kernel32.GetCurrentThread.restype = wintypes.HANDLE
            kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
            if not kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1):  # THREAD_PRIORITY_BELOW_NORMAL, as x265's pool
                raise OSError(ctypes.get_last_error(), "lowering this thread's priority failed")
    x265_threads_for(n)
    return cpus


def pin_to_one_core() -> None:
    """Run on one CPU core, like the 1-CPU server: pin_to_cores(1)."""
    pin_to_cores(1)


_X265_POOL_OPTIONS, _X265_FRAME_THREADS = ("pools", "numa-pools"), "frame-threads"
_X265_CPUS = {}  # the VideoCutter class patched -> the CPUs x265 is sized for


def x265_frame_threads(cpus: int, height: int) -> int:
    """The frame threads x265 picks for this many CPUs (ThreadPool::getFrameThreadsCount, with wavefront parallel
    processing, its default)."""
    if cpus >= 32:
        return 6 if height > 2000 else 5
    return 4 if cpus >= 16 else 3 if cpus >= 8 else 2 if cpus >= 4 else 1


def x265_threads_for(cpus: int) -> None:
    """x265 re-encodes get the threads x265 picks by itself on a machine with this many CPUs: a pool of that many
    threads and x265_frame_threads(). Threads set explicitly (e.g. reused from the source's x265 options) are kept,
    as there."""
    from smartcut import video_cutter  # noqa: PLC0415

    cutter = video_cutter.VideoCutter
    first = cutter not in _X265_CPUS
    _X265_CPUS[cutter] = cpus
    if not first:
        return
    init = cutter.init_encoder

    def init_encoder(self: video_cutter.VideoCutter) -> None:
        init(self)
        params = self.encoding_options.get("x265-params")
        if params is not None:
            n = _X265_CPUS[cutter]
            names = {p.split("=", 1)[0].replace("_", "-") for p in params.split(":")}
            extra = [] if names & set(_X265_POOL_OPTIONS) else [f"pools={n}"]
            height = self.in_stream.codec_context.height or 0
            extra += [] if _X265_FRAME_THREADS in names else [f"frame-threads={x265_frame_threads(n, height)}"]
            self.encoding_options["x265-params"] = ":".join([params, *extra])

    cutter.init_encoder = init_encoder


def x265_one_cpu() -> None:
    """x265 re-encodes get a 1-CPU machine's threads: x265_threads_for(1)."""
    x265_threads_for(1)


def fades_of(seg):
    """(video_in, video_out, audio_in, audio_out) durations, with CutSegmentConfig's precedence."""
    vi = seg.get("fadein") if seg.get("fadein") is not None else seg.get("video_fadein")
    ai = seg.get("fadein") if seg.get("fadein") is not None else seg.get("audio_fadein")
    vo = seg.get("fadeout") if seg.get("fadeout") is not None else seg.get("video_fadeout")
    ao = seg.get("fadeout") if seg.get("fadeout") is not None else seg.get("audio_fadeout")
    return vi or 0, vo or 0, ai or 0, ao or 0


def ideal_gain(u, length, fin, fout):
    g = np.ones_like(u, dtype=float)
    if fin:
        g = np.minimum(g, np.clip(u / fin, 0, 1))
    if fout:
        g = np.minimum(g, np.clip((length - u) / fout, 0, 1))
    return g


def read_media(path):
    """Video luma means/hashes per frame, and per audio track: mono samples, sample rate, start time."""
    v_t, v_mean, v_hash = [], [], []
    with av.open(str(path)) as c:
        vs = c.streams.video[0] if c.streams.video else None
        audio_streams = list(c.streams.audio)
        as_ = audio_streams[0] if audio_streams else None
        a_chunks = {s.index: [] for s in audio_streams}
        a_rate, a_start = {}, {}
        info = {"vcodec": vs.codec_context.name if vs else None,
                "avg_rate": float(vs.average_rate) if vs and vs.average_rate else None,
                "acodec": as_.codec_context.name if as_ else None, "abitrate": as_.bit_rate if as_ else None,
                "atracks": [(s.codec_context.name, s.codec_context.profile) for s in audio_streams],
                "start_time": (c.start_time or 0) / av.time_base,
                # As players see it (in MP4 the edit list); None in Matroska
                "v_stream_s": float(vs.duration * vs.time_base) if vs and vs.duration is not None and vs.time_base else None}
        # Decoding drops the frames outside an MP4 edit list, as FFmpeg and players do: the frame count is the same
        # as ffprobe -count_frames
        decoded = ((packet.stream.index, frame) for packet in c.demux(*[s for s in (vs, *audio_streams) if s is not None])
                   for frame in packet.decode())
        for i, frame in decoded:
            if isinstance(frame, av.VideoFrame):
                p = frame.planes[0]
                if frame.format.name.endswith(("10le", "12le")):  # scale high bit depth luma to 8-bit
                    shift = 2 if frame.format.name.endswith("10le") else 4
                    y = (np.frombuffer(p, "<u2").reshape(frame.height, p.line_size // 2)[:, :frame.width] >> shift).astype(np.uint8)
                else:
                    y = np.frombuffer(p, np.uint8).reshape(frame.height, p.line_size)[:, :frame.width]
                v_t.append(float(frame.pts * frame.time_base))
                v_mean.append(float(y.mean()))
                v_hash.append(hashlib.md5(y.tobytes()).hexdigest())
            else:
                arr = frame.to_ndarray()
                mono = (arr.astype(np.float32).mean(axis=0) if frame.format.is_planar
                        else arr.reshape(-1, len(frame.layout.channels)).astype(np.float32).mean(axis=1))
                if np.issubdtype(arr.dtype, np.integer):
                    mono /= np.iinfo(arr.dtype).max
                if i not in a_start:
                    a_start[i] = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                a_rate[i] = frame.sample_rate
                a_chunks[i].append(mono)
    audios = [(np.concatenate(a_chunks[s.index]) if a_chunks[s.index] else np.zeros(0, np.float32),
               a_rate.get(s.index), a_start.get(s.index, 0.0)) for s in audio_streams]
    return info, np.array(v_t), np.array(v_mean), v_hash, audios


def rms_windows(x, rate, win):
    n = int(rate * win)
    k = len(x) // n
    return np.sqrt((x[: k * n].reshape(k, n).astype(np.float64) ** 2).mean(axis=1))


def measure_audio(o_track, s_track, o_offset, tone, segs, starts, lengths, v_duration):
    """One audio track vs the same source track: length vs the video, fade curve error, gain outside fades, dips."""
    (o_audio, o_rate, _), (s_audio, s_rate, s_astart) = o_track, s_track
    win = 0.01 if tone else 0.05
    o_rms, s_rms = rms_windows(o_audio, o_rate, win), rms_windows(s_audio, s_rate, win)
    at = np.arange(len(o_rms)) * win + win / 2 + o_offset
    a_gain, a_ideal = np.full(len(at), np.nan), np.full(len(at), np.nan)
    for k, s in enumerate(segs):
        _, _, ai, ao = fades_of(s)
        m = (at >= starts[k]) & (at < starts[k] + lengths[k])
        u = at[m] - starts[k]
        idx = np.clip(((s["start"] + u - s_astart) / win).astype(int), 0, len(s_rms) - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_gain[m] = np.where(s_rms[idx] > 1e-3, o_rms[m] / s_rms[idx], np.nan)
        a_ideal[m] = ideal_gain(u, lengths[k], ai, ao)
    a_err, afade = np.abs(a_gain - a_ideal), a_ideal < 0.999
    # Windows touching the 3 ms fades at joins (re-encoded audio) are neither: join_sync.py measures the joins
    near_join = np.zeros(len(at), bool)
    for k in range(1, len(segs)):
        if segs[k]["start"] != segs[k - 1]["end"]:
            near_join |= np.abs(at - starts[k]) < win / 2 + join_sync.JOIN_FADE
    afade &= ~near_join
    flat = ~afade & ~near_join & np.isfinite(a_gain)
    r = {"a_duration": len(o_audio) / o_rate if o_rate else 0}
    # Where the decoded audio ends against the end of the last picture (both from the first frame). Not the lengths:
    # in Matroska the encoder priming decodes before the first frame, at negative times.
    r["av_diff_ms"] = (o_offset + r["a_duration"] - v_duration) * 1000
    r["a_fade_max_err"] = float(np.nanmax(a_err[afade])) if afade.any() and np.isfinite(a_err[afade]).any() else None
    r["a_flat_min_gain"] = float(np.nanmin(a_gain[flat])) if flat.any() else None
    r["a_flat_max_gain"] = float(np.nanmax(a_gain[flat])) if flat.any() else None
    if tone:
        dips = at[flat & (a_gain < 0.9)]
        r["a_dips_at"] = [round(float(x), 2) for x in dips][:10]
    return r, at, a_gain, a_ideal


_sources = {}


def analyse(media, out_dir, case, src_name, segs, out_path):
    if src_name not in _sources:
        _sources[src_name] = read_media(media / src_name)
    s_info, s_vt, s_vmean, _, s_audios = _sources[src_name]
    o_info, o_vt, o_vmean, o_hash, o_audios = read_media(out_path)
    segs = sorted(segs, key=lambda s: s["start"])
    lengths = [s["end"] - s["start"] for s in segs]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    expected = float(sum(lengths))
    fps = s_info["avg_rate"] or 30
    res = {"expected_s": expected, "out_info": o_info}

    t0 = o_vt[0] if len(o_vt) else 0
    vt = o_vt - t0
    d = np.diff(vt)
    res["v_frames"] = len(vt)
    # Exactly the source frames inside the segments (smartcut keeps start <= t < end)
    st = s_info["start_time"]
    res["v_frames_expected"] = int(sum(((s_vt >= st + s["start"] - 1e-9) & (s_vt < st + s["end"] - 1e-9)).sum() for s in segs))
    res["v_duration"] = float(vt[-1] + 1 / fps) if len(vt) else 0
    res["v_stream_s"] = o_info["v_stream_s"]
    res["v_max_gap_frames"] = float(d.max() * fps) if len(d) else 0
    res["v_nonmonotonic"] = int((d <= 0).sum())
    res["v_identical_frames"] = int(sum(o_hash[i] == o_hash[i - 1] for i in range(1, len(o_hash))))

    # Video fade: brightness relative to the matching source frame
    v_alpha, v_ideal = np.full(len(vt), np.nan), np.full(len(vt), np.nan)
    for k, s in enumerate(segs):
        vi, vo, _, _ = fades_of(s)
        m = (vt >= starts[k] - 1e-6) & (vt < starts[k] + lengths[k] - 1e-6)
        u = vt[m] - starts[k]
        idx = np.clip(np.searchsorted(s_vt, s["start"] + u - 0.5 / fps), 0, len(s_vt) - 1)
        base = s_vmean[idx] - 16
        with np.errstate(divide="ignore", invalid="ignore"):
            v_alpha[m] = np.where(base > 8, (o_vmean[m] - 16) / base, np.nan)
        v_ideal[m] = ideal_gain(u, lengths[k], vi, vo)
    err, fading = np.abs(v_alpha - v_ideal), v_ideal < 0.999
    res["v_fade_max_err"] = float(np.nanmax(err[fading])) if fading.any() and np.isfinite(err[fading]).any() else None
    res["v_flat_min_alpha"] = float(np.nanmin(v_alpha[~fading])) if (~fading).any() else None

    # Audio, every track (the first one's numbers fill the table columns): RMS relative to the same source track.
    # Tone tracks (synthetic sources, not HE-AAC real audio) get finer windows and the dip check.
    problems = []
    if len(o_audios) != len(s_audios):
        problems.append(f"{len(o_audios)} of {len(s_audios)} audio tracks")
    res.update(a_duration=0, av_diff_ms=0, a_fade_max_err=None, a_flat_min_gain=None, a_flat_max_gain=None, tracks=[])
    for t, (o_track, s_track) in enumerate(zip(o_audios, s_audios)):
        tone = src_name.startswith("synth_") and s_info["atracks"][t][1] != "HE-AAC"
        tr, at, a_gain, a_ideal = measure_audio(o_track, s_track, o_track[2] - t0, tone, segs, starts, lengths,
                                                res["v_duration"])
        res["tracks"].append(tr)
        if t == 0:
            res.update(tr)
            np.savez(out_dir / f"{case}.npz", vt=vt, v_alpha=v_alpha, v_ideal=v_ideal, at=at, a_gain=a_gain, a_ideal=a_ideal)
        prefix = f"track {t + 1}: " if t else ""
        if abs(tr["av_diff_ms"]) > MAX_AV_MS:
            problems.append(prefix + "A/V length")
        if tr["a_fade_max_err"] is not None and tr["a_fade_max_err"] > (MAX_FADE_ERR_SYNTH if tone else MAX_FADE_ERR_REAL):
            problems.append(prefix + "a_fade_max_err")
        if tr.get("a_dips_at"):
            problems.append(prefix + "audio dips")

    tone = src_name.startswith("synth_") and "heaac" not in src_name
    max_err = MAX_FADE_ERR_SYNTH if tone else MAX_FADE_ERR_REAL

    # A/V sync per segment (real audio, first track): the output audio around the middle of each segment's video
    # against the source. Not for tones (ambiguous) or resampled audio (Opus in WebM from 44.1 kHz).
    res["sync_ms"] = None
    if o_audios and s_audios and not tone and o_audios[0][1] == s_audios[0][1]:
        (ox, rate, oa0), (sx, _, sa0) = o_audios[0], s_audios[0]
        n, worst = 0, 0.0
        for s in segs:
            kept = s_vt[(s_vt >= st + s["start"] - 1e-9) & (s_vt < st + s["end"] - 1e-9)]
            if not len(kept) or n >= len(o_vt):
                break
            S, V = float(kept[0]), float(o_vt[n])  # the segment's first frame, in the source and in the output
            n += len(kept)
            u = max(0.0, (float(kept[-1]) - S - join_sync.WIN) / 2)
            lag, _ = join_sync.find(sx.astype(np.float64), sa0, rate, ox.astype(np.float64), oa0, V + u, S + u)
            if np.isfinite(lag):
                worst = max(worst, abs(lag))
        res["sync_ms"] = worst * 1000
        if res["sync_ms"] > MAX_SYNC_MS:
            problems.append("A/V sync")
    if res["v_frames"] != res["v_frames_expected"]:
        problems.append("frame count")
    if res["v_stream_s"] is not None and abs(res["v_stream_s"] - res["v_duration"]) * fps > MAX_STREAM_END_FRAMES:
        problems.append("stream duration")
    if res["v_max_gap_frames"] > MAX_GAP_FRAMES or res["v_nonmonotonic"]:
        problems.append("frame timing")
    if res["v_fade_max_err"] is not None and res["v_fade_max_err"] > max_err:
        problems.append("v_fade_max_err")
    res["problems"] = problems
    return res


def other_tracks(r):
    """Table notes for the audio tracks after the first: A-V ms and fade error."""
    notes = []
    for t, tr in enumerate(r.get("tracks", [])[1:], start=2):
        fade = "-" if tr["a_fade_max_err"] is None else f"{tr['a_fade_max_err']:.2f}"
        notes.append(f"track {t}: A-V {tr['av_diff_ms']:.0f} aFade {fade}")
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]), help="smartcut repo root")
    ap.add_argument("--one-core", action="store_true", help="pin to one CPU core")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    args = ap.parse_args()
    if args.list or not args.media_dir:
        for name, (src, segs) in CASES.items():
            print(f"{name:<16} {src:<24} {segs}")
        return 0

    sys.path.insert(0, args.repo)
    import smartcut  # noqa: E402

    if args.one_core:
        pin_to_one_core()
    media, out_dir = Path(args.media_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case, (src, segs) in CASES.items():
        if args.cases and case not in args.cases:
            continue
        if not (media / src).exists():
            results[case] = {"skipped": f"no {src}"}
            continue
        out_path = out_dir / f"{case}{OUTPUT_EXT.get(case, '.mp4')}"
        out_path.unlink(missing_ok=True)
        calls = []
        t = time.perf_counter()
        try:
            ok = smartcut.process_segments(str(media / src), str(out_path), segments=[dict(s) for s in segs],
                                           progress_callback=lambda c, n: calls.append((c, n)))
            err = None
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        r = {"ok": ok, "error": err, "seconds": round(time.perf_counter() - t, 2), "progress_calls": len(calls)}
        if ok and out_path.exists():
            try:
                r.update(analyse(media, out_dir, case, src, segs, out_path))
            except Exception as e:
                r["analysis_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        results[case] = r
        print(f"done {case}: ok={ok} {r['seconds']:.1f}s", flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    def f(x, nd=2):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    print(f"\n{'case':<20}{'verdict':<9}{'sec':>7}{'dur/exp':>13}{'stream':>7}{'frames/exp':>12}{'gap':>5}{'same':>5}{'A-V ms':>8}"
          f"{'sync ms':>8}{'vFadeErr':>9}{'aFadeErr':>9}{'aFlat min/max':>15}  notes")
    failures = 0
    for case, r in results.items():
        if "skipped" in r:
            print(f"{case:<20}{'skip':<9}  {r['skipped']}")
            continue
        expected_fail = case in EXPECTED_FAILURES
        if "expected_s" not in r:
            verdict = "XFAIL" if expected_fail and not r["ok"] else "FAIL"
            failures += verdict == "FAIL"
            print(f"{case:<20}{verdict:<9}{r['seconds']:>7.1f}  {r.get('error') or r.get('analysis_error') or 'returned False'}")
            continue
        verdict = "PASS" if not r["problems"] else "FAIL"
        if expected_fail:  # XPASS: started working, update EXPECTED_FAILURES
            verdict = "XFAIL" if r["problems"] else "XPASS"
        failures += verdict == "FAIL"
        print(f"{case:<20}{verdict:<9}{r['seconds']:>7.1f}{r['v_duration']:>7.2f}/{r['expected_s']:<5.1f}{f(r['v_stream_s']):>7}"
              f"{r['v_frames']:>6}/{r['v_frames_expected']:<5}{f(r['v_max_gap_frames'], 1):>5}{r['v_identical_frames']:>5}"
              f"{r['av_diff_ms']:>8.0f}{f(r.get('sync_ms'), 1):>8}{f(r['v_fade_max_err']):>9}{f(r['a_fade_max_err']):>9}"
              f"{f(r['a_flat_min_gain']):>8}/{f(r['a_flat_max_gain'])}  {', '.join(r['problems'] + [f'dips {d}' for d in r.get('a_dips_at', [])][:1] + other_tracks(r))}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
