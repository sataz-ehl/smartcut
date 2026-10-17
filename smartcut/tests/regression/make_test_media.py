#!/usr/bin/env python3
"""
Generate the test media used by run_cases.py.

Usage:
    python make_test_media.py <media dir> [--ffmpeg PATH] [--font PATH] [--real CLIP.mp4]

Synthetic clips (20 s, grey picture with a large frame counter and timestamp, 440 Hz stereo tone, fixed
keyframe interval) are always generated:

    synth_h264.mp4            H.264 High, 640x360 30 fps, 2 s GOPs, AAC-LC 128k   (the reference clip)
    synth_vp9.mp4             VP9 profile 0
    synth_av1.mp4             AV1 Main 8-bit (SVT-AV1)
    synth_vp9_10bit.mp4       VP9 profile 2, 10-bit (smartcut can't re-encode this: expected failure)
    synth_av1_10bit.mp4       AV1 Main 10-bit
    synth_h264_baseline.mp4   H.264 Constrained Baseline (like YouTube format 18)
    synth_h264_60fps.mp4      60 fps, 2 s GOPs
    synth_portrait.mp4        720x1280
    synth_2audio.mp4          synth_h264.mp4 + a second audio track: AC-3 880 Hz mono, 44.1 kHz
    synth_2audio_mp3.mp4      the same with an MP3 second track
    synth_vp9_opus.webm       synth_vp9.mp4's video with the tone as Opus (for .webm outputs)

--real CLIP.mp4 (any H.264 + AAC clip, e.g. a downloaded music video) additionally derives:

    real_h264.mp4 (copy), real_vp9.mp4, real_av1.mp4 (5 s-ish GOPs, like YouTube),
    real_h264_720p.mp4, real_vp9_720p.mp4, real_av1_720p.mp4 (upscaled, for speed comparisons)

Not generated (FFmpeg can't encode HE-AAC, and real-world encoders matter): synth_heaac.mp4 / real_fb_vp9.mp4 /
fb_sd_h264.mp4. To make synth_heaac.mp4, mux any HE-AAC track (e.g. Facebook audio) with synth_h264.mp4's video:
    ffmpeg -i synth_h264.mp4 -ss 30 -i heaac.m4a -map 0:v -map 1:a -c copy -shortest synth_heaac.mp4
and synth_2audio_heaac.mp4 (two audio tracks, the second HE-AAC) likewise:
    ffmpeg -i synth_h264.mp4 -ss 30 -i heaac.m4a -map 0:v -map 0:a -map 1:a -c copy -shortest synth_2audio_heaac.mp4
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def run(ffmpeg, args, cwd):
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}\n{result.stderr}", file=sys.stderr)
        return False
    return True


def counter_clip(ffmpeg, out_dir, name, size, rate, gop, fontsize, extra_video_args=()):
    """Grey clip with a frame counter (+ timestamp) and a 440 Hz tone."""
    text = "%{frame_num}  %{pts\\:hms}" if size[0] > size[1] else "%{frame_num}"
    graph = (f"[0:v]drawtext=fontfile=font.ttf:text='{text}':fontsize={fontsize}:fontcolor=white:"
             f"x=(w-tw)/2:y=(h-th)/2[v];[1:a]pan=stereo|c0=c0|c1=c0[a]")
    return run(ffmpeg, [
        "-f", "lavfi", "-i", f"color=c=gray:s={size[0]}x{size[1]}:r={rate}:d=20",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=20",
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", *extra_video_args, "-c:a", "aac", "-b:a", "128k", name,
    ], out_dir)


def transcode(ffmpeg, out_dir, src, name, video_args):
    return run(ffmpeg, ["-i", src, "-map", "0", *video_args, "-c:a", "copy", name], out_dir)


def add_audio_track(ffmpeg, out_dir, src, name, audio_args):
    """src plus a second audio track: an 880 Hz mono tone at 44.1 kHz."""
    return run(ffmpeg, ["-i", src, "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=44100:duration=20",
                        "-map", "0:v", "-map", "0:a", "-map", "1:a", "-c:v", "copy", "-c:a:0", "copy", *audio_args,
                        "-shortest", name], out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media_dir")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--font", help="TTF font for the counter (auto-detected)")
    ap.add_argument("--real", help="H.264 + AAC clip to derive real_* VP9/AV1/720p variants from")
    args = ap.parse_args()

    out = Path(args.media_dir)
    out.mkdir(parents=True, exist_ok=True)
    font = args.font or next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    if not font:
        sys.exit("No font found; pass --font /path/to/font.ttf")
    shutil.copy(font, out / "font.ttf")
    ff = args.ffmpeg
    vp9 = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-g", "60", "-keyint_min", "60", "-cpu-used", "4", "-row-mt", "1"]
    av1 = ["-c:v", "libsvtav1", "-crf", "35", "-preset", "8", "-g", "60"]

    ok = {
        "synth_h264.mp4": counter_clip(ff, out, "synth_h264.mp4", (640, 360), 30, 60, 48),
        "synth_h264_60fps.mp4": counter_clip(ff, out, "synth_h264_60fps.mp4", (640, 360), 60, 120, 48),
        "synth_portrait.mp4": counter_clip(ff, out, "synth_portrait.mp4", (720, 1280), 30, 60, 96),
        "synth_h264_baseline.mp4": counter_clip(ff, out, "synth_h264_baseline.mp4", (640, 360), 30, 60, 48,
                                                ("-profile:v", "baseline")),
    }
    ok["synth_vp9.mp4"] = transcode(ff, out, "synth_h264.mp4", "synth_vp9.mp4", vp9)
    ok["synth_vp9_10bit.mp4"] = transcode(ff, out, "synth_h264.mp4", "synth_vp9_10bit.mp4",
                                          [*vp9, "-pix_fmt", "yuv420p10le", "-profile:v", "2"])
    ok["synth_vp9_opus.webm"] = run(ff, ["-i", "synth_vp9.mp4", "-c:v", "copy", "-c:a", "libopus", "-b:a", "96k",
                                         "synth_vp9_opus.webm"], out)
    ok["synth_av1.mp4"] = transcode(ff, out, "synth_h264.mp4", "synth_av1.mp4", av1)
    ok["synth_av1_10bit.mp4"] = transcode(ff, out, "synth_h264.mp4", "synth_av1_10bit.mp4", [*av1, "-pix_fmt", "yuv420p10le"])
    ok["synth_2audio.mp4"] = add_audio_track(ff, out, "synth_h264.mp4", "synth_2audio.mp4", ["-c:a:1", "ac3", "-b:a:1", "192k"])
    ok["synth_2audio_mp3.mp4"] = add_audio_track(ff, out, "synth_h264.mp4", "synth_2audio_mp3.mp4",
                                                 ["-c:a:1", "libmp3lame", "-b:a:1", "128k"])

    if args.real:
        real = Path(args.real).resolve()
        shutil.copy(real, out / "real_h264.mp4")
        up = ["-vf", "scale=1280:720"]
        rvp9 = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33", "-g", "150", "-cpu-used", "4", "-row-mt", "1"]
        rav1 = ["-c:v", "libsvtav1", "-crf", "35", "-preset", "8", "-g", "150"]
        for name, vargs in (("real_vp9.mp4", rvp9), ("real_av1.mp4", rav1),
                            ("real_h264_720p.mp4", [*up, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "150"]),
                            ("real_vp9_720p.mp4", [*up, *rvp9]), ("real_av1_720p.mp4", [*up, *rav1])):
            ok[name] = transcode(ff, out, "real_h264.mp4", name, vargs)

    for name, good in ok.items():
        print(f"{'ok  ' if good else 'FAIL'} {name}")
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()
