"""Treble-focused A/B files for comparing FFmpeg's default and fast AAC coders.

Usage: python audio_coder_ab.py <out dir> <real media file>

Used in Oct 2026 to decide that RecodeAudioCutter keeps FFmpeg's default AAC coder: aac_coder=fast was ~1.8x
faster but left visible holes above ~6 kHz in the spectrogram and audibly duller / swishy cymbals.

Two sets:
  drums_*  : a synthetic drum loop (closed/open hi-hats, ride, crash, kick, snare) made from noise and tones.
             0_lossless.wav is the generated signal; 1_source.m4a is it encoded to AAC-LC 128 kbps (like a
             YouTube download); 2_default / 3_fast re-encode that source again, as a Smart Edit with audio fades does.
  music_*  : the 15 s of the real file with the most energy above 8 kHz (usually cymbals/hi-hats);
             1_original.m4a copies the source packets, 2_default / 3_fast re-encode them.
For each re-encode, *_lost_x4.wav is (source - re-encode) amplified 4x: only the detail the re-encode removed.
spectrograms.png shows source / default / fast side by side.
"""
import sys
from contextlib import closing  # with closing(av.open()): PyAV types a container's __exit__ as maybe swallowing errors
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import matplotlib
from av import AudioStream

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, real_src = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


# ---------- synthetic drums ----------
def drums(rate: int = 48000, seconds: float = 20.0, bpm: int = 110) -> tuple[np.ndarray, int]:
    n = int(rate * seconds)
    left, right = np.zeros(n), np.zeros(n)

    def noise(length: int, lo: float, hi: float | None = None) -> np.ndarray:
        spec = np.fft.rfft(rng.standard_normal(length))
        f = np.fft.rfftfreq(length, 1 / rate)
        spec[(f < lo) | ((f > hi) if hi else False)] = 0
        x = np.fft.irfft(spec, length)
        return x / (np.abs(x).max() + 1e-9)

    def hit(at: float, sig: np.ndarray, gain: float, pan: float) -> None:  # pan: -1 left .. +1 right
        i = int(at * rate)
        j = min(n, i + len(sig))
        left[i:j] += sig[: j - i] * gain * (1 - pan) / 2
        right[i:j] += sig[: j - i] * gain * (1 + pan) / 2

    def env(length: int, decay: float) -> np.ndarray:
        return np.exp(-np.arange(length) / rate / decay)

    def partials(length: int, freqs: list[int]) -> np.ndarray | float:  # sum(): float for no freqs
        tt = np.arange(length) / rate
        return sum(np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.28)) for f in freqs) / len(freqs)

    beat = 60 / bpm
    bar = 4 * beat
    for b in range(int(seconds / bar)):
        t0 = b * bar
        for k in range(8):  # eighth-note hi-hats, open on the last "and" of every bar
            t = t0 + k * beat / 2
            if k == 7:
                L = int(0.45 * rate)
                hit(t, noise(L, 6000) * env(L, 0.15), 0.32, -0.4)
            else:
                L = int(0.09 * rate)
                hit(t, noise(L, 7000) * env(L, 0.018), 0.30 if k % 2 == 0 else 0.22, -0.4)
        for k in range(4):  # ride on quarter notes
            L = int(0.7 * rate)
            ride = 0.6 * partials(L, [2730, 3920, 5480, 7310, 9840, 12650]) + 0.4 * noise(L, 5000)
            hit(t0 + k * beat, ride * env(L, 0.3), 0.18, 0.5)
        if b % 2 == 0:  # crash every two bars
            L = int(2.6 * rate)
            crash = 0.55 * noise(L, 3500) + 0.45 * partials(L, [3170, 4740, 6290, 8910, 11380, 13720, 15900])
            hit(t0, crash * env(L, 0.7), 0.42, 0.2)
        for k in (0, 2):  # kick
            L = int(0.3 * rate)
            tt = np.arange(L) / rate
            f = 45 + 75 * np.exp(-tt / 0.04)
            hit(t0 + k * beat, np.sin(2 * np.pi * np.cumsum(f) / rate) * env(L, 0.12), 0.85, 0.0)
        for k in (1, 3):  # snare
            L = int(0.25 * rate)
            tt = np.arange(L) / rate
            sn = 0.6 * noise(L, 1000, 9000) * env(L, 0.07) + 0.4 * np.sin(2 * np.pi * 190 * tt) * env(L, 0.05)
            hit(t0 + k * beat, sn, 0.55, 0.05)
    stereo = np.stack([left, right]).astype(np.float32)
    return stereo / np.abs(stereo).max() * 0.89, rate  # peak -1 dBFS


# ---------- encode / decode helpers (PyAV, same FFmpeg as smartcut) ----------
def write(samples: np.ndarray, rate: int, path: Path, codec: str = "aac", coder: str | None = None) -> None:
    with av.open(str(path), "w") as o:
        s = cast(AudioStream, o.add_stream(codec, rate=rate))
        s.codec_context.layout = "stereo"
        if codec == "aac":
            s.codec_context.bit_rate = 128_000
            if coder:
                s.codec_context.options = {"aac_coder": coder}
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples, np.float32), format="fltp", layout="stereo")
        frame.sample_rate, frame.time_base, frame.pts = rate, Fraction(1, rate), 0
        for p in s.encode(frame):
            o.mux(p)
        for p in s.encode(None):
            o.mux(p)


def read(path: Path) -> tuple[np.ndarray, int]:
    with closing(av.open(str(path))) as c:
        a = c.streams.audio[0]
        frames = [f for f in c.decode(a)]
    arr = np.concatenate([f.to_ndarray().astype(np.float32) for f in frames], axis=1)
    if not frames[0].format.is_planar:
        arr = arr.reshape(-1, 2).T
    return arr, frames[0].sample_rate


def align(ref: np.ndarray, x: np.ndarray, max_lag: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Shift x so it lines up with ref (cross-correlation on the first 2 s)."""
    n = min(ref.shape[1], x.shape[1], 96000)
    r, y = ref[0, :n], x[0, :n]
    corr = np.fft.irfft(np.fft.rfft(r, 2 * n) * np.conj(np.fft.rfft(y, 2 * n)))
    lags = np.concatenate([corr[: max_lag + 1], corr[-max_lag:]])
    lag = int(np.argmax(lags))
    lag = lag if lag <= max_lag else lag - (2 * max_lag + 1)
    x = np.roll(x, lag, axis=1)
    m = min(ref.shape[1], x.shape[1])
    return x[:, :m], ref[:, :m]


def snr(ref: np.ndarray, x: np.ndarray, rate: int, min_hz: float = 0) -> float:
    r, y = ref.mean(axis=0), x.mean(axis=0)
    if min_hz:
        R, Y = np.fft.rfft(r), np.fft.rfft(y)
        cut = int(min_hz / (rate / 2) * len(R))
        R[:cut] = 0
        Y[:cut] = 0
        r, y = np.fft.irfft(R, len(r)), np.fft.irfft(Y, len(y))
    return 10 * np.log10(np.sum(r ** 2) / max(np.sum((r - y) ** 2), 1e-20))


def copy_excerpt(src: Path, t0: float, dur: float, dst: Path) -> None:
    with av.open(str(src)) as c, av.open(str(dst), "w") as o:
        a = c.streams.audio[0]
        s = o.add_stream_from_template(a)
        c.seek(int(max(t0 - 1, 0) / cast(Fraction, a.time_base)), stream=a)
        first = None
        for p in c.demux(a):
            if p.pts is None:
                continue
            t = float(p.pts * cast(Fraction, a.time_base))
            if t < t0:
                continue
            if t >= t0 + dur:
                break
            first = p.pts if first is None else first
            p.pts -= first
            p.dts = cast(int, p.dts) - first
            p.stream = s
            o.mux(p)


def most_treble_window(src: Path, seconds: int = 15) -> tuple[int, float]:
    x, rate = read(src)
    mono = x.mean(axis=0)
    hop = rate
    energy = []
    for i in range(0, len(mono) - hop, hop):
        spec = np.abs(np.fft.rfft(mono[i: i + hop]))
        f = np.fft.rfftfreq(hop, 1 / rate)
        energy.append((spec[f > 8000] ** 2).sum() / max((spec ** 2).sum(), 1e-20))
    energy = np.array(energy)
    sums = np.convolve(energy, np.ones(seconds), mode="valid")
    return int(np.argmax(sums)), float(energy[np.argmax(sums): np.argmax(sums) + seconds].mean())


results = []
spectro = []

# Set 1: synthetic drums
lossless, rate = drums()
write(lossless, rate, out / "drums_0_lossless.wav", codec="pcm_s16le")
write(lossless, rate, out / "drums_1_source.m4a")
source, _ = read(out / "drums_1_source.m4a")
sets = [("drums", source, rate)]

# Set 2: most cymbal-heavy part of the real file
start, share = most_treble_window(real_src)
copy_excerpt(real_src, start, 15, out / "music_1_original.m4a")
orig, mrate = read(out / "music_1_original.m4a")
sets.append(("music", orig, mrate))
print(f"music excerpt: {real_src.name} @ {start} s (energy above 8 kHz: {share * 100:.1f}% of the total)")

for label, ref, r in sets:
    row = [(f"{label}: source", ref)]
    for name, coder in (("2_default", None), ("3_fast", "fast")):
        path = out / f"{label}_{name}.m4a"
        write(ref, r, path, coder=coder)
        x, _ = read(path)
        x, ref_m = align(ref, x)
        lost = (ref_m - x) * 4
        write(np.clip(lost, -1, 1), r, out / f"{label}_{name}_lost_x4.wav", codec="pcm_s16le")
        results.append(f"  {label}_{name}.m4a: SNR {snr(ref_m, x, r):5.1f} dB overall, {snr(ref_m, x, r, 10000):5.1f} dB above 10 kHz")
        row.append((f"{label}: re-encoded, {'default' if coder is None else 'fast'} coder", x))
    spectro.append((row, r))
print("\n".join(results))

# Spectrograms: 4 s of each version
fig, axes = plt.subplots(len(spectro), 3, figsize=(16, 4.2 * len(spectro)), sharey=True)
for i, (row, r) in enumerate(spectro):
    for j, (title, x) in enumerate(row):
        ax = axes[i][j]
        seg = x.mean(axis=0)[2 * r: 6 * r]
        ax.specgram(seg, NFFT=2048, Fs=r, noverlap=1536, cmap="magma", vmin=-130, vmax=-20)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, r / 2)
        ax.set_xlabel("seconds")
        if j == 0:
            ax.set_ylabel("Hz")
fig.suptitle("AAC 128 kbps re-encode: default vs fast coder (brighter = more energy; look above ~10 kHz)", fontsize=13)
fig.tight_layout()
fig.savefig(out / "spectrograms.png", dpi=90)
print(f"wrote {out}")
