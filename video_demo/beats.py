"""Musical beats of a video's sound, so cuts can land on them.

The onset strength is the spectral flux of the log magnitude spectrum (how
much louder each band gets from one frame to the next); its autocorrelation
gives the beat period within MIN_BPM-MAX_BPM, and the phase is the offset
whose beat grid collects the most onset strength. Dance music keeps a
steady tempo, which a grid fits.
"""

from __future__ import annotations

import numpy as np

RATE = 22050
FRAME = 1024
HOP = 256
MIN_BPM, MAX_BPM = 70.0, 180.0
OCTAVE_SHARE = 0.8


def onset_strength(samples: np.ndarray, rate: int = RATE) -> np.ndarray:
    """Positive spectral flux per hop of mono ``samples``, normalised to a
    maximum of 1."""
    if samples.size < FRAME * 2:
        raise ValueError("too little sound to find beats")
    n = 1 + (samples.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    window = np.hanning(FRAME).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples[idx] * window, axis=1))
    log = np.log1p(100.0 * spectrum)
    flux = np.maximum(np.diff(log, axis=0), 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    top = flux.max()
    if top <= 0:
        raise ValueError("no sound changes to find beats in")
    return flux / top


def find_beats(samples: np.ndarray, rate: int = RATE) -> tuple[float, list[float]]:
    """(beat period in seconds, beat times in seconds) of mono ``samples``.

    The period is the shortest autocorrelation peak within OCTAVE_SHARE of
    the strongest (every multiple of the beat correlates too); the phase is
    the offset whose grid at that exact period collects the most onset
    strength. Times are the middles of the analysis frames."""
    env = onset_strength(samples, rate)
    env = env - env.mean()
    hop_sec = HOP / rate
    lo = int(round(60.0 / MAX_BPM / hop_sec))
    hi = int(round(60.0 / MIN_BPM / hop_sec))
    # Smoothed over a few hops, so a period that falls between two hops
    # keeps its peak as high as its multiples that fall on one.
    smooth = np.convolve(env, np.hanning(7)[1:-1], mode="same")
    corr = np.correlate(smooth, smooth, mode="full")[env.size - 1 :]
    peaks = [
        k for k in range(max(lo, 1), hi + 1) if corr[k - 1] <= corr[k] >= corr[k + 1]
    ]
    if not peaks:
        raise ValueError("no beat period in the sound")
    best = max(corr[k] for k in peaks)
    lag = min(k for k in peaks if corr[k] >= OCTAVE_SHARE * best)
    # The peak lies between two hops: refine it with its neighbours.
    a, b, c = corr[lag - 1], corr[lag], corr[lag + 1]
    denom = a - 2 * b + c
    period = (lag + (0.5 * (a - c) / denom if denom else 0.0)) * hop_sec
    grid = np.arange(0.0, env.size * hop_sec, period)
    phases = np.arange(0.0, period, hop_sec)
    scores = [
        env[np.minimum(((grid + ph) / hop_sec).round().astype(int), env.size - 1)].sum()
        for ph in phases
    ]
    phase = float(phases[int(np.argmax(scores))]) + FRAME / 2 / rate
    times = np.arange(phase, env.size * hop_sec, period)
    return float(period), [round(float(t), 3) for t in times]


def snap(
    start: float,
    length: float,
    lo: float,
    hi: float,
    period: float,
    times: list[float],
) -> tuple[float, float]:
    """A window of about ``length`` inside [lo, hi] that starts on a beat
    and lasts a whole number of beats (at least one); the window as it was
    when no beat leaves room for one."""
    beats = max(1, round(length / period))
    room = [t for t in times if lo <= t and t + period <= hi + 1e-6]
    if not room:
        return start, start + length
    first = min(room, key=lambda t: abs(t - start))
    while beats > 1 and first + beats * period > hi + 1e-6:
        beats -= 1
    return first, first + beats * period
