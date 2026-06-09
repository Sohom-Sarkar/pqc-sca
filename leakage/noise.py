"""
Noise generators for the leakage simulation.

Two noise processes:
  - Gaussian (thermal noise floor): white noise with standard deviation σ
  - 1/f (flicker noise):  colored noise via autoregressive filter on white noise

Both are parameterised by SNR (dB) relative to a reference signal power.
"""

import numpy as np


def gaussian_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Pure Gaussian (thermal) noise: N(0, σ²) vector of length n."""
    return rng.standard_normal(n) * sigma


def flicker_noise(n: int, sigma: float, rng: np.random.Generator,
                  alpha: float = 1.0) -> np.ndarray:
    """1/f (flicker) noise approximated via autoregressive filter.

    Produces correlated noise whose power spectral density decays as 1/f^alpha.
    The AR approach (Kasdin 1995) uses precomputed coefficients:
        h[k] = prod_{i=1}^{k} (i - 1 - alpha/2) / i
    applied to a white Gaussian input.

    alpha=1.0 → 1/f noise (pink noise)
    alpha=0.0 → white noise
    alpha=2.0 → Brownian (1/f²) noise
    """
    # AR coefficients (truncated to min(n, 512) terms)
    order = min(n, 512)
    h = np.zeros(order)
    h[0] = 1.0
    for k in range(1, order):
        h[k] = h[k - 1] * (k - 1.0 - alpha / 2.0) / k

    white = rng.standard_normal(n + order) * sigma
    # Convolve via linear filter (causal AR)
    out = np.zeros(n)
    padded = np.concatenate([np.zeros(order), white[:n]])
    for i in range(n):
        for j in range(1, order):
            if i - j >= 0:
                padded[order + i] -= h[j] * padded[order + i - j]
        out[i] = padded[order + i]
    return out


def flicker_noise_fast(n: int, sigma: float, rng: np.random.Generator,
                       alpha: float = 1.0) -> np.ndarray:
    """Fast 1/f noise via FFT shaping (Timmer & König method).

    Shapes the power spectrum of white noise to follow f^{-alpha}.
    Faster than the AR approach for large n; slight edge effects.
    """
    # Frequency-domain shaping
    white = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0   # avoid division by zero at DC
    power = freqs ** (-alpha / 2.0)
    power[0] = 0.0   # zero DC component
    shaped = np.fft.irfft(power * np.fft.rfft(rng.standard_normal(n)), n=n)
    # Normalise to unit variance, then scale
    shaped /= (shaped.std() + 1e-12)
    return shaped * sigma


def snr_to_sigma(signal_power: float, snr_db: float) -> float:
    """Convert SNR in dB to noise standard deviation given signal power.

    SNR_dB = 10 · log10(P_signal / P_noise) = 10 · log10(P_signal / σ²)
    → σ = sqrt(P_signal / 10^{SNR_dB/10})
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(np.sqrt(max(signal_power, 1e-12) / snr_linear))
