"""
Correlation Power Analysis (CPA) — Kocher et al. 1999.

Targets ML-KEM NTT intermediate values during decapsulation.

The hypothesis model:
    H(k_guess, ct_byte) = leakage_model(NTT_output(k_guess ⊕ ct_byte))

Attack procedure:
1.  For each candidate key byte k ∈ {0, …, 255}:
    a.  Compute hypothesis H[i] for each trace i using the known ciphertext byte.
    b.  Compute Pearson correlation between H[i] and trace[i, t] for each
        time sample t.
2.  The key guess with the highest peak |correlation| at any time sample is
    the recovered key byte.

Returns:
    correlation_matrix : (256, T) — one row per key guess
    recovered_key      : int       — argmax of peak correlation
    peak_corr          : float     — peak |correlation| value
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def _hw(v: int) -> int:
    return bin(v).count('1')


class CPA:
    """Correlation Power Analysis engine.

    Parameters
    ----------
    traces      : (N, T) array of power traces
    ciphertext_bytes : (N,) array of the targeted ciphertext byte for each trace
    model       : 'hw' (Hamming weight) or 'hd' (Hamming distance from 0)
    """

    def __init__(
        self,
        traces: np.ndarray,
        ciphertext_bytes: np.ndarray,
        model: str = "hw",
    ) -> None:
        self.traces = np.asarray(traces, dtype=np.float64)
        self.ct_bytes = np.asarray(ciphertext_bytes, dtype=np.uint8)
        self.model = model
        self._corr_matrix: Optional[np.ndarray] = None

    # ── Hypothesis functions ──────────────────────────────────────────────────

    def _hypothesis(self, k_guess: int) -> np.ndarray:
        """Return hypothesis vector H[i] = leakage(k_guess ⊕ ct_byte[i])."""
        intermediate = k_guess ^ self.ct_bytes
        if self.model == "hw":
            return np.array([_hw(int(v)) for v in intermediate], dtype=np.float64)
        elif self.model == "hd":
            # HD from zero (when previous value unknown): equivalent to HW
            return np.array([_hw(int(v)) for v in intermediate], dtype=np.float64)
        raise ValueError(f"Unknown CPA model: {self.model!r}")

    # ── Core attack ───────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run CPA over all 256 key guesses.

        Returns
        -------
        dict with keys:
          correlation_matrix : (256, T) float array
          recovered_key      : int
          peak_corr          : float
          key_rank           : dict mapping each guess to its peak |corr|
        """
        N, T = self.traces.shape
        corr = np.zeros((256, T), dtype=np.float64)

        # Precompute trace stats (avoid redundant computation per guess)
        trace_mean = self.traces.mean(axis=0)           # (T,)
        trace_std  = self.traces.std(axis=0) + 1e-12    # (T,)
        traces_centered = self.traces - trace_mean       # (N, T)

        for k in range(256):
            h = self._hypothesis(k)                     # (N,)
            h_centered = h - h.mean()
            h_std = h.std() + 1e-12
            # Pearson correlation for all T simultaneously
            cov = (traces_centered.T @ h_centered) / N  # (T,)
            corr[k] = cov / (trace_std * h_std)

        self._corr_matrix = corr

        peak_per_key = np.abs(corr).max(axis=1)         # (256,)
        recovered    = int(peak_per_key.argmax())
        peak_corr    = float(peak_per_key[recovered])
        key_rank     = {k: float(peak_per_key[k]) for k in range(256)}

        return {
            "correlation_matrix": corr,
            "recovered_key":      recovered,
            "peak_corr":          peak_corr,
            "key_rank":           key_rank,
        }

    # ── Incremental TTD (traces-to-disclosure) ────────────────────────────────

    def run_incremental(self, true_key: int,
                        step: int = 50) -> tuple[list[int], list[float]]:
        """Run CPA with increasing N to find traces-to-disclosure (TTD).

        Returns (n_values, corr_of_true_key) for plotting the convergence curve.
        """
        N = self.traces.shape[0]
        ns, corrs = [], []
        for n in range(step, N + 1, step):
            sub = CPA(self.traces[:n], self.ct_bytes[:n], self.model)
            result = sub.run()
            corr = float(np.abs(result["correlation_matrix"][true_key]).max())
            ns.append(n)
            corrs.append(corr)
        return ns, corrs

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        return self._corr_matrix


# ── ML-KEM NTT-Coefficient CPA ────────────────────────────────────────────────

class NTTCoeffCPA:
    """Rigorous ML-KEM side-channel attack targeting secret polynomial coefficients.

    Attack model (FIPS 203 §5.3, k-PKE.Decrypt):
        Intermediate = s_hat[poly_idx][2*coeff_idx] * u_hat[poly_idx][2*coeff_idx] mod q
    where:
        u_hat  = NTT(Decompress(c1, du))  — computable from ciphertext alone
        s_hat  = NTT(s)                   — secret key NTT coefficients

    The secret coefficient s_hat[*][j] belongs to {-eta1, ..., eta1} (7 candidates
    for ML-KEM-512 with eta1=3), making this attack orders of magnitude more
    efficient than a 256-candidate byte-level attack.

    Parameters
    ----------
    traces       : (N, T) power trace array
    u_hat_coeffs : (N,) array of the target u_hat coefficient for each trace
                   (extracted from ciphertext via _decode_u_hat_coeff())
    eta          : int  bound on secret coefficients (eta1 from ML-KEM params)
    q            : int  modulus (3329 for ML-KEM)
    target_sample: int  index into the trace to correlate against (auto-selected
                   as the kem_intt region start if None)
    """

    def __init__(
        self,
        traces:        np.ndarray,
        u_hat_coeffs:  np.ndarray,
        eta:           int = 3,
        q:             int = 3329,
    ) -> None:
        self.traces        = np.asarray(traces, dtype=np.float64)
        self.u_hat_coeffs  = np.asarray(u_hat_coeffs, dtype=np.int64)
        self.eta           = eta
        self.q             = q
        self._corr_matrix: Optional[np.ndarray] = None

    def _hypothesis(self, s_guess: int) -> np.ndarray:
        """HW(s_guess * u_hat_coeff mod q) for each trace."""
        intermediates = (s_guess * self.u_hat_coeffs) % self.q
        return np.array([_hw(int(v)) for v in intermediates], dtype=np.float64)

    def run(self) -> dict:
        """Run CPA over all 2*eta+1 secret coefficient guesses.

        Returns
        -------
        dict with:
          correlation_matrix : (2*eta+1, T) — one row per s_guess
          s_candidates       : list of candidate values [-eta, ..., eta]
          recovered_s        : int  — recovered secret coefficient
          peak_corr          : float
          key_rank           : dict mapping each s_guess to its peak |corr|
        """
        N, T = self.traces.shape
        candidates = list(range(-self.eta, self.eta + 1))
        n_cand = len(candidates)
        corr = np.zeros((n_cand, T), dtype=np.float64)

        trace_mean     = self.traces.mean(axis=0)
        trace_std      = self.traces.std(axis=0) + 1e-12
        traces_centered = self.traces - trace_mean

        for idx, s in enumerate(candidates):
            h = self._hypothesis(s)
            h_centered = h - h.mean()
            h_std = h.std() + 1e-12
            cov = (traces_centered.T @ h_centered) / N
            corr[idx] = cov / (trace_std * h_std)

        self._corr_matrix = corr
        peak_per_cand = np.abs(corr).max(axis=1)
        best_idx      = int(peak_per_cand.argmax())
        recovered_s   = candidates[best_idx]
        peak_corr     = float(peak_per_cand[best_idx])
        key_rank      = {s: float(peak_per_cand[i]) for i, s in enumerate(candidates)}

        return {
            "correlation_matrix": corr,
            "s_candidates":       candidates,
            "recovered_s":        recovered_s,
            "peak_corr":          peak_corr,
            "key_rank":           key_rank,
        }

    def run_incremental(self, true_s: int, step: int = 50) -> tuple[list[int], list[float]]:
        """Convergence curve: peak |corr| of true_s vs N."""
        N = self.traces.shape[0]
        ns, corrs = [], []
        for n in range(step, N + 1, step):
            sub = NTTCoeffCPA(self.traces[:n], self.u_hat_coeffs[:n], self.eta, self.q)
            r = sub.run()
            cand_idx = r["s_candidates"].index(true_s)
            corrs.append(float(np.abs(r["correlation_matrix"][cand_idx]).max()))
            ns.append(n)
        return ns, corrs


def decode_u_hat_coeff(ct: bytes, poly_idx: int = 0, coeff_idx: int = 0,
                       k: int = 2, du: int = 10, q: int = 3329) -> int:
    """Decode one NTT coefficient of u from the ML-KEM ciphertext.

    Reconstructs u_hat[poly_idx][2*coeff_idx] — the ciphertext-side value
    used in the CPA hypothesis function.

    Parameters
    ----------
    ct         : raw ciphertext bytes
    poly_idx   : which u polynomial (0 or 1 for k=2)
    coeff_idx  : which base-multiplication pair (0..127)
    k, du, q   : ML-KEM parameters
    """
    # Avoid importing the full algorithm module here
    c1_len = k * 256 * du // 8
    poly_len = 256 * du // 8
    c1_poly = ct[poly_idx * poly_len : (poly_idx + 1) * poly_len]

    # Decode du-bit packed values
    n = 256
    F = [0] * n
    for i in range(n * du):
        bit = (c1_poly[i >> 3] >> (i & 7)) & 1
        F[i // du] |= bit << (i % du)

    # Decompress: round(y * q / 2^du)
    u_coeffs = [(y * q + (1 << (du - 1))) >> du for y in F]

    # NTT of u_coeffs — use the ntt_kem function
    from algorithms.ntt import ntt_kem
    u_hat = ntt_kem(u_coeffs)

    return u_hat[2 * coeff_idx]
