"""
Test Vector Leakage Assessment (TVLA) — Goodwill et al. 2011.

Compares power traces from a fixed-input set vs. a random-input set using
Welch's two-sample t-test at each time sample.

Decision rule (standard threshold):
    |t(sample)| > 4.5  →  leakage detected at that sample

Returns:
    t_statistic  : (T,) float array — t-value at each time sample
    leaking_mask : (T,) bool array  — True where |t| > threshold
    max_t        : float            — peak |t| value
    leakage_detected : bool         — any sample above threshold
"""

from __future__ import annotations
import numpy as np
from scipy import stats


class TVLA:
    """Welch t-test leakage assessment.

    Parameters
    ----------
    fixed_traces  : (N1, T) traces with fixed (constant) input
    random_traces : (N2, T) traces with random inputs
    threshold     : float  |t| threshold for leakage (default 4.5)
    """

    def __init__(
        self,
        fixed_traces:  np.ndarray,
        random_traces: np.ndarray,
        threshold: float = 4.5,
    ) -> None:
        self.fixed  = np.asarray(fixed_traces,  dtype=np.float64)
        self.random = np.asarray(random_traces, dtype=np.float64)
        self.threshold = threshold
        self._result: dict | None = None

    def run(self) -> dict:
        """Run Welch t-test at every time sample.

        Returns dict with:
          t_statistic      : (T,)
          leaking_mask     : (T,) bool
          max_t            : float
          leakage_detected : bool
          leaking_samples  : list[int]  — indices where |t| > threshold
        """
        T = self.fixed.shape[1]
        t_stat = np.zeros(T, dtype=np.float64)

        # Vectorised Welch t-test across all T samples simultaneously
        mu1  = self.fixed.mean(axis=0)
        mu2  = self.random.mean(axis=0)
        var1 = self.fixed.var(axis=0, ddof=1) + 1e-20
        var2 = self.random.var(axis=0, ddof=1) + 1e-20
        n1   = self.fixed.shape[0]
        n2   = self.random.shape[0]

        t_stat = (mu1 - mu2) / np.sqrt(var1 / n1 + var2 / n2)

        leaking_mask    = np.abs(t_stat) > self.threshold
        leaking_samples = list(np.where(leaking_mask)[0])
        max_t           = float(np.abs(t_stat).max())

        self._result = {
            "t_statistic":      t_stat,
            "leaking_mask":     leaking_mask,
            "max_t":            max_t,
            "leakage_detected": bool(leaking_mask.any()),
            "leaking_samples":  leaking_samples,
            "threshold":        self.threshold,
            "n_fixed":          n1,
            "n_random":         n2,
        }
        return self._result

    @property
    def result(self) -> dict | None:
        return self._result

    # ── Convenience: run from simulator output ────────────────────────────────

    @classmethod
    def from_simulator(cls, algorithm: str, model: str,
                       n_traces: int = 1000, snr_db: float = 20.0,
                       seed: int = 42, threshold: float = 4.5) -> "TVLA":
        """Build a TVLA instance using TraceSimulator to generate both sets."""
        from leakage.simulator import TraceSimulator
        fixed, random = TraceSimulator.tvla_sets(
            algorithm=algorithm, model=model,
            n_traces=n_traces, snr_db=snr_db, seed=seed,
        )
        return cls(fixed, random, threshold=threshold)
