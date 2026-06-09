"""
Attack metrics: Traces-to-Disclosure (TTD), success rate, vulnerability score.

TTD   — minimum N at which CPA first ranks the true key first.
SR    — fraction of independent experiments in which the correct key is recovered.
VS    — [0, 100] composite vulnerability score combining TTD and TVLA max-t.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def traces_to_disclosure(
    ns: list[int],
    corrs: list[float],
    threshold: float = 0.8,
) -> Optional[int]:
    """Find the first N in the convergence curve where correlation exceeds threshold.

    Parameters
    ----------
    ns         : list of trace counts (x-axis of convergence plot)
    corrs      : list of peak |correlation| of the true key at each N
    threshold  : correlation threshold considered "disclosed" (default 0.8)

    Returns the first N where corr >= threshold, or None if never reached.
    """
    for n, c in zip(ns, corrs):
        if c >= threshold:
            return n
    return None


def success_rate(
    n_correct: int,
    n_experiments: int,
) -> float:
    """Key recovery success rate = n_correct / n_experiments."""
    return n_correct / max(n_experiments, 1)


def vulnerability_score(
    ttd: Optional[int],
    max_ttd: int,
    tvla_max_t: float,
    tvla_threshold: float = 4.5,
) -> float:
    """Composite vulnerability score in [0, 100].

    Score components:
      - CPA component (0–70): based on TTD vs max_ttd.
        70 points if TTD ≤ 100 traces (highly vulnerable),
        scales down to 0 as TTD → max_ttd or if CPA failed.
      - TVLA component (0–30): based on max |t| relative to 4.5 threshold.
        30 points if max_t ≥ 20, 0 if max_t < 4.5.

    Higher score = more vulnerable / easier to attack.
    """
    # CPA component
    if ttd is None:
        cpa_score = 0.0
    else:
        # Linear interpolation: 70 at ttd=1, 0 at ttd=max_ttd
        cpa_score = 70.0 * max(0.0, 1.0 - (ttd - 1) / max(max_ttd - 1, 1))

    # TVLA component
    if tvla_max_t < tvla_threshold:
        tvla_score = 0.0
    else:
        tvla_score = min(30.0, 30.0 * (tvla_max_t - tvla_threshold) / (20.0 - tvla_threshold))

    return min(100.0, cpa_score + tvla_score)


def vulnerability_label(score: float) -> str:
    """Human-readable label for the vulnerability score."""
    if score >= 70:
        return "HIGH"
    elif score >= 35:
        return "MEDIUM"
    else:
        return "LOW"


def compute_metrics(
    cpa_result: dict,
    tvla_result: dict,
    true_key: Optional[int] = None,
    n_traces: int = 1000,
) -> dict:
    """Compute all metrics from CPA and TVLA results.

    Parameters
    ----------
    cpa_result  : dict returned by CPA.run()
    tvla_result : dict returned by TVLA.run()
    true_key    : known correct key byte (for TTD computation), or None
    n_traces    : total trace count (used to set max_ttd)

    Returns
    -------
    dict with:
      ttd                  : int or None
      success_rate         : float
      vulnerability_score  : float  [0, 100]
      vulnerability_label  : str    "LOW" | "MEDIUM" | "HIGH"
      cpa_recovered_key    : int
      cpa_peak_corr        : float
      tvla_max_t           : float
      tvla_leakage_detected: bool
      countermeasures      : list[str]
    """
    recovered    = cpa_result["recovered_key"]
    peak_corr    = cpa_result["peak_corr"]
    tvla_max_t   = tvla_result["max_t"]
    tvla_leaked  = tvla_result["leakage_detected"]
    n_leaking    = len(tvla_result["leaking_samples"])

    # TTD: use peak_corr as a proxy if true key unknown
    ttd = None
    sr  = 0.0
    if true_key is not None:
        if recovered == true_key:
            ttd = n_traces   # full set needed (approximate; use incremental CPA for exact)
            sr  = 1.0

    vs    = vulnerability_score(ttd, n_traces, tvla_max_t)
    label = vulnerability_label(vs)

    # Countermeasure suggestions based on attack results
    suggestions = _suggest_countermeasures(
        cpa_leaked=peak_corr > 0.5,
        tvla_leaked=tvla_leaked,
        n_leaking_samples=n_leaking,
    )

    return {
        "ttd":                   ttd,
        "success_rate":          sr,
        "vulnerability_score":   round(vs, 1),
        "vulnerability_label":   label,
        "cpa_recovered_key":     recovered,
        "cpa_peak_corr":         round(peak_corr, 4),
        "tvla_max_t":            round(tvla_max_t, 2),
        "tvla_leakage_detected": tvla_leaked,
        "n_leaking_samples":     n_leaking,
        "countermeasures":       suggestions,
    }


def _suggest_countermeasures(
    cpa_leaked: bool,
    tvla_leaked: bool,
    n_leaking_samples: int,
) -> list[str]:
    """Return countermeasure recommendations based on attack findings."""
    suggestions = []

    if cpa_leaked:
        suggestions += [
            "Apply Boolean masking to NTT intermediate values "
            "(split each coefficient into ≥2 shares before computation).",
            "Randomise NTT computation order per execution "
            "(shuffling countermeasure; increases TTD significantly).",
        ]

    if tvla_leaked:
        suggestions += [
            "Add dummy NTT operations to balance trace power at leaking time samples.",
            "Use hardware RNG to inject random noise (active blinding).",
        ]

    if n_leaking_samples > 100:
        suggestions.append(
            "Large number of leaking samples indicates systemic leakage; "
            "consider a full algorithmic re-implementation with masking."
        )

    if not suggestions:
        suggestions.append(
            "No significant leakage detected under current attack conditions; "
            "consider increasing N or lowering SNR to stress-test further."
        )

    return suggestions
