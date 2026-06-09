"""
Structured vulnerability report generator.

Aggregates results from CPA, TVLA, and the metrics module into a
JSON-serialisable dict that can be served by the API or saved to disk.
"""

from __future__ import annotations
import datetime
import numpy as np
from typing import Optional


def generate_report(
    algorithm:    str,
    model:        str,
    n_traces:     int,
    snr_db:       float,
    cpa_result:   dict,
    tvla_result:  dict,
    metrics:      dict,
    traces:       Optional[np.ndarray] = None,
) -> dict:
    """Build the full vulnerability report.

    Parameters
    ----------
    algorithm   : "ML_KEM_512" | "ML_KEM_768" | "ML_DSA_44"
    model       : "hamming_weight" | "physics"
    n_traces    : total trace count used
    snr_db      : simulation SNR in dB
    cpa_result  : CPA.run() output
    tvla_result : TVLA.run() output
    metrics     : compute_metrics() output
    traces      : optional (N, T) array (first 50 rows sent to UI)

    Returns
    -------
    JSON-serialisable dict.
    """
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    # Leakage map: which time windows are leaking according to TVLA
    leaking_samples = tvla_result.get("leaking_samples", [])
    leaking_density = (len(leaking_samples) /
                       max(len(tvla_result.get("t_statistic", [1])), 1))

    # Top-5 key guesses by CPA correlation
    key_rank = cpa_result.get("key_rank", {})
    top5 = sorted(key_rank.items(), key=lambda x: x[1], reverse=True)[:5]

    report = {
        "timestamp":    timestamp,
        "algorithm":    algorithm,
        "leakage_model": model,
        "simulation": {
            "n_traces":  n_traces,
            "snr_db":    snr_db,
            "trace_length": int(traces.shape[1]) if traces is not None else None,
        },
        "cpa": {
            "recovered_key":  cpa_result.get("recovered_key"),
            "peak_correlation": cpa_result.get("peak_corr"),
            "top5_guesses": [
                {"key": f"0x{k:02X}", "peak_corr": round(v, 4)}
                for k, v in top5
            ],
        },
        "tvla": {
            "max_t":              tvla_result.get("max_t"),
            "threshold":          tvla_result.get("threshold", 4.5),
            "leakage_detected":   tvla_result.get("leakage_detected"),
            "n_leaking_samples":  len(leaking_samples),
            "leaking_density":    round(leaking_density, 4),
        },
        "vulnerability": {
            "score": metrics.get("vulnerability_score"),
            "label": metrics.get("vulnerability_label"),
            "ttd":   metrics.get("ttd"),
        },
        "countermeasures": metrics.get("countermeasures", []),
        "traces_preview": (
            traces[:50].tolist() if traces is not None else None
        ),
    }

    return report
