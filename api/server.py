"""
FastAPI server — exposes the PQC SCA pipeline as REST endpoints.

Start with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /run       — run full pipeline with given parameters
    GET  /traces    — first 50 traces as JSON
    GET  /attack    — CPA correlation matrix + TVLA t-statistic
    GET  /report    — full vulnerability report
    GET  /health    — liveness check
"""

from __future__ import annotations
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from leakage.simulator import TraceSimulator
from attacks.cpa import CPA
from attacks.tvla import TVLA
from attacks.metrics import compute_metrics
from output.report import generate_report
from output.plots import plot_traces, plot_cpa, plot_tvla, fig_to_base64
from config import DEFAULT_N_TRACES, DEFAULT_SNR_DB

app = FastAPI(
    title="PQC Side-Channel Analyzer",
    description="Physics-informed ML-KEM / ML-DSA leakage analysis tool.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory state (single-user research tool) ──────────────────────────────

_state: dict = {
    "traces":      None,
    "metadata":    None,
    "cpa_result":  None,
    "tvla_result": None,
    "metrics":     None,
    "report":      None,
    "params":      None,
}


# ─── Request / Response models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    algorithm:  str   = Field("ML_KEM_512", description="ML_KEM_512 | ML_KEM_768 | ML_DSA_44")
    n_traces:   int   = Field(DEFAULT_N_TRACES, ge=10, le=10000)
    model:      str   = Field("physics",    description="hamming_weight | physics")
    snr_db:     float = Field(DEFAULT_SNR_DB, ge=0.0, le=60.0)
    seed:       int   = Field(42)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run_pipeline(req: RunRequest) -> dict:
    """Run the full leakage simulation + CPA + TVLA pipeline."""
    _state["params"] = req.model_dump()

    valid_algos = ("ML_KEM_512", "ML_KEM_768", "ML_DSA_44")
    if req.algorithm not in valid_algos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unknown algorithm {req.algorithm!r}. Valid: {valid_algos}")

    # 1. Simulate traces
    sim = TraceSimulator(
        algorithm=req.algorithm,
        model=req.model,
        n_traces=req.n_traces,
        snr_db=req.snr_db,
        seed=req.seed,
        fixed_key=False,
    )
    traces = sim.run()
    _state["traces"]   = traces
    _state["metadata"] = sim.metadata

    if traces.shape[0] == 0:
        raise HTTPException(status_code=500, detail="Trace simulation produced no data")

    # 2. Extract ciphertext bytes for CPA
    #    For ML-KEM: use first byte of each ciphertext; for ML-DSA: use first message byte
    ct_bytes = _extract_ct_bytes(sim.metadata, req.algorithm)

    # 3. CPA
    cpa = CPA(traces, ct_bytes, model="hw")
    cpa_result = cpa.run()
    _state["cpa_result"] = cpa_result

    # 4. TVLA
    fixed_traces, random_traces = TraceSimulator.tvla_sets(
        algorithm=req.algorithm, model=req.model,
        n_traces=req.n_traces, snr_db=req.snr_db, seed=req.seed,
    )
    tvla = TVLA(fixed_traces, random_traces)
    tvla_result = tvla.run()
    _state["tvla_result"] = tvla_result

    # 5. Metrics
    metrics = compute_metrics(cpa_result, tvla_result, n_traces=req.n_traces)
    _state["metrics"] = metrics

    # 6. Report
    report = generate_report(
        algorithm=req.algorithm, model=req.model,
        n_traces=req.n_traces, snr_db=req.snr_db,
        cpa_result=cpa_result, tvla_result=tvla_result,
        metrics=metrics, traces=traces,
    )
    _state["report"] = report

    return {
        "status":     "ok",
        "n_traces":   traces.shape[0],
        "trace_length": traces.shape[1],
        "vulnerability_score": metrics["vulnerability_score"],
        "vulnerability_label": metrics["vulnerability_label"],
    }


@app.get("/traces")
def get_traces() -> dict:
    """Return first 50 traces as nested list + base64 PNG."""
    traces = _state["traces"]
    if traces is None:
        raise HTTPException(status_code=404, detail="No traces — run /run first")
    subset = traces[:50]
    fig = plot_traces(traces, n_show=50, title="Simulated Power Traces")
    return {
        "n":      int(subset.shape[0]),
        "T":      int(subset.shape[1]),
        "traces": subset.tolist(),
        "plot_png": fig_to_base64(fig),
    }


@app.get("/attack")
def get_attack() -> dict:
    """Return CPA correlation matrix + TVLA t-statistic + plots."""
    cpa_result  = _state["cpa_result"]
    tvla_result = _state["tvla_result"]
    if cpa_result is None or tvla_result is None:
        raise HTTPException(status_code=404, detail="No attack results — run /run first")

    cpa_fig  = plot_cpa(cpa_result["correlation_matrix"],
                        cpa_result["recovered_key"])
    tvla_fig = plot_tvla(tvla_result["t_statistic"],
                         tvla_result["threshold"])

    return {
        "cpa": {
            "recovered_key": cpa_result["recovered_key"],
            "peak_corr":     cpa_result["peak_corr"],
            "corr_matrix":   cpa_result["correlation_matrix"].tolist(),
            "plot_png":      fig_to_base64(cpa_fig),
        },
        "tvla": {
            "max_t":            tvla_result["max_t"],
            "leakage_detected": tvla_result["leakage_detected"],
            "t_statistic":      tvla_result["t_statistic"].tolist(),
            "plot_png":         fig_to_base64(tvla_fig),
        },
    }


@app.get("/report")
def get_report() -> dict:
    """Return the full vulnerability report."""
    report = _state["report"]
    if report is None:
        raise HTTPException(status_code=404, detail="No report — run /run first")
    # Remove traces_preview from API response (too large)
    r = dict(report)
    r.pop("traces_preview", None)
    return r


# ─── Helper ───────────────────────────────────────────────────────────────────

def _extract_ct_bytes(metadata: list[dict], algorithm: str) -> np.ndarray:
    """Extract a single 'ciphertext byte' per trace for CPA targeting."""
    result = []
    for m in metadata:
        if algorithm.startswith("ML_KEM"):
            ct = m.get("ct", b"\x00")
            result.append(ct[0] if ct else 0)
        else:
            msg = m.get("msg", b"\x00")
            result.append(msg[0] if msg else 0)
    return np.array(result, dtype=np.uint8)
