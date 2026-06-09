"""
Visualization functions for PQC SCA results.

All functions return matplotlib Figure objects (or plotly dicts when plotly=True)
so they can be embedded in the Jupyter notebook or served via the API.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


# ─── Power traces ────────────────────────────────────────────────────────────

def plot_traces(
    traces: np.ndarray,
    n_show: int = 20,
    title: str = "Power Traces",
    highlight_key: Optional[int] = None,
) -> plt.Figure:
    """Plot a stack of raw power traces.

    Parameters
    ----------
    traces      : (N, T) array
    n_show      : number of traces to overlay (max)
    title       : figure title
    highlight_key : if given, add a vertical line at sample = highlight_key
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    N, T = traces.shape
    n = min(n_show, N)
    t = np.arange(T)
    for i in range(n):
        ax.plot(t, traces[i], linewidth=0.5, alpha=0.6)
    ax.plot(t, traces[:n].mean(axis=0), color='black', linewidth=1.5,
            label='Mean trace')
    if highlight_key is not None:
        ax.axvline(highlight_key, color='red', linestyle='--', linewidth=1.0,
                   label=f'Target sample {highlight_key}')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time sample (operation index)")
    ax.set_ylabel("Simulated power (a.u.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ─── CPA correlation map ─────────────────────────────────────────────────────

def plot_cpa(
    corr_matrix: np.ndarray,
    recovered_key: int,
    true_key: Optional[int] = None,
    title: str = "CPA Correlation Map",
) -> plt.Figure:
    """Plot the CPA correlation matrix as a heatmap + peak curve.

    Parameters
    ----------
    corr_matrix  : (256, T) Pearson correlation values
    recovered_key: key guess with highest peak correlation
    true_key     : actual key byte (if known), for validation marking
    """
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Heatmap
    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(
        np.abs(corr_matrix), aspect='auto', cmap='viridis',
        vmin=0, vmax=np.abs(corr_matrix).max(),
        extent=[0, corr_matrix.shape[1], 255, 0],
    )
    plt.colorbar(im, ax=ax_heat, label="|Pearson ρ|")
    ax_heat.set_title(f"{title}\nRecovered key = 0x{recovered_key:02X}", fontsize=11)
    ax_heat.set_xlabel("Time sample")
    ax_heat.set_ylabel("Key guess (0–255)")

    # Mark recovered key
    ax_heat.axhline(recovered_key, color='cyan', linewidth=1.5,
                    linestyle='--', label=f'Recovered 0x{recovered_key:02X}')
    if true_key is not None:
        ax_heat.axhline(true_key, color='lime', linewidth=1.5,
                        linestyle=':', label=f'True 0x{true_key:02X}')
    ax_heat.legend(fontsize=8)

    # Peak correlation per key guess (bar chart)
    ax_bar = fig.add_subplot(gs[1])
    peak_per_key = np.abs(corr_matrix).max(axis=1)
    ax_bar.barh(np.arange(256), peak_per_key, height=1.0, color='steelblue', alpha=0.7)
    ax_bar.axhline(recovered_key, color='cyan',  linewidth=2, linestyle='--')
    if true_key is not None:
        ax_bar.axhline(true_key, color='lime', linewidth=2, linestyle=':')
    ax_bar.set_xlabel("Peak |ρ|")
    ax_bar.set_ylabel("Key guess")
    ax_bar.set_title("Peak correlation\nper key guess")
    ax_bar.invert_yaxis()

    fig.tight_layout()
    return fig


# ─── TVLA t-statistic ────────────────────────────────────────────────────────

def plot_tvla(
    t_statistic: np.ndarray,
    threshold: float = 4.5,
    title: str = "TVLA — Welch t-statistic",
) -> plt.Figure:
    """Plot TVLA t-statistic over time with threshold bands."""
    fig, ax = plt.subplots(figsize=(12, 4))
    T = len(t_statistic)
    t = np.arange(T)

    ax.plot(t, t_statistic, linewidth=0.8, color='steelblue', label='t-statistic')
    ax.axhline( threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'+{threshold} threshold')
    ax.axhline(-threshold, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)

    # Shade leaking regions
    leaking = np.abs(t_statistic) > threshold
    if leaking.any():
        ax.fill_between(t, t_statistic, 0,
                        where=leaking, color='red', alpha=0.25,
                        label='Leaking samples')

    ax.set_title(f"{title}\nMax |t| = {np.abs(t_statistic).max():.2f}", fontsize=11)
    ax.set_xlabel("Time sample (operation index)")
    ax.set_ylabel("Welch t-statistic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ─── Convergence plot (TTD) ───────────────────────────────────────────────────

def plot_convergence(
    ns: list[int],
    corrs_hw: list[float],
    corrs_physics: Optional[list[float]] = None,
    threshold: float = 0.8,
    title: str = "CPA Convergence (Traces-to-Disclosure)",
) -> plt.Figure:
    """Plot CPA correlation convergence vs. number of traces."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ns, corrs_hw, marker='o', markersize=3, linewidth=1.5,
            label='Hamming Weight model', color='steelblue')
    if corrs_physics is not None:
        ax.plot(ns, corrs_physics, marker='s', markersize=3, linewidth=1.5,
                label='Physics-informed model', color='darkorange')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.0,
               label=f'TTD threshold ({threshold})')
    ax.set_xlabel("Number of traces")
    ax.set_ylabel("Peak |correlation| of true key")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG string (for API/frontend)."""
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')
