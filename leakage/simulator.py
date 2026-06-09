"""
Trace simulator: runs N algorithm executions and collects power traces.

Output shape: (N, T) numpy array, where:
  N = number of traces
  T = trace length (number of instrumented operations per execution)

The simulator ensures T is identical across all N traces by padding/truncating
to the length of the first trace. In practice T is deterministic for a given
algorithm and parameter set.
"""

from __future__ import annotations
import numpy as np
import os
from typing import Optional

from algorithms.ntt import Tracer
from algorithms.mlkem import MLKEM
from algorithms.mldsa import MLDSA
from leakage.models import HammingWeightModel, PhysicsModel
from config import ALGORITHM, LEAKAGE_MODEL, PHYSICS_PARAMS


def _make_model(model_name: str, snr_db: float, seed: Optional[int]):
    if model_name == "hamming_weight":
        return HammingWeightModel(snr_db=snr_db, seed=seed)
    elif model_name == "physics":
        return PhysicsModel(snr_db=snr_db, seed=seed)
    raise ValueError(f"Unknown leakage model: {model_name!r}")


def _run_mlkem_decaps(kem: MLKEM, tracer: Tracer) -> dict:
    """Run one ML-KEM encaps+decaps cycle with a fresh random key.
    Returns the tracer log and the ciphertext (for CPA targeting)."""
    ek, dk = kem.keygen()
    m = os.urandom(32)
    K, ct = kem.encaps_internal(ek, m)
    tracer.reset()
    # Decapsulation is the leakage-critical path (secret key in use)
    kem.decaps_internal(dk, ct, tracer=tracer)
    return {"ct": ct, "K": K}


def _run_mldsa_sign(dsa: MLDSA, tracer: Tracer) -> dict:
    """Run one ML-DSA sign cycle with a fresh random key."""
    pk, sk = dsa.keygen()
    msg = os.urandom(32)
    tracer.reset()
    sig = dsa.sign_deterministic(sk, msg, tracer=tracer)
    return {"msg": msg, "sig": sig}


class TraceSimulator:
    """Collects N power traces from a chosen algorithm and leakage model.

    Parameters
    ----------
    algorithm   : str   "ML_KEM_512" | "ML_KEM_768" | "ML_DSA_44"
    model       : str   "hamming_weight" | "physics"
    n_traces    : int   Number of traces to collect
    snr_db      : float Signal-to-noise ratio in dB
    seed        : int   RNG seed for reproducibility
    fixed_key   : bool  If True, reuse the same key across all traces
    fixed_input : bool  If True, reuse the same message/plaintext (TVLA fixed set)
    """

    def __init__(
        self,
        algorithm:   str = ALGORITHM,
        model:       str = LEAKAGE_MODEL,
        n_traces:    int = 1000,
        snr_db:      float = 20.0,
        seed:        Optional[int] = 42,
        fixed_key:   bool = False,
        fixed_input: bool = False,
    ) -> None:
        self.algorithm   = algorithm
        self.model_name  = model
        self.n_traces    = n_traces
        self.snr_db      = snr_db
        self.seed        = seed
        self.fixed_key   = fixed_key
        self.fixed_input = fixed_input

        self._leakage_model = _make_model(model, snr_db, seed)
        self._traces: Optional[np.ndarray] = None
        self._metadata: list[dict] = []

    # ── Main simulation entry point ───────────────────────────────────────────

    def run(self) -> np.ndarray:
        """Run all N traces and return array of shape (N, T)."""
        if self.algorithm in ("ML_KEM_512", "ML_KEM_768"):
            self._traces, self._metadata = self._run_kem()
        elif self.algorithm == "ML_DSA_44":
            self._traces, self._metadata = self._run_dsa()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm!r}")
        return self._traces

    def _run_kem(self) -> tuple[np.ndarray, list[dict]]:
        kem = MLKEM(self.algorithm)
        fixed_ek, fixed_dk = kem.keygen() if (self.fixed_key or self.fixed_input) else (None, None)
        fixed_m = os.urandom(32) if self.fixed_input else None
        traces, metadata = [], []
        for i in range(self.n_traces):
            tracer = Tracer()
            if self.fixed_key or self.fixed_input:
                m = fixed_m if self.fixed_input else os.urandom(32)
                ek = fixed_ek if fixed_ek is not None else kem.keygen()[0]
                dk = fixed_dk if fixed_dk is not None else kem.keygen()[1]
                K, ct = kem.encaps_internal(ek, m)
                tracer.reset()
                kem.decaps_internal(dk, ct, tracer=tracer)
                meta = {"ct": ct, "K": K}
            else:
                meta = _run_mlkem_decaps(kem, tracer)
            power = self._leakage_model.compute_trace(tracer.log)
            traces.append(power)
            metadata.append(meta)
        return self._stack(traces), metadata

    def _run_dsa(self) -> tuple[np.ndarray, list[dict]]:
        dsa = MLDSA(self.algorithm)
        fixed_pk, fixed_sk = dsa.keygen() if (self.fixed_key or self.fixed_input) else (None, None)
        fixed_msg = os.urandom(32) if self.fixed_input else None
        traces, metadata = [], []
        for i in range(self.n_traces):
            tracer = Tracer()
            if self.fixed_key or self.fixed_input:
                msg = fixed_msg if self.fixed_input else os.urandom(32)
                sk  = fixed_sk if fixed_sk is not None else dsa.keygen()[1]
                tracer.reset()
                sig = dsa.sign_deterministic(sk, msg, tracer=tracer)
                meta = {"msg": msg, "sig": sig}
            else:
                meta = _run_mldsa_sign(dsa, tracer)
            power = self._leakage_model.compute_trace(tracer.log)
            traces.append(power)
            metadata.append(meta)
        return self._stack(traces), metadata

    @staticmethod
    def _stack(traces: list[np.ndarray]) -> np.ndarray:
        """Pad/truncate traces to a common length and stack into (N, T)."""
        if not traces:
            return np.empty((0, 0))
        T = len(traces[0])
        out = np.zeros((len(traces), T), dtype=np.float64)
        for i, t in enumerate(traces):
            l = min(len(t), T)
            out[i, :l] = t[:l]
        return out

    # ── TVLA helper ───────────────────────────────────────────────────────────

    @classmethod
    def tvla_sets(cls, algorithm: str = ALGORITHM, model: str = LEAKAGE_MODEL,
                  n_traces: int = 1000, snr_db: float = 20.0,
                  seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Return (fixed_traces, random_traces) each of shape (n_traces//2, T).

        Fixed set: same key AND same plaintext every trace (maximum leakage distinction).
        Random set: fresh random key and plaintext every trace.
        """
        half = n_traces // 2
        fixed = cls(algorithm, model, half, snr_db, seed,     fixed_key=True, fixed_input=True)
        rand  = cls(algorithm, model, half, snr_db, seed + 1, fixed_key=False, fixed_input=False)
        return fixed.run(), rand.run()

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def traces(self) -> Optional[np.ndarray]:
        return self._traces

    @property
    def metadata(self) -> list[dict]:
        return self._metadata
