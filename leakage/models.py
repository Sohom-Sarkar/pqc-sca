"""
Leakage models for PQC side-channel analysis.

Model A — Hamming Weight (baseline):
    P(v) = HW(v) + N(0, σ²)

Model B — Physics-informed (novel contribution):
    P(v_new, v_old) = α · HD(v_new, v_old) · C · V²
                    + β · HW(v_new) · V
                    + γ · N_thermal
                    + δ · N_flicker

The physics model maps directly to CMOS switching energy:
  - HD term  → dynamic power: E_dyn = C · V² · ΔN_transitions
  - HW term  → static leakage: P_static ∝ HW · V (current through on-state devices)
  - Thermal  → Johnson-Nyquist noise floor (kT/C noise)
  - Flicker  → 1/f noise from interface traps (prominent in sub-micron nodes)

All parameters are exposed via config.PHYSICS_PARAMS and can be tuned
by the user based on target device physics.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from config import PHYSICS_PARAMS
from leakage.noise import gaussian_noise, flicker_noise_fast, snr_to_sigma


def _hw(v: int) -> int:
    """Hamming weight (popcount) of integer v."""
    return bin(v).count('1')


def _hd(v_new: int, v_old: int) -> int:
    """Hamming distance: number of bit transitions."""
    return _hw(v_new ^ v_old)


class HammingWeightModel:
    """Baseline leakage model: power ∝ HW(value) + Gaussian noise.

    This is the standard model used in most published PQC SCA papers.
    We include it as a comparison baseline.
    """

    def __init__(self, snr_db: float = 20.0, seed: Optional[int] = None) -> None:
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    def compute_trace(self, trace_log: list[tuple[str, int, int]]) -> np.ndarray:
        """Convert a Tracer log → power trace array.

        Each log entry contributes one sample = HW(new_value) + noise.
        """
        if not trace_log:
            return np.array([])

        values = np.array([entry[2] for entry in trace_log], dtype=np.float64)
        hw_vals = np.array([_hw(int(v)) for v in values], dtype=np.float64)

        signal_power = float(np.var(hw_vals)) if np.var(hw_vals) > 0 else 1.0
        sigma = snr_to_sigma(signal_power, self.snr_db)
        noise = gaussian_noise(len(hw_vals), sigma, self.rng)

        return hw_vals + noise

    def hypothesis(self, key_guess: int, ciphertext_byte: int) -> float:
        """CPA hypothesis: expected leakage given a key guess and ciphertext byte."""
        intermediate = key_guess ^ ciphertext_byte   # XOR model for key-byte attack
        return float(_hw(intermediate))


class PhysicsModel:
    """Physics-informed leakage model (novel research contribution).

    Power equation:
        P(v_new, v_old) = α · HD(v_new, v_old) · C · V²
                        + β · HW(v_new) · V
                        + γ · N_thermal
                        + δ · N_flicker

    Physical parameter mapping (tunable via config.PHYSICS_PARAMS):
      C     — effective node capacitance (F)
      V     — supply voltage (V)
      alpha — dynamic power coefficient (switching energy weight)
      beta  — static leakage coefficient (quiescent current weight)
      gamma — thermal noise weight (Johnson-Nyquist floor)
      delta — flicker noise weight (1/f noise amplitude)
    """

    def __init__(self, snr_db: float = 20.0, seed: Optional[int] = None,
                 params: Optional[dict] = None) -> None:
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)
        p = params or PHYSICS_PARAMS
        self.C     = p["C"]
        self.V     = p["V"]
        self.alpha = p["alpha"]
        self.beta  = p["beta"]
        self.gamma = p["gamma"]
        self.delta = p["delta"]

    def _sample_power(self, v_new: int, v_old: int) -> float:
        """Deterministic (noise-free) power for a single transition."""
        dynamic = self.alpha * _hd(v_new, v_old) * self.C * (self.V ** 2)
        static  = self.beta  * _hw(v_new) * self.V
        return dynamic + static

    def compute_trace(self, trace_log: list[tuple[str, int, int]]) -> np.ndarray:
        """Convert a Tracer log → physics-informed power trace array.

        Each log entry (op, old_val, new_val) contributes:
            P_det(new_val, old_val)  +  γ·N_thermal  +  δ·N_flicker
        """
        if not trace_log:
            return np.array([])

        n = len(trace_log)
        det = np.array(
            [self._sample_power(int(entry[2]), int(entry[1])) for entry in trace_log],
            dtype=np.float64,
        )

        signal_power = float(np.var(det)) if np.var(det) > 0 else 1.0
        sigma_base   = snr_to_sigma(signal_power, self.snr_db)

        thermal  = gaussian_noise(n, self.gamma * sigma_base, self.rng)
        flicker  = flicker_noise_fast(n, self.delta * sigma_base, self.rng)

        return det + thermal + flicker

    def hypothesis(self, key_guess: int, ciphertext_byte: int,
                   prev_value: int = 0) -> float:
        """CPA hypothesis: expected physics-informed leakage."""
        intermediate = key_guess ^ ciphertext_byte
        return self._sample_power(intermediate, prev_value)
