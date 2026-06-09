"""
Number Theoretic Transform (NTT) for ML-KEM (FIPS 203) and ML-DSA (FIPS 204).

Both algorithms share the same Cooley-Tukey butterfly structure but differ in:
  - Modulus q
  - Primitive root ζ
  - Depth (7 levels for ML-KEM, 8 levels for ML-DSA)
  - Inverse scaling factor

The Tracer class is the instrumentation hook used by the leakage simulator.
"""

from __future__ import annotations
from typing import Optional

# ─── Tracer ──────────────────────────────────────────────────────────────────

class Tracer:
    """Records (op_name, old_value, new_value) for every instrumented operation.

    One Tracer instance per algorithm execution.  The leakage simulator passes
    tracer.log to the leakage model to generate a synthetic power trace.
    """

    __slots__ = ("log",)

    def __init__(self) -> None:
        self.log: list[tuple[str, int, int]] = []

    def record(self, op_name: str, old_val: int, new_val: int) -> int:
        self.log.append((op_name, old_val, new_val))
        return new_val

    def reset(self) -> None:
        self.log = []

    def __len__(self) -> int:
        return len(self.log)


# ─── Bit-reversal helpers ────────────────────────────────────────────────────

def _bitrev7(n: int) -> int:
    """Reverse the 7 least significant bits of n."""
    r = 0
    for _ in range(7):
        r = (r << 1) | (n & 1)
        n >>= 1
    return r


def _bitrev8(n: int) -> int:
    """Reverse the 8 least significant bits of n."""
    r = 0
    for _ in range(8):
        r = (r << 1) | (n & 1)
        n >>= 1
    return r


# ─── ML-KEM NTT constants (FIPS 203) ─────────────────────────────────────────
# q = 3329, ζ = 17 (primitive 256th root of unity mod q).
# 7-level NTT over the ring ℤ_q[x]/(x^256 + 1).

Q_KEM = 3329
_ZETA_KEM = 17

# ZETAS_KEM[k] = 17^{BitRev7(k)} mod 3329,  k ∈ [0, 127]
# Using the bitrev formula directly — verified to give correct NTT round-trips.
ZETAS_KEM: list[int] = [pow(_ZETA_KEM, _bitrev7(k), Q_KEM) for k in range(128)]

# γ_i = ζ^{2·BitRev7(i)+1} mod q — used in base-case multiplication (Alg 11)
GAMMAS_KEM: list[int] = [pow(_ZETA_KEM, 2 * _bitrev7(i) + 1, Q_KEM) for i in range(128)]

# 128^{-1} mod 3329 (inverse scaling for INTT; 2^7 = 128 layers of 2× scaling)
_INV128_KEM = pow(128, -1, Q_KEM)   # = 3303


# ─── ML-DSA NTT constants (FIPS 204) ─────────────────────────────────────────
# q = 8380417, ζ = 1753 (primitive 512th root of unity mod q).
# 8-level NTT over the ring ℤ_q[x]/(x^256 + 1).

Q_DSA = 8380417
_ZETA_DSA = 1753

# ZETAS_DSA[k] = 1753^{BitRev8(k)} mod q, k ∈ [0, 255]; index 0 is never used.
ZETAS_DSA: list[int] = [pow(_ZETA_DSA, _bitrev8(k), Q_DSA) for k in range(256)]

# 256^{-1} mod 8380417 (inverse scaling for INTT; 2^8 = 256 layers)
_INV256_DSA = pow(256, -1, Q_DSA)


# ─── ML-KEM NTT / INTT ───────────────────────────────────────────────────────

def ntt_kem(f: list[int], tracer: Optional[Tracer] = None) -> list[int]:
    """Forward NTT for ML-KEM.  FIPS 203 Algorithm 9.

    Input:  polynomial coefficients f ∈ ℤ_q^256
    Output: NTT representation f̂ ∈ ℤ_q^256 (in-place, returns same list)
    """
    f = list(f)  # work on a copy
    k = 1
    length = 128
    while length >= 2:
        for start in range(0, 256, 2 * length):
            zeta = ZETAS_KEM[k]
            k += 1
            for j in range(start, start + length):
                old_hi = f[j + length]
                t = (zeta * f[j + length]) % Q_KEM
                new_hi = (f[j] - t) % Q_KEM
                new_lo = (f[j] + t) % Q_KEM
                if tracer is not None:
                    tracer.record("kem_ntt_hi", old_hi, new_hi)
                    tracer.record("kem_ntt_lo", f[j], new_lo)
                f[j + length] = new_hi
                f[j] = new_lo
        length >>= 1
    return f


def intt_kem(f: list[int], tracer: Optional[Tracer] = None) -> list[int]:
    """Inverse NTT for ML-KEM.  FIPS 203 Algorithm 10 (GS butterfly)."""
    f = list(f)
    k = 127
    length = 2
    while length <= 128:
        for start in range(0, 256, 2 * length):
            zeta = ZETAS_KEM[k]   # positive zeta — GS butterfly inverts CT
            k -= 1
            for j in range(start, start + length):
                old_j = f[j]
                t = f[j]
                new_j = (t + f[j + length]) % Q_KEM
                new_ji = (zeta * (f[j + length] - t)) % Q_KEM
                if tracer is not None:
                    tracer.record("kem_intt", old_j, new_j)
                f[j] = new_j
                f[j + length] = new_ji
        length <<= 1
    return [(_INV128_KEM * x) % Q_KEM for x in f]


def multiply_ntts_kem(f_hat: list[int], g_hat: list[int],
                      tracer: Optional[Tracer] = None) -> list[int]:
    """Coefficient-wise multiply in NTT domain (base-case).  FIPS 203 Algorithm 11."""
    h = [0] * 256
    for i in range(128):
        a0, a1 = f_hat[2 * i], f_hat[2 * i + 1]
        b0, b1 = g_hat[2 * i], g_hat[2 * i + 1]
        gamma = GAMMAS_KEM[i]
        h[2 * i]     = (a0 * b0 + a1 * b1 * gamma) % Q_KEM
        h[2 * i + 1] = (a0 * b1 + a1 * b0) % Q_KEM
        if tracer is not None:
            tracer.record("kem_basemul", a0, h[2 * i])
    return h


# ─── ML-DSA NTT / INTT ───────────────────────────────────────────────────────

def ntt_dsa(w: list[int], tracer: Optional[Tracer] = None) -> list[int]:
    """Forward NTT for ML-DSA.  FIPS 204 Algorithm 41.

    Input:  polynomial coefficients w ∈ ℤ_q^256
    Output: NTT representation ŵ ∈ ℤ_q^256
    """
    w = list(w)
    k = 0          # pre-incremented before first use → first zeta = ZETAS_DSA[1]
    length = 128
    while length >= 1:
        for start in range(0, 256, 2 * length):
            k += 1
            zeta = ZETAS_DSA[k]
            for j in range(start, start + length):
                old_hi = w[j + length]
                t = (zeta * w[j + length]) % Q_DSA
                new_hi = (w[j] - t) % Q_DSA
                new_lo = (w[j] + t) % Q_DSA
                if tracer is not None:
                    tracer.record("dsa_ntt_hi", old_hi, new_hi)
                    tracer.record("dsa_ntt_lo", w[j], new_lo)
                w[j + length] = new_hi
                w[j] = new_lo
        length >>= 1
    return w


def intt_dsa(w: list[int], tracer: Optional[Tracer] = None) -> list[int]:
    """Inverse NTT for ML-DSA.  FIPS 204 Algorithm 42 (GS butterfly)."""
    w = list(w)
    k = 256         # pre-decremented before first use → first zeta = ZETAS_DSA[255]
    length = 1
    while length <= 128:
        for start in range(0, 256, 2 * length):
            k -= 1
            zeta = ZETAS_DSA[k]   # positive zeta — GS butterfly inverts CT
            for j in range(start, start + length):
                old_j = w[j]
                t = w[j]
                new_j = (t + w[j + length]) % Q_DSA
                new_ji = (zeta * (w[j + length] - t)) % Q_DSA
                if tracer is not None:
                    tracer.record("dsa_intt", old_j, new_j)
                w[j] = new_j
                w[j + length] = new_ji
        length <<= 1
    return [(_INV256_DSA * x) % Q_DSA for x in w]


def multiply_ntts_dsa(f_hat: list[int], g_hat: list[int]) -> list[int]:
    """Pointwise multiply in ML-DSA NTT domain (full 256-point, no base-case)."""
    return [(a * b) % Q_DSA for a, b in zip(f_hat, g_hat)]
