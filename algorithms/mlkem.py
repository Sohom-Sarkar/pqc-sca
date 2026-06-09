"""
ML-KEM (CRYSTALS-Kyber) — NIST FIPS 203 (2024).

Implements ML-KEM-512 and ML-KEM-768 with full instrumentation via Tracer.
All internal functions are pure (no global state); the Tracer is passed
explicitly so the leakage simulator can collect one trace per call.

Algorithm numbering follows FIPS 203.
"""

from __future__ import annotations
import hashlib
import os
from typing import Optional

from .ntt import (
    Tracer, ntt_kem, intt_kem, multiply_ntts_kem, Q_KEM
)
from config import MLKEM_PARAMS

# ─── Parameter helper ────────────────────────────────────────────────────────

def _params(variant: str) -> dict:
    p = MLKEM_PARAMS[variant]
    return p


# ─── Hash / XOF wrappers (FIPS 203 §4.1) ────────────────────────────────────

def _H(data: bytes) -> bytes:
    """H = SHA3-256 (32-byte output)."""
    return hashlib.sha3_256(data).digest()


def _G(data: bytes) -> tuple[bytes, bytes]:
    """G = SHA3-512 (64-byte output), split 32|32."""
    out = hashlib.sha3_512(data).digest()
    return out[:32], out[32:]


def _J(data: bytes) -> bytes:
    """J = SHAKE-256 with 32-byte output."""
    h = hashlib.shake_256(data)
    return h.digest(32)


def _XOF(rho: bytes, j: int, i: int) -> "hashlib._Hash":
    """XOF = SHAKE-128, seeded with ρ ∥ j ∥ i.  Returns a streaming object."""
    seed = rho + bytes([j, i])
    return hashlib.shake_128(seed)


def _PRF(sigma: bytes, b: int, length: int) -> bytes:
    """PRF_η = SHAKE-256(σ ∥ b), output `length` bytes."""
    return hashlib.shake_256(sigma + bytes([b])).digest(length)


# ─── Encoding / decoding (FIPS 203 Algorithms 5–6) ───────────────────────────

def _byte_encode(F: list[int], d: int) -> bytes:
    """Encode 256 d-bit integers as bytes.  Algorithm 5."""
    out = bytearray(256 * d // 8)
    bit = 0
    for f in F:
        for i in range(d):
            b_idx = bit >> 3
            b_off = bit & 7
            out[b_idx] |= ((f >> i) & 1) << b_off
            bit += 1
    return bytes(out)


def _byte_decode(B: bytes, d: int) -> list[int]:
    """Decode bytes to 256 d-bit integers.  Algorithm 6."""
    n = len(B) * 8 // d
    F = [0] * n
    for i in range(n * d):
        bit = (B[i >> 3] >> (i & 7)) & 1
        F[i // d] |= bit << (i % d)
    return F


def _compress(x: int, d: int, q: int = Q_KEM) -> int:
    """Compress a single coefficient: round(x · 2^d / q) mod 2^d."""
    return ((x * (1 << d) + q // 2) // q) % (1 << d)


def _decompress(y: int, d: int, q: int = Q_KEM) -> int:
    """Decompress: round(y · q / 2^d)."""
    return (y * q + (1 << (d - 1))) >> d


def _compress_poly(f: list[int], d: int) -> list[int]:
    return [_compress(c, d) for c in f]


def _decompress_poly(f: list[int], d: int) -> list[int]:
    return [_decompress(c, d) for c in f]


# ─── Sampling (FIPS 203 Algorithms 7–8) ──────────────────────────────────────

def _sample_ntt(rho: bytes, j: int, i: int) -> list[int]:
    """Sample a uniform polynomial in NTT domain.  Algorithm 7.
    Note: input is ρ ∥ j ∥ i (j first, then i), as per FIPS 203 §4.2.2.
    """
    xof = _XOF(rho, j, i)
    a_hat: list[int] = []
    buf = b""
    buf_pos = 0
    while len(a_hat) < 256:
        if buf_pos + 3 > len(buf):
            buf = xof.digest(buf_pos + 96)   # read ahead in chunks
        b = buf[buf_pos:buf_pos + 3]
        buf_pos += 3
        d1 = b[0] + 256 * (b[1] & 0x0F)
        d2 = (b[1] >> 4) + 16 * b[2]
        if d1 < Q_KEM:
            a_hat.append(d1)
        if d2 < Q_KEM and len(a_hat) < 256:
            a_hat.append(d2)
    return a_hat


def _sample_poly_cbd(B: bytes, eta: int) -> list[int]:
    """Sample from centered binomial distribution CBD_η.  Algorithm 8."""
    bits = []
    for byte in B:
        for k in range(8):
            bits.append((byte >> k) & 1)
    f = []
    for i in range(256):
        x = sum(bits[2 * i * eta + j] for j in range(eta))
        y = sum(bits[(2 * i + 1) * eta + j] for j in range(eta))
        f.append((x - y) % Q_KEM)
    return f


# ─── Matrix / vector helpers ─────────────────────────────────────────────────

def _expand_A(rho: bytes, k: int) -> list[list[list[int]]]:
    """Build k×k matrix Â in NTT domain: Â[i][j] = SampleNTT(ρ, j, i)."""
    return [[_sample_ntt(rho, j, i) for j in range(k)] for i in range(k)]


def _mat_vec_mul_ntt(A_hat: list[list[list[int]]], s_hat: list[list[int]]) -> list[list[int]]:
    """Matrix × vector product in NTT domain."""
    k = len(A_hat)
    result = []
    for i in range(k):
        acc = [0] * 256
        for j in range(k):
            prod = multiply_ntts_kem(A_hat[i][j], s_hat[j])
            acc = [(acc[c] + prod[c]) % Q_KEM for c in range(256)]
        result.append(acc)
    return result


def _mat_T_vec_mul_ntt(A_hat: list[list[list[int]]], r_hat: list[list[int]]) -> list[list[int]]:
    """Transpose(A) × vector in NTT domain: result[i] = Σ_j A[j][i] * r[j]."""
    k = len(A_hat)
    result = []
    for i in range(k):
        acc = [0] * 256
        for j in range(k):
            prod = multiply_ntts_kem(A_hat[j][i], r_hat[j])
            acc = [(acc[c] + prod[c]) % Q_KEM for c in range(256)]
        result.append(acc)
    return result


def _inner_product_ntt(a_hat: list[list[int]], b_hat: list[list[int]]) -> list[int]:
    """Inner product of two NTT-domain vectors: returns a single NTT polynomial."""
    k = len(a_hat)
    acc = [0] * 256
    for i in range(k):
        prod = multiply_ntts_kem(a_hat[i], b_hat[i])
        acc = [(acc[c] + prod[c]) % Q_KEM for c in range(256)]
    return acc


def _add_polys(a: list[int], b: list[int], q: int = Q_KEM) -> list[int]:
    return [(a[i] + b[i]) % q for i in range(256)]


def _sub_polys(a: list[int], b: list[int], q: int = Q_KEM) -> list[int]:
    return [(a[i] - b[i]) % q for i in range(256)]


# ─── K-PKE (internal PKE) ─────────────────────────────────────────────────────

def _kpke_keygen(d: bytes, k: int, eta1: int, tracer: Optional[Tracer] = None
                 ) -> tuple[bytes, bytes]:
    """K-PKE.KeyGen.  FIPS 203 Algorithm 13."""
    rho, sigma = _G(d + bytes([k]))          # Algorithm 13 step 1
    A_hat = _expand_A(rho, k)

    s: list[list[int]] = []
    e: list[list[int]] = []
    for i in range(k):
        prf_bytes = _PRF(sigma, i, 64 * eta1)
        s.append(_sample_poly_cbd(prf_bytes, eta1))
    for i in range(k):
        prf_bytes = _PRF(sigma, k + i, 64 * eta1)
        e.append(_sample_poly_cbd(prf_bytes, eta1))

    s_hat = [ntt_kem(s[i], tracer) for i in range(k)]
    e_hat = [ntt_kem(e[i], tracer) for i in range(k)]

    t_hat = _mat_vec_mul_ntt(A_hat, s_hat)
    t_hat = [_add_polys(t_hat[i], e_hat[i]) for i in range(k)]

    # ek_PKE = ByteEncode12(t̂) ∥ ρ
    ek = b"".join(_byte_encode(t_hat[i], 12) for i in range(k)) + rho
    # dk_PKE = ByteEncode12(ŝ)
    dk = b"".join(_byte_encode(s_hat[i], 12) for i in range(k))
    return ek, dk


def _kpke_encrypt(ek: bytes, m: bytes, r: bytes,
                  k: int, eta1: int, eta2: int, du: int, dv: int,
                  tracer: Optional[Tracer] = None) -> bytes:
    """K-PKE.Encrypt.  FIPS 203 Algorithm 14."""
    # Decode ek
    t_hat = [_byte_decode(ek[i * 384:(i + 1) * 384], 12) for i in range(k)]
    rho = ek[k * 384: k * 384 + 32]
    A_hat = _expand_A(rho, k)

    r_vec: list[list[int]] = []
    e1: list[list[int]] = []
    for i in range(k):
        r_vec.append(_sample_poly_cbd(_PRF(r, i, 64 * eta1), eta1))
    for i in range(k):
        e1.append(_sample_poly_cbd(_PRF(r, k + i, 64 * eta2), eta2))
    e2 = _sample_poly_cbd(_PRF(r, 2 * k, 64 * eta2), eta2)

    r_hat = [ntt_kem(r_vec[i], tracer) for i in range(k)]

    # u = NTT⁻¹(Aᵀ ∘ r̂) + e1
    u_ntt = _mat_T_vec_mul_ntt(A_hat, r_hat)
    u = [_add_polys(intt_kem(u_ntt[i], tracer), e1[i]) for i in range(k)]

    # μ = Decompress_1(ByteDecode_1(m))
    mu_bits = _byte_decode(m, 1)
    mu = _decompress_poly(mu_bits, 1)

    # v = NTT⁻¹(t̂ · r̂) + e2 + μ
    v = intt_kem(_inner_product_ntt(t_hat, r_hat), tracer)
    v = _add_polys(_add_polys(v, e2), mu)

    c1 = b"".join(_byte_encode(_compress_poly(u[i], du), du) for i in range(k))
    c2 = _byte_encode(_compress_poly(v, dv), dv)
    return c1 + c2


def _kpke_decrypt(dk: bytes, c: bytes,
                  k: int, du: int, dv: int,
                  tracer: Optional[Tracer] = None) -> bytes:
    """K-PKE.Decrypt.  FIPS 203 Algorithm 15."""
    c1_len = k * 256 * du // 8
    c1, c2 = c[:c1_len], c[c1_len:]

    u = [_decompress_poly(_byte_decode(c1[i * 256 * du // 8:(i + 1) * 256 * du // 8], du), du)
         for i in range(k)]
    v = _decompress_poly(_byte_decode(c2, dv), dv)

    s_hat = [_byte_decode(dk[i * 384:(i + 1) * 384], 12) for i in range(k)]

    # w = v − NTT⁻¹(ŝ · NTT(u))
    u_hat = [ntt_kem(u[i], tracer) for i in range(k)]
    sv = intt_kem(_inner_product_ntt(s_hat, u_hat), tracer)
    w = _sub_polys(v, sv)

    return _byte_encode(_compress_poly(w, 1), 1)


# ─── ML-KEM (public API) ──────────────────────────────────────────────────────

class MLKEM:
    """ML-KEM key encapsulation mechanism.  FIPS 203.

    Usage:
        kem = MLKEM("ML_KEM_512")
        ek, dk = kem.keygen()             # random key generation
        K, c   = kem.encaps(ek)           # encapsulate
        K2     = kem.decaps(dk, c)        # decapsulate; K == K2

    For instrumented simulation:
        ek, dk = kem.keygen_internal(d, z)
        K, c   = kem.encaps_internal(ek, m, tracer=t)
        K2     = kem.decaps_internal(dk, c, tracer=t)
    """

    def __init__(self, variant: str = "ML_KEM_512") -> None:
        p = _params(variant)
        self.variant = variant
        self.k    = p["k"]
        self.eta1 = p["eta1"]
        self.eta2 = p["eta2"]
        self.du   = p["du"]
        self.dv   = p["dv"]
        self.ek_size = p["ek_size"]
        self.dk_size = p["dk_size"]
        self.ct_size = p["ct_size"]

    # ── Internal (deterministic) ──────────────────────────────────────────────

    def keygen_internal(self, d: bytes, z: bytes,
                        tracer: Optional[Tracer] = None) -> tuple[bytes, bytes]:
        """ML-KEM.KeyGen_internal(d, z).  FIPS 203 Algorithm 15."""
        assert len(d) == 32 and len(z) == 32
        ek_pke, dk_pke = _kpke_keygen(d, self.k, self.eta1, tracer)
        ek = ek_pke
        dk = dk_pke + ek + _H(ek) + z
        return ek, dk

    def encaps_internal(self, ek: bytes, m: bytes,
                        tracer: Optional[Tracer] = None) -> tuple[bytes, bytes]:
        """ML-KEM.Encaps_internal(ek, m).  FIPS 203 Algorithm 17."""
        assert len(m) == 32
        h_ek = _H(ek)
        K, r = _G(m + h_ek)
        c = _kpke_encrypt(ek, m, r, self.k, self.eta1, self.eta2,
                          self.du, self.dv, tracer)
        return K, c

    def decaps_internal(self, dk: bytes, c: bytes,
                        tracer: Optional[Tracer] = None) -> bytes:
        """ML-KEM.Decaps_internal(dk, c).  FIPS 203 Algorithm 18 (implicit rejection)."""
        dk_pke_len = self.k * 384
        dk_pke = dk[:dk_pke_len]
        ek      = dk[dk_pke_len: dk_pke_len + self.ek_size]
        h       = dk[dk_pke_len + self.ek_size: dk_pke_len + self.ek_size + 32]
        z       = dk[dk_pke_len + self.ek_size + 32: dk_pke_len + self.ek_size + 64]

        m_prime = _kpke_decrypt(dk_pke, c, self.k, self.du, self.dv, tracer)
        K_prime, r_prime = _G(m_prime + h)
        K_bar   = _J(z + c)

        c_prime = _kpke_encrypt(ek, m_prime, r_prime, self.k, self.eta1, self.eta2,
                                self.du, self.dv, tracer)

        # Constant-time select (for Python simulation purposes, branching is fine)
        return K_prime if c == c_prime else K_bar

    # ── Randomized wrappers ───────────────────────────────────────────────────

    def keygen(self, tracer: Optional[Tracer] = None) -> tuple[bytes, bytes]:
        """Randomized key generation."""
        d = os.urandom(32)
        z = os.urandom(32)
        return self.keygen_internal(d, z, tracer)

    def encaps(self, ek: bytes, tracer: Optional[Tracer] = None) -> tuple[bytes, bytes]:
        """Randomized encapsulation."""
        m = os.urandom(32)
        return self.encaps_internal(ek, m, tracer)

    def decaps(self, dk: bytes, c: bytes,
               tracer: Optional[Tracer] = None) -> bytes:
        """Decapsulation (deterministic given dk, c)."""
        return self.decaps_internal(dk, c, tracer)
