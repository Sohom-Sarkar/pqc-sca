"""
ML-DSA (CRYSTALS-Dilithium) — NIST FIPS 204 (2024).

Implements ML-DSA-44 with full instrumentation via Tracer.
Algorithm numbering follows FIPS 204.
"""

from __future__ import annotations
import hashlib
import os
from typing import Optional

from .ntt import (
    Tracer, ntt_dsa, intt_dsa, multiply_ntts_dsa, Q_DSA
)
from config import MLDSA_PARAMS

# ─── Parameter helper ────────────────────────────────────────────────────────

def _params(variant: str) -> dict:
    return MLDSA_PARAMS[variant]


# ─── Hash / XOF wrappers (FIPS 204 §4.1) ────────────────────────────────────

def _H(data: bytes, length: int) -> bytes:
    """H = SHAKE-256 with `length` byte output."""
    return hashlib.shake_256(data).digest(length)


def _G(data: bytes, length: int) -> bytes:
    """G = SHAKE-128 with `length` byte output."""
    return hashlib.shake_128(data).digest(length)


# ─── Polynomial arithmetic helpers ───────────────────────────────────────────

def _add_polys(a: list[int], b: list[int]) -> list[int]:
    return [(a[i] + b[i]) % Q_DSA for i in range(256)]


def _sub_polys(a: list[int], b: list[int]) -> list[int]:
    return [(a[i] - b[i]) % Q_DSA for i in range(256)]


def _scale_poly(c: list[int], z: int) -> list[int]:
    return [(z * x) % Q_DSA for x in c]


def _mat_vec_mul_ntt_dsa(A_hat: list[list[list[int]]], s_hat: list[list[int]]) -> list[list[int]]:
    k = len(A_hat)
    l = len(s_hat)
    result = []
    for i in range(k):
        acc = [0] * 256
        for j in range(l):
            prod = multiply_ntts_dsa(A_hat[i][j], s_hat[j])
            acc = [(acc[c] + prod[c]) % Q_DSA for c in range(256)]
        result.append(acc)
    return result


def _poly_inf_norm(f: list[int]) -> int:
    """∞-norm: max absolute value of centered coefficients."""
    half_q = Q_DSA // 2
    return max(abs(c if c <= half_q else c - Q_DSA) for c in f)


def _vec_inf_norm(v: list[list[int]]) -> int:
    return max(_poly_inf_norm(p) for p in v)


def _coeff_from_three_bytes(b: bytes) -> int:
    """Decode 3 bytes to a coefficient < q."""
    assert len(b) == 3
    z = b[0] | (b[1] << 8) | (b[2] << 16)
    z &= 0x7FFFFF
    return z


def _centered(x: int, q: int = Q_DSA) -> int:
    """Reduce x to the centered representative in (−q/2, q/2]."""
    x %= q
    if x > q // 2:
        x -= q
    return x


# ─── Rounding / decomposition (FIPS 204 §6.1) ────────────────────────────────

def _power2round(r: int, d: int = 13) -> tuple[int, int]:
    """Power2Round_d(r): split r = r1·2^d + r0."""
    r = r % Q_DSA
    r0 = r % (1 << d)
    if r0 > (1 << (d - 1)):
        r0 -= (1 << d)
    r1 = (r - r0) >> d
    return r1, r0


def _decompose(r: int, gamma2: int) -> tuple[int, int]:
    """Decompose(r): r = r1·(2·γ2) + r0,  r0 ∈ (−γ2, γ2].  FIPS 204 Alg 35."""
    r = r % Q_DSA
    r0 = r % (2 * gamma2)
    if r0 > gamma2:
        r0 -= 2 * gamma2
    r1 = (r - r0) // (2 * gamma2)
    # When r1 = (q-1)/(2*gamma2) the top of the range wraps: r1←0, r0←r0-1
    if r1 == (Q_DSA - 1) // (2 * gamma2):
        r1 = 0
        r0 = r0 - 1
    return r1, r0


def _high_bits(r: int, gamma2: int) -> int:
    r1, _ = _decompose(r, gamma2)
    return r1


def _low_bits(r: int, gamma2: int) -> int:
    _, r0 = _decompose(r, gamma2)
    return r0


def _make_hint(z: int, r: int, gamma2: int) -> int:
    """MakeHint(z, r) = 1 if HighBits(r) ≠ HighBits(r+z)."""
    r1 = _high_bits(r, gamma2)
    v1 = _high_bits(r + z, gamma2)
    return int(r1 != v1)


def _use_hint(h: int, r: int, gamma2: int) -> int:
    """UseHint(h, r): recover r1 from hint and r+z."""
    m = (Q_DSA - 1) // (2 * gamma2)
    r1, r0 = _decompose(r, gamma2)
    if h == 1 and r0 > 0:
        return (r1 + 1) % m
    if h == 1 and r0 <= 0:
        return (r1 - 1) % m
    return r1


def _power2round_poly(f: list[int], d: int = 13) -> tuple[list[int], list[int]]:
    r1s, r0s = [], []
    for c in f:
        r1, r0 = _power2round(c, d)
        r1s.append(r1); r0s.append(r0)
    return r1s, r0s


def _high_bits_poly(f: list[int], gamma2: int) -> list[int]:
    return [_high_bits(c, gamma2) for c in f]


def _low_bits_poly(f: list[int], gamma2: int) -> list[int]:
    return [_low_bits(c, gamma2) for c in f]


def _make_hint_poly(z: list[int], r: list[int], gamma2: int) -> list[int]:
    return [_make_hint(z[i], r[i], gamma2) for i in range(256)]


def _use_hint_poly(h: list[int], r: list[int], gamma2: int) -> list[int]:
    return [_use_hint(h[i], r[i], gamma2) for i in range(256)]


def _hint_count(h_vec: list[list[int]]) -> int:
    return sum(sum(p) for p in h_vec)


# ─── Sampling functions ───────────────────────────────────────────────────────

def _expand_A(rho: bytes, k: int, l: int) -> list[list[list[int]]]:
    """ExpandA: k×l matrix Â in NTT domain.  FIPS 204 Algorithm 32."""
    A_hat = []
    for i in range(k):
        row = []
        for j in range(l):
            seed = rho + bytes([j, i])   # j first, then i per FIPS 204 §6.3
            row.append(_rej_ntt_poly(seed))
        A_hat.append(row)
    return A_hat


def _rej_ntt_poly(seed: bytes) -> list[int]:
    """RejNTTPoly: rejection-sample a uniform polynomial in NTT domain.  Algorithm 24."""
    xof = hashlib.shake_128(seed)
    a: list[int] = []
    buf_len = 840
    buf = xof.digest(buf_len)
    pos = 0
    while len(a) < 256:
        if pos + 3 > len(buf):
            buf_len += 168
            buf = xof.digest(buf_len)
        c = _coeff_from_three_bytes(buf[pos:pos + 3])
        pos += 3
        if c < Q_DSA:
            a.append(c)
    return a


def _rej_bounded_poly(rho_prime: bytes, nonce: int, eta: int) -> list[int]:
    """RejBoundedPoly: sample polynomial with coefficients in [−η, η].  Algorithm 25."""
    seed = rho_prime + bytes([nonce & 0xFF, (nonce >> 8) & 0xFF])
    xof = hashlib.shake_256(seed)
    f: list[int] = []
    buf_len = 136
    buf = xof.digest(buf_len)
    pos = 0
    while len(f) < 256:
        if pos >= len(buf):
            buf_len += 136
            buf = xof.digest(buf_len)
        b = buf[pos]; pos += 1
        z0 = b & 0x0F
        z1 = b >> 4
        if eta == 2:
            if z0 < 15:
                c0 = z0 - (205 * z0 >> 10) * 5
                if c0 < 5:
                    f.append((Q_DSA + 2 - c0) % Q_DSA)
            if len(f) < 256 and z1 < 15:
                c1 = z1 - (205 * z1 >> 10) * 5
                if c1 < 5:
                    f.append((Q_DSA + 2 - c1) % Q_DSA)
        elif eta == 4:
            if z0 < 9:
                f.append((Q_DSA + 4 - z0) % Q_DSA)
            if len(f) < 256 and z1 < 9:
                f.append((Q_DSA + 4 - z1) % Q_DSA)
    return f


def _expand_s(rho_prime: bytes, eta: int, l: int, k: int
              ) -> tuple[list[list[int]], list[list[int]]]:
    """ExpandS: sample s1 ∈ S_η^l, s2 ∈ S_η^k.  Algorithm 26."""
    s1 = [_rej_bounded_poly(rho_prime, i,       eta) for i in range(l)]
    s2 = [_rej_bounded_poly(rho_prime, l + i,   eta) for i in range(k)]
    return s1, s2


def _expand_mask(rho_prime: bytes, kappa: int, l: int, gamma1: int) -> list[list[int]]:
    """ExpandMask: sample y ∈ S_{γ1-1}^l.  Algorithm 27."""
    y = []
    for i in range(l):
        nonce = kappa + i
        seed = rho_prime + bytes([nonce & 0xFF, (nonce >> 8) & 0xFF])
        y.append(_sample_mask_poly(seed, gamma1))
    return y


def _sample_mask_poly(seed: bytes, gamma1: int) -> list[int]:
    """Sample a polynomial with coefficients in (−γ1, γ1]."""
    if gamma1 == 1 << 17:
        bits_per, n_bytes, mask = 18, 576, 0x3FFFF
    elif gamma1 == 1 << 19:
        bits_per, n_bytes, mask = 20, 640, 0xFFFFF
    else:
        raise ValueError(f"Unsupported gamma1={gamma1}")
    buf = hashlib.shake_256(seed).digest(n_bytes)
    f = []
    for i in range(256):
        bit_offset  = i * bits_per
        byte_offset = bit_offset >> 3
        bit_shift   = bit_offset & 7
        raw = (buf[byte_offset]
               | (buf[byte_offset + 1] << 8)
               | (buf[byte_offset + 2] << 16))
        val = (raw >> bit_shift) & mask
        f.append((gamma1 - val) % Q_DSA)
    return f


def _sample_in_ball(rho: bytes, tau: int) -> list[int]:
    """SampleInBall: sample polynomial with τ nonzero ±1 coefficients.  Algorithm 29."""
    xof = hashlib.shake_256(rho)
    buf = xof.digest(8 + 256)
    signs = int.from_bytes(buf[:8], 'little')
    pos = 8
    c = [0] * 256
    for i in range(256 - tau, 256):
        # rejection sample j ∈ [0, i]
        while True:
            if pos >= len(buf):
                buf = xof.digest(len(buf) + 64)
            j = buf[pos]; pos += 1
            if j <= i:
                break
        c[i] = c[j]
        c[j] = 1 - 2 * (signs & 1)
        signs >>= 1
    return c


# ─── Encoding / decoding ──────────────────────────────────────────────────────

def _polyeta_pack(f: list[int], eta: int) -> bytes:
    """Pack polynomial with |coeff| ≤ η into bytes (3 bits per coeff for η=2)."""
    if eta == 2:
        # 8 coefficients × 3 bits each = 24 bits = 3 bytes
        out = bytearray(96)
        for i in range(256 // 8):
            t = [(eta - _centered(f[8 * i + j])) for j in range(8)]
            out[3 * i]     =  t[0]       | (t[1] << 3) | ((t[2] & 0x3) << 6)
            out[3 * i + 1] = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | ((t[5] & 0x1) << 7)
            out[3 * i + 2] = (t[5] >> 1) | (t[6] << 2) |  (t[7] << 5)
        return bytes(out)
    elif eta == 4:
        out = bytearray(128)  # 256 * 4 / 8
        for i in range(256 // 2):
            t0 = eta - _centered(f[2 * i])
            t1 = eta - _centered(f[2 * i + 1])
            out[i] = t0 | (t1 << 4)
        return bytes(out)
    raise ValueError(f"Unsupported eta={eta}")


def _polyeta_unpack(buf: bytes, eta: int) -> list[int]:
    if eta == 2:
        f = []
        for i in range(96 // 3):  # 32 groups of 3 bytes
            b = buf[3 * i: 3 * i + 3]
            t = [0] * 8
            t[0] =  b[0] & 7
            t[1] = (b[0] >> 3) & 7
            t[2] = (b[0] >> 6) | ((b[1] & 1) << 2)
            t[3] = (b[1] >> 1) & 7
            t[4] = (b[1] >> 4) & 7
            t[5] = (b[1] >> 7) | ((b[2] & 3) << 1)
            t[6] = (b[2] >> 2) & 7
            t[7] = (b[2] >> 5)
            f.extend([(eta - v) % Q_DSA for v in t])
        return f
    elif eta == 4:
        f = []
        for byte in buf[:128]:
            f.append((eta - (byte & 0x0F)) % Q_DSA)
            f.append((eta - (byte >> 4))   % Q_DSA)
        return f
    raise ValueError(f"Unsupported eta={eta}")


def _polyt1_pack(f: list[int]) -> bytes:
    """Pack t1 coefficients (10 bits each, fits in range 0..1023)."""
    # 256 * 10 / 8 = 320 bytes
    out = bytearray(320)
    for i in range(256 // 4):
        t = f[4 * i: 4 * i + 4]
        out[5 * i]     =  t[0] & 0xFF
        out[5 * i + 1] = (t[0] >> 8) | ((t[1] & 0x3F) << 2)
        out[5 * i + 2] = (t[1] >> 6) | ((t[2] & 0x0F) << 4)
        out[5 * i + 3] = (t[2] >> 4) | ((t[3] & 0x03) << 6)
        out[5 * i + 4] =  t[3] >> 2
    return bytes(out)


def _polyt1_unpack(buf: bytes) -> list[int]:
    f = []
    for i in range(256 // 4):
        b = buf[5 * i: 5 * i + 5]
        f.append(((b[0]       ) | (b[1] << 8)) & 0x3FF)
        f.append(((b[1] >> 2  ) | (b[2] << 6)) & 0x3FF)
        f.append(((b[2] >> 4  ) | (b[3] << 4)) & 0x3FF)
        f.append(((b[3] >> 6  ) | (b[4] << 2)) & 0x3FF)
    return f


def _polyt0_pack(f: list[int], d: int = 13) -> bytes:
    """Pack t0 coefficients (signed, d bits each; d=13 → 416 bytes for 256 coeffs)."""
    shift = 1 << (d - 1)
    out = bytearray(256 * d // 8)
    for i in range(256 // 8):
        t = [(shift - _centered(f[8 * i + j])) for j in range(8)]
        acc = 0
        for j, v in enumerate(t):
            acc |= v << (d * j)
        out[d * i: d * i + d] = acc.to_bytes(d, 'little')
    return bytes(out)


def _polyt0_unpack(buf: bytes, d: int = 13) -> list[int]:
    shift = 1 << (d - 1)
    mask  = (1 << d) - 1
    f = []
    for i in range(256 // 8):
        acc = int.from_bytes(buf[13 * i: 13 * i + 13], 'little')
        for j in range(8):
            val = (acc >> (13 * j)) & mask
            f.append((Q_DSA + shift - val) % Q_DSA)
    return f


def _polyz_pack(f: list[int], gamma1: int) -> bytes:
    """Pack z polynomial (|z| < γ1); γ1=2^17 → 18 bits/coeff → 576 bytes."""
    if gamma1 == 1 << 17:
        out = bytearray(576)
        for i in range(256):
            val = gamma1 - _centered(f[i])
            bit_offset = i * 18
            byte_offset = bit_offset >> 3
            bit_shift   = bit_offset & 7
            # Write 18 bits at bit_shift
            mask3 = val << bit_shift
            out[byte_offset]     = (out[byte_offset]     | (mask3 & 0xFF)) & 0xFF
            out[byte_offset + 1] = (out[byte_offset + 1] | ((mask3 >> 8)  & 0xFF)) & 0xFF
            out[byte_offset + 2] = (out[byte_offset + 2] | ((mask3 >> 16) & 0xFF)) & 0xFF
        return bytes(out)
    elif gamma1 == 1 << 19:
        out = bytearray(640)
        for i in range(256):
            val = gamma1 - _centered(f[i])
            bit_offset = i * 20
            byte_offset = bit_offset >> 3
            bit_shift   = bit_offset & 7
            mask3 = val << bit_shift
            out[byte_offset]     = (out[byte_offset]     | (mask3 & 0xFF)) & 0xFF
            out[byte_offset + 1] = (out[byte_offset + 1] | ((mask3 >> 8)  & 0xFF)) & 0xFF
            out[byte_offset + 2] = (out[byte_offset + 2] | ((mask3 >> 16) & 0xFF)) & 0xFF
        return bytes(out)
    raise ValueError(f"Unsupported gamma1={gamma1}")


def _polyz_unpack(buf: bytes, gamma1: int) -> list[int]:
    f = []
    if gamma1 == 1 << 17:
        for i in range(256):
            bit_offset = i * 18
            byte_offset = bit_offset >> 3
            bit_shift   = bit_offset & 7
            raw = (buf[byte_offset]
                   | (buf[byte_offset + 1] << 8)
                   | (buf[byte_offset + 2] << 16))
            val = (raw >> bit_shift) & 0x3FFFF
            f.append((gamma1 - val) % Q_DSA)
    elif gamma1 == 1 << 19:
        for i in range(256):
            bit_offset = i * 20
            byte_offset = bit_offset >> 3
            bit_shift   = bit_offset & 7
            raw = (buf[byte_offset]
                   | (buf[byte_offset + 1] << 8)
                   | (buf[byte_offset + 2] << 16))
            val = (raw >> bit_shift) & 0xFFFFF
            f.append((gamma1 - val) % Q_DSA)
    return f


def _encode_w1(w1_vec: list[list[int]], gamma2: int) -> bytes:
    """Encode w1 for challenge hash; coefficient width depends on γ2."""
    m = (Q_DSA - 1) // (2 * gamma2)
    bits_per = m.bit_length()
    out = bytearray()
    for w1 in w1_vec:
        poly_bytes = bytearray(256 * bits_per // 8)
        for i, v in enumerate(w1):
            bit_offset = i * bits_per
            byte_offset = bit_offset >> 3
            bit_shift   = bit_offset & 7
            poly_bytes[byte_offset] |= (v << bit_shift) & 0xFF
            if bit_shift + bits_per > 8:
                poly_bytes[byte_offset + 1] |= (v >> (8 - bit_shift)) & 0xFF
        out.extend(poly_bytes)
    return bytes(out)


def _hint_pack(h_vec: list[list[int]], omega: int, k: int) -> bytes:
    """Pack hint vector into omega + k bytes."""
    buf = bytearray(omega + k)
    idx = 0
    for i, h_poly in enumerate(h_vec):
        for j, bit in enumerate(h_poly):
            if bit:
                buf[idx] = j; idx += 1
        buf[omega + i] = idx
    return bytes(buf)


def _hint_unpack(buf: bytes, omega: int, k: int) -> list[list[int]]:
    h_vec = [[0] * 256 for _ in range(k)]
    idx = 0
    for i in range(k):
        end = buf[omega + i]
        if end < idx or end > omega:
            return None   # malformed
        for j in range(idx, end):
            if j > idx and buf[j] <= buf[j - 1]:
                return None  # non-monotone indices
            h_vec[i][buf[j]] = 1
        idx = end
    return h_vec


# ─── Key encoding / decoding ─────────────────────────────────────────────────

def _pk_encode(rho: bytes, t1_vec: list[list[int]]) -> bytes:
    return rho + b"".join(_polyt1_pack(p) for p in t1_vec)


def _pk_decode(pk: bytes, k: int) -> tuple[bytes, list[list[int]]]:
    rho = pk[:32]
    t1_vec = [_polyt1_unpack(pk[32 + i * 320: 32 + (i + 1) * 320]) for i in range(k)]
    return rho, t1_vec


def _sk_encode(rho: bytes, K: bytes, tr: bytes,
               s1: list[list[int]], s2: list[list[int]],
               t0: list[list[int]], eta: int, k: int, l: int) -> bytes:
    return (rho + K + tr
            + b"".join(_polyeta_pack(p, eta) for p in s1)
            + b"".join(_polyeta_pack(p, eta) for p in s2)
            + b"".join(_polyt0_pack(p) for p in t0))


def _sk_decode(sk: bytes, eta: int, k: int, l: int):
    pos = 0
    rho = sk[pos:pos + 32]; pos += 32
    K   = sk[pos:pos + 32]; pos += 32
    tr  = sk[pos:pos + 64]; pos += 64
    eta_bytes = 96 if eta == 2 else 128
    s1 = [_polyeta_unpack(sk[pos + i * eta_bytes: pos + (i + 1) * eta_bytes], eta) for i in range(l)]
    pos += l * eta_bytes
    s2 = [_polyeta_unpack(sk[pos + i * eta_bytes: pos + (i + 1) * eta_bytes], eta) for i in range(k)]
    pos += k * eta_bytes
    t0 = [_polyt0_unpack(sk[pos + i * 416: pos + (i + 1) * 416]) for i in range(k)]
    return rho, K, tr, s1, s2, t0


# ─── ML-DSA (public API) ──────────────────────────────────────────────────────

class MLDSA:
    """ML-DSA digital signature scheme.  FIPS 204.

    Usage:
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen()
        sig    = dsa.sign(sk, message)
        valid  = dsa.verify(pk, message, sig)

    For instrumented simulation:
        pk, sk = dsa.keygen_internal(xi, tracer=t)
        sig    = dsa.sign_internal(sk, M_prime, rnd=bytes(32), tracer=t)
    """

    def __init__(self, variant: str = "ML_DSA_44") -> None:
        p = _params(variant)
        self.variant = variant
        self.n       = p["n"]
        self.k       = p["k"]
        self.l       = p["l"]
        self.q       = p["q"]
        self.d       = p["d"]
        self.lambda_ = p["lambda_"]
        self.gamma1  = p["gamma1"]
        self.gamma2  = p["gamma2"]
        self.tau     = p["tau"]
        self.beta    = p["beta"]
        self.omega   = p["omega"]
        self.eta     = p["eta"]
        self.c_tilde_len = self.lambda_ // 4   # bytes

    # ── Internal ──────────────────────────────────────────────────────────────

    def keygen_internal(self, xi: bytes, tracer: Optional[Tracer] = None
                        ) -> tuple[bytes, bytes]:
        """ML-DSA.KeyGen_internal(ξ).  FIPS 204 Algorithm 6."""
        assert len(xi) == 32
        seed = _H(xi + bytes([self.k, self.l]), 128)
        rho, rho_prime, K = seed[:32], seed[32:96], seed[96:128]

        A_hat = _expand_A(rho, self.k, self.l)
        s1, s2 = _expand_s(rho_prime, self.eta, self.l, self.k)

        s1_hat = [ntt_dsa(s, tracer) for s in s1]
        As1_hat = _mat_vec_mul_ntt_dsa(A_hat, s1_hat)
        t = [_add_polys(intt_dsa(As1_hat[i], tracer), s2[i]) for i in range(self.k)]

        t1_vec = []; t0_vec = []
        for poly in t:
            r1s, r0s = _power2round_poly(poly, self.d)
            t1_vec.append(r1s); t0_vec.append(r0s)

        pk = _pk_encode(rho, t1_vec)
        tr = _H(pk, 64)
        sk = _sk_encode(rho, K, tr, s1, s2, t0_vec, self.eta, self.k, self.l)
        return pk, sk

    def sign_internal(self, sk: bytes, M_prime: bytes, rnd: bytes,
                      tracer: Optional[Tracer] = None) -> bytes:
        """ML-DSA.SignInternal(sk, M', rnd).  FIPS 204 Algorithm 7 (deterministic if rnd=0^32)."""
        rho, K, tr, s1, s2, t0 = _sk_decode(sk, self.eta, self.k, self.l)
        A_hat = _expand_A(rho, self.k, self.l)

        mu      = _H(tr + M_prime, 64)
        rho_pp  = _H(K + rnd + mu, 64)

        s1_hat = [ntt_dsa(s, tracer) for s in s1]
        s2_hat = [ntt_dsa(s, tracer) for s in s2]
        t0_hat = [ntt_dsa(t, tracer) for t in t0]

        kappa = 0
        while True:
            y     = _expand_mask(rho_pp, kappa, self.l, self.gamma1)
            y_hat = [ntt_dsa(p, tracer) for p in y]
            Ay_hat = _mat_vec_mul_ntt_dsa(A_hat, y_hat)
            w = [intt_dsa(p, tracer) for p in Ay_hat]

            w1 = [_high_bits_poly(w[i], self.gamma2) for i in range(self.k)]
            c_tilde = _H(mu + _encode_w1(w1, self.gamma2), self.c_tilde_len)
            c = _sample_in_ball(c_tilde, self.tau)
            c_hat = ntt_dsa(c, tracer)

            cs1 = [intt_dsa(multiply_ntts_dsa(c_hat, s1_hat[j]), tracer) for j in range(self.l)]
            cs2 = [intt_dsa(multiply_ntts_dsa(c_hat, s2_hat[i]), tracer) for i in range(self.k)]
            ct0 = [intt_dsa(multiply_ntts_dsa(c_hat, t0_hat[i]), tracer) for i in range(self.k)]

            z = [_add_polys(y[j], cs1[j]) for j in range(self.l)]
            r = [_sub_polys(w[i], cs2[i]) for i in range(self.k)]

            if (_vec_inf_norm(z) >= self.gamma1 - self.beta
                    or _vec_inf_norm([_low_bits_poly(r[i], self.gamma2) for i in range(self.k)]) >= self.gamma2 - self.beta):
                kappa += self.l; continue

            r_ct0 = [_add_polys(r[i], ct0[i]) for i in range(self.k)]
            h = [_make_hint_poly(
                    [(-ct0[i][j]) % Q_DSA for j in range(256)],
                    r_ct0[i], self.gamma2)
                 for i in range(self.k)]

            if (_vec_inf_norm(ct0) >= self.gamma2
                    or _hint_count(h) > self.omega):
                kappa += self.l; continue

            # Encode signature: c̃ ∥ z ∥ h
            z_mod = [[(v + Q_DSA // 2) % Q_DSA - Q_DSA // 2 for v in p] for p in z]
            sig = (c_tilde
                   + b"".join(_polyz_pack(z_mod[j], self.gamma1) for j in range(self.l))
                   + _hint_pack(h, self.omega, self.k))
            return sig

    def verify_internal(self, pk: bytes, M_prime: bytes, sig: bytes,
                        tracer: Optional[Tracer] = None) -> bool:
        """ML-DSA.VerifyInternal.  FIPS 204 Algorithm 3 (internal)."""
        rho, t1_vec = _pk_decode(pk, self.k)
        A_hat = _expand_A(rho, self.k, self.l)

        # Decode signature
        c_tilde_len = self.c_tilde_len
        z_len = self.l * (576 if self.gamma1 == 1 << 17 else 640)
        hint_len = self.omega + self.k

        if len(sig) != c_tilde_len + z_len + hint_len:
            return False

        c_tilde = sig[:c_tilde_len]
        z_bytes = sig[c_tilde_len: c_tilde_len + z_len]
        h_bytes = sig[c_tilde_len + z_len:]

        z_poly_len = 576 if self.gamma1 == 1 << 17 else 640
        z = [_polyz_unpack(z_bytes[j * z_poly_len: (j + 1) * z_poly_len], self.gamma1)
             for j in range(self.l)]
        h = _hint_unpack(h_bytes, self.omega, self.k)
        if h is None:
            return False

        if _vec_inf_norm(z) >= self.gamma1 - self.beta:
            return False

        tr = _H(pk, 64)
        mu = _H(tr + M_prime, 64)
        c  = _sample_in_ball(c_tilde, self.tau)

        c_hat   = ntt_dsa(c, tracer)
        z_hat   = [ntt_dsa(p, tracer) for p in z]
        t1_scaled = [[((t1_vec[i][j] << self.d) % Q_DSA) for j in range(256)]
                     for i in range(self.k)]
        t1_hat  = [ntt_dsa(t1_scaled[i], tracer) for i in range(self.k)]
        ct1_hat = [multiply_ntts_dsa(c_hat, t1_hat[i]) for i in range(self.k)]

        Az_hat  = _mat_vec_mul_ntt_dsa(A_hat, z_hat)
        w_prime = [intt_dsa(
                       [( Az_hat[i][j] - ct1_hat[i][j]) % Q_DSA for j in range(256)],
                       tracer)
                   for i in range(self.k)]

        w1_prime = [_use_hint_poly(h[i], w_prime[i], self.gamma2) for i in range(self.k)]
        c_tilde_prime = _H(mu + _encode_w1(w1_prime, self.gamma2), self.c_tilde_len)

        return c_tilde == c_tilde_prime and _hint_count(h) <= self.omega

    # ── Public wrappers ───────────────────────────────────────────────────────

    def keygen(self, tracer: Optional[Tracer] = None) -> tuple[bytes, bytes]:
        """Randomized key generation."""
        return self.keygen_internal(os.urandom(32), tracer)

    def sign(self, sk: bytes, message: bytes, ctx: bytes = b"",
             tracer: Optional[Tracer] = None) -> bytes:
        """Randomized signing (hedged).  FIPS 204 Algorithm 2."""
        if len(ctx) > 255:
            raise ValueError("Context too long")
        M_prime = bytes([0, len(ctx)]) + ctx + message
        rnd = os.urandom(32)
        return self.sign_internal(sk, M_prime, rnd, tracer)

    def sign_deterministic(self, sk: bytes, message: bytes, ctx: bytes = b"",
                           tracer: Optional[Tracer] = None) -> bytes:
        """Deterministic signing (rnd = 0^32).  Used for KAT validation."""
        if len(ctx) > 255:
            raise ValueError("Context too long")
        M_prime = bytes([0, len(ctx)]) + ctx + message
        return self.sign_internal(sk, M_prime, bytes(32), tracer)

    def verify(self, pk: bytes, message: bytes, sig: bytes, ctx: bytes = b"",
               tracer: Optional[Tracer] = None) -> bool:
        """Signature verification.  FIPS 204 Algorithm 3."""
        if len(ctx) > 255:
            return False
        M_prime = bytes([0, len(ctx)]) + ctx + message
        return self.verify_internal(pk, M_prime, sig, tracer)
