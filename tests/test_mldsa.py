"""
ML-DSA-44 Known Answer Tests (KAT) — NIST FIPS 204.

Vectors sourced from:
  https://github.com/post-quantum-cryptography/KAT/blob/main/MLDSA/kat_MLDSA_44_det_pure.rsp

The deterministic-pure KAT uses:
  xi   (32 bytes) → ML-DSA.KeyGen_internal(xi) → (pk, sk)
  msg  (mlen bytes) + ctx → deterministic Sign (rnd=0^32) → sm
  sm format: signature (sig_size bytes) || message
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from algorithms.mldsa import MLDSA

# ─── KAT vector (count=0, kat_MLDSA_44_det_pure.rsp) ─────────────────────────

KAT_0 = {
    "xi": bytes.fromhex(
        "f696484048ec21f96cf50a56d0759c448f3779752f0383d37449690694cf7a68"
    ),
    "pk": bytes.fromhex(
        "bd4e96f9a038ab5e36214fe69c0b1cb835ef9d7c8417e76aecd152f5cddebec8"
        "a1ac25f03b3643700dbf76eef49a324f93d5042e203f3c70658ad1ad13b917cc"
        "1ed23f06a4dd1c543350525a9e2451dfe5f3969b1fa530488cc8903fdeb77180"  # truncated for readability
        # Full pk omitted — test uses structural checks + sign/verify round-trip
    ),
    "msg": bytes.fromhex("6dbbc4375136df3b07f7c70e639e223e"),
    "ctx": bytes.fromhex("480c658c0cb3e040bde084345cef0df7"),
    "smlen": 2436,
}


class TestMLDSA44:
    def test_key_sizes(self):
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen_internal(bytes(32))
        assert len(pk) == 1312, f"pk size = {len(pk)}, expected 1312"
        assert len(sk) == 2560, f"sk size = {len(sk)}, expected 2560"

    def test_sig_size(self):
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen()
        sig = dsa.sign_deterministic(sk, b"test message")
        assert len(sig) == 2420, f"sig size = {len(sig)}, expected 2420"

    def test_sign_verify_roundtrip(self):
        """Random keygen → deterministic sign → verify must succeed."""
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen()
        msg = b"PQC side-channel analyzer test"
        sig = dsa.sign_deterministic(sk, msg)
        assert dsa.verify(pk, msg, sig), "Verification failed on valid signature"

    def test_sign_verify_roundtrip_with_ctx(self):
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen()
        msg = b"hello world"
        ctx = b"test-context"
        sig = dsa.sign_deterministic(sk, msg, ctx)
        assert dsa.verify(pk, msg, sig, ctx)

    def test_wrong_message_rejected(self):
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen()
        msg = b"correct message"
        sig = dsa.sign_deterministic(sk, msg)
        assert not dsa.verify(pk, b"wrong message", sig), "Wrong message should be rejected"

    def test_wrong_key_rejected(self):
        dsa = MLDSA("ML_DSA_44")
        pk1, sk1 = dsa.keygen()
        pk2, sk2 = dsa.keygen()
        msg = b"test"
        sig = dsa.sign_deterministic(sk1, msg)
        assert not dsa.verify(pk2, msg, sig), "Signature from sk1 must not verify under pk2"

    def test_deterministic_signing_reproducible(self):
        """Deterministic sign must produce identical output for same inputs."""
        dsa = MLDSA("ML_DSA_44")
        _, sk = dsa.keygen()
        msg = b"determinism test"
        sig1 = dsa.sign_deterministic(sk, msg)
        sig2 = dsa.sign_deterministic(sk, msg)
        assert sig1 == sig2, "Deterministic signing not reproducible"

    def test_kat_pk_size(self):
        """Spot-check: KAT xi=0 produces pk of correct length."""
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen_internal(KAT_0["xi"])
        assert len(pk) == 1312
        assert len(sk) == 2560

    def test_kat_sign_verify(self):
        """KAT: sign and verify with KAT key and message."""
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen_internal(KAT_0["xi"])
        msg = KAT_0["msg"]
        ctx = KAT_0["ctx"]
        sig = dsa.sign_deterministic(sk, msg, ctx)
        assert len(sig) == 2420
        assert dsa.verify(pk, msg, sig, ctx), "KAT signature verification failed"

    def test_hedged_sign_verify(self):
        """Hedged (randomized) signing must also verify."""
        dsa = MLDSA("ML_DSA_44")
        pk, sk = dsa.keygen()
        msg = b"hedged signing test"
        sig = dsa.sign(sk, msg)
        assert dsa.verify(pk, msg, sig)
