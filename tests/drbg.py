"""
AES-256-CTR DRBG (NIST SP 800-90A) — KAT test harness.

This is the exact DRBG used in NIST PQC submission KAT vectors.
Requires pycryptodome: pip install pycryptodome
"""

from Crypto.Cipher import AES


def _increment_v(V: bytearray) -> None:
    """Increment V as a 128-bit big-endian counter (in-place)."""
    for i in range(15, -1, -1):
        if V[i] == 0xFF:
            V[i] = 0x00
        else:
            V[i] += 1
            break


def _aes256_ecb(key: bytes, block: bytes) -> bytes:
    return AES.new(key, AES.MODE_ECB).encrypt(block)


class NIST_DRBG:
    """AES-256-CTR DRBG matching the NIST PQC KAT rng.c implementation."""

    def __init__(self, entropy_input: bytes) -> None:
        assert len(entropy_input) == 48, "Entropy input must be 48 bytes"
        self._key = bytearray(32)   # initial key = 0^32
        self._V   = bytearray(16)   # initial V   = 0^16
        self._update(entropy_input)

    def _update(self, provided_data: bytes | None) -> None:
        """AES256_CTR_DRBG_Update: produce 48 bytes, XOR with provided_data."""
        temp = bytearray()
        for _ in range(3):
            _increment_v(self._V)
            temp += _aes256_ecb(bytes(self._key), bytes(self._V))
        if provided_data is not None:
            for i in range(48):
                temp[i] ^= provided_data[i]
        self._key = temp[:32]
        self._V   = temp[32:48]

    def randombytes(self, length: int) -> bytes:
        """Generate `length` random bytes and reseed with zeros."""
        out = bytearray()
        while len(out) < length:
            _increment_v(self._V)
            out += _aes256_ecb(bytes(self._key), bytes(self._V))
        out = bytes(out[:length])
        self._update(None)   # reseed with zero data
        return out
