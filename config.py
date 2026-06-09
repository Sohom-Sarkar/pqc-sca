"""
Central configuration for the PQC Side-Channel Analyzer.
All algorithm constants and tunable physics model parameters live here.
"""

# ─── Algorithm selector ──────────────────────────────────────────────────────
# Change this to switch algorithms throughout the entire pipeline.
ALGORITHM: str = "ML_KEM_512"   # "ML_KEM_512" | "ML_KEM_768" | "ML_DSA_44"

# ─── Leakage model selector ───────────────────────────────────────────────────
LEAKAGE_MODEL: str = "physics"  # "hamming_weight" | "physics"

# ─── Simulation defaults ─────────────────────────────────────────────────────
DEFAULT_N_TRACES: int = 1000
DEFAULT_SNR_DB: float = 20.0
RANDOM_SEED: int = 42

# ─── ML-KEM parameter sets (NIST FIPS 203) ───────────────────────────────────
MLKEM_PARAMS = {
    "ML_KEM_512": {
        "k": 2, "n": 256, "q": 3329,
        "eta1": 3, "eta2": 2, "du": 10, "dv": 4,
        "ek_size": 800, "dk_size": 1632, "ct_size": 768, "ss_size": 32,
        "variant_byte": 2,
    },
    "ML_KEM_768": {
        "k": 3, "n": 256, "q": 3329,
        "eta1": 2, "eta2": 2, "du": 10, "dv": 4,
        "ek_size": 1184, "dk_size": 2400, "ct_size": 1088, "ss_size": 32,
        "variant_byte": 3,
    },
}

# ─── ML-DSA parameter sets (NIST FIPS 204) ───────────────────────────────────
MLDSA_PARAMS = {
    "ML_DSA_44": {
        "n": 256, "k": 4, "l": 4, "q": 8380417, "d": 13,
        "lambda_": 128, "gamma1": 1 << 17, "gamma2": (8380417 - 1) // 88,
        "tau": 39, "beta": 78, "omega": 80, "eta": 2,
        "pk_size": 1312, "sk_size": 2560, "sig_size": 2420,
    },
}

# ─── Physics-informed leakage model parameters ───────────────────────────────
# These map to real device physics: CMOS switching energy ∝ C·V²·ΔN_transitions
# plus static leakage and noise terms.
#
#  P(v_new, v_old) = α·HD(v_new,v_old)·C·V²
#                  + β·HW(v_new)·V
#                  + γ·N_thermal
#                  + δ·N_flicker
#
# Tunable by Sohom based on target device physics.

PHYSICS_PARAMS = {
    "C":     1e-15,   # effective node capacitance (F) — 1 fF, typical sub-micron CMOS node
    "V":     1.0,     # supply voltage (V)
    "alpha": 1.0,     # dynamic power weight (switching energy coefficient)
    "beta":  0.1,     # static/leakage power weight
    "gamma": 0.05,    # thermal noise weight
    "delta": 0.01,    # 1/f (flicker) noise weight
}

# ─── CPA / TVLA attack parameters ────────────────────────────────────────────
CPA_KEY_BYTE_TARGET: int = 0   # which NTT output coefficient to target first
TVLA_THRESHOLD: float = 4.5    # standard |t| threshold for leakage detection

# ─── API / server settings ───────────────────────────────────────────────────
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
