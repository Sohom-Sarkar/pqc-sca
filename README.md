# PQC Side-Channel Analyzer

ML-KEM (FIPS 203) and ML-DSA (FIPS 204) implemented from scratch, with every NTT butterfly instrumented for power leakage simulation. Two leakage models are tested: a Hamming Weight baseline and a CMOS physics model. Both are attacked with CPA and TVLA.

The physics model attenuates CPA correlation by about 6% at N=500. The reason is that HD-based switching noise doesn't line up with the HW hypothesis an attacker applies. That's the main finding.

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Python 3.11+. No Node.js needed — the frontend is a plain HTML file.

## Layout

```
algorithms/   mlkem.py, mldsa.py, ntt.py (+ Tracer)
leakage/      models.py (HW + physics), simulator.py
attacks/      cpa.py, tvla.py, metrics.py
api/          server.py  (FastAPI)
frontend/     index.html (React via CDN, no build step)
output/       plots.py, report.py
notebooks/    full_pipeline.ipynb
tests/        test_mlkem.py (15), test_mldsa.py (10)
```

## Tests

```bash
python -m pytest tests/ -v
```

## Notebook

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

Runs the full pipeline end-to-end: KAT correctness, trace simulation, CPA on both models, TVLA on ML-KEM and ML-DSA, convergence curves, vulnerability report.

To execute headlessly:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
    --output notebooks/full_pipeline_executed.ipynb notebooks/full_pipeline.ipynb
```

## API + Frontend

```bash
# Terminal 1
uvicorn api.server:app --port 8000

# Terminal 2
python -m http.server 3000 --directory frontend
# open http://localhost:3000
```

Endpoints: `POST /run`, `GET /traces`, `GET /attack`, `GET /report`, `GET /health`.

## Results

| | HW model | Physics model |
|---|---|---|
| CPA peak \|ρ\| (N=500) | 0.489 | 0.459 |
| TVLA max\|t\| (N=500) | 81 | 82 |
| ML-DSA TVLA max\|t\| (N=200) | 70 | n/a |

TVLA threshold is 4.5. Both models leak substantially — neither algorithm has countermeasures.

## Algorithm parameters

ML-KEM-512: k=2, η₁=3, η₂=2, d_u=10, d_v=4, q=3329, ζ=17^BitRev7(k) mod q  
ML-KEM-768: k=3, η₁=2, η₂=2, d_u=10, d_v=4  
ML-DSA-44:  k=4, l=4, η=2, γ₁=2¹⁷, γ₂=(q−1)/88, τ=39, q=8380417, ζ=1753^BitRev8(k) mod q
