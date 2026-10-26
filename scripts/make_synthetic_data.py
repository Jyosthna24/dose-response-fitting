"""Synthetic potentiator dose-response data with KNOWN parameters.

Two CFTR variants respond to the same compound with different potency, so
the fitting and the EC50 comparison can be checked against the truth.

    python scripts/make_synthetic_data.py --out data/synthetic_dose_response.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doseresp import hill4  # noqa: E402

TRUTH = {
    "WT":      {"bottom": 5.0, "top": 100.0, "logEC50": np.log10(0.3), "hill": 1.2},
    "F508del": {"bottom": 3.0, "top": 45.0,  "logEC50": np.log10(2.0), "hill": 1.0},
}
CONCS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]  # µM, half-log series + vehicle


def make(replicates: int = 3, noise_sd: float = 4.0, seed: int = 0, truth=TRUTH) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cond, p in truth.items():
        for c in CONCS:
            mean = p["bottom"] if c == 0 else hill4(np.log10(c), p["bottom"], p["top"],
                                                    p["logEC50"], p["hill"])
            for r in range(replicates):
                rows.append({"condition": cond, "concentration": c, "replicate": r + 1,
                             "response": round(mean + rng.normal(0, noise_sd), 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_dose_response.csv"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    make(seed=a.seed).to_csv(a.out, index=False)
    print(f"Wrote {a.out}")
