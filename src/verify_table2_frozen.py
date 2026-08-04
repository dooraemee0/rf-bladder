"""
verify_table2_frozen.py -- Reproduce the exact Table II numbers (MAE 35.46 mL,
RMSE 42.12 mL, R^2 0.901, +-50 mL acc 82.35%) from the frozen, verified
per-sample predictions.

Why this script exists (read this before treating the numbers below as a
"live model evaluation"): the original Table II run selected its
channel-selection augmentation via a seed drawn from the global NumPy RNG
inside each DataLoader worker (see the history of src/test.py). That draw
was never logged, and is not reconstructable from the checkpoint alone --
we confirmed this by testing multiple plausible worker-seed reconstructions
(sequential single-stream, forked-worker-with-round-robin-scheduling, and
variations of both) against these exact per-sample values; none reproduced
them, including one that landed within 0.1 mL on a single sample only to
diverge sharply everywhere else, consistent with adjacent-scan-line
correlation rather than a matching seed. src/test.py and
src/evaluate_clinical_bland_altman.py have since been fixed to use
deterministic per-sample seeding, but a deterministic scheme necessarily
lands on a *different* specific channel draw than the original
non-deterministic one, so a fresh forward pass now reproducibly gives
~MAE 45 mL / R^2 0.81 instead -- not bit-identical to Table II.

This script instead verifies Table II directly from
figures/table2_reference/clinical_test_predictions_n17.csv, which stores the
model's actual saved output (actual volume, predicted volume, and both
augmentation-pass predictions) for that specific historical run, for the
same checkpoint (best_model.pt, md5 9d00bea8d44e4ecd3a682e01fe090a3f) and the
same held-out split (test_idx.pt). It recomputes the regression metrics and
Bland-Altman statistics from those saved numbers -- this is a verification
of a real recorded result, not a re-derivation via a fresh forward pass.

Usage:
  cd src
  python verify_table2_frozen.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
CSV_PATH = _REPO_ROOT / "figures" / "table2_reference" / "clinical_test_predictions_n17.csv"
OUT_DIR = _REPO_ROOT / "figures" / "table2_reference"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found. This file ships with the repository under "
            "figures/table2_reference/ -- if it's missing, your checkout is incomplete."
        )

    df = pd.read_csv(CSV_PATH)
    actual = df["actual"].to_numpy(dtype=float)
    pred = df["pred"].to_numpy(dtype=float)
    n = len(df)

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    acc50 = float(np.mean(np.abs(pred - actual) <= 50.0) * 100.0)

    print("=" * 60)
    print("Table II verification (frozen predictions)")
    print("=" * 60)
    print(f"n        : {n}")
    print(f"MAE      : {mae:.2f} mL")
    print(f"RMSE     : {rmse:.2f} mL")
    print(f"R^2      : {r2:.4f}")
    print(f"+-50 mL acc : {acc50:.2f}%")
    print()
    print("Expected (Table II / README): MAE 35.46 mL, RMSE 42.12 mL, "
          "R^2 0.901, +-50 mL acc 82.35%")

    mean_volume = (actual + pred) / 2.0
    diff = pred - actual
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    print(f"\nBland-Altman: bias={bias:.2f} mL, LoA=[{loa_lower:.2f}, {loa_upper:.2f}] mL")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, alpha=0.7, label="Predictions")
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, "k--", label="Ideal (y=x)")
    ax.plot(lims, [v + 50 for v in lims], "r--", alpha=0.7, label="±50 mL range")
    ax.plot(lims, [v - 50 for v in lims], "r--", alpha=0.7)
    ax.set_xlabel("Actual Volume (mL)")
    ax.set_ylabel("Predicted Volume (mL)")
    ax.set_title(f"Table II verification (n={n}, R²={r2:.3f}, MAE={mae:.2f} mL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "table2_verified_scatter.png"
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
