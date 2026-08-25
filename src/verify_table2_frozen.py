#!/usr/bin/env python3
"""Verify the recorded Stage 1 result from its frozen prediction artifact.

This command reproduces the numerical result reported for the historical
17-acquisition development evaluation. It intentionally does not perform a
new model forward pass: the two stochastic scan-line draws used by that run
were not logged, so they cannot be reconstructed from the checkpoint alone.

The verifier uses only the Python standard library. It checks the SHA-256
hashes of the prediction CSV, released checkpoint, and recorded split before
recomputing MAE, RMSE, R2, within-50-mL accuracy, and Bland--Altman statistics.
Any artifact or metric mismatch causes a non-zero exit status.

Usage from the repository root:

    python src/verify_table2_frozen.py
    python src/verify_table2_frozen.py --json-out table2_verification.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = (
    REPO_ROOT / "figures" / "table2_reference" / "table2_artifact_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot compute a mean from an empty sequence")
    return sum(values) / len(values)


def compute_metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and equal length")

    errors = [prediction - target for target, prediction in zip(actual, predicted)]
    target_mean = mean(actual)
    residual_sum_squares = sum(error * error for error in errors)
    total_sum_squares = sum((target - target_mean) ** 2 for target in actual)
    if total_sum_squares == 0:
        raise ValueError("R2 is undefined because every target is identical")

    bias = mean(errors)
    error_sd = statistics.stdev(errors)
    return {
        "n": len(actual),
        "mae_ml": mean(abs(error) for error in errors),
        "rmse_ml": math.sqrt(residual_sum_squares / len(errors)),
        "r2": 1.0 - residual_sum_squares / total_sum_squares,
        "within_50_ml_pct": 100.0 * mean(abs(error) <= 50.0 for error in errors),
        "bland_altman_bias_ml": bias,
        "bland_altman_loa_lower_ml": bias - 1.96 * error_sd,
        "bland_altman_loa_upper_ml": bias + 1.96 * error_sd,
    }


def assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"Metric mismatch for {name}: observed {actual:.12g}, "
            f"expected {expected:.12g} (absolute tolerance {tolerance:g})"
        )


def verify() -> Dict[str, object]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing release manifest: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    verified_artifacts: Dict[str, Dict[str, object]] = {}
    for relative_path, expected in manifest["artifacts"].items():
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing release artifact: {path}")
        observed_hash = sha256(path)
        observed_size = path.stat().st_size
        if observed_hash != expected["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative_path}: observed {observed_hash}, "
                f"expected {expected['sha256']}"
            )
        if observed_size != expected["size_bytes"]:
            raise RuntimeError(
                f"Size mismatch for {relative_path}: observed {observed_size}, "
                f"expected {expected['size_bytes']}"
            )
        verified_artifacts[relative_path] = {
            "sha256": observed_hash,
            "size_bytes": observed_size,
        }

    prediction_path = REPO_ROOT / manifest["prediction_artifact"]
    with prediction_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required_columns = {"actual", "pred"}
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            f"{prediction_path} must contain the columns {sorted(required_columns)}"
        )

    actual = [float(row["actual"]) for row in rows]
    predicted = [float(row["pred"]) for row in rows]
    metrics = compute_metrics(actual, predicted)

    expected_metrics = manifest["expected_metrics"]
    if metrics["n"] != expected_metrics["n"]:
        raise RuntimeError(
            f"Row-count mismatch: observed {metrics['n']}, expected {expected_metrics['n']}"
        )
    tolerance = float(manifest["metric_absolute_tolerance"])
    for name, expected_value in expected_metrics.items():
        if name == "n":
            continue
        assert_close(name, float(metrics[name]), float(expected_value), tolerance)

    return {
        "status": "PASS",
        "interpretation": manifest["interpretation"],
        "metrics": metrics,
        "verified_artifacts": verified_artifacts,
        "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optionally write the complete machine-readable verification result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify()
    metrics = result["metrics"]

    print("Stage 1 frozen-result verification: PASS")
    print(f"n                 : {metrics['n']}")
    print(f"MAE               : {metrics['mae_ml']:.2f} mL")
    print(f"RMSE              : {metrics['rmse_ml']:.2f} mL")
    print(f"R2                : {metrics['r2']:.6f}")
    print(f"Within 50 mL      : {metrics['within_50_ml_pct']:.2f}%")
    print(f"Bland-Altman bias : {metrics['bland_altman_bias_ml']:.2f} mL")
    print(
        "Bland-Altman LoA  : "
        f"[{metrics['bland_altman_loa_lower_ml']:.2f}, "
        f"{metrics['bland_altman_loa_upper_ml']:.2f}] mL"
    )
    print(f"Manifest SHA-256  : {result['manifest_sha256']}")
    print("Interpretation    : recorded-result verification; not a fresh forward pass")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"JSON report       : {args.json_out}")


if __name__ == "__main__":
    main()
