"""
evaluate_clinical_bland_altman.py
==================================

Generate the held-out clinical test-set prediction CSV and Bland--Altman plot
for the clinical-only RFNet model.

This script is intended for Supplementary Figure S2.
It uses:
  - checkpoints/best_model.pt
  - checkpoints/test_idx.pt
  - src/dataloader.py
  - src/model.py

Example:
  cd rf-bladder
  python src/evaluate_clinical_bland_altman.py \
      --data-path /path/to/converted_rf_csv_folder \
      --index-csv /path/to/Z_volume_selection.csv \
      --output-dir figures/supplementary_bland_altman

Expected output:
  clinical_test_predictions.csv
  supp_bland_altman_clinical.pdf
  supp_bland_altman_clinical.png
  metrics_summary.txt

NOTE on numbers: this script seeds channel-selection augmentation
deterministically per sample (original_index + aug offset), so it is fully
reproducible run to run, and matches test.py's (now also deterministic)
result -- approximately MAE 45 mL / R^2 0.81 on the n=17 held-out set. This
differs from the Table II point estimate (MAE 35.46 mL, R^2 0.901), which
came from test.py's earlier, non-deterministic worker-seeded augmentation
and cannot be reconstructed after the fact. The exact frozen predictions
behind Table II are in figures/table2_reference/clinical_test_predictions_n17.csv.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Allow execution from repository root or from src/.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
import sys
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dataloader import FolderDataset, PreprocessTransform, load_full_dataframe  # noqa: E402
from model import RFNet  # noqa: E402


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_pair_exists(row: pd.Series, data_path: str) -> bool:
    """Return True only when both upright H and S RF CSV files exist."""
    if pd.isna(row.get("Upright_H")) or pd.isna(row.get("Upright_S")):
        return False
    h_pattern = os.path.join(data_path, str(row["Upright_H"]) + "*.csv")
    s_pattern = os.path.join(data_path, str(row["Upright_S"]) + "*.csv")
    return bool(glob.glob(h_pattern)) and bool(glob.glob(s_pattern))


def load_index_dataframe(index_csv: str, data_path: str | None, check_files: bool) -> pd.DataFrame:
    """Load the clinical index CSV and prepare original_index for test_idx matching."""
    df = load_full_dataframe(index_csv)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["volume"]).copy()
    df["original_index"] = df.index

    if check_files:
        if data_path is None:
            raise ValueError("--check-files requires --data-path.")
        valid_mask = df.apply(lambda row: file_pair_exists(row, data_path), axis=1)
        df = df.loc[valid_mask].copy()

    return df.reset_index(drop=True)


def select_test_dataframe(df: pd.DataFrame, test_idx: Sequence[int]) -> pd.DataFrame:
    """
    Select held-out test rows robustly.

    In the released repository, test_idx.pt is described as storing original_index
    values. Some older test scripts used it as positional iloc indices. To avoid
    accidental split mismatch, this function first tries exact original_index
    matching and falls back to iloc only when needed.
    """
    test_idx = [int(x) for x in test_idx]
    original_index_set = set(df["original_index"].astype(int).tolist())
    test_idx_set = set(test_idx)

    if test_idx_set.issubset(original_index_set):
        selected = df[df["original_index"].astype(int).isin(test_idx)].copy()
        # Preserve the order saved in test_idx.pt.
        order = {idx: i for i, idx in enumerate(test_idx)}
        selected["_test_order"] = selected["original_index"].astype(int).map(order)
        selected = selected.sort_values("_test_order").drop(columns=["_test_order"])
        return selected.reset_index(drop=True)

    max_idx = max(test_idx) if test_idx else -1
    if max_idx < len(df):
        return df.iloc[test_idx].copy().reset_index(drop=True)

    missing = sorted(test_idx_set - original_index_set)[:10]
    raise ValueError(
        "Could not match test_idx.pt to the loaded dataframe. "
        f"Examples of missing indices: {missing}. Check that --index-csv and --data-path "
        "correspond to the checkpoint/test split."
    )


class DeterministicAugmentedClinicalDataset(Dataset):
    """Clinical test dataset with deterministic channel-selection augmentation."""

    def __init__(self, df: pd.DataFrame, data_path: str, aug_seed_offset: int):
        self.df = df.reset_index(drop=True)
        self.transform_h = PreprocessTransform(num_select=3, seed_offset=0)
        self.transform_s = PreprocessTransform(num_select=4, seed_offset=0)
        self.dataset = FolderDataset(self.df, data_path, self.transform_h, self.transform_s)
        self.aug_seed_offset = int(aug_seed_offset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Load raw CSVs using the same logic as FolderDataset, but avoid the
        # random seed inside FolderDataset.__getitem__ so the output is reproducible.
        h_file = glob.glob(os.path.join(self.dataset.folder_path, str(row["Upright_H"]) + "*.csv"))[0]
        s_file = glob.glob(os.path.join(self.dataset.folder_path, str(row["Upright_S"]) + "*.csv"))[0]

        h_df = pd.read_csv(h_file, header=None, low_memory=False)
        h_data = h_df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()[: self.dataset.max_depth, :]

        s_df = pd.read_csv(s_file, header=None, low_memory=False)
        s_data = s_df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()[: self.dataset.max_depth, :]

        original_index = int(row["original_index"])
        seed = original_index + self.aug_seed_offset

        h_data = self.transform_h(h_data, seed=seed)
        s_data = self.transform_s(s_data, seed=seed)

        return {
            "horizon": torch.tensor(h_data, dtype=torch.float32),
            "sagittal": torch.tensor(s_data, dtype=torch.float32),
            "volume_gt": torch.tensor(float(row["volume"]), dtype=torch.float32),
            "index": original_index,
        }


def predict_once(
    model: RFNet,
    df_test: pd.DataFrame,
    data_path: str,
    device: torch.device,
    batch_size: int,
    aug_seed_offset: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = DeterministicAugmentedClinicalDataset(df_test, data_path, aug_seed_offset=aug_seed_offset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    preds: List[float] = []
    targets: List[float] = []
    indices: List[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            rf_h = [batch["horizon"][:, i, :].unsqueeze(1).to(device) for i in range(3)]
            rf_s = [batch["sagittal"][:, i, :].unsqueeze(1).to(device) for i in range(4)]
            y = batch["volume_gt"].to(device)
            pred, _, _ = model(rf_h, rf_s)
            pred = pred.squeeze(-1)

            preds.extend(pred.detach().cpu().numpy().astype(float).tolist())
            targets.extend(y.detach().cpu().numpy().astype(float).tolist())
            indices.extend(batch["index"].detach().cpu().numpy().astype(int).tolist())

    return np.array(indices), np.array(targets), np.array(preds)


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(actual)),
        "MAE_mL": float(mean_absolute_error(actual, pred)),
        "RMSE_mL": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
        "within_50_mL_percent": float(np.mean(np.abs(pred - actual) <= 50.0) * 100.0),
    }


def save_bland_altman(actual: np.ndarray, pred: np.ndarray, out_pdf: Path, out_png: Path) -> dict:
    mean_volume = (actual + pred) / 2.0
    diff = pred - actual
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(mean_volume, diff, alpha=0.8)
    ax.axhline(bias, linestyle="-", linewidth=1.5, label=f"Bias = {bias:.1f} mL")
    ax.axhline(loa_upper, linestyle="--", linewidth=1.2, label=f"+1.96 SD = {loa_upper:.1f} mL")
    ax.axhline(loa_lower, linestyle="--", linewidth=1.2, label=f"-1.96 SD = {loa_lower:.1f} mL")
    ax.axhline(0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Mean of actual and predicted volume (mL)")
    ax.set_ylabel("Prediction error (predicted - actual, mL)")
    ax.set_title("Clinical test-set Bland--Altman analysis")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "bias_mL": bias,
        "sd_error_mL": sd,
        "loa_lower_mL": float(loa_lower),
        "loa_upper_mL": float(loa_upper),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clinical predictions CSV and Bland--Altman plot.")
    parser.add_argument("--data-path", required=True, help="Directory containing converted clinical RF CSV files.")
    parser.add_argument("--index-csv", required=True, help="Path to Z_volume_selection.csv or equivalent index CSV.")
    parser.add_argument("--model-path", default=str(_REPO_ROOT / "checkpoints" / "best_model.pt"), help="Path to best_model.pt.")
    parser.add_argument("--test-idx-path", default=str(_REPO_ROOT / "checkpoints" / "test_idx.pt"), help="Path to test_idx.pt.")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "figures" / "supplementary_bland_altman"), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for maximum reproducibility.")
    parser.add_argument("--max-volume", type=float, default=700.0, help="Evaluation volume upper bound in mL.")
    parser.add_argument("--check-files", action="store_true", help="Filter rows to those with existing Upright_H/Upright_S files before matching test_idx.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_index_dataframe(args.index_csv, args.data_path, check_files=args.check_files)
    test_idx = torch.load(args.test_idx_path, map_location="cpu")
    if isinstance(test_idx, torch.Tensor):
        test_idx = test_idx.detach().cpu().numpy().tolist()
    df_test = select_test_dataframe(df, test_idx)

    model = RFNet(dropout_p=0.5).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    aug_offsets = [0, 1000]
    all_aug_preds = []
    indices_ref = None
    targets_ref = None
    for offset in aug_offsets:
        indices, targets, preds = predict_once(
            model=model,
            df_test=df_test,
            data_path=args.data_path,
            device=device,
            batch_size=args.batch_size,
            aug_seed_offset=offset,
            num_workers=args.num_workers,
        )
        if indices_ref is None:
            indices_ref = indices
            targets_ref = targets
        else:
            if not np.array_equal(indices_ref, indices):
                raise RuntimeError("Augmentation passes returned different sample orders.")
            if not np.allclose(targets_ref, targets):
                raise RuntimeError("Augmentation passes returned different targets.")
        all_aug_preds.append(preds)

    pred_mean = np.mean(np.vstack(all_aug_preds), axis=0)
    actual = targets_ref.astype(float)
    indices = indices_ref.astype(int)

    mask = actual <= args.max_volume
    actual = actual[mask]
    pred_mean = pred_mean[mask]
    indices = indices[mask]
    pred_aug0 = all_aug_preds[0][mask]
    pred_aug1 = all_aug_preds[1][mask]

    results = pd.DataFrame(
        {
            "sample_order": np.arange(len(actual)),
            "original_index": indices,
            "actual": actual,
            "pred": pred_mean,
            "pred_aug0": pred_aug0,
            "pred_aug1": pred_aug1,
            "error_pred_minus_actual": pred_mean - actual,
            "abs_error": np.abs(pred_mean - actual),
            "within_50_mL": np.abs(pred_mean - actual) <= 50.0,
        }
    )
    csv_path = out_dir / "clinical_test_predictions.csv"
    results.to_csv(csv_path, index=False)

    metrics = regression_metrics(actual, pred_mean)
    ba_stats = save_bland_altman(
        actual=actual,
        pred=pred_mean,
        out_pdf=out_dir / "supp_bland_altman_clinical.pdf",
        out_png=out_dir / "supp_bland_altman_clinical.png",
    )

    summary_path = out_dir / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Clinical-only held-out test evaluation (volume <= %.1f mL)\n" % args.max_volume)
        f.write("Device used for inference: %s\n\n" % device)
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBland--Altman statistics\n")
        for k, v in ba_stats.items():
            f.write(f"{k}: {v}\n")

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {out_dir / 'supp_bland_altman_clinical.pdf'}")
    print(f"  {out_dir / 'supp_bland_altman_clinical.png'}")
    print(f"  {summary_path}")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nBland--Altman:")
    for k, v in ba_stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
