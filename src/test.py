import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from dataloader import FolderDataset, PreprocessTransform, load_full_dataframe
from model import RFNet

# ===== Seed Initialization =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== Configuration =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
USE_LOG1P = False
SAVE_DIR = "{address}"
DATA_PATH ="{address}"
EXCEL_PATH ="{address}"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
TEST_IDX_PATH = "{address}"

# ===== Load Test Indices =====
df = load_full_dataframe(EXCEL_PATH)
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
df.dropna(subset=['volume'], inplace=True)
df['original_index'] = df.index

if not os.path.exists(TEST_IDX_PATH):
    print("test_idx.pt not found. Please run train.py first.")
    exit()

test_idx = torch.load(TEST_IDX_PATH)

# ===== Augmented Test Dataset Class =====
# NOTE (reproducibility fix): the previous version of this class delegated
# to FolderDataset.__getitem__, which draws its per-sample augmentation seed
# via `np.random.randint(0, 100000)` -- a call that consumes the *global*
# NumPy RNG state. With num_workers>0, each DataLoader worker is a fork of
# the main process and inherits an *independent copy* of that global state;
# without an explicit per-worker reseed (this codebase has none), several
# workers can end up drawing the same "random" numbers, and the exact
# sequence depends on worker-to-index scheduling, prefetch timing, and OS
# scheduling. None of that was logged, so the specific channel-selection
# draw behind the historical Table II numbers (MAE 35.46 mL, R^2=0.901)
# cannot be reconstructed after the fact -- confirmed by testing multiple
# plausible worker-seed reconstructions, none of which reproduced the saved
# per-sample predictions in figures/table2_reference/.
#
# This version seeds each sample deterministically from its own
# `original_index`, independent of worker count, batch size, or call order.
# Re-running this script now gives the SAME result every time and on every
# machine -- but that result will generally differ from the historical
# Table II point estimate (which came from one particular, unrecorded
# random draw). For the exact frozen predictions behind Table II, see
# figures/table2_reference/clinical_test_predictions_n17.csv.
class AugmentedTestDataset(Dataset):
    def __init__(self, df, data_path, test_idx, seed_offset):
        self.df = df.iloc[test_idx].reset_index(drop=True)
        self.data_path = data_path
        self.transform_h = PreprocessTransform(num_select=3, seed_offset=0)
        self.transform_s = PreprocessTransform(num_select=4, seed_offset=0)
        self.seed_offset = seed_offset
        # Reuse FolderDataset only for its file-loading logic; its own
        # (non-deterministic) transform calls are bypassed below.
        self.dataset = FolderDataset(self.df, data_path, transform_h=None, transform_s=None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        original_index = int(sample['index'])
        seed = original_index + self.seed_offset
        h = self.transform_h(sample['horizon'].numpy(), seed=seed)
        s = self.transform_s(sample['sagittal'].numpy(), seed=seed)
        sample['horizon'] = torch.tensor(h, dtype=torch.float32)
        sample['sagittal'] = torch.tensor(s, dtype=torch.float32)
        return sample

# ===== Load Model =====
model = RFNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== Predict with 2x Augmentation and Average =====
all_preds_1, all_preds_2, all_targets = [], [], []

for seed_offset in [0, 1000]:
    test_dataset = AugmentedTestDataset(df, DATA_PATH, test_idx, seed_offset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    current_preds = []
    with torch.no_grad():
        for batch in test_loader:
            rf_h = [batch['horizon'][:, i, :].unsqueeze(1).to(DEVICE) for i in range(3)]
            rf_s = [batch['sagittal'][:, i, :].unsqueeze(1).to(DEVICE) for i in range(4)]
            targets = batch['volume_gt'].to(DEVICE)

            preds, _, _ = model(rf_h, rf_s)
            preds = preds.squeeze(-1)

            if USE_LOG1P:
                preds = torch.expm1(preds)

            current_preds.extend(preds.cpu().numpy())

            if seed_offset == 0:
                all_targets.extend(targets.cpu().numpy())

    if seed_offset == 0:
        all_preds_1 = current_preds
    else:
        all_preds_2 = current_preds

# ===== Compute Averaged Results =====
final_preds = (np.array(all_preds_1) + np.array(all_preds_2)) / 2
final_targets = np.array(all_targets)

# ===== Filter to ≤700ml =====
mask = final_targets <= 700
final_preds = final_preds[mask]
final_targets = final_targets[mask]

# ===== Regression Evaluation =====
mae = mean_absolute_error(final_targets, final_preds)
rmse = np.sqrt(mean_squared_error(final_targets, final_preds))
r2 = r2_score(final_targets, final_preds)
within_50 = np.mean(np.abs(final_preds - final_targets) <= 50) * 100

print("\n===== Regression Evaluation (≤700ml) =====")
print(f"MAE        : {mae:.2f}")
print(f"RMSE       : {rmse:.2f}")
print(f"R²         : {r2:.4f}")
print(f"±50 Accuracy   : {within_50:.2f}%")

# ===== Scatter Plot Visualization =====
plt.figure(figsize=(6, 6))
plt.scatter(final_targets, final_preds, alpha=0.6, label='Predictions')
plt.plot([min(final_targets), max(final_targets)], [min(final_targets), max(final_targets)], 'k--', label='Ideal (y=x)')
plt.plot([min(final_targets), max(final_targets)], [min(final_targets)+50, max(final_targets)+50], 'r--', alpha=0.7, label='±50 Range')
plt.plot([min(final_targets), max(final_targets)], [min(final_targets)-50, max(final_targets)-50], 'r--', alpha=0.7)
plt.xlabel("Actual Volume")
plt.ylabel("Predicted Volume")
plt.title("Prediction vs Actual (2x Aug, ≤700ml)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "scatter_testset_filtered.png"))
plt.close()

# ===== Class-based Evaluation (50ml bins) =====
acc = (1 - np.mean(abs(final_preds-final_targets)/final_targets)) * 100

print("\n===== Class-wise Evaluation (≤700ml) =====")
print(f"Accuracy (exact)    : {acc:.2f}%")
