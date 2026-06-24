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

# ===== 시드 고정 =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== 설정 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
USE_LOG1P = False
SAVE_DIR = "{address}"
DATA_PATH = "{address}"
EXCEL_PATH = "{address}"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
TEST_IDX_PATH = "{address}"

# ===== 테스트 인덱스 불러오기 =====
df = load_full_dataframe(EXCEL_PATH)
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
df.dropna(subset=['volume'], inplace=True)
df['original_index'] = df.index

if not os.path.exists(TEST_IDX_PATH):
    print("test_idx.pt 파일이 없습니다. train.py 실행 후 테스트하세요.")
    exit()

test_idx = torch.load(TEST_IDX_PATH)

# ===== 증강된 테스트셋 클래스 =====
class AugmentedTestDataset(Dataset):
    def __init__(self, df, data_path, test_idx, seed_offset):
        self.df = df.iloc[test_idx].reset_index(drop=True)
        self.transform_h = PreprocessTransform(num_select=3, seed_offset=seed_offset)
        self.transform_s = PreprocessTransform(num_select=4, seed_offset=seed_offset)
        self.dataset = FolderDataset(self.df, data_path, self.transform_h, self.transform_s)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# ===== 모델 로드 =====
model = RFNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== 두 번 증강 후 평균 예측 =====
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

# ===== 평균 결과 계산 =====
final_preds = (np.array(all_preds_1) + np.array(all_preds_2)) / 2
final_targets = np.array(all_targets)

# ===== 600ml 이하만 필터링 =====
mask = final_targets <= 700
final_preds = final_preds[mask]
final_targets = final_targets[mask]

# ===== 회귀 평가 =====
mae = mean_absolute_error(final_targets, final_preds)
rmse = np.sqrt(mean_squared_error(final_targets, final_preds))
r2 = r2_score(final_targets, final_preds)
within_50 = np.mean(np.abs(final_preds - final_targets) <= 50) * 100

print("\n===== 회귀 기반 평가 (≤700ml) =====")
print(f"MAE        : {mae:.2f}")
print(f"RMSE       : {rmse:.2f}")
print(f"R²         : {r2:.4f}")
print(f"±50 정확도 : {within_50:.2f}%")

# ===== 산점도 시각화 =====
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

# ===== 클래스 기반 평가 (50ml 단위) =====
acc = (1 - np.mean(abs(final_preds-final_targets)/final_targets)) * 100

print("\n===== 클래스 분류 평가 (≤700ml) =====")
print(f"Accuracy (exact)    : {acc:.2f}%")
