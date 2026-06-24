import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

from dataloader import FolderDataset, PreprocessTransform, load_full_dataframe
from model import RFNet

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= 실험 파라미터 조합 =========================
param_grid = {
    "lr": [1e-3],
    "weight_decay": [1e-2],
    "dropout_p": [0.3],
    "delta": [5.0],
    "repeat_factor": [2]
}
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# ========================= 고정 설정 =========================
EPOCHS = 100
BATCH_SIZE = 16
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
DATA_PATH = "{address}"
EXCEL_PATH = "{address}"
BASE_SAVE_DIR = "./grid_results"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ========================= 공통 전처리 및 분할 =========================
df_initial = load_full_dataframe(EXCEL_PATH)
df_initial['volume'] = pd.to_numeric(df_initial['volume'], errors='coerce')
df_initial.dropna(subset=['volume'], inplace=True)
df_initial['original_index'] = df_initial.index

def check_files_exist(row):
    if pd.isna(row['Upright_H']) or pd.isna(row['Upright_S']): return False
    h_path = os.path.join(DATA_PATH, str(row['Upright_H']) + '*.csv')
    s_path = os.path.join(DATA_PATH, str(row['Upright_S']) + '*.csv')
    return bool(glob.glob(h_path) and glob.glob(s_path))

valid_mask = df_initial.apply(check_files_exist, axis=1)
df = df_initial[valid_mask].reset_index(drop=True)

indices = list(range(len(df)))
volume_bins = pd.cut(df['volume'], bins=5, labels=False, include_lowest=True)
train_idx, val_test_idx, _, _ = train_test_split(indices, df['volume'], test_size=(VAL_RATIO + TEST_RATIO), random_state=42, stratify=volume_bins)

val_test_df = df.iloc[val_test_idx].copy()
val_test_bins = pd.cut(val_test_df['volume'], bins=5, labels=False, include_lowest=True)
bin_counts = val_test_bins.value_counts()

if (bin_counts < 2).any():
    print("Stratification Error 방지: 최소 클래스 멤버가 2개 미만인 구간이 있어 복제합니다.")
    rows_to_add = []
    for bin_val in bin_counts[bin_counts < 2].index:
        idx_to_duplicate = val_test_bins[val_test_bins == bin_val].index[0]
        row_df = val_test_df.loc[[idx_to_duplicate]].copy()
        rows_to_add.append(row_df)
        print(f"-> ID {val_test_df.loc[idx_to_duplicate]['original_index']} 복제")
    val_test_df = pd.concat([val_test_df] + rows_to_add, ignore_index=True)
    val_test_bins = pd.cut(val_test_df['volume'], bins=5, labels=False, include_lowest=True)

val_idx, test_idx, _, _ = train_test_split(val_test_df.index, val_test_df['volume'], test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), random_state=42, stratify=val_test_bins)

test_original_indices = df.loc[test_idx]['original_index'].tolist()
torch.save(test_original_indices, os.path.join(BASE_SAVE_DIR, "test_idx.pt"))

transform_h = PreprocessTransform(num_select=3)
transform_s = PreprocessTransform(num_select=4)
full_dataset = FolderDataset(df, DATA_PATH, transform_h=transform_h, transform_s=transform_s)

# ========================= 실험 반복 =========================
for i, combo in enumerate(param_combinations):
    params = dict(zip(param_names, combo))
    exp_name = f"lr{params['lr']}_wd{params['weight_decay']}_do{params['dropout_p']}_r{params['repeat_factor']}"
    save_dir = os.path.join(BASE_SAVE_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n==== 실험 {i+1}/{len(param_combinations)}: {exp_name} ====")

    augmented_train_idx = []
    for idx in train_idx:
        volume = df.iloc[idx]['volume']
        rep = params['repeat_factor']
        if volume <= 150 or volume >= 600:
            rep *= 2
        augmented_train_idx.extend([idx] * rep)

    train_dataset = Subset(full_dataset, augmented_train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = RFNet(dropout_p=params['dropout_p']).to(device)
    criterion = nn.HuberLoss(delta=params['delta'])
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_val_mae = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            rf_h = [batch['horizon'][:, i, :].unsqueeze(1).to(device) for i in range(3)]
            rf_s = [batch['sagittal'][:, i, :].unsqueeze(1).to(device) for i in range(4)]
            targets = batch['volume_gt'].to(device)

            preds, _, _ = model(rf_h, rf_s)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, all_preds, all_targets = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                rf_h = [batch['horizon'][:, i, :].unsqueeze(1).to(device) for i in range(3)]
                rf_s = [batch['sagittal'][:, i, :].unsqueeze(1).to(device) for i in range(4)]
                targets = batch['volume_gt'].to(device)
                preds, _, _ = model(rf_h, rf_s)
                val_loss += criterion(preds, targets).item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        if mae < best_val_mae:
            best_val_mae = mae
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        print(f"[Exp {i+1} Epoch {epoch+1}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f} | Best MAE: {best_val_mae:.2f}")

    with open(os.path.join(BASE_SAVE_DIR, "summary.txt"), 'a') as f:
        f.write(f"{exp_name}: MAE {best_val_mae:.2f}\n")
