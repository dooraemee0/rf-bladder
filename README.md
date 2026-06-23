# Lightweight RF Bladder Volume Estimation

Reference implementation for the clinical (SNUH) bladder-volume regression
model described in our paper, *"Lightweight Domain-Adaptive Deep Learning for
Bladder Volume Estimation via Direct RF Signal Processing."*

The model (**RFStudent**) estimates bladder volume **directly from raw
seven-channel RF ultrasound** (three horizontal + four sagittal channels),
with no B-mode image reconstruction. Each channel is encoded by an independent
1D-CNN; the seven embeddings are concatenated and regressed to a scalar volume.

This repository contains the code and the trained checkpoint needed to
**reproduce the clinical-cohort result** (Table II in the paper).

## Results (clinical SNUH cohort, held-out test set)

| Metric | Value |
|---|---|
| MAE | 35.46 mL |
| RMSE | 42.12 mL |
| R² | 0.901 |
| ±50 mL accuracy | 82.35 % |

Evaluation uses two-pass channel-randomized augmentation (averaged) and is
restricted to samples with volume ≤ 700 mL, as in `src/test.py`.

> **Note on data.** The clinical RF dataset was collected at Seoul National
> University Hospital under IRB No. H-2107-024-1233 and contains identifiable
> clinical information, so it **cannot be publicly released**. The code and the
> trained weights are provided so the architecture, training procedure, and
> reported metrics can be inspected and re-run on equivalently formatted data.
> De-identified data may be available from the corresponding author on
> reasonable request, subject to IRB approval and a data-transfer agreement.

## Repository layout

```
rf-bladder/
├── src/
│   ├── model.py            # RFStudent (per-channel 1D-CNN encoders + regressor)
│   ├── dataloader.py       # RF CSV loading + preprocessing + channel-select augmentation
│   ├── train.py            # original training driver (grid over hyperparameters)
│   ├── train_reproduce.py  # single-config script reproducing the Table II model
│   └── test.py             # evaluation (2x aug averaging, ≤700 mL filter)
├── checkpoints/
│   └── best_model.pt       # trained weights for the Table II result
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Tested with Python 3.8 and PyTorch (CUDA optional; CPU works for inference).

## Reproducing the reported result

### Option A — verify with the released checkpoint (recommended, fastest)

The checkpoint in `checkpoints/best_model.pt` is the exact model behind Table II.
Point `test.py` at it and run:

1. Edit the paths near the top of `src/test.py`:
   - `DATA_PATH`  → directory of converted RF CSV files (one CSV per acquisition,
     shape `(4160 samples, 384 scan lines)`).
   - `EXCEL_PATH` → `Z_volume_selection.csv` (columns `id, Upright_H, Upright_S,
     volume, ...`).
   - `SAVE_DIR`   → directory containing `best_model.pt`.
   - `TEST_IDX_PATH` → matching `test_idx.pt` (held-out split; see below).
2. Run:
   ```bash
   cd src
   python test.py
   ```

`test_idx.pt` stores the held-out test indices (as `original_index` values from
the dataframe). It is produced by the training scripts; if you only have the
checkpoint, regenerate the split with `train_reproduce.py` (same seed) to obtain
an identical `test_idx.pt`.

### Option B — retrain from scratch

```bash
cd src
python train_reproduce.py     # writes best_model.pt + test_idx.pt to its SAVE_DIR
python test.py                # point SAVE_DIR/TEST_IDX_PATH at that folder
```

Recovered training configuration (Table II model):

| Hyperparameter | Value |
|---|---|
| learning rate | 1e-3 |
| weight decay (AdamW) | 1e-3 |
| dropout | 0.3 |
| Huber δ | 10.0 |
| augmentation repeat factor | 3 (×2 for volume ≤150 or ≥600 mL) |
| epochs | 100 |
| batch size | 16 |
| split | stratified 60/20/20 by 5 volume bins, seed 42 |

> Exact bit-for-bit reproduction can vary with library versions, GPU
> nondeterminism, and the stochastic channel-selection augmentation. For an
> exact match, use the released checkpoint (Option A).

## Data format

- **RF CSV**: one file per acquisition, numeric matrix of shape
  `(4160 depth samples, 384 scan lines)`. Files are matched by the
  `Upright_H` / `Upright_S` timestamp prefixes listed in the index CSV.
- **Index CSV** (`Z_volume_selection.csv`): one row per patient with columns
  `id`, `Upright_H`, `Upright_S`, `volume` (ground-truth catheterized volume in
  mL), and related fields.

Preprocessing (`dataloader.py`): Hilbert envelope → Gaussian smoothing →
downsample → length normalization → per-acquisition standardization, followed
by grouped random channel selection (3 of the H lines, 4 of the S lines) that
also serves as augmentation.

## Citation

```bibtex
@article{kim_rf_bladder,
  title   = {Lightweight Domain-Adaptive Deep Learning for Bladder Volume
             Estimation via Direct RF Signal Processing},
  author  = {Kim, Minseong and others},
  year    = {2026},
  note    = {Preprint / under review}
}
```

## License

Code released under the MIT License (see `LICENSE`). The trained weights and any
derived clinical data are subject to the data-use restrictions described above.
