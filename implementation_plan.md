# Surpass GTE-PPIS Paper Metrics

## Root Cause Analysis: 5 Critical Differences Found

I cloned the original GTE-PPIS repo and compared every line. Here is what's different:

### 🔴 Diff 1: Loss Function — CrossEntropyLoss vs FocalLoss
- **Original paper**: `nn.CrossEntropyLoss()` (standard, unweighted)
- **Your code**: `FocalLoss(gamma=2.0, class_weights=sqrt_ratio)` 
- **Impact**: FocalLoss with class_weights is **double-penalizing** — both the focal term AND the class weights try to handle imbalance, causing training instability.

### 🔴 Diff 2: Learning Rate — 1e-3 vs 1e-4  
- **Original paper**: `lr=LEARNING_RATE` which is `1E-3` from `data_generator.py`
- **Your code**: `lr=1e-4` (hardcoded)
- **Impact**: **10x slower learning**. Combined with early stopping, the model barely trains.

### 🔴 Diff 3: LR Scheduler — ReduceLROnPlateau vs CosineAnnealing
- **Original paper**: `ReduceLROnPlateau(mode='max', factor=0.6, patience=5)` monitoring validation AUPRC
- **Your code**: `CosineAnnealingLR(T_max=50, eta_min=1e-6)`
- **Impact**: Cosine decay is too aggressive for this small dataset.

### 🔴 Diff 4: Early Stopping — None vs patience=15
- **Original paper**: NO early stopping. Runs full 50 epochs. Saves best checkpoint by AUPRC.
- **Your code**: EarlyStopping with patience=15, causing training to stop at epochs 7–10.
- **Impact**: Model gets ~20% of the training the paper model gets.

### 🔴 Diff 5: EGNN — No tanh/normalize vs tanh=True, normalize=True
- **Original paper**: `EGNN(..., attention=True, residual=False)` — no tanh, no normalize
- **Your code**: `EGNN(..., attention=True, residual=False, tanh=True, normalize=True)`
- **Impact**: tanh constrains coordinate updates which may limit model expressiveness.

### 🟡 Diff 6: Threshold Search — starts at 0 vs starts at 1
- **Original paper**: `range(0, 100)` — threshold starts at 0.0
- **Your code**: `range(1, 100)` — threshold starts at 0.01
- **Impact**: Minor, but the paper's MCC/F1 numbers are computed differently.

## Proposed Fix Strategy

### Phase 1: Restore Original Baseline (fusion_mode=none)
Fix all 5 differences to match the original paper exactly when `fusion_mode=none`. This should reproduce the paper's numbers.

### Phase 2: Add ESM-2 + Curriculum on Top
Once baseline matches, add the gated fusion + curriculum learning on top of the correctly-configured model.

### Specific Changes

#### [MODIFY] [final_model.py](file:///home/aftab/gte-zip/final_model.py)
- When `fusion_mode='none'`: use `CrossEntropyLoss()`, `lr=1e-3`, `ReduceLROnPlateau`, NO tanh/normalize on EGNN
- When `fusion_mode='gated'` + `use_curriculum`: keep FocalLoss + curriculum but with `lr=1e-3` and `ReduceLROnPlateau`

#### [MODIFY] [train.py](file:///home/aftab/gte-zip/train.py)  
- Remove EarlyStopping entirely — run all 50 epochs
- Save best model by validation AUPRC (like original)
- Fix threshold search to match original paper: `range(0, 100)`

## Verification Plan
1. Run `fusion_mode=none` → expect AUROC ~0.87, AUPRC ~0.61, MCC ~0.50 (matching paper)
2. Run `fusion_mode=gated --use_curriculum` → expect to surpass paper numbers
3. Run `test.py` on all 3 benchmarks
