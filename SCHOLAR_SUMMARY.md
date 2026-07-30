# GTE-PPIS Feature Fusion Module (FFM) Research Summary

This document summarizes the implementation and evaluation setup for the Feature Fusion Module (FFM) in the GTE-PPIS model. It integrates evolutionary/PLM-derived features (ESM-2 embeddings) with classical handcrafted sequence features (PSSM + HMM) via a learned fusion mechanism.

## Specific Gap and Motivation

Prior work (e.g., DHEG, Briefings in Bioinformatics 2026) showed that naive concatenation or direct replacement of PSSM/HMM handcrafted features with ESM-2 embeddings underperforms classical handcrafted feature baselines (with MCC dropping as low as 0.086 for standalone PLMs, and PLM-replacing-PSSM setups losing to baseline). The unresolved question is whether a **learned fusion mechanism**—as opposed to static concatenation or replacement—enables PLM embeddings to contribute a complementary signal.

This work implements and evaluates a Feature Fusion Module (FFM) placed between raw node features and downstream GNN branches (`EGNN` and `GraphTransformer`), specifically isolating evolutionary streams for learned fusion while leaving structural/geometric streams separate and untouched.

---

## Architectural Specification (FFM)

Two input streams per residue $i$ are processed:
1. **`classical_i`**: Handcrafted evolutionary features: PSSM (20d) + HMM (20d) concatenated to form a 40d vector.
2. **`plm_i`**: Pre-extracted ESM-2 (650M) per-residue embeddings (1280d).

*Note: Structural/geometric features (DSSP (14d) and residue Atom Features (resAF) (7d)) are kept separate and passed directly to downstream branches without modification to isolate the impact of evolutionary fusion.*

### Steps and Modes:

1. **Projection**: Both streams are projected into a shared latent dimension $d$ (default: $d=128$, configurable via CLI flag `--d_proj`):
   $$\text{classical\_proj}_i = \text{Linear}(40, d)(\text{classical}_i)$$
   $$\text{plm\_proj}_i = \text{Linear}(1280, d)(\text{plm}_i)$$

2. **Fusion Variants (selectable via `--fusion_mode`)**:
   - **`none`**: Standard classical-only baseline (no PLM features used; fallback to standard 61d node features).
   - **`concat`**: Naive fusion by concatenating the projected streams:
     $$\text{fused}_i = \text{concat}(\text{classical\_proj}_i, \text{plm\_proj}_i) \in \mathbb{R}^{2d}$$
   - **`gated`**: Per-residue scalar gate $g_i \in [0, 1]$ computed via a linear projection of the concatenated features:
     $$g_i = \sigma(\text{Linear}([\text{classical\_proj}_i, \text{plm\_proj}_i]) \to 1)$$
     $$\text{fused}_i = g_i \cdot \text{classical\_proj}_i + (1 - g_i) \cdot \text{plm\_proj}_i \in \mathbb{R}^d$$
   - **`cross_attn`**: Single-head cross-attention treating each residue as a 1-token sequence. The query is $\text{classical\_proj}_i$, and the key/value is $\text{plm\_proj}_i$. The cross-attention output is added residually to $\text{classical\_proj}_i$:
     $$\text{fused}_i = \text{classical\_proj}_i + \text{MultiheadAttention}(\text{query}=\text{classical\_proj}_i, \text{key}=\text{plm\_proj}_i, \text{value}=\text{plm\_proj}_i) \in \mathbb{R}^d$$

3. **Re-concatenation**: The final node feature vector is formed by concatenating the fused representation with the unmodified structural features:
   $$\text{node\_features}_i = \text{concat}(\text{fused}_i, \text{DSSP}_i, \text{AF}_i)$$
   This concatenated representation is fed directly into the existing model downstream (`h0_i` and `e0_ij` linear layers) without changing the network's downstream branches.

---

## File modification directory

### Created Files:
- **[generate_esm2_embeddings.py](file:///home/pranav/GTE-PPIS/generate_esm2_embeddings.py)**: Extracts per-residue embeddings from the `facebook/esm2_t33_650M_UR50D` model (fp16 on GPU, falling back to float32 on CPU if CUDA out-of-memory occurs) and caches them in `./Feature/esm2/{ID}.npy` as float16.
- **[fusion_module.py](file:///home/pranav/GTE-PPIS/fusion_module.py)**: Implements the `FeatureFusionModule` class covering projections and the fusion modes (`none`, `concat`, `gated`, `cross_attn`).

### Modified Files:
- **[data_generator.py](file:///home/pranav/GTE-PPIS/data_generator.py)**: Updates `ProDataset` to dynamically load cached ESM-2 embeddings from disk, perform padding/slicing checks to align sequence lengths, and return them. Updates `graph_collate` to collate and batch the PLM embedding tensors.
- **[final_model.py](file:///home/pranav/GTE-PPIS/final_model.py)**: Integrates the `FeatureFusionModule` into the `FinalModel` pipeline. Dynamically adjusts the expected input dimension for the `EGNN` and `GraphTransformer` branches depending on `--fusion_mode`.
- **[train.py](file:///home/pranav/GTE-PPIS/train.py)**: Adds CLI flags to select the fusion mode, project dimension, and smoke test setup. Updates paths to save checkpoints/logs in mode-specific directories. Performs memory cleanup at the end of each fold to avoid fragmentation.
- **[test.py](file:///home/pranav/GTE-PPIS/test.py)**: Adds CLI flags. Performs testing on all three datasets (`Test_60`, `Test_315-28`, `UBtest_31-6`) across all fold checkpoints. Logs and outputs per-residue gate value metrics (along with site ground truth and DSSP RSA) to a CSV in `gated` mode.

---

## CLI Flag Specifications

The following CLI options were introduced to configure model execution:
- `--fusion_mode`: Sets the FFM variant to use (`none`, `concat`, `gated`, `cross_attn`).
- `--d_proj`: Integer size for the shared project dimension $d$ (default: `128`).
- `--smoke_test`: Bounded validation mode that restricts datasets to 2 samples, runs a single fold, and terminates training after 1 epoch (or tests a single fold model in `test.py`) for rapid verification.

---

## Experiment Status

- **`none` (baseline)**: Passed. Verified on disk; full 5-fold cross-validation training and evaluation completed successfully, with checkpoint models (`Fold1` through `Fold5`) and full results stored in `Log/fusion_none_d128_2026-07-01-09-34-00/`.
- **`concat`**: Passed. Completed training and evaluation smoke test without crashing.
- **`cross_attn`**: Passed. Completed training and evaluation smoke test without crashing. Resolved initial NaN issues by adding LayerNorm to the classical and PLM projection layers, using standard Xavier initialization with a weight gain of 1.0 (an initial attempt to use a scaled-down gain of 0.1 was reverted as it amplified gradients through LayerNorm's inverse variance scaling).
- **`gated`**: Passed. Completed training and evaluation smoke test without crashing (resolved numerical instability in the downstream `EGNN` block by enabling `tanh=True` for coordinate updates to bound messaging offsets).



