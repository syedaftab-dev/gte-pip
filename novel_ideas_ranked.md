# Novel Publishable Ideas for GTE-PPIS — Ranked by Feasibility & Novelty

> Based on your current codebase (`fusion_module.py`, `final_model.py`, `EGNN_model.py`, `GraphTransformer_Block.py`, `train.py`, `test.py`, `loss.py`) and extensive literature search (August 2026).

---

## Summary Table

| Rank | Idea | Novelty Confidence | Implementation Effort | Publication Strength |
|:---:|:---|:---:|:---:|:---:|
| **1** | Biophysics-Supervised Gate (RSA auxiliary loss on gate) | ★★★★★ | Low (1–2 days) | High |
| **2** | Branch-Disagreement Regularization (EGNN ↔ GT consistency loss) | ★★★★☆ | Low (1 day) | High |
| **3** | Residue-Level Curriculum Learning via Focal-Gate Coupling | ★★★★★ | Medium (2–3 days) | High |
| **4** | Depth-Adaptive PLM Injection (multi-layer ESM-2 fusion) | ★★★★☆ | Medium (2–3 days) | Medium–High |
| **5** | Gate-Entropy Regularization for Interpretable Feature Selection | ★★★☆☆ | Low (1 day) | Medium |

---

## Idea 1 — Biophysics-Supervised Gating (★★★★★ Novelty)

### What
Add an **auxiliary loss** that teaches the FFM gate value $g_i$ to correlate with a known biophysical property — specifically **Relative Solvent Accessibility (RSA)**.

### Why it's novel
- **No published PPIS paper** applies an auxiliary supervision signal directly to a learned modality-fusion gate.
- DHEG uses gates for intra-layer state fusion but never supervises them with biophysical labels.
- Multi-task PPIS papers (GPSite, SaMCL, MPBind) predict RSA as a **separate head**, but never use RSA as a **gate supervision target** to control modality selection.
- Your data already has RSA in `node_features[:, 11]` and you're already saving gate values to CSV — you have everything you need.

### Biological justification
Interface residues are predominantly **surface-exposed** (high RSA). The gate $g_i$ controls how much classical evolutionary info vs. PLM contextual info to use. Surface residues should rely more on PLM context (which captures surface evolutionary pressure better), while buried residues should rely on classical PSSM/HMM features (which capture core conservation). A soft auxiliary loss:

$$\mathcal{L}_{\text{gate}} = \lambda_g \cdot \text{MSE}\left(g_i, \; 1 - \text{RSA}_i\right)$$

teaches the gate to use classical features ($g_i \to 1$) for buried residues (RSA ≈ 0) and PLM features ($g_i \to 0$) for surface residues (RSA ≈ 1).

### Implementation in your code
```python
# In final_model.py forward():
self.last_gate_val = gate_val  # already exists
self.last_rsa = node_features[:, 11]  # already available

# In train.py loss computation:
rsa_target = 1.0 - rsa_values  # invert: buried → 1, surface → 0
gate_loss = F.mse_loss(gate_val.squeeze(), rsa_target)
total_loss = focal_loss + lambda_gate * gate_loss
```

### Literature search confirmation
Searched: "protein-protein interaction site" + "gate supervision" / "gate regularization" / "biophysically-supervised gate" / "solvent accessibility label guided gating" — **No published results found** that apply RSA-supervised gating in any PPIS method (2024–2026).

### Publication angle
*"We propose the first biophysically-supervised modality gate for protein interaction site prediction, where the fusion gate between handcrafted evolutionary features and PLM embeddings is explicitly guided by solvent accessibility, enforcing the biophysical prior that surface-exposed residues should preferentially leverage contextual PLM representations."*

---

## Idea 2 — Branch-Disagreement Regularization (★★★★☆ Novelty)

### What
Add a **consistency loss** between the EGNN branch output $x_1$ and the GT branch output $x_2$ to regularize training:

$$\mathcal{L}_{\text{agree}} = \lambda_a \cdot \text{KL}\!\left(\sigma(x_1) \;\|\; \sigma(x_2)\right)$$

Currently your model simply averages: `x = (x1 + x2) / 2`. This wastes the dual-branch structure — neither branch is encouraged to agree or complement the other.

### Why it's novel
- GTE-PPIS (Wang et al., 2025) uses naive averaging of the two branches.
- AEG-PPIS (2024) uses a similar dual-branch but no inter-branch loss.
- M3Site (2025) uses "Adaptive Weighted Fusion" but no consistency regularization.
- **No PPIS paper** applies KL-divergence or mutual-information-based branch-agreement loss between an EGNN and a Graph Transformer.
- Deep Mutual Learning (Zhang et al., 2018) exists for vision but has never been applied to dual-branch EGNN+GT for residue-level protein tasks.

### Implementation in your code
```python
# In final_model.py forward():
x1 = self.Egnn(...)  # shape (N, 2)
x2 = self.GT(...)    # shape (N, 2)
self.branch_kl = F.kl_div(
    F.log_softmax(x1, dim=1), F.softmax(x2, dim=1), reduction='batchmean'
)
x = (x1 + x2) / 2
return x

# In train.py:
total_loss = focal_loss + lambda_agree * model.branch_kl
```

### Publication angle
*"We introduce branch-disagreement regularization for dual-pathway geometric+topological protein graph architectures, demonstrating that enforcing soft prediction consistency between the equivariant and transformer branches acts as an implicit ensemble regularizer."*

---

## Idea 3 — Residue-Level Curriculum Learning via Focal-Gate Coupling (★★★★★ Novelty)

### What
Use the **gate value** $g_i$ and the **Focal Loss per-residue difficulty** together to implement a **curriculum**: in early epochs, focus on "easy" residues (where the gate is confident, i.e., $g_i \approx 0$ or $g_i \approx 1$); in later epochs, shift focus to "hard" residues (where $g_i \approx 0.5$, indicating the gate is uncertain about which modality to trust).

### Why it's novel
- Curriculum learning exists in DTI prediction (ESP-DTI, AAAI 2026) but uses **sample-level** difficulty scheduling, not **residue-level** gate-derived difficulty.
- **No PPIS paper** uses the fusion gate's uncertainty as a curriculum signal.
- This couples two existing mechanisms (Focal Loss + gate) in a novel way that requires **zero new parameters**.

### Implementation in your code
```python
# Schedule: epoch_weight = min(1.0, epoch / warmup_epochs)
# Per-residue difficulty: d_i = 1 - |2*g_i - 1| (peaks at g_i=0.5)
# Focal gamma modulated per residue: gamma_eff = gamma * (1 - epoch_weight*d_i)
```

### Publication angle
*"We propose gate-uncertainty curriculum learning, a zero-parameter training schedule that leverages the modality-fusion gate's entropy as a per-residue difficulty signal, progressively exposing the model to residues where the optimal feature modality is ambiguous."*

---

## Idea 4 — Depth-Adaptive PLM Injection (★★★★☆ Novelty)

### What
Instead of injecting ESM-2 embeddings only at the **input** (before message passing), inject them at **multiple EGNN layers** with learned per-layer gates. ESM-2's 33 transformer layers capture different linguistic scales — early layers capture local motifs, later layers capture global context. Inject layer-$k$ ESM-2 hidden states into EGNN layer $k$.

### Why it's novel
- DHEG, GTE-PPIS, EGCPPIS, AEG-PPIS all inject PLM embeddings at the **input stage only**.
- No PPIS paper performs **depth-wise PLM injection** into an EGNN.
- This is analogous to FPN (Feature Pyramid Networks) in vision but applied to EGNN+PLM.

### Implementation
- Requires saving intermediate ESM-2 hidden states during `generate_esm2_embeddings.py` (instead of just the last layer).
- Add a per-layer gate in `EGNN_model.py` that mixes the current EGNN hidden state with the corresponding ESM-2 intermediate representation.
- **Effort**: Medium — need to modify embedding generation + EGNN forward pass.

### Caveat
Increases storage (10 layers × 1280d per residue). Can mitigate by PCA-compressing intermediate ESM-2 states to 128d.

### Publication angle
*"We propose depth-adaptive PLM injection, where intermediate representations from different ESM-2 transformer layers are fused into corresponding EGNN message-passing layers via per-layer gates, enabling the geometric network to access scale-appropriate linguistic context at each processing depth."*

---

## Idea 5 — Gate-Entropy Regularization for Interpretable Feature Selection (★★★☆☆ Novelty)

### What
Add an entropy penalty on the gate distribution to push it toward binary decisions:

$$\mathcal{L}_{\text{ent}} = -\lambda_e \sum_i \left[g_i \log g_i + (1-g_i)\log(1-g_i)\right]$$

This forces the gate to **decide** — either classical or PLM — rather than averaging, making the fusion interpretable and potentially improving performance.

### Why it's less novel
- Gate entropy/sparsity regularization is a known technique in Mixture-of-Experts (MoE) literature and general multi-modal fusion.
- However, it has **not been applied to PPIS modality fusion gates** specifically.
- Weaker novelty than Ideas 1–3 because the technique itself is not new, only the application domain.

### Implementation
Trivial — 3 lines in `train.py`.

### Publication angle
*"We apply gate-entropy regularization to encourage binary modality selection at the residue level, revealing that interface residues exhibit systematically different modality preferences compared to non-interface residues."*

---

## Recommended Strategy for Publication

> [!IMPORTANT]
> **Combine Ideas 1 + 2 as a single paper.** This gives you two genuinely novel contributions:
> 1. **Biophysics-supervised gating** (Idea 1) — a new loss formulation that is specific to protein biophysics and has zero precedent in PPIS.
> 2. **Branch-disagreement regularization** (Idea 2) — a new training objective for the dual-branch EGNN+GT architecture class.
>
> Together, these transform a "feature engineering extension" (adding ESM-2) into a **methodological contribution** (how to train modality gates with biophysical priors + how to regularize dual-branch architectures).

### Ablation study for the combined paper

| Condition | Gate supervision | Branch consistency | Expected contribution |
|:---|:---:|:---:|:---|
| Baseline (`none`) | ✗ | ✗ | GTE-PPIS original |
| +ESM-2 concat | ✗ | ✗ | Naive fusion baseline |
| +ESM-2 gated | ✗ | ✗ | Your current FFM |
| +ESM-2 gated + RSA gate loss | ✔ | ✗ | Idea 1 isolated |
| +ESM-2 gated + branch KL | ✗ | ✔ | Idea 2 isolated |
| +ESM-2 gated + RSA gate loss + branch KL | ✔ | ✔ | **Full proposed method** |

This is a clean 6-row ablation table that any reviewer would accept.

### Target venues
- **Briefings in Bioinformatics** (where DHEG was published — directly comparable)
- **Bioinformatics** (Oxford, where GTE-PPIS was published)
- **ICML/NeurIPS Workshop on Computational Biology** (for faster turnaround)
- **IEEE BIBM** or **ACM-BCB** (solid computational biology conferences)

---

## What NOT to pursue (based on literature search)

| Idea | Why it's already done |
|:---|:---|
| Partner-aware PPIS | SPPIDER-seq (2026) already does cross-attention partner-conditioned prediction |
| Self-supervised contrastive PPIS | HIPPO (2026), CSGNN (2026), and biology-aware GCL (ICML 2026) cover this |
| Structure-free distillation | GeoARG (2026) already distills structure-aware teacher to sequence-only student |
| Multi-task RSA prediction head | GPSite (2025), MPBind (2025) already do RSA as auxiliary prediction target |

> [!WARNING]
> The key differentiator is: existing papers predict RSA as a **separate auxiliary head**. Idea 1 uses RSA to **supervise the gate itself** — this is architecturally and conceptually distinct. Make sure your paper clearly articulates this difference.
