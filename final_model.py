import torch
import torch.nn as nn
import torch.nn.functional as F

from data_generator import *
from EGNN_model import *
from GraphTransformer_Block import *

from fusion_module import FeatureFusionModule


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification in bioinformatics.
    Optionally coupled with Focal-Gate Curriculum Learning (FG-Curriculum).

    Reference: Lin et al. (2017) 'Focal Loss for Dense Object Detection'.
    Widely adopted in PPI binding-site prediction to handle ~5:1 neg:pos imbalance.

    gamma (focusing parameter): down-weights loss contribution from easy (well-classified)
      negatives, forcing the network to focus training signal on hard positives.
      gamma=2 is the standard value from the original paper.

    class_weights: optional per-class weight tensor [w_neg, w_pos], applied before
      the focal term — combines class-frequency balancing with hard-sample focusing.

    use_curriculum: when True, modulates per-residue loss by gate uncertainty U_i = 1 - |2*g_i - 1|
      scaled by training epoch pacing p_epoch = min(1.0, (epoch + 1) / warmup_epochs).
    """
    def __init__(self, gamma=2.0, class_weights=None, use_curriculum=False, warmup_epochs=15):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights  # shape [C]
        self.use_curriculum = use_curriculum
        self.warmup_epochs = warmup_epochs

    def forward(self, inputs, targets, gate_val=None, epoch=0):
        # inputs: (N, C) logits; targets: (N,) long
        # Per-sample cross-entropy loss (no reduction) with optional class weights
        ce = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        # p_t: probability assigned to the correct class
        p_t = torch.exp(-ce)
        # Focal weight: (1 - p_t)^gamma — approaches 0 for easy samples
        focal_loss = ((1 - p_t) ** self.gamma) * ce

        if self.use_curriculum and gate_val is not None:
            # Gate uncertainty U_i in [0, 1]: 1 when g_i=0.5 (uncertain), 0 when g_i=0 or 1 (certain)
            # CRITICAL: .detach() prevents gradients from flowing back through the curriculum
            # weight into the gate network. Without this, the gate learns to minimize uncertainty
            # as an unintended auxiliary objective, corrupting the learned representations.
            g = gate_val.detach().squeeze()
            U = 1.0 - torch.abs(2.0 * g - 1.0)

            # Pacing parameter p_epoch in [0, 1]
            p_epoch = min(1.0, float(epoch + 1) / float(self.warmup_epochs))

            # Curriculum weight w_i: early in training, down-weight uncertain residues;
            # as epoch -> warmup_epochs, w_i -> 1.0 for all residues.
            w = 1.0 - (1.0 - p_epoch) * U

            return (w * focal_loss).sum() / (w.sum() + 1e-8)
        else:
            return focal_loss.mean()

class FinalModel(nn.Module):
    def __init__(self, input_size, hidden_size, fliter_size, output_size, dropout_rate, n_layers,
                 fusion_mode='none', d_proj=128, class_weights=None, use_curriculum=False, warmup_epochs=15):
        super(FinalModel, self).__init__()
        self.fusion_mode = fusion_mode
        self.d_proj = d_proj
        self.use_curriculum = use_curriculum
        if use_curriculum and fusion_mode != 'gated':
            print("WARNING: --use_curriculum only has effect with --fusion_mode gated. "
                  f"Current fusion_mode='{fusion_mode}' — curriculum will be silently ignored.")

        # Calculate actual input dimension for EGNN and GT branches
        if fusion_mode == 'none':
            self.actual_input_size = input_size  # 61
            self.fusion_module = None
        elif fusion_mode == 'concat':
            self.actual_input_size = 2 * d_proj + 21
            self.fusion_module = FeatureFusionModule(fusion_mode=fusion_mode, d_proj=d_proj)
        elif fusion_mode in ['gated', 'cross_attn']:
            self.actual_input_size = d_proj + 21
            self.fusion_module = FeatureFusionModule(fusion_mode=fusion_mode, d_proj=d_proj)
        else:
            raise ValueError(f"Unknown fusion mode {fusion_mode}")

        self.Egnn = EGNN(in_node_nf=self.actual_input_size, hidden_nf=hidden_size,
                         out_node_nf=output_size, in_edge_nf=2, n_layers=10,
                         attention=True, residual=False, tanh=True, normalize=True)
        self.GT = GraghTransformer(in_channels=self.actual_input_size, edge_features=2,
                                   dropout_rate=dropout_rate, num_layers=4,
                                   transformer_residual=False)

        # Focal Loss: down-weights easy negatives, focusing training signal on hard
        # positive (binding-site) residues. gamma=2 is the standard value.
        # class_weights further compensates for the ~5:1 neg:pos imbalance.
        if class_weights is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            class_weights = class_weights.to(device)
        self.criterion = FocalLoss(gamma=2.0, class_weights=class_weights,
                                   use_curriculum=use_curriculum, warmup_epochs=warmup_epochs)

        # Adam optimizer with stabilized learning rate (1e-4) for attention layers
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)

        # CosineAnnealingLR: smooth LR decay over the full training horizon.
        # Avoids ReduceLROnPlateau getting stuck when AUPRC collapses near 0.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUMBER_EPOCHS, eta_min=1e-6)

    def forward(self, node_features, xyz_feats, edges, edge_att, edge_feat, adj,
                plm_features=None):
        self.last_gate_val = None

        if self.fusion_module is not None and plm_features is not None:
            classical_i = node_features[:, 14:54]
            dssp_i      = node_features[:, 0:14]
            af_i        = node_features[:, 54:61]

            fused_i, gate_val = self.fusion_module(classical_i, plm_features, edges=edges)
            self.last_gate_val = gate_val

            node_features = torch.cat([fused_i, dssp_i, af_i], dim=-1)

        x1 = self.Egnn(node_features, xyz_feats, edges, edge_feat)
        x2 = self.GT(node_features, edge_feat, edges)
        x  = (x1 + x2) / 2
        return x

