import torch
import torch.nn as nn
import torch.nn.functional as F

from data_generator import *
from EGNN_model import *
from GraphTransformer_Block import *

from fusion_module import FeatureFusionModule, MultiStreamFusionModule


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification in bioinformatics.
    Optionally coupled with Focal-Gate Curriculum Learning (FG-Curriculum).
    Label smoothing prevents overconfident predictions on training data.
    """
    def __init__(self, gamma=2.0, class_weights=None, use_curriculum=False, warmup_epochs=15,
                 label_smoothing=0.05, delay_epochs=3):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights  # shape [C]
        self.use_curriculum = use_curriculum
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        # Gap 4: Delay curriculum onset to let the GNN form stable initial representations
        # before uncertain/hard samples are scheduled. First delay_epochs use no curriculum.
        self.delay_epochs = delay_epochs

    def forward(self, inputs, targets, gate_val=None, epoch=0):
        ce = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none',
                             label_smoothing=self.label_smoothing)
        p_t = torch.exp(-ce)
        focal_loss = ((1 - p_t) ** self.gamma) * ce

        if self.use_curriculum and gate_val is not None:
            g = gate_val.detach().squeeze()
            U = 1.0 - torch.abs(2.0 * g - 1.0)
            # Gap 4: Skip curriculum for first delay_epochs — confidence head is untrained then.
            # Without this, near-random g_i≈0.5 → U_i≈1 → all samples near-zero weighted,
            # starving the GNN of gradient in the critical early epochs.
            effective_epoch = max(0, epoch - self.delay_epochs)
            p_epoch = min(1.0, float(effective_epoch + 1) / float(self.warmup_epochs))
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
        elif fusion_mode == 'multistream':
            # Gap 1: GNN sees: fused (d_proj=128) + PSSM/HMM pass-through (40) + DSSP (14) + AF (7) = d_proj+61
            # The raw 40d classical_i is preserved alongside fused features so the GNN
            # backbone always has direct access to PSSM+HMM conservation signals.
            self.actual_input_size = d_proj + 61
            self.fusion_module = MultiStreamFusionModule(d_proj=d_proj)
        else:
            raise ValueError(f"Unknown fusion mode {fusion_mode}")

        self.Egnn = EGNN(in_node_nf=self.actual_input_size, hidden_nf=hidden_size,
                         out_node_nf=output_size, in_edge_nf=2, n_layers=10,
                         attention=True, residual=False, tanh=False, normalize=False)
        self.GT = GraghTransformer(in_channels=self.actual_input_size, edge_features=2,
                                   dropout_rate=dropout_rate, num_layers=4,
                                   transformer_residual=False)

        # Apply class weights for ALL fusion modes to handle imbalance (~16% positives)
        if class_weights is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            class_weights = class_weights.to(device)

        # FocalLoss for all modes: gamma=2 down-weights easy negatives, naturally handles imbalance
        # curriculum is only active for fusion modes that have a confidence gate
        _use_curriculum = use_curriculum and (fusion_mode != 'none')
        self.criterion = FocalLoss(gamma=2.0, class_weights=class_weights,
                                   use_curriculum=_use_curriculum, warmup_epochs=warmup_epochs,
                                   delay_epochs=3)

        # Adam optimizer: lr=1e-4 for all modes
        # 1e-3 caused gradient explosion in epoch 1 (train acc 0.74 → val loss explodes to 2.18)
        # weight_decay=1e-5 adds mild L2 regularization to combat overfitting
        lr = 1e-4
        wd = 1e-5
        # Gap 5: PSSM/HMM projection (40d, only 5,120 params) gets 2× LR vs ESM-2 projection
        # (1280d bottleneck, ~51K params). Without this, larger ESM-2 gradients naturally
        # dominate the optimizer update, further marginalizing the PSSM/HMM signal.
        if fusion_mode == 'multistream' and self.fusion_module is not None:
            classical_param_ids = {id(p) for p in self.fusion_module.classical_proj.parameters()}
            classical_proj_params = list(self.fusion_module.classical_proj.parameters())
            other_params = [p for p in self.parameters() if id(p) not in classical_param_ids]
            self.optimizer = torch.optim.Adam([
                {'params': classical_proj_params, 'lr': 2e-4, 'weight_decay': wd},
                {'params': other_params,          'lr': lr,   'weight_decay': wd}
            ])
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        # ReduceLROnPlateau monitoring validation metric (matching original GTE-PPIS paper)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)

    def forward(self, node_features, xyz_feats, edges, edge_att, edge_feat, adj,
                plm_features=None):
        self.last_gate_val = None

        if self.fusion_module is not None and plm_features is not None:
            classical_i = node_features[:, 14:54]   # (N, 40)  PSSM+HMM
            dssp_i      = node_features[:, 0:14]    # (N, 14)  DSSP (includes RSA in col 4)
            af_i        = node_features[:, 54:61]   # (N, 7)   Atom Features

            fused_i, gate_val = self.fusion_module(classical_i, plm_features, edges=edges)
            self.last_gate_val = gate_val

            # Gap 1: Keep raw classical_i (PSSM+HMM) alongside fused features.
            # The baseline achieves 0.461 MCC by directly seeing PSSM+HMM.
            # Without pass-through, the GNN only sees a diluted ESM-2-dominated fused vector.
            node_features = torch.cat([fused_i, classical_i, dssp_i, af_i], dim=-1)  # (N, d_proj+61)

        x1 = self.Egnn(node_features, xyz_feats, edges, edge_feat)
        x2 = self.GT(node_features, edge_feat, edges)
        x  = (x1 + x2) / 2
        return x

