import torch
import torch.nn as nn

class FeatureFusionModule(nn.Module):
    def __init__(self, fusion_mode='gated', d_proj=128):
        super(FeatureFusionModule, self).__init__()
        self.fusion_mode = fusion_mode
        self.d_proj = d_proj
        
        if fusion_mode == 'none':
            return
            
        # 1. Projections
        self.classical_proj = nn.Linear(40, d_proj)
        self.plm_proj = nn.Linear(1280, d_proj)
        
        self.classical_ln = nn.LayerNorm(d_proj)
        self.plm_ln = nn.LayerNorm(d_proj)
        
        # 2. Fusion variant components
        if fusion_mode == 'gated':
            self.gate_linear = nn.Linear(2 * d_proj, 1)
        elif fusion_mode == 'cross_attn':
            # Single-head cross-attention, classical_proj as query, plm_proj as key/value
            self.cross_attention = nn.MultiheadAttention(embed_dim=d_proj, num_heads=1, batch_first=True)
            self.post_attn_ln = nn.LayerNorm(d_proj)
        elif fusion_mode == 'concat':
            pass
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
            
        # Initialize projections with standard gain
        nn.init.xavier_uniform_(self.classical_proj.weight, gain=1.0)
        nn.init.zeros_(self.classical_proj.bias)
        nn.init.xavier_uniform_(self.plm_proj.weight, gain=1.0)
        nn.init.zeros_(self.plm_proj.bias)
        
        if fusion_mode == 'gated':
            nn.init.xavier_uniform_(self.gate_linear.weight, gain=1.0)
            nn.init.zeros_(self.gate_linear.bias)
        elif fusion_mode == 'cross_attn':
            pass
            
    def forward(self, classical_i, plm_i, edges=None):
        if torch.isnan(classical_i).any() or torch.isnan(plm_i).any():
            print(f"[DEBUG] NaN detected in raw FeatureFusionModule inputs! classical: {torch.isnan(classical_i).any()}, plm: {torch.isnan(plm_i).any()}")
            
        # Project both to shared dim d
        c_proj = self.classical_ln(self.classical_proj(classical_i))  # (N, d)
        p_proj = self.plm_ln(self.plm_proj(plm_i))                  # (N, d)
        
        if torch.isnan(c_proj).any() or torch.isnan(p_proj).any():
            print(f"[DEBUG] NaN detected in projected streams! c_proj: {torch.isnan(c_proj).any()}, p_proj: {torch.isnan(p_proj).any()}")
            
        gate_val = None
        if self.fusion_mode == 'concat':
            fused_i = torch.cat([c_proj, p_proj], dim=-1)  # (N, 2d)
        elif self.fusion_mode == 'gated':
            # g_i = sigmoid(Linear(concat(classical_proj, plm_proj)))
            concat_proj = torch.cat([c_proj, p_proj], dim=-1)  # (N, 2d)
            gate_val = torch.sigmoid(self.gate_linear(concat_proj))  # (N, 1)
            if torch.isnan(gate_val).any():
                print("[DEBUG] NaN detected in gate_val!")
            fused_i = gate_val * c_proj + (1.0 - gate_val) * p_proj  # (N, d)
        elif self.fusion_mode == 'cross_attn':
            L = c_proj.size(0)
            device = c_proj.device
            
            # Construct 3D spatial attention mask: shape (L, L)
            # True means masked out (no attention), False means allowed to attend.
            attn_mask = torch.ones((L, L), dtype=torch.bool, device=device)
            # A residue can always attend to itself (self-attention)
            attn_mask.fill_diagonal_(False)
            
            if edges is not None and edges.numel() > 0:
                # Allow attention between spatially neighboring residues
                attn_mask[edges[0], edges[1]] = False
                attn_mask[edges[1], edges[0]] = False
            
            # Reshape query and key/value to (1, L, d) where Batch Size = 1, Sequence Length = L
            q = c_proj.unsqueeze(0)    # (1, L, d)
            kv = p_proj.unsqueeze(0)   # (1, L, d)
            
            # attn_output shape: (1, L, d)
            attn_output, _ = self.cross_attention(q, kv, kv, attn_mask=attn_mask)
            attn_output = attn_output.squeeze(0)  # (L, d)
            
            # Residual connection + LayerNorm
            fused_i = self.post_attn_ln(c_proj + attn_output)  # (L, d)
            
        return fused_i, gate_val


class MultiStreamFusionModule(nn.Module):
    """Multi-Stream Interaction Fusion Module (MSF).
    Replaces scalar residue-level convex combination with a learned interaction MLP
    and residual connection:
        interaction = LayerNorm(Linear128->128(GELU(Linear256->128(concat(c_proj, p_proj)))))
        fused = LayerNorm(interaction + c_proj + p_proj)
    Provides an internal confidence signal for FG-Curriculum compatibility:
        confidence = Sigmoid(Linear128->1(fused))
    """
    def __init__(self, d_proj=128):
        super(MultiStreamFusionModule, self).__init__()
        self.d_proj = d_proj
        
        # 1. Projections
        self.classical_proj = nn.Linear(40, d_proj)
        self.plm_proj       = nn.Linear(1280, d_proj)
        
        self.classical_ln   = nn.LayerNorm(d_proj)
        self.plm_ln         = nn.LayerNorm(d_proj)
        
        # 2. Multi-Stream Interaction MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
            nn.LayerNorm(d_proj)
        )
        
        # 3. Residual LayerNorm
        self.fused_ln = nn.LayerNorm(d_proj)
        
        # 4. Confidence Head for FG-Curriculum compatibility
        self.confidence_head = nn.Linear(d_proj, 1)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.classical_proj.weight, gain=1.0)
        nn.init.zeros_(self.classical_proj.bias)
        nn.init.xavier_uniform_(self.plm_proj.weight, gain=1.0)
        nn.init.zeros_(self.plm_proj.bias)
        
    def forward(self, classical_i, plm_i, edges=None):
        # Project both streams to shared dim d
        c_proj = self.classical_ln(self.classical_proj(classical_i.float()))  # (N, d)
        p_proj = self.plm_ln(self.plm_proj(plm_i.float()))                  # (N, d)
        
        # Concatenate projected streams
        concat_cp = torch.cat([c_proj, p_proj], dim=-1)  # (N, 2d)
        
        # Interaction MLP
        interaction = self.interaction_mlp(concat_cp)   # (N, d)
        
        # Residual fusion
        fused_i = self.fused_ln(interaction + c_proj + p_proj)  # (N, d)
        
        # Confidence signal for FG-Curriculum
        confidence = torch.sigmoid(self.confidence_head(fused_i))  # (N, 1)
        
        return fused_i, confidence
