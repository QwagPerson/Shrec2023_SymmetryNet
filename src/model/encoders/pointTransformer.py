import torch
from point_transformer_pytorch import PointTransformerLayer

attn = PointTransformerLayer(
    dim=3,
    pos_mlp_hidden_dim=64,
    attn_mlp_hidden_mult=4
)

feats = torch.randn(1, 16, 3)
pos = torch.randn(1, 16, 3)
mask = torch.ones(1, 16).bool()

out = attn(feats, pos, mask=mask)  # (1, 16, 128)
print(out.shape)
