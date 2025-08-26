import torch
import torch.nn as nn
from functools import partial
from typing import List
import torch.nn.functional as F

# ---------------------------
# Patch Embedding
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x

# ---------------------------
# ViT Block
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = h + self.mlp(x)
        return x

# ---------------------------
# Vision Transformer
# ---------------------------

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 初始化時還是用 224×224 的 pos_embed
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

    def interpolate_pos_encoding(self, x, H, W):
        """
        動態插值位置編碼以支援任意解析度
        x: [B, N+1, C], N=patch數
        H, W: 輸入影像大小
        """
        npatch = x.shape[1] - 1  # 除掉 cls token
        N = self.pos_embed.shape[1] - 1
        if npatch == N:  # 如果剛好符合，就不用插值
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]   # [1, 1, C]
        patch_pos_embed = self.pos_embed[:, 1:]  # [1, N, C]
        dim = x.shape[-1]

        # 原本的 grid size
        w0 = h0 = int(N**0.5)
        # 目前輸入對應的 grid size
        w1 = W // self.patch_size
        h1 = H // self.patch_size

        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h1, w1), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        print("Input:", x.shape)
        x = self.patch_embed(x)  # [B, N, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat((cls_tokens, x), dim=1)

        # 🔑 插值位置編碼
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def get_intermediate_layers(self, x, n=4) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = x + pos_embed

        intermediates = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                intermediates.append(self.norm(x)[:, 1:, :])  # 去掉 cls token
        return intermediates