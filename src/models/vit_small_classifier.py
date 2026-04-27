"""ViT-Small binary classifier.

The backbone is a self-contained ViT-S/16 (depth=12, embed_dim=384, heads=6,
mlp_ratio=4) with a [CLS] head. The class also exposes an optional
training-time random patch masking knob (``mask_ratio``); it is disabled by
default (``mask_ratio=0.0``).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def random_patch_mask(
    patch_tokens: torch.Tensor, mask_ratio: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop a `mask_ratio` fraction of patch tokens, per-sample.

    Returns the kept tokens (B, K, D) and a boolean mask (B, N) where True
    marks tokens that were *dropped*.
    """
    batch, num_tokens, _ = patch_tokens.shape
    num_keep = max(1, int(round(num_tokens * (1.0 - mask_ratio))))

    noise = torch.rand(batch, num_tokens, device=patch_tokens.device, generator=generator)
    shuffled = torch.argsort(noise, dim=1)
    keep_idx = shuffled[:, :num_keep]

    gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
    kept = torch.gather(patch_tokens, dim=1, index=gather_idx)

    dropped_mask = torch.ones(batch, num_tokens, dtype=torch.bool, device=patch_tokens.device)
    dropped_mask.scatter_(1, keep_idx, False)
    return kept, dropped_mask


class ViTSmallRandomMaskClassifier(nn.Module):
    """ViT-Small backbone with [CLS] head and an optional random-masking knob."""

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                nn.init.normal_(module.weight, std=math.sqrt(2.0 / fan_in))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        apply_masking: bool | None = None,
    ) -> torch.Tensor:
        if apply_masking is None:
            apply_masking = self.training and self.mask_ratio > 0.0

        patch_tokens = self.patch_embed(x)
        patch_pos = self.pos_embed[:, 1:, :]
        patch_tokens = patch_tokens + patch_pos

        if apply_masking:
            patch_tokens, _ = random_patch_mask(patch_tokens, self.mask_ratio)

        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, :1, :]
        tokens = torch.cat([cls, patch_tokens], dim=1)
        tokens = self.pos_drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        return self.head(tokens[:, 0]).squeeze(-1)
