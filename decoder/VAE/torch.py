"""VAE-декодер: латентный вектор z -> последовательность эмбеддингов.

Контракт (его ожидают train_vae.py и run_vae.py):
    class Decoder(nn.Module):
        __init__(self, d_model, latent_dim, max_len, n_layers, n_heads)
        forward(z, length=None) -> (B, length or max_len, d_model)
            z  (B, latent_dim)

Декодер не авторегрессионный: все позиции получаются за один проход из одного z.
Перевод эмбеддингов в токены/логиты и семплирование z делает train (run).
"""
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, d_model=512, latent_dim=256, max_len=64, n_layers=4, n_heads=8,
                 ff_mult=4, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.from_z = nn.Linear(latent_dim, d_model)
        # обучаемые позиционные "запросы": z + pos[i] -> заготовка для i-й позиции
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, z, length=None):
        L = length or self.max_len
        if L > self.max_len:
            raise ValueError(f"длина {L} больше max_len={self.max_len}")
        h = self.from_z(z).unsqueeze(1) + self.pos[:, :L]               # (B, L, d)
        return self.norm_out(self.blocks(h))

