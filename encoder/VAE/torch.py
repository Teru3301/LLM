"""VAE-энкодер: последовательность эмбеддингов -> параметры VAE (mu, logvar).

Контракт (его ожидают train_vae.py и run_vae.py):
    class Encoder(nn.Module):
        __init__(self, d_model, latent_dim, max_len, n_layers, n_heads)
        forward(x, pad_mask=None) -> (mu, logvar)
            x         (B, L, d_model), L <= max_len
            pad_mask  (B, L) bool, True = позиция-заполнитель (игнорируется)
            mu, logvar  (B, latent_dim)

Энкодер ничего не знает ни о токенах, ни о лоссе, ни о сэмплировании z -
этим занимается train.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_model=512, latent_dim=256, max_len=64, n_layers=4, n_heads=8,
                 ff_mult=4, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        # обучаемый CLS-токен: его выход после трансформера = сжатое представление всей последовательности
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

        self.norm_in = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d_model)
        self.to_stats = nn.Linear(d_model, 2 * latent_dim)

    def forward(self, x, pad_mask=None):
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"длина {L} больше max_len={self.max_len}")
        h = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)          # (B, L+1, d)
        h = self.norm_in(h + self.pos[:, : L + 1])
        if pad_mask is not None:
            pad_mask = torch.cat([pad_mask.new_zeros(B, 1), pad_mask], dim=1)   # CLS не маскируется
        h = self.blocks(h, src_key_padding_mask=pad_mask)
        mu, logvar = self.to_stats(self.norm_out(h[:, 0])).chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)
