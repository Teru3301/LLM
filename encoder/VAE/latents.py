"""VAE-энкодер с несколькими латентными токенами: эмбеддинги -> (mu, logvar).

Отличие от torch.py: вместо одного CLS в последовательность добавляются `n_latents`
обучаемых латентных токенов. Каждый собирает свою часть смысла и отдаёт свои mu/logvar
размером latent_dim // n_latents. Наружу они выходят склеенными в один вектор (B, latent_dim),
поэтому контракт прежний, а декодер (decoder/VAE/ar.py) раскладывает его обратно на
n_latents векторов для cross-attention.

Один вектор на 64 токена - узкое горлышко: вся последовательность должна пройти через
одну позицию attention. С несколькими латентными токенами энкодер распределяет информацию,
а декодер может смотреть на разные её части.

Контракт (его ожидают train_vae.py и run_vae.py):
    class Encoder(nn.Module):
        __init__(self, d_model, latent_dim, max_len, n_layers, n_heads)
        forward(x, pad_mask=None) -> (mu, logvar)
            x         (B, L, d_model), L <= max_len
            pad_mask  (B, L) bool, True = позиция-заполнитель (игнорируется)
            mu, logvar  (B, latent_dim)

n_latents должен совпадать с декодером (по умолчанию 16 у обоих).
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_model=512, latent_dim=256, max_len=64, n_layers=4, n_heads=8,
                 n_latents=16, ff_mult=4, dropout=0.1):
        super().__init__()
        if latent_dim % n_latents:
            raise ValueError(f"latent_dim={latent_dim} не делится на n_latents={n_latents}")
        self.max_len, self.n_latents = max_len, n_latents
        self.d_lat = latent_dim // n_latents

        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.norm_in = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d_model)
        self.to_stats = nn.Linear(d_model, 2 * self.d_lat)
        # старт с малой дисперсией (sigma ~ 0.05): в начале z почти детерминирован и несёт
        # информацию. При logvar ~ 0 сэмплы z были бы чистым шумом, и авторегрессионный декодер
        # быстрее научился бы обходиться без z (коллапс), чем энкодер - что-то в него класть.
        # Шум и гладкость приходят позже, по мере роста веса KL.
        nn.init.normal_(self.to_stats.weight, std=0.02)
        nn.init.zeros_(self.to_stats.bias)
        with torch.no_grad():
            self.to_stats.bias[self.d_lat:] = -6.0

    def forward(self, x, pad_mask=None):
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"длина {L} больше max_len={self.max_len}")
        K = self.n_latents
        h = torch.cat([self.latents.expand(B, -1, -1), x + self.pos[:, :L]], dim=1)   # (B, K+L, d)
        h = self.norm_in(h)
        if pad_mask is not None:
            pad_mask = torch.cat([pad_mask.new_zeros(B, K), pad_mask], dim=1)        # латенты не маскируются
        h = self.blocks(h, src_key_padding_mask=pad_mask)[:, :K]                    # (B, K, d)
        mu, logvar = self.to_stats(self.norm_out(h)).chunk(2, dim=-1)              # (B, K, d_lat)
        return mu.reshape(B, -1), logvar.clamp(-10.0, 10.0).reshape(B, -1)
