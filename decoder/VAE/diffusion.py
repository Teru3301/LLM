"""Диффузионный VAE-декодер: латентный вектор z -> последовательность эмбеддингов.

Интерфейс для run_vae.py и будущей рекуррентной модели тот же, что у decoder.py:
    forward(z, length=None) -> (B, L, d_model)      # DDIM-сэмплирование из шума

Дополнительно (по наличию метода `denoise` train_vae.py включает диффузионный режим):
    space(e)            эмбеддинги токенов -> пространство диффузии (нормировка на RMS = 1)
    sample_t(B, dev)    случайные моменты времени t in (0, 1]
    q_sample(x0, t, n)  прямой процесс: x_t = sqrt(ab) * x0 + sqrt(1 - ab) * n
    denoise(x_t, t, z)  предсказание чистого x0 (x-prediction), (B, L, d_model)

Зачем нормировка: у обучаемой таблицы эмбеддингов нет фиксированного масштаба, а шум
единичный. Без нормировки таблица может "сжаться", и MSE станет тривиальным.
Нормированные векторы имеют RMS = 1, как и шум; величину и направление слов дальше
задаёт логит-проекция `emb.logits` (в ней остаётся масштаб самой таблицы).

Расписание шума - косинусное (Nichol & Dhariwal), сдвинутое в сторону сильного шума:
SNR делится на `snr_shift`. Это принципиально: слово из 512 чисел читается из зашумлённого
вектора даже при очень малом SNR, и при обычном расписании на большей части t декодер
восстанавливает токены сам по x_t и игнорирует z (KL схлопывается к ~0). Со сдвигом на
значительной доле t токены нечитаемы, и без z не обойтись. Сэмплирование - детерминированный
DDIM (eta=0) из N(0, I); число шагов - атрибут `n_steps` (можно менять после загрузки).
forward() выполняется под no_grad; при обучении train вызывает только `denoise`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """t (B,) in [0, 1] -> (B, dim) синусоидальное представление."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return F.pad(emb, (0, dim - emb.size(-1)))            # если dim нечётный


class Decoder(nn.Module):
    def __init__(self, d_model=512, latent_dim=256, max_len=64, n_layers=4, n_heads=8,
                 ff_mult=4, dropout=0.1, n_steps=32, snr_shift=32.0):
        super().__init__()
        self.d_model, self.max_len, self.n_steps = d_model, max_len, n_steps
        self.snr_shift = snr_shift

        self.from_z = nn.Linear(latent_dim, d_model)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        # позиции: 0 - токен условия (z + t), 1..L - шумные эмбеддинги
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------ прямой процесс
    def space(self, e):
        return F.normalize(e, dim=-1) * math.sqrt(self.d_model)

    def sample_t(self, batch, device):
        return torch.rand(batch, device=device).clamp(min=1e-3)

    def alpha_bar(self, t, s=0.008):
        """Косинусное расписание со сдвигом SNR: ab(0) ~ 1, ab(1) ~ 0 (с зажимом для устойчивости)."""
        f = lambda u: torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        ab = (f(t) / math.cos(s / (1 + s) * math.pi / 2) ** 2).clamp(1e-6, 1 - 1e-6)
        snr = ab / (1 - ab) / self.snr_shift
        return (snr / (1 + snr)).clamp(1e-7, 1 - 1e-5)

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar(t)[:, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    # ------------------------------------------------------------ обратный процесс
    def denoise(self, xt, t, z):
        """(x_t, t, z) -> предсказание x0. xt (B, L, d_model), t (B,), z (B, latent_dim)."""
        B, L, _ = xt.shape
        if L > self.max_len:
            raise ValueError(f"длина {L} больше max_len={self.max_len}")
        c = self.from_z(z) + self.t_mlp(timestep_embedding(t, self.d_model))       # (B, d)
        h = torch.cat([c.unsqueeze(1), self.in_proj(xt)], dim=1)                  # (B, L+1, d)
        h = h + self.pos[:, : L + 1] + c.unsqueeze(1)
        h = self.blocks(h)[:, 1:]                                                  # убрать токен условия
        return self.out(self.norm_out(h))

    @torch.no_grad()
    def forward(self, z, length=None):
        L = length or self.max_len
        B = z.size(0)
        x = torch.randn(B, L, self.d_model, device=z.device, dtype=z.dtype)
        ts = torch.linspace(1.0, 0.0, self.n_steps + 1, device=z.device)
        for i in range(self.n_steps):
            x0 = self.denoise(x, ts[i].expand(B), z)
            if i == self.n_steps - 1:
                return x0
            ab, ab_next = self.alpha_bar(ts[i]), self.alpha_bar(ts[i + 1])
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
