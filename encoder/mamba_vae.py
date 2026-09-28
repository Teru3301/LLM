"""
Энкодер VAE на Mamba.

К эмбеддингам текста в КОНЦЕ добавляются n_slots обучаемых query-токенов. Mamba каузальна,
поэтому query "видят" весь текст; их выходы -> Linear -> (mu, logvar) для каждого слота.
(Не avg-pool по окнам, как в mamba_ae.py: там слот видел бы только своё окно, и латент был бы
упакованным текстом, а не глобальным представлением.)
"""
import torch
import torch.nn as nn

from blocks import MambaStack


class MambaVAEEncoder(nn.Module):
    def __init__(self, d_model=512, d_state_ssm=64, n_slots=16, d_latent=32, n_layers=4):
        super().__init__()
        self.n_slots, self.d_latent = n_slots, d_latent
        self.queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.stack = MambaStack(d_model, d_state_ssm, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.to_stats = nn.Linear(d_model, 2 * d_latent)
        # старт с малой дисперсией постериора (logvar=-4): декодеру проще сразу начать пользоваться z;
        # KL при этом велик, но первые шаги идут с beta=0 (см. train_vae.py)
        self.to_stats.bias.data[d_latent:] = -4.0

    def forward(self, x):
        B = x.size(0)
        h = torch.cat([x, self.queries[None].expand(B, -1, -1).to(x.dtype)], 1)
        h = self.norm(self.stack(h))[:, -self.n_slots:]
        mu, logvar = self.to_stats(h).float().chunk(2, -1)
        return mu, logvar.clamp(-10.0, 2.0)

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
