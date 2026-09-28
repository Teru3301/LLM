"""
Авторегрессионный декодер на Mamba.

Латент z (n_slots, d_latent) -> Linear -> n_slots токенов памяти ПЕРЕД текстом (+ метка слота)
-> стек Mamba -> голова с общими весами из tokenizer/.
Позиционных эмбеддингов нет: Mamba рекуррентна, порядок знает сама.
"""
import torch
import torch.nn as nn

from blocks import MambaStack
from . import head


class MambaARDecoder(nn.Module):
    def __init__(self, emb, d_model=512, d_state_ssm=64, n_slots=16, d_latent=32, n_layers=4, d_conv=2):
        super().__init__()
        # Ссылка на таблицу из tokenizer/ ХРАНИТСЯ В СПИСКЕ, чтобы nn.Module её не зарегистрировал:
        # иначе её веса попали бы и в чекпойнт токенайзера, и в чекпойнт декодера.
        self._emb = [emb]
        self.n_slots = n_slots
        self.z_proj = nn.Linear(d_latent, d_model)
        self.slot_id = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.stack = MambaStack(d_model, d_state_ssm, n_layers, d_conv=d_conv)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    @property
    def emb(self):
        return self._emb[0]

    def hidden(self, z, dec_in):
        mem = self.z_proj(z.to(self.z_proj.weight.dtype)) + self.slot_id[None]   # (B, n_slots, d)
        x = torch.cat([mem, self.emb(dec_in)], 1)                                 # (B, n_slots+L, d)
        h = self.norm(self.stack(x))[:, self.n_slots:]                            # отбрасываем позиции памяти
        return self.out(h)

    # ---- голова (веса берутся из tokenizer/)
    def logits(self, h):
        return head.logits(h, self.emb.out_weight)

    def predict(self, h):
        return head.predict(h, self.emb.out_weight)

    def ce_sum(self, h, tgt):
        return head.ce_sum(h, tgt, self.emb.out_weight)

    @torch.no_grad()
    def generate(self, z, bos, max_new):
        """Жадная генерация. Простая O(L^2) версия (пересчёт префикса), как в mamba_ae.py."""
        cur = torch.full((z.size(0), 1), bos, dtype=torch.long, device=z.device)
        for _ in range(max_new):
            hid = self.hidden(z, cur)
            nxt = self.logits(hid[:, -1]).argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], 1)
        return cur[:, 1:]
