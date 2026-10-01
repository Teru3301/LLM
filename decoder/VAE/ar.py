"""Авторегрессионный VAE-декодер: z -> последовательность эмбеддингов (по токену за шаг).

Почему авторегрессия. Латент нужен как промежуточное представление: рекуррентная модель
будет выдавать z, которые неизбежно чуть отличаются от выходов энкодера. Непараллельный
(one-shot) декодер предсказывает каждую позицию независимо, поэтому неточный z превращается
в набор слов без связи. Авторегрессионный декодер видит уже сгенерированный префикс, и из
соседних z получается связный текст: это и есть гладкость латентного пространства.

Цена - риск "коллапса": декодер может научиться предсказывать текст по префиксу и
игнорировать z. От этого защищают:
  * word dropout: train заменяет часть входных токенов на MASK, и декодер вынужден брать
    недостающее из z (--word_drop);
  * два пути от z: cross-attention на n_latents латентных токенов + добавка z к каждой позиции;
  * KL-annealing и free bits в train.
train печатает CE с родным z и с z от чужого примера: если они близки, декодер игнорирует z.

Контракт (как у decoder/VAE/torch.py):
    __init__(self, d_model, latent_dim, max_len, n_layers, n_heads)
    forward(z, length=None) -> (B, L, d_model)     # жадная генерация; argmax(emb.logits) = токены

Дополнительно (по наличию `teacher` train_vae.py включает авторегрессионный режим):
    attach(emb)       дать декодеру таблицу эмбеддингов (нужна для генерации; в state_dict не попадает)
    teacher(z, x)     teacher forcing: x (B, L, d_model) - эмбеддинги целевых токенов (их может
                      испортить word dropout); выход i-й позиции предсказывает i-й токен.

n_latents должен совпадать с энкодером (по умолчанию 16 у обоих).
"""
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, d_model=512, latent_dim=256, max_len=64, n_layers=4, n_heads=8,
                 n_latents=16, ff_mult=4, dropout=0.1):
        super().__init__()
        if latent_dim % n_latents:
            raise ValueError(f"latent_dim={latent_dim} не делится на n_latents={n_latents}")
        self.max_len, self.n_latents = max_len, n_latents
        self.d_lat = latent_dim // n_latents

        # z -> память для cross-attention (n_latents векторов) и глобальная добавка ко входу
        self.mem_proj = nn.Linear(self.d_lat, d_model)
        self.mem_pos = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.mem_norm = nn.LayerNorm(d_model)
        self.glob = nn.Linear(latent_dim, d_model)

        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(layer, n_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.register_buffer(
            "causal", torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
            persistent=False)

    # ------------------------------------------------------------ служебное
    def attach(self, emb):
        """Ссылка на эмбеддер без регистрации подмодулем (его веса хранит train отдельно)."""
        self.__dict__["_emb"] = emb

    def _memory(self, z):
        B = z.size(0)
        mem = self.mem_proj(z.view(B, self.n_latents, self.d_lat)) + self.mem_pos
        return self.mem_norm(mem), self.glob(z).unsqueeze(1)

    def _run(self, inp, mem, g):
        """inp (B, L, d): BOS + эмбеддинги предыдущих токенов -> выходные векторы (B, L, d)."""
        L = inp.size(1)
        if L > self.max_len:
            raise ValueError(f"длина {L} больше max_len={self.max_len}")
        h = inp + self.pos[:, :L] + g
        h = self.blocks(h, mem, tgt_mask=self.causal[:L, :L], tgt_is_causal=True)
        return self.norm_out(h)

    # ------------------------------------------------------------ обучение
    def teacher(self, z, x):
        mem, g = self._memory(z)
        inp = torch.cat([self.bos.expand(x.size(0), -1, -1), x[:, :-1]], dim=1)
        return self._run(inp, mem, g)

    # ------------------------------------------------------------ генерация
    @torch.no_grad()
    def forward(self, z, length=None):
        emb = self.__dict__.get("_emb")
        if emb is None:
            raise RuntimeError("перед генерацией вызовите dec.attach(emb)")
        L = length or self.max_len
        B = z.size(0)
        mem, g = self._memory(z)
        inp = self.bos.expand(B, -1, -1)
        outs = []
        for _ in range(L):
            h = self._run(inp, mem, g)[:, -1:]                       # (B, 1, d)
            outs.append(h)
            nxt = emb.logits(h).argmax(-1)                           # (B, 1)
            inp = torch.cat([inp, emb(nxt)], dim=1)
        return torch.cat(outs, dim=1)
