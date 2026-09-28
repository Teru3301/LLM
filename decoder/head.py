"""Выходная голова с общими (tied) весами: скрытые векторы -> токены. weight = tokenizer.out_weight."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def logits(h, weight):
    return F.linear(h, weight)


@torch.no_grad()
def predict(h, weight, chunk=4096):
    """argmax по словарю без материализации полных логитов (B, L, V)."""
    shape = h.shape[:-1]
    flat = h.reshape(-1, h.size(-1))
    out = torch.cat([F.linear(flat[i:i + chunk], weight).argmax(-1) for i in range(0, flat.size(0), chunk)])
    return out.reshape(shape)


def ce_sum(h, tgt, weight, chunk=2048):
    """Сумма кросс-энтропии (в натах) по N токенам, h: (N, d), tgt: (N,).
    Кусками с checkpoint - полные логиты не хранятся."""
    def f(hc, tc):
        return F.cross_entropy(F.linear(hc, weight).float(), tc, reduction="sum")

    total = 0.0
    for i in range(0, h.size(0), chunk):
        total = total + checkpoint(f, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
    return total
