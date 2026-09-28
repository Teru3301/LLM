"""
Тексты -> токены -> батчи.

Сами тексты и кэши токенов лежат в datasets/ (только данные!). В этой папке НЕЛЬЗЯ держать .py-файлы
и __init__.py: папка с именем datasets перекрыла бы установленный пакет HuggingFace datasets.

source может быть:
    "wikitext103"            - datasets/wikitext103.txt (при отсутствии скачивается через HF)
    "book.txt"               - свой текстовый файл (datasets/book.txt или путь)
    "tokens.npy"             - готовый массив gpt2-токенов (uint16), например ваш прежний tokens.npy
Токенизация текстов кэшируется в datasets/<имя>.<токенизатор>.npy.
"""
from pathlib import Path

import numpy as np
import torch

DATASETS = Path(__file__).resolve().parent / "datasets"


# ------------------------------------------------------------------ тексты -> токены
def _download_wikitext(txt_path, max_chars):
    from datasets import load_dataset          # пакет HuggingFace (см. предупреждение выше)
    print(f"Скачиваю wikitext-103 (streaming) -> {txt_path} (~{max_chars:,} символов)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    part = txt_path.with_suffix(".part")
    total = 0
    with open(part, "w", encoding="utf-8") as f:
        for row in ds:
            t = row["text"].strip()
            if not t or t.startswith("="):
                continue
            t = t.replace("\n", " ")
            f.write(t + "\n")
            total += len(t) + 1
            if total >= max_chars:
                break
    part.replace(txt_path)


def _tokenize_file(txt_path, tok, n_tokens):
    assert tok.vocab_size < 65536, "кэш токенов хранится как uint16"
    print(f"Токенизирую {txt_path.name} (до {n_tokens:,} токенов)...")
    buf, total, lines = [], 0, []

    def flush():
        nonlocal total, lines
        for ids in tok.encode_batch(lines):
            buf.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
        lines = []

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            lines.append(line + "\n")
            if len(lines) == 512:
                flush()
                if total >= n_tokens:
                    break
    if lines and total < n_tokens:
        flush()
    if not buf:
        raise SystemExit(f"В {txt_path} нет текста.")
    return np.concatenate(buf)[:n_tokens]


def load_tokens(source, tok, n_tokens):
    p = Path(source)
    if p.suffix == ".npy":
        return np.load(p if p.exists() else DATASETS / p.name)
    if p.suffix == ".txt":
        txt = p if p.exists() else DATASETS / p.name
        name = txt.stem
    else:
        name, txt = source, DATASETS / f"{source}.txt"
    npy = DATASETS / f"{name}.{tok.tokname}.npy"
    if npy.exists():
        return np.load(npy)
    if not txt.exists():
        if name != "wikitext103":
            raise SystemExit(f"Не найден текст {txt}. Положите файл в datasets/ или укажите путь.")
        _download_wikitext(txt, max_chars=n_tokens * 5)      # ~4-4.5 символа на gpt2-токен, берём с запасом
    arr = _tokenize_file(txt, tok, n_tokens)
    np.save(npy, arr)
    return arr


# ------------------------------------------------------------------ батчи
def make_batch(data, offs, lens, L, eos, dev):
    """Окно из L токенов, у каждого примера своя длина lens (остальное - EOS-паддинг).
    Возвращает (enc_ids (B,L), dec_in (B,L+1), tgt (B,L+1), -100 = не считать)."""
    idx = offs[:, None] + np.arange(L)[None]
    w = torch.from_numpy(data[idx].astype(np.int64)).to(dev)
    B = w.size(0)
    eos_col = torch.full((B, 1), eos, device=dev)
    w_ext = torch.cat([w, eos_col], 1)
    pos = torch.arange(L + 1, device=dev)[None]
    n = lens[:, None]
    ids_ext = torch.where(pos < n, w_ext, torch.full_like(w_ext, eos))
    tgt = torch.where(pos < n, w_ext,
                      torch.where(pos == n, torch.full_like(w_ext, eos),
                                  torch.full_like(w_ext, -100)))
    enc_ids = ids_ext[:, :L]
    dec_in = torch.cat([eos_col, ids_ext[:, :L]], 1)       # BOS = EOS
    return enc_ids, dec_in, tgt


class TextData:
    """Токенизированный корпус + train/val сплит (последние val_frac - валидация)."""

    def __init__(self, source, tok, n_tokens, val_frac=0.05):
        self.eos = tok.eos_id
        self.arr = load_tokens(source, tok, n_tokens)
        self.split = int(len(self.arr) * (1 - val_frac))

    def __len__(self):
        return len(self.arr)

    def train_batch(self, B, L, dev, full_prob=0.5):
        """Случайные окна. Половина примеров - полной длины L, остальные - случайной
        длины 8..L (хвост добивается EOS, чтобы декодер учился останавливаться)."""
        offs = np.random.randint(0, self.split - L - 1, B)
        lens = torch.where(torch.rand(B, device=dev) < full_prob,
                           torch.full((B,), L, device=dev),
                           torch.randint(8, L + 1, (B,), device=dev))
        return make_batch(self.arr, offs, lens, L, self.eos, dev)

    def val_batch(self, n, L, dev):
        """Детерминированные валидационные окна полной длины L."""
        offs = self.split + np.arange(n) * (L + 7)
        assert offs[-1] + L < len(self.arr), "валидационная часть слишком мала для n окон"
        lens = torch.full((n,), L, device=dev)
        return make_batch(self.arr, offs, lens, L, self.eos, dev)

    def sequence(self, B, n_chunks, T, dev, split="train"):
        """B последовательностей из n_chunks ПОДРЯД идущих чанков по T токенов: (B, n_chunks, T).
        Понадобится на 2-й стадии (рекуррентная модель над латентами)."""
        span = n_chunks * T
        lo, hi = (0, self.split - span - 1) if split == "train" else (self.split, len(self.arr) - span - 1)
        offs = np.random.randint(lo, hi, B)
        idx = offs[:, None] + np.arange(span)[None]
        x = torch.from_numpy(self.arr[idx].astype(np.int64)).to(dev)
        return x.view(B, n_chunks, T)
