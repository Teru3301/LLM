"""GPT-2 токенизатор (HuggingFace) + обучаемая таблица эмбеддингов.

Слой "текст <-> вектор эмбеддингов":

    текст --encode--> ids --embed--> векторы (L, d_model)      # прямой путь
    векторы --logits/nearest--> ids --decode--> текст          # обратный путь
"""
import torch
import torch.nn as nn


class GPT2Embedder(nn.Module):
    def __init__(self, d_model=512, hf_name="gpt2", n_special=1):
        """n_special: служебные строки в конце таблицы (нет в словаре токенизатора).
        Сейчас нужен один - MASK (id = vocab_size) для word dropout в декодере."""
        super().__init__()
        from transformers import AutoTokenizer
        self._hf = AutoTokenizer.from_pretrained(hf_name)   # не параметр -> в чекпойнт не попадает
        self.tokname = hf_name.replace("/", "_")            # для имён кэшей токенов
        self.vocab_size = len(self._hf)
        self.eos_id = self._hf.eos_token_id
        self.d_model = d_model
        self.mask_id = self.vocab_size if n_special > 0 else None
        self.table = nn.Embedding(self.vocab_size + n_special, d_model)
        nn.init.normal_(self.table.weight, std=0.02)

    # ------------------------------------------------------------ текст <-> токены
    def encode(self, text):
        return self._hf(text)["input_ids"]

    def encode_batch(self, texts):
        return self._hf(texts)["input_ids"]

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self._hf.decode(ids)

    def decode_batch(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self._hf.batch_decode(ids)

    # ------------------------------------------------------------ токены -> векторы
    def forward(self, ids):
        return self.table(ids)

    def embed(self, ids):
        return self.table(ids)

    def text_to_vectors(self, text):
        """Строка -> тензор (L, d_model)."""
        ids = torch.tensor(self.encode(text), device=self.table.weight.device)
        return self.table(ids)

    # ------------------------------------------------------------ векторы -> токены
    @property
    def out_weight(self):
        """Веса для выходной проекции (без служебных строк - MASK не может быть сгенерирован)."""
        return self.table.weight[: self.vocab_size]

    def logits(self, h):
        """Векторы (..., d_model) -> логиты по словарю (..., vocab_size) (tied weights)."""
        return h @ self.out_weight.T

    @torch.no_grad()
    def vectors_to_ids(self, h, metric="dot"):
        """Векторы (..., d_model) -> ids (...).
        metric="dot"    - argmax по логитам (как это делает декодер с tied-проекцией);
        metric="cosine" - ближайший эмбеддинг по косинусу;
        metric="l2"     - ближайший эмбеддинг по евклидову расстоянию."""
        w = self.out_weight
        if metric == "dot":
            return (h @ w.T).argmax(-1)
        if metric == "cosine":
            h = nn.functional.normalize(h, dim=-1)
            w = nn.functional.normalize(w, dim=-1)
            return (h @ w.T).argmax(-1)
        if metric == "l2":
            flat = h.reshape(-1, h.shape[-1])
            return torch.cdist(flat, w).argmin(-1).reshape(h.shape[:-1])
        raise ValueError(f"unknown metric: {metric}")

    @torch.no_grad()
    def vectors_to_text(self, h, metric="dot"):
        """(L, d_model) -> строка; (B, L, d_model) -> список строк."""
        ids = self.vectors_to_ids(h, metric)
        return self.decode(ids) if ids.dim() == 1 else self.decode_batch(ids)
