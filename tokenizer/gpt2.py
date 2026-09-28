"""GPT-2 токенизатор (HuggingFace) + обучаемая таблица эмбеддингов."""
import torch.nn as nn


class GPT2Embedder(nn.Module):
    def __init__(self, d_model=512, hf_name="gpt2", n_special=1):
        """n_special: служебные строки в конце таблицы (нет в словаре токенизатора).
        Сейчас нужен один - MASK (id = vocab_size) для word dropout в декодере."""
        super().__init__()
        from transformers import AutoTokenizer
        self._hf = AutoTokenizer.from_pretrained(hf_name)       # не параметр -> в чекпойнт не попадает
        self.tokname = hf_name.replace("/", "_")                # для имён кэшей токенов
        self.vocab_size = len(self._hf)
        self.eos_id = self._hf.eos_token_id
        self.d_model = d_model
        self.mask_id = self.vocab_size if n_special > 0 else None
        self.table = nn.Embedding(self.vocab_size + n_special, d_model)
        nn.init.normal_(self.table.weight, std=0.02)

    # ---- текст <-> токены
    def encode(self, text):
        return self._hf(text)["input_ids"]

    def encode_batch(self, texts):
        return self._hf(texts)["input_ids"]

    def decode(self, ids):
        return self._hf.decode(ids)

    # ---- токены -> векторы
    def forward(self, ids):
        return self.table(ids)

    @property
    def out_weight(self):
        """Веса для выходной проекции (без служебных строк - MASK не может быть сгенерирован)."""
        return self.table.weight[: self.vocab_size]
