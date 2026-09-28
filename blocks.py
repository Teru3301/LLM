"""Общие низкоуровневые блоки для encoder/ и decoder/: детект версии Mamba + стек с pre-norm residual."""
import torch.nn as nn

_MAMBA_IMPL = None  # кэш: (имя, класс, доп. kwargs), детектится один раз


def _detect_mamba():
    """Пробуем Mamba3 -> Mamba2 -> Mamba(v1), берём первую доступную."""
    global _MAMBA_IMPL
    if _MAMBA_IMPL is not None:
        return _MAMBA_IMPL
    try:
        from mamba_ssm import Mamba3
        # headdim=64 требует d_model % headdim == 0 (d_model=512 -> ок)
        _MAMBA_IMPL = ("Mamba3", Mamba3, dict(headdim=64, chunk_size=32))
    except ImportError:
        try:
            from mamba_ssm import Mamba2
            _MAMBA_IMPL = ("Mamba2", Mamba2, dict(d_conv=4, expand=2, headdim=64))
        except ImportError:
            try:
                from mamba_ssm import Mamba
                # Mamba(v1) не поддерживает d_state > 16 в CUDA-ядре
                _MAMBA_IMPL = ("Mamba (v1)", Mamba, dict(d_conv=4, expand=2))
            except ImportError as e:
                raise SystemExit("Пакет mamba_ssm не найден (pip install mamba-ssm).") from e
    print(f"[blocks] используется блок: {_MAMBA_IMPL[0]}")
    return _MAMBA_IMPL


def make_mamba_block(d_model, d_state, d_conv=None):
    """d_conv=None -> дефолт библиотеки. Меньшее значение сужает окно depthwise-conv
    (нужно в декодере, чтобы не давать дешёвый локальный шорткат мимо латента)."""
    name, cls, extra = _detect_mamba()
    kwargs = dict(d_model=d_model, d_state=d_state)
    if name == "Mamba (v1)":
        kwargs["d_state"] = min(d_state, 16)
    kwargs.update(extra)
    if d_conv is not None:
        kwargs["d_conv"] = d_conv
    try:
        return cls(**kwargs)
    except TypeError as e:
        # некоторые версии блока могут не принимать d_conv - не падаем, а предупреждаем
        if d_conv is not None and "d_conv" in str(e):
            print(f"[blocks] {name} не принимает d_conv, использую значение по умолчанию")
            kwargs.pop("d_conv")
            return cls(**kwargs)
        raise


class MambaStack(nn.Module):
    """Несколько блоков Mamba с pre-norm residual. Каузальность - от самой Mamba."""

    def __init__(self, d_model, d_state, n_layers, d_conv=None):
        super().__init__()
        self.blocks = nn.ModuleList([make_mamba_block(d_model, d_state, d_conv) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x):                      # (B, L, d_model)
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))
        return x
