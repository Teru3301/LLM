"""
Сборка модели из блоков и чекпойнты.

Модель = Pipeline из именованных частей (tokenizer / encoder / decoder / позже - ещё что-то).
Каждая часть описывается СПЕКОЙ:
    {"file": "encoder/mamba_vae.py", "class": "MambaVAEEncoder", "cfg": {...}, "deps": {...}}
    file, class - какой файл проекта и какой класс из него собирать
    cfg         - kwargs конструктора
    deps        - {аргумент конструктора: имя ранее собранной части}, напр. {"emb": "tokenizer"}

Чекпойнт - папка checkpoints/<name>/:
    manifest.json        стадия обучения, шаг, для каждой части: file/class/cfg/deps + sha256 файла
                         с кодом блока на момент сохранения + имя файла с весами
    <part>.safetensors   веса каждой части ОТДЕЛЬНО (можно взять энкодер из одного чекпойнта,
                         декодер из другого, для 2-й стадии заморозить энкодер+декодер и т.д.)
    optimizer.pt         состояние оптимизатора (для --resume)
"""
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"


class Pipeline(nn.Module):
    """Контейнер именованных частей: pipe.tokenizer, pipe.encoder, pipe.decoder, ..."""

    def __init__(self, **parts):
        super().__init__()
        self.part_names = list(parts)
        for name, module in parts.items():
            self.add_module(name, module)


# ------------------------------------------------------------------ сборка
def _sha(file):
    return hashlib.sha256((ROOT / file).read_bytes()).hexdigest()[:16]


def load_class(file, cls):
    if os.path.isabs(file) or ".." in Path(file).parts or not file.endswith(".py"):
        raise ValueError(f"file должен быть относительным путём к .py внутри проекта: {file}")
    module = importlib.import_module(file[:-3].replace("/", "."))
    return getattr(module, cls)


def build_block(spec, **deps):
    return load_class(spec["file"], spec["class"])(**spec.get("cfg", {}), **deps)


def build_pipeline(specs):
    """specs - dict в порядке сборки (части, от которых зависят другие, идут раньше)."""
    parts = {}
    for name, spec in specs.items():
        deps = {arg: parts[src] for arg, src in spec.get("deps", {}).items()}
        parts[name] = build_block(spec, **deps)
    return Pipeline(**parts)


# ------------------------------------------------------------------ чекпойнты
def checkpoint_dir(name):
    return CHECKPOINTS / name


def specs_from_manifest(manifest):
    return {n: {k: c[k] for k in ("file", "class", "cfg", "deps") if k in c}
            for n, c in manifest["components"].items()}


def save_checkpoint(path, pipe, specs, stage, step, opt=None, extra=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    components = {}
    for name in pipe.part_names:
        weights = f"{name}.safetensors"
        sd = {k: v.detach().cpu().contiguous() for k, v in getattr(pipe, name).state_dict().items()}
        save_file(sd, str(path / weights))
        components[name] = dict(specs[name], sha256=_sha(specs[name]["file"]), weights=weights)
    if opt is not None:
        torch.save(opt.state_dict(), path / "optimizer.pt")
    manifest = dict(stage=stage, step=step, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    components=components, extra=extra or {})
    tmp = path / "manifest.json.tmp"                       # манифест пишем последним и атомарно
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path / "manifest.json")


def read_manifest(path):
    return json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))


def load_checkpoint(path, dev):
    """Собирает pipeline по манифесту и загружает веса. Возвращает (pipe, manifest)."""
    path = Path(path)
    manifest = read_manifest(path)
    for name, c in manifest["components"].items():
        if _sha(c["file"]) != c["sha256"]:
            print(f"[!] {c['file']} ({name}) изменился с момента сохранения чекпойнта. "
                  f"Если менялась архитектура, загрузка весов упадёт с ошибкой формы/ключей.")
    pipe = build_pipeline(specs_from_manifest(manifest))
    for name, c in manifest["components"].items():
        getattr(pipe, name).load_state_dict(load_file(str(path / c["weights"])))
    return pipe.to(dev), manifest


def load_part(pipe, name, path):
    """Загрузить веса ОДНОЙ части из другого чекпойнта в уже собранный pipeline
    (загрузка строгая: ключи и формы должны совпасть, т.е. блок и его cfg те же)."""
    manifest = read_manifest(path)
    c = manifest["components"][name]
    module = getattr(pipe, name)
    module.load_state_dict(load_file(str(Path(path) / c["weights"])))
    return manifest


def load_optimizer_state(path, dev):
    p = Path(path) / "optimizer.pt"
    return torch.load(p, map_location=dev, weights_only=True) if p.exists() else None
