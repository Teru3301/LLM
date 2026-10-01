"""Обучение VAE.

    текст -> токены -> эмбеддинги -> Encoder -> (mu, logvar) -> z -> Decoder -> эмбеддинги -> логиты -> CE

Запуск:
    python autoencoder/VAE/train_vae.py --en=encoder --de=decoder --ch=vae --epochs=10

--en, --de  файлы энкодера/декодера (имя из encoder/VAE, decoder/VAE или путь)
--ch        файл чекпоинта (имя из checkpoints/VAE или путь).
            Нет файла -> обучение с нуля и создание; есть -> дообучение.
            Если чекпоинт создан с другими --en/--de, скрипт останавливается.

Train берёт на себя всё остальное: токенизацию и кэш датасета, сэмплирование z, лосс,
KL-annealing, оптимизатор, статистику, сохранение и продолжение обучения.

Тип декодера определяется по интерфейсу:
  * обычный (decoder.py): dec(z) -> эмбеддинги, лосс = CE по логитам;
  * авторегрессионный (есть метод teacher, ar.py): teacher forcing по эмбеддингам целевых
    токенов, часть которых заменена на MASK (--word_drop); лосс = CE по позициям до первого
    EOS включительно. dec(z) - жадная генерация (нужен dec.attach(emb), его делает build_models);
  * диффузионный (есть метод denoise, decoder_diffusion.py): на каждом шаге берётся
    случайное t, эмбеддинги зашумляются, dec.denoise предсказывает чистые эмбеддинги;
    лосс = CE(предсказание) + mse_w * MSE + anchor_w * CE(чистые эмбеддинги).
    dec(z) (сэмплирование DDIM) используется только для примеров и в run_vae.py.
Этот же файл содержит общие функции для run_vae.py.
"""
import argparse
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]      # .../LLM
ARCH = "VAE"

# параметры архитектуры; при дообучении берутся из чекпоинта
DEFAULTS = dict(tok="gpt2", d_model=512, latent_dim=256, max_len=64, n_layers=4, n_heads=8)


# =============================================================== общие функции (используются и в run_vae.py)
def die(msg):
    print(f"ОШИБКА: {msg}", file=sys.stderr)
    sys.exit(1)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_id(path):
    """Каноническое имя файла для сравнения с чекпоинтом (путь относительно корня проекта)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_part(kind, name):
    """--en=encoder / --en=encoder.py / --en=путь -> Path файла."""
    p = Path(name)
    if p.suffix != ".py":
        p = p.with_name(p.name + ".py")
    if not p.exists():
        p = ROOT / kind / ARCH / p.name
    if not p.exists():
        die(f"файл {kind} не найден: {name}")
    return p


def resolve_ckpt(name):
    p = Path(name)
    if p.suffix == "":
        p = p.with_suffix(".pt")
    if len(p.parts) == 1:
        p = ROOT / "checkpoints" / ARCH / p
    return p


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_files(ck, en_id, de_id, ch_path):
    """Чекпоинт должен быть создан именно с этими файлами энкодера и декодера."""
    if ck["encoder_file"] != en_id or ck["decoder_file"] != de_id:
        die(f"чекпоинт {ch_path} создан с другими файлами:\n"
            f"  в чекпоинте: encoder={ck['encoder_file']}, decoder={ck['decoder_file']}\n"
            f"  переданы:    encoder={en_id}, decoder={de_id}\n"
            f"Укажите те же --en/--de или другое имя --ch.")


def warn_if_changed(ck, en_path, de_path):
    """Имена совпали, но содержимое могло быть отредактировано - только предупреждаем."""
    for kind, path in (("encoder", en_path), ("decoder", de_path)):
        if ck[f"{kind}_sha"] != file_sha(path):
            print(f"ВНИМАНИЕ: {path.name} изменён после создания чекпоинта; "
                  f"если форма весов не совпадёт, загрузка упадёт.")


def build_models(cfg, en_path, de_path, device):
    GPT2Embedder = load_module(ROOT / "tokenizer" / "gpt2.py", "vae_tokenizer_gpt2").GPT2Embedder
    kw = dict(d_model=cfg["d_model"], latent_dim=cfg["latent_dim"], max_len=cfg["max_len"],
              n_layers=cfg["n_layers"], n_heads=cfg["n_heads"])
    emb = GPT2Embedder(d_model=cfg["d_model"], hf_name=cfg["tok"]).to(device)
    enc = load_module(en_path, "vae_encoder_impl").Encoder(**kw).to(device)
    dec = load_module(de_path, "vae_decoder_impl").Decoder(**kw).to(device)
    if hasattr(dec, "attach"):          # авторегрессионному декодеру таблица нужна для генерации
        dec.attach(emb)
    return emb, enc, dec


def decoder_kind(dec):
    if hasattr(dec, "teacher"):
        return "ar"
    if hasattr(dec, "denoise"):
        return "diffusion"
    return "direct"


def target_mask(pad):
    """Позиции, по которым считается лосс/точность: весь текст + первый EOS после него.
    Дальше идут заполнители: их содержимое неважно, генерация всё равно обрезается по EOS."""
    valid = torch.ones_like(pad)
    valid[:, 1:] = ~pad[:, :-1]
    return valid


def load_states(ck, emb, enc, dec):
    emb.load_state_dict(ck["embedder"])
    enc.load_state_dict(ck["encoder"])
    dec.load_state_dict(ck["decoder"])


def trim_eos(ids, eos_id):
    """Тензор/список ids -> список до первого EOS."""
    ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


# =============================================================== данные
def build_cache(txt, npy, emb, block=1_000_000):
    """datasets/<name>.txt -> datasets/<name>.<tok>.npy (плоский массив id токенов)."""
    emb._hf.model_max_length = int(1e12)          # убрать предупреждение о длине > 1024
    dtype = np.uint16 if emb.vocab_size < 2 ** 16 else np.uint32
    parts, buf, size = [], [], 0

    def flush():
        nonlocal buf, size
        if buf:
            parts.append(np.asarray(emb.encode("".join(buf)), dtype=dtype))
            buf, size = [], 0

    with open(txt, encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= block:
                flush()
    flush()
    tmp = npy.with_suffix(".tmp.npy")
    np.save(tmp, np.concatenate(parts))
    os.replace(tmp, npy)


def load_tokens(emb, data):
    npy = ROOT / "datasets" / f"{data}.{emb.tokname}.npy"
    if not npy.exists():
        txt = ROOT / "datasets" / f"{data}.txt"
        if not txt.exists():
            die(f"нет ни {npy.name}, ни {txt.name} в datasets/")
        print(f"токенизирую {txt.name} -> {npy.name} (один раз) ...")
        build_cache(txt, npy, emb)
    return np.load(npy, mmap_mode="r")


def make_batch(rows, eos_id, short_p, device):
    """rows (B, L) -> ids, pad_mask.
    В части примеров (short_p) текст обрезается до случайной длины, хвост = EOS и маскируется
    в энкодере: так декодер учится заканчивать текст, а run может принимать короткие строки."""
    ids = torch.from_numpy(rows.astype(np.int64))
    B, L = ids.shape
    pad = torch.zeros(B, L, dtype=torch.bool)
    if short_p > 0 and L > 1:
        lens = torch.randint(max(1, L // 8), L, (B,))
        lens = torch.where(torch.rand(B) < short_p, lens, torch.full_like(lens, L))
        pad = torch.arange(L)[None, :] >= lens[:, None]
        ids = ids.masked_fill(pad, eos_id)
    return ids.to(device), pad.to(device)


# =============================================================== чекпоинт
def save_checkpoint(path, en, de, cfg, emb, enc, dec, opt, epoch, step, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = dict(
        encoder_file=file_id(en), decoder_file=file_id(de),
        encoder_sha=file_sha(en), decoder_sha=file_sha(de),
        cfg=cfg, embedder=emb.state_dict(), encoder=enc.state_dict(), decoder=dec.state_dict(),
        optimizer=opt.state_dict(), epoch=epoch, step=step, train_args=vars(args),
    )
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)               # атомарно: обрыв записи не портит старый чекпоинт


# =============================================================== статистика
@torch.no_grad()
def show_examples(emb, enc, dec, ids, pad, k=2, k_eval=8):
    """Настоящее восстановление (генерация без подсказок) из z = mu и из z ~ q(z|x).
    Точность считается по тексту и первому EOS (заполнители после него не в счёт).
    Разрыв mu -> z~q показывает гладкость: насколько декодер терпит шум в латенте."""
    enc.eval(); dec.eval()
    n = min(k_eval, ids.size(0))
    ids, pad = ids[:n], pad[:n]
    valid = target_mask(pad)
    mu, logvar = enc(emb(ids), pad)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    out = emb.vectors_to_ids(dec(mu))
    acc = ((out == ids) & valid).sum().item() / valid.sum().item()
    acc_z = ((emb.vectors_to_ids(dec(z)) == ids) & valid).sum().item() / valid.sum().item()
    print(f"    точность восстановления ({n} строк): из mu {acc:.3f} | из z~q {acc_z:.3f}")
    if hasattr(dec, "teacher"):
        # проверка коллапса: teacher forcing с родным z и с z соседнего примера
        def ce_with(zz):
            ce = F.cross_entropy(emb.logits(dec.teacher(zz, emb(ids))).transpose(1, 2), ids, reduction="none")
            return (ce * valid).sum().item() / valid.sum().item()
        print(f"    CE/ток при teacher forcing: свой z {ce_with(mu):.3f} | чужой z {ce_with(mu.roll(1, 0)):.3f}"
              f"  (близкие значения = декодер игнорирует z)")
    for i in range(min(k, n)):
        print(f"    исходный: {emb.decode(trim_eos(ids[i], emb.eos_id))[:200]!r}")
        print(f"    восстан.: {emb.decode(trim_eos(out[i], emb.eos_id))[:200]!r}")
    enc.train(); dec.train()


@torch.no_grad()
def evaluate(emb, enc, dec, val, eos_id, device, bs=64):
    """Отложенная выборка (модель её не видит): полные окна + такие же окна, обрезанные вдвое.
    Печатает CE/ток при teacher forcing из mu (без word dropout), KL и точность генерации
    из mu и из z~q. Это честные цифры; примеры из show_examples взяты из обучающего батча."""
    enc.eval(); dec.eval()
    ids = torch.from_numpy(val.astype(np.int64))
    L = ids.size(1)
    lens = torch.full((ids.size(0),), L)
    lens[1::2] = L // 2                                        # половина строк - короткие, с EOS
    pad = torch.arange(L)[None, :] >= lens[:, None]
    ids = ids.masked_fill(pad, eos_id)
    kind = decoder_kind(dec)
    tot = dict(ce=0.0, kl=0.0, mu=0.0, zq=0.0, n=0, rows=0)
    for i in range(0, ids.size(0), bs):
        x, p = ids[i:i + bs].to(device), pad[i:i + bs].to(device)
        valid = target_mask(p)
        mu, logvar = enc(emb(x), p)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        if kind != "diffusion":
            h = dec.teacher(mu, emb(x)) if kind == "ar" else dec(mu)
            ce = F.cross_entropy(emb.logits(h).transpose(1, 2), x, reduction="none")
            tot["ce"] += (ce * valid).sum().item()
        tot["kl"] += (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum().item()
        tot["mu"] += ((emb.vectors_to_ids(dec(mu)) == x) & valid).sum().item()
        tot["zq"] += ((emb.vectors_to_ids(dec(z)) == x) & valid).sum().item()
        tot["n"] += valid.sum().item()
        tot["rows"] += x.size(0)
    n = tot["n"]
    ce = f"ce/ток (mu) {tot['ce'] / n:.3f} | " if kind != "diffusion" else ""
    print(f"    [валидация, {tot['rows']} строк] {ce}kl {tot['kl'] / tot['rows']:.1f} | "
          f"восстановление: из mu {tot['mu'] / n:.3f} | из z~q {tot['zq'] / n:.3f}")
    enc.train(); dec.train()


def fmt_stats(m, diffusion):
    s = f"loss {m[0]:.2f} ce/ток {m[1]:.3f} kl {m[2]:.1f} acc {m[3]:.3f}"
    return s + (f" mse {m[4]:.2f}" if diffusion else "")


# =============================================================== main
def parse_args():
    ap = argparse.ArgumentParser(description="Обучение VAE (см. README)")
    ap.add_argument("--en", default="encoder", help="файл энкодера")
    ap.add_argument("--de", default="decoder", help="файл декодера")
    ap.add_argument("--ch", default="vae", help="файл чекпоинта")
    ap.add_argument("--data", default="wikitext103", help="имя датасета в datasets/")
    ap.add_argument("--device", default="auto")
    # архитектура (игнорируется/сверяется при дообучении)
    ap.add_argument("--tok", default=None, help=f"HF-токенизатор (по умолч. {DEFAULTS['tok']})")
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--latent_dim", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None, help="длина последовательности в токенах")
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    # обучение
    ap.add_argument("--epochs", type=int, default=1,
                    help="сколько эпох выполнить в этом запуске (0 = только оценка на валидации)")
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="0 = проход по всему датасету")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_warmup", type=int, default=1000, help="шагов разогрева lr")
    ap.add_argument("--beta", type=float, default=0.1, help="итоговый вес KL")
    ap.add_argument("--free_bits", type=float, default=0.0,
                    help="нат на латентную размерность (среднее по батчу), ниже которого KL не штрафуется")
    ap.add_argument("--kl_warmup", type=int, default=5000, help="шагов, за которые вес KL растёт от 0 до --beta")
    ap.add_argument("--short_p", type=float, default=0.3, help="доля коротких (обрезанных) примеров")
    ap.add_argument("--mse_w", type=float, default=1.0, help="вес MSE (только диффузионный декодер)")
    ap.add_argument("--anchor_w", type=float, default=0.1,
                    help="вес CE по чистым эмбеддингам, держит таблицу различимой (только диффузионный декодер)")
    ap.add_argument("--word_drop", type=float, default=0.3,
                    help="доля входных токенов декодера, заменяемых на MASK (только авторегрессионный декодер)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=1, help="печатать статистику каждые N эпох")
    ap.add_argument("--save_every", type=int, default=1, help="сохранять чекпоинт каждые M эпох")
    ap.add_argument("--val_tokens", type=int, default=200_000,
                    help="хвост датасета, отложенный для валидации (0 = без валидации)")
    ap.add_argument("--val_data", default=None,
                    help="отдельный датасет для валидации в datasets/ (тогда обучающий не урезается)")
    ap.add_argument("--val_rows", type=int, default=512, help="сколько окон валидации проверять")
    ap.add_argument("--progress", type=int, default=500, help="строка прогресса каждые K шагов (0 = выкл)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)

    en_path, de_path = resolve_part("encoder", args.en), resolve_part("decoder", args.de)
    en_id, de_id = file_id(en_path), file_id(de_path)
    ch_path = resolve_ckpt(args.ch)
    explicit = {k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None}

    ck = None
    if ch_path.exists():
        ck = torch.load(ch_path, map_location=device, weights_only=True)
        check_files(ck, en_id, de_id, ch_path)
        warn_if_changed(ck, en_path, de_path)
        cfg = ck["cfg"]
        for k, v in explicit.items():
            if cfg[k] != v:
                die(f"--{k}={v}, но чекпоинт создан с {k}={cfg[k]}. Уберите флаг или используйте другой --ch.")
    else:
        cfg = {**DEFAULTS, **explicit}

    emb, enc, dec = build_models(cfg, en_path, de_path, device)
    kind = decoder_kind(dec)
    diffusion = kind == "diffusion"
    params = list(emb.parameters()) + list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    epoch_done, step = 0, 0
    if ck is not None:
        load_states(ck, emb, enc, dec)
        opt.load_state_dict(ck["optimizer"])
        epoch_done, step = ck["epoch"], ck["step"]

    tokens = load_tokens(emb, args.data)
    L, B, V = cfg["max_len"], args.batch, emb.vocab_size
    val = None
    if args.val_data:
        vt = load_tokens(emb, args.val_data)
        val = np.asarray(vt[: min(len(vt) // L, args.val_rows) * L]).reshape(-1, L)
    elif args.val_tokens > 0:
        # фиксированный хвост: не зависит от запуска, в обучение не попадает
        vt = tokens[len(tokens) - args.val_tokens:]
        val = np.asarray(vt[: min(len(vt) // L, args.val_rows) * L]).reshape(-1, L)
        tokens = tokens[: len(tokens) - args.val_tokens]
        if ck is not None and ck.get("train_args", {}).get("val_tokens", 0) != args.val_tokens:
            print(f"ВНИМАНИЕ: чекпоинт обучался без этой валидационной части - "
                  f"он её уже видел, цифры валидации будут завышены.")
    if args.epochs == 0:
        if val is None:
            die("--epochs=0 - режим оценки, но валидация выключена (--val_tokens=0 и нет --val_data)")
        evaluate(emb, enc, dec, val, emb.eos_id, device)
        return
    print(f"устройство: {device} | параметров: {sum(p.numel() for p in params) / 1e6:.1f}M | "
          f"токенов в датасете: {len(tokens):,} | max_len={L}")
    print(f"{'дообучение' if ck else 'новое обучение'}: {ch_path} (эпох выполнено: {epoch_done}, шаг {step})")
    print(f"декодер: {dict(ar='авторегрессионный', diffusion='диффузионный', direct='прямой')[kind]}")

    rng = np.random.default_rng()
    last_saved = epoch_done
    enc.train(); dec.train()
    t0 = time.time()

    def save():
        nonlocal last_saved
        save_checkpoint(ch_path, en_path, de_path, cfg, emb, enc, dec, opt, epoch_done, step, args)
        last_saved = epoch_done
        print(f"  сохранено: {ch_path} (эпоха {epoch_done}, шаг {step})")

    try:
        for _ in range(args.epochs):
            ep = epoch_done + 1
            off = int(rng.integers(0, L))                       # случайный сдвиг окон в каждой эпохе
            n_win = (len(tokens) - off) // L
            data = tokens[off: off + n_win * L].reshape(n_win, L)
            perm = rng.permutation(n_win)
            n_steps = n_win // B
            if args.steps_per_epoch:
                n_steps = min(n_steps, args.steps_per_epoch)

            sums = torch.zeros(5, device=device)                # loss, ce/токен, kl, точность, mse
            for i in range(n_steps):
                rows = data[np.sort(perm[i * B:(i + 1) * B])]
                ids, pad = make_batch(rows, emb.eos_id, args.short_p, device)

                lr = args.lr * min(1.0, (step + 1) / max(1, args.lr_warmup))
                beta = args.beta * min(1.0, step / max(1, args.kl_warmup))
                for g in opt.param_groups:
                    g["lr"] = lr

                e = emb(ids)
                mu, logvar = enc(e, pad)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)      # reparameterization
                mse = torch.zeros((), device=device)
                if diffusion:
                    x0 = dec.space(e)                                        # эмбеддинги в масштабе шума
                    t = dec.sample_t(ids.size(0), device)
                    xt = dec.q_sample(x0.detach(), t, torch.randn_like(x0))
                    x0p = dec.denoise(xt, t, z)
                    logits = emb.logits(x0p)                                 # (B, L, V)
                    mse = (x0p - x0.detach()).pow(2).mean(-1).sum(1).mean()
                    anchor = F.cross_entropy(emb.logits(x0).reshape(-1, V), ids.reshape(-1),
                                             reduction="none").view_as(ids).sum(1).mean()
                elif kind == "ar":
                    inp = ids
                    if args.word_drop > 0:
                        drop = torch.rand(ids.shape, device=device) < args.word_drop
                        inp = ids.masked_fill(drop, emb.mask_id)
                    logits = emb.logits(dec.teacher(z, emb(inp)))            # (B, L, V)
                else:
                    logits = emb.logits(dec(z))                              # (B, L, V)
                ce = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1), reduction="none").view_as(ids)
                valid = target_mask(pad) if kind == "ar" else torch.ones_like(pad)
                ce = ce * valid
                recon = ce.sum(1).mean()                                     # nats на последовательность
                if diffusion:
                    recon = recon + args.mse_w * mse + args.anchor_w * anchor
                kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())      # (B, latent_dim)
                kl = kl_dim.sum(1).mean()                                    # настоящий KL (для статистики)
                kl_loss = kl_dim.mean(0).clamp(min=args.free_bits).sum() if args.free_bits > 0 else kl
                loss = recon + beta * kl_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                step += 1

                n_valid = valid.sum()
                acc = ((logits.argmax(-1) == ids) & valid).sum() / n_valid
                sums += torch.stack([loss.detach(), (ce.sum() / n_valid).detach(), kl.detach(),
                                     acc.detach(), mse.detach()])
                if args.progress and (i + 1) % args.progress == 0:
                    m = (sums / (i + 1)).tolist()
                    print(f"  эпоха {ep} шаг {i + 1}/{n_steps} | {fmt_stats(m, diffusion)} "
                          f"| beta {beta:.3f} lr {lr:.1e}", flush=True)

            epoch_done = ep
            if epoch_done % args.log_every == 0:
                m = (sums / max(1, n_steps)).tolist()
                print(f"[эпоха {epoch_done}] шаг {step} | {fmt_stats(m, diffusion)} "
                      f"| beta {beta:.3f} | {time.time() - t0:.0f}с")
                show_examples(emb, enc, dec, ids, pad)
                if val is not None:
                    evaluate(emb, enc, dec, val, emb.eos_id, device)
            if epoch_done % args.save_every == 0:
                save()
    except KeyboardInterrupt:
        print("\nостановка по Ctrl+C: сохраняю текущее состояние (незавершённая эпоха будет повторена)")
        save()
        return

    if last_saved != epoch_done:
        save()


if __name__ == "__main__":
    main()
