"""
Стадия 1: обучение автоэнкодера (VAE) текст -> латент -> текст.

    python train_vae.py train                       # wikitext103 -> checkpoints/vae_stage1/
    python train_vae.py train --source tokens.npy   # ваш готовый файл токенов (в datasets/ или путь)
    python train_vae.py train --resume
    python train_vae.py demo --text "Some text..." [--text2 "Other text"]

Модель собирается из блоков (assemble.py). Другой блок = другой файл/класс:
    --encoder_block encoder/my_enc.py:MyEncoder --enc_cfg '{"d_model": 512, ...}'
(--enc_cfg/--dec_cfg/--tok_cfg, если заданы, ЗАМЕНЯЮТ cfg, собранный из флагов ниже, целиком.)
"""
import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from assemble import (build_pipeline, checkpoint_dir, load_checkpoint, load_optimizer_state,
                      save_checkpoint, specs_from_manifest)
from corpus import TextData

STAGE = "1_autoencoder"
DEFAULT_BLOCKS = dict(tokenizer="tokenizer/gpt2.py:GPT2Embedder",
                      encoder="encoder/mamba_vae.py:MambaVAEEncoder",
                      decoder="decoder/mamba_ar.py:MambaARDecoder")


def autocast(dev):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")


def make_optimizer(params, lr, use_8bit):
    if use_8bit:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.05)


def cut_eos(g, eos):
    return g[:g.index(eos)] if eos in g else g


def specs_from_args(a):
    def block(arg, cfg, deps=None):
        file, cls = arg.split(":")
        s = {"file": file, "class": cls, "cfg": cfg}
        if deps:
            s["deps"] = deps
        return s

    shape = dict(d_model=a.d_model, n_slots=a.n_slots, d_latent=a.d_latent)
    tok_cfg = json.loads(a.tok_cfg) if a.tok_cfg else dict(d_model=a.d_model)
    enc_cfg = json.loads(a.enc_cfg) if a.enc_cfg else dict(**shape, d_state_ssm=a.d_state_ssm, n_layers=a.n_enc_layers)
    dec_cfg = json.loads(a.dec_cfg) if a.dec_cfg else dict(**shape, d_state_ssm=a.d_state_ssm,
                                                            n_layers=a.n_dec_layers, d_conv=a.dec_d_conv)
    return {"tokenizer": block(a.tokenizer_block, tok_cfg),                       # порядок важен: сначала tokenizer
            "encoder": block(a.encoder_block, enc_cfg),
            "decoder": block(a.decoder_block, dec_cfg, deps={"emb": "tokenizer"})}


# ---------------------------------------------------------------- ELBO
def elbo(pipe, enc_ids, dec_in, tgt, beta, free_bits=0.02, word_dropout=0.3, ss_prob=0.0):
    """loss = (rec + beta * KL_free_bits) / средняя_длина, rec и KL - в натах на последовательность.
    Деление на длину держит масштаб градиента как у per-token CE; на баланс rec/KL не влияет."""
    tok, enc, dec = pipe.tokenizer, pipe.encoder, pipe.decoder
    B = enc_ids.size(0)
    mu, logvar = enc(tok(enc_ids))
    z = enc.sample(mu, logvar)                     # декодер учится на СЭМПЛАХ, не на mu

    inp = dec_in
    if ss_prob > 0:                                # scheduled sampling: часть входов = собственные предсказания
        with torch.no_grad():
            pred = dec.predict(dec.hidden(z, dec_in))                       # (B, L+1)
        swap = torch.rand(dec_in[:, 1:].shape, device=dec_in.device) < ss_prob
        inp = dec_in.clone()
        inp[:, 1:] = torch.where(swap, pred[:, :-1], dec_in[:, 1:])
    if word_dropout > 0:                           # часть входных токенов декодера -> MASK
        if tok.mask_id is None:
            raise SystemExit("word_dropout > 0, но у tokenizer нет служебного MASK-токена (n_special=0)")
        drop = torch.rand(inp[:, 1:].shape, device=inp.device) < word_dropout
        inp = inp.clone()
        inp[:, 1:] = inp[:, 1:].masked_fill(drop, tok.mask_id)

    hid = dec.hidden(z, inp)
    m = tgt != -100
    n_valid = m.sum()
    rec_sum = dec.ce_sum(hid[m], tgt[m])                                     # нат, сумма по батчу

    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)                     # (B, K, dz)
    kl_dim = kl.mean(0)                                                      # по батчу -> (K, dz)
    kl_fb = kl_dim.clamp(min=free_bits).sum()                                # free bits

    loss = (rec_sum + beta * B * kl_fb) / n_valid
    stats = dict(rec_tok=(rec_sum / n_valid).detach(), kl=kl_dim.sum().detach())
    return loss, stats, hid.detach(), m


# ---------------------------------------------------------------- оценка
@torch.no_grad()
def evaluate(pipe, data, dev, maxlen, n=64):
    """Честная авторегрессивная генерация из mu и из сэмпла z + диагностика латента."""
    pipe.eval()
    tok, enc, dec = pipe.tokenizer, pipe.encoder, pipe.decoder
    eos = data.eos
    enc_ids = data.val_batch(n, maxlen, dev)
    with autocast(dev):
        mu, logvar = enc(tok(enc_ids))
        z = enc.sample(mu, logvar)
        gen = dec.generate(torch.cat([mu, z]), eos, maxlen + 1)          # первые n - из mu, вторые n - из z

    def score(g_all):
        exact = acc = 0.0
        for i in range(n):
            g = cut_eos(g_all[i].tolist(), eos)
            ref = enc_ids[i].tolist()
            exact += g == ref
            acc += sum(a == b for a, b in zip(g, ref)) / maxlen
        return exact / n, acc / n

    em_mu, acc_mu = score(gen[:n])
    em_z, acc_z = score(gen[n:])
    kl = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)).mean(0).sum().item()
    active = (mu.var(0) > 0.01).sum().item()             # размерности, где mu реально зависит от входа
    print(f">>> VAL: из mu: exact {em_mu:.2%} tok-acc {acc_mu:.2%} | из z~q: exact {em_z:.2%} "
          f"tok-acc {acc_z:.2%} | KL {kl:.1f} нат | active {active}/{mu[0].numel()} | "
          f"mu mean {mu.mean().item():+.3f} std {mu.std().item():.3f}", flush=True)
    show_latent_samples(pipe, mu, enc_ids, dev, eos, maxlen)
    pipe.train()
    return acc_mu


@torch.no_grad()
def show_latent_samples(pipe, mu, enc_ids, dev, eos, maxlen, steps=5, n_prior=3):
    """Интерполяция между двумя текстами и сэмплы из приора N(0, I): проверка гладкости."""
    tok, dec = pipe.tokenizer, pipe.decoder
    alphas = torch.linspace(0, 1, steps, device=mu.device).view(-1, 1, 1)
    z_int = (1 - alphas) * mu[0:1] + alphas * mu[1:2]
    z_pri = torch.randn(n_prior, *mu.shape[1:], device=mu.device)
    with autocast(dev):
        gen = dec.generate(torch.cat([z_int, z_pri]), eos, maxlen + 1)
    show = lambda t: repr(tok.decode(cut_eos(t.tolist(), eos))[:160])
    print("  A:", show(enc_ids[0]))
    print("  B:", show(enc_ids[1]))
    for al, g in zip(alphas.flatten().tolist(), gen[:steps]):
        print(f"  a={al:.2f}:", show(g))
    for i, g in enumerate(gen[steps:]):
        print(f"  prior#{i}:", show(g))


# ---------------------------------------------------------------- обучение
def cmd_train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckdir = checkpoint_dir(a.name)
    opt_state, start_step = None, 1

    if a.resume and (ckdir / "manifest.json").exists():
        pipe, manifest = load_checkpoint(ckdir, dev)
        specs = specs_from_manifest(manifest)
        opt_state = load_optimizer_state(ckdir, dev)
        start_step = manifest["step"] + 1
        print(f"Продолжаю {ckdir.name} (стадия {manifest['stage']}) с шага {start_step}. "
              f"Архитектура взята из чекпойнта, флаги модели проигнорированы.")
    else:
        if (ckdir / "manifest.json").exists():
            print(f"[!] {ckdir} будет перезаписан (нет --resume). Ctrl+C, если не то.")
            time.sleep(3)
        specs = specs_from_args(a)
        pipe = build_pipeline(specs).to(dev)

    opt = make_optimizer(pipe.parameters(), a.lr, a.adam8bit)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    data = TextData(a.source, pipe.tokenizer, a.tokens)
    n_par = sum(p.numel() for p in pipe.parameters()) / 1e6
    n_emb = pipe.tokenizer.table.weight.numel() / 1e6
    print(f"Данные: {len(data):,} токенов | устройство: {dev} | параметры: {n_par:.1f}M (эмбеддинги {n_emb:.1f}M)")
    for name in pipe.part_names:
        s = specs[name]
        print(f"  {name:9s} {s['file']}:{s['class']} {s['cfg']}")

    t0 = time.time()
    for step in range(start_step, a.steps + 1):
        lr = a.lr * min(1, step / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        # beta: 0 первые beta_delay шагов (автоэнкодер учится пользоваться z), затем линейный рост
        beta = a.beta * min(1.0, max(0.0, (step - a.beta_delay) / max(1, a.beta_warmup)))
        ss_prob = 0.0 if a.ss_max <= 0 else a.ss_max * min(1.0, max(0.0, (step - a.beta_delay) / (0.5 * a.steps)))

        enc_ids, dec_in, tgt = data.train_batch(a.bs, a.maxlen, dev)
        with autocast(dev):
            loss, st, hid, m = elbo(pipe, enc_ids, dec_in, tgt, beta, a.free_bits, a.word_dropout, ss_prob)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(pipe.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            with torch.no_grad(), autocast(dev):
                acc = (pipe.decoder.predict(hid[m]) == tgt[m]).float().mean().item()
            print(f"step {step:6d} | loss {loss.item():.3f} | rec/tok {st['rec_tok'].item():.3f} "
                  f"| KL {st['kl'].item():7.1f} | beta {beta:.2f} | tok-acc {acc:.3f} | lr {lr:.2e} "
                  f"| {(time.time() - t0) / (step - start_step + 1):.2f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(pipe, data, dev, a.maxlen)
            save_checkpoint(ckdir, pipe, specs, stage=STAGE, step=step, opt=opt,
                            extra=dict(maxlen=a.maxlen, train_args=vars(a)))


# ------------------------------------------------------------------ demo
@torch.no_grad()
def encode_text(pipe, text, dev, maxlen):
    tok, eos = pipe.tokenizer, pipe.tokenizer.eos_id
    ids = tok.encode(text)
    if len(ids) > maxlen:
        print(f"[!] Текст {len(ids)} токенов, обрезаю до {maxlen}")
        ids = ids[:maxlen]
    ids = ids + [eos] * (maxlen - len(ids))
    with autocast(dev):
        return pipe.encoder(tok(torch.tensor([ids], device=dev)))


def cmd_demo(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe, manifest = load_checkpoint(checkpoint_dir(a.name), dev)
    pipe.eval()
    maxlen, eos = manifest["extra"]["maxlen"], pipe.tokenizer.eos_id
    text = open(a.file, encoding="utf-8").read() if a.file else a.text
    mu, logvar = encode_text(pipe, text, dev, maxlen)
    zs, labels = [mu, pipe.encoder.sample(mu, logvar)], ["из mu", "из z~q(z|x)"]
    if a.text2:
        mu2, _ = encode_text(pipe, a.text2, dev, maxlen)
        for al in (0.25, 0.5, 0.75):
            zs.append((1 - al) * mu + al * mu2)
            labels.append(f"интерполяция a={al}")
    if a.save_latent:
        torch.save(mu.cpu(), a.save_latent)
    with autocast(dev):
        gen = pipe.decoder.generate(torch.cat(zs), eos, maxlen + 1)
    tok = pipe.tokenizer
    print("--- ОРИГИНАЛ ---\n" + tok.decode(tok.encode(text)[:maxlen]))
    for lab, g in zip(labels, gen):
        print(f"\n--- {lab} ---\n" + tok.decode(cut_eos(g.tolist(), eos)))


# ------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--name", default="vae_stage1", help="папка в checkpoints/")
    t.add_argument("--source", default="wikitext103", help="см. докстринг corpus.py")
    t.add_argument("--tokens", type=int, default=10_000_000)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--maxlen", type=int, default=256, help="токенов в чанке")
    # блоки
    t.add_argument("--tokenizer_block", default=DEFAULT_BLOCKS["tokenizer"])
    t.add_argument("--encoder_block", default=DEFAULT_BLOCKS["encoder"])
    t.add_argument("--decoder_block", default=DEFAULT_BLOCKS["decoder"])
    t.add_argument("--tok_cfg", default="", help="JSON, заменяет cfg токенайзера целиком")
    t.add_argument("--enc_cfg", default="", help="JSON, заменяет cfg энкодера целиком")
    t.add_argument("--dec_cfg", default="", help="JSON, заменяет cfg декодера целиком")
    # размеры (для блоков по умолчанию)
    t.add_argument("--d_model", type=int, default=512)
    t.add_argument("--d_state_ssm", type=int, default=64)
    t.add_argument("--n_slots", type=int, default=16, help="число латентных векторов на чанк")
    t.add_argument("--d_latent", type=int, default=32, help="размерность одного латентного вектора")
    t.add_argument("--n_enc_layers", type=int, default=4)
    t.add_argument("--n_dec_layers", type=int, default=4)
    t.add_argument("--dec_d_conv", type=int, default=2)
    # VAE
    t.add_argument("--beta", type=float, default=0.5, help="итоговый вес KL")
    t.add_argument("--beta_delay", type=int, default=1000, help="шагов с beta=0 перед разгоном")
    t.add_argument("--beta_warmup", type=int, default=8000, help="шагов линейного роста beta")
    t.add_argument("--free_bits", type=float, default=0.02, help="нат на латентную размерность")
    t.add_argument("--word_dropout", type=float, default=0.3)
    t.add_argument("--ss_max", type=float, default=0.0, help="scheduled sampling, 0 - выкл.")
    t.add_argument("--eval_every", type=int, default=1000)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--adam8bit", action="store_true")

    d = sub.add_parser("demo")
    d.add_argument("--name", default="vae_stage1")
    d.add_argument("--text", default="")
    d.add_argument("--text2", default="", help="второй текст для интерполяции")
    d.add_argument("--file", default="")
    d.add_argument("--save_latent", default="")

    args = p.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_demo(args)


if __name__ == "__main__":
    main()
