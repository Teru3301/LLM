"""Запуск обученного VAE с ручным вводом текста: текст -> z -> восстановленный текст.

    python autoencoder/VAE/run_vae.py --en=encoder --de=decoder --ch=vae

Выход: Ctrl+D, Ctrl+C или строка `exit`.
Строка вида `текст A || текст B` - интерполяция: z идёт от mu(A) к mu(B) за --interp шагов,
на каждом шаге печатается декодированный текст. Так проверяется гладкость латента:
промежуточные точки должны давать осмысленный текст, плавно переходящий от A к B.
--en/--de должны быть теми же, с которыми создан чекпоинт (иначе скрипт остановится).
"""
import argparse

import torch

from train_vae import (build_models, check_files, die, file_id, load_states, pick_device,
                       resolve_ckpt, resolve_part, trim_eos, warn_if_changed)


def main():
    ap = argparse.ArgumentParser(description="Запуск VAE (см. README)")
    ap.add_argument("--en", default="encoder", help="файл энкодера")
    ap.add_argument("--de", default="decoder", help="файл декодера")
    ap.add_argument("--ch", default="vae", help="файл чекпоинта")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--metric", default="dot", choices=["dot", "cosine", "l2"],
                    help="как переводить векторы декодера в токены")
    ap.add_argument("--steps", type=int, default=0,
                    help="число шагов сэмплирования диффузионного декодера (0 = как в файле декодера)")
    ap.add_argument("--interp", type=int, default=6, help="число точек интерполяции для `A || B`")
    ap.add_argument("--sample", action="store_true",
                    help="брать z ~ N(mu, sigma) вместо z = mu")
    args = ap.parse_args()
    device = pick_device(args.device)

    en_path, de_path = resolve_part("encoder", args.en), resolve_part("decoder", args.de)
    ch_path = resolve_ckpt(args.ch)
    if not ch_path.exists():
        die(f"чекпоинт не найден: {ch_path}")
    ck = torch.load(ch_path, map_location=device, weights_only=True)
    check_files(ck, file_id(en_path), file_id(de_path), ch_path)
    warn_if_changed(ck, en_path, de_path)

    cfg = ck["cfg"]
    emb, enc, dec = build_models(cfg, en_path, de_path, device)
    load_states(ck, emb, enc, dec)
    emb.eval(); enc.eval(); dec.eval()
    if args.steps and hasattr(dec, "n_steps"):
        dec.n_steps = args.steps
    L, eos = cfg["max_len"], emb.eos_id
    print(f"загружено: {ch_path} (эпох: {ck['epoch']}, шагов: {ck['step']}); "
          f"длина до {L} токенов. Введите текст (Ctrl+D - выход).")

    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip() == "exit":
            break
        if not text.strip():
            continue

        if "||" in text:
            a, b = (p.strip() for p in text.split("||", 1))
            (ia, pa), (ib, pb) = prepare(emb, a, L, device), prepare(emb, b, L, device)
            with torch.no_grad():
                mu, _ = enc(emb(torch.cat([ia, ib])), torch.cat([pa, pb]))
                w = torch.linspace(0, 1, max(2, args.interp), device=device)[:, None]
                zs = (1 - w) * mu[:1] + w * mu[1:]
                outs = emb.vectors_to_ids(dec(zs), args.metric)
            for wi, o in zip(w[:, 0].tolist(), outs):
                print(f"  {wi:.2f}: {emb.decode(trim_eos(o, eos))}")
            continue

        ids_t, pad = prepare(emb, text, L, device)
        n = int((~pad).sum())
        with torch.no_grad():
            mu, logvar = enc(emb(ids_t), pad)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if args.sample else mu
            out = emb.vectors_to_ids(dec(z), args.metric)[0]

        match = (out[:n] == ids_t[0, :n]).float().mean().item()
        note = f", обрезано до {L}" if len(emb.encode(text)) > L else ""
        print(f"  токенов: {n}{note} | совпало позиций: {match:.0%}")
        print(f"  {emb.decode(trim_eos(out, eos))}")


def prepare(emb, text, L, device):
    """Строка -> ids (1, L) и pad (1, L), как в обучении: хвост = EOS, замаскирован в энкодере."""
    ids = emb.encode(text)[:L]
    n = len(ids)
    ids_t = torch.tensor(ids + [emb.eos_id] * (L - n), device=device)[None]
    return ids_t, (torch.arange(L, device=device) >= n)[None]


if __name__ == "__main__":
    main()
