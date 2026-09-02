#!/usr/bin/env python3
"""
RARE26 — pool the RN50 aug-diverse ensemble into ONE held-out OOF score.

The grid trains 4 aug presets (top1..top4) x 5 folds. Each image is a validation
sample in exactly ONE fold, and appears in that fold's OOF for EVERY aug preset.
The ensemble prediction for an image = aggregate (mean or noisy-OR) of the per-aug
models' positive-class scores for that image. We then have ONE score per image over
all 3,095 -> full-pool PPV@90R + 1:100 bootstrap (the defensible ensemble number).

NB: score_oof.py CONCATENATES csvs (right for pooling 5 disjoint folds of ONE model,
WRONG for an ensemble where each image recurs across aug presets). This script does
the correct per-image aggregation across aug presets, then across folds.

Per-model recalibration (--recalibrate): fit the winner's affine prior-recal to
[100/101,1/101] on each (aug,fold) model's own val logits BEFORE aggregating — matches
the winner's per-model recal + noisy-OR pipeline.

Usage:
  python pool_ensemble.py \
    --model_dir /scratch/.../RARE26/work/rn50_first:top1 \
    --model_dir /scratch/.../RARE26/work/rn50:top2,top3,top4 \
    --pool mean --recalibrate --out /scratch/.../RARE26/work/rn50/ENSEMBLE
Each --model_dir is BASE[:aug1,aug2,...]; we glob BASE/<aug>/fold*/**/oof_val_predictions.csv.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ppv import full_pool_metrics, bootstrap_ppv_at_90recall


def _softmax2(l0, l1):
    z = np.stack([l0, l1], axis=1); z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e[:, 1] / e.sum(axis=1)


def model_scores(csv, recalibrate):
    df = pd.read_csv(csv)
    labels = df["target"].values.astype(int)
    ids = df["sample_id"].astype(str).values
    l0 = df["logits_0"].values.astype(float); l1 = df["logits_1"].values.astype(float)
    if recalibrate:
        try:
            import torch
            from psrcal.calibration import calibrate, AffineCalLogLoss
            preds = np.stack([l0, l1], axis=1)
            cal, _ = calibrate(trnscores=torch.tensor(preds, dtype=torch.float64),
                               trnlabels=torch.tensor(labels),
                               tstscores=torch.tensor(preds, dtype=torch.float64),
                               calclass=AffineCalLogLoss, bias=True,
                               priors=[100/101, 1/101], quiet=True)
            cal = cal.detach().cpu().numpy()
            sc = _softmax2(cal[:, 0], cal[:, 1])
            return ids, labels, sc
        except Exception as e:
            print(f"[pool] WARN recal failed {csv}: {e}; raw softmax")
    if np.allclose(l0, 0.0):
        return ids, labels, 1.0 / (1.0 + np.exp(-l1))
    return ids, labels, _softmax2(l0, l1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", action="append", default=[],
                    help="BASE[:aug1,aug2,...]; globs BASE/<aug>/fold*/**/oof_val_predictions.csv")
    ap.add_argument("--glob", action="append", default=[], dest="globs",
                    help="extra recursive glob(s) for OOF csvs, e.g. the ViT "
                         "'.../vitl/RARE_dinov3_vitl16/*/test_predictions_fold_*.csv'")
    ap.add_argument("--pool", choices=["mean", "noisy_or"], default="mean")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csvs = []
    for spec in args.model_dir:
        base, _, augs = spec.partition(":")
        auglist = augs.split(",") if augs else ["*"]
        for a in auglist:
            csvs += glob.glob(os.path.join(base, a, "fold*", "**", "oof_val_predictions.csv"), recursive=True)
    for g in args.globs:
        csvs += glob.glob(g, recursive=True)
    csvs = sorted(set(csvs))
    if not csvs:
        sys.exit("No oof_val_predictions.csv found for the given --model_dir specs.")
    print(f"[pool] {len(csvs)} model OOF csvs ({args.pool}, recal={args.recalibrate}):")
    for c in csvs:
        print("   ", c)

    # accumulate per-image scores across models
    lab_by_id, sc_by_id = {}, {}
    for c in csvs:
        ids, labels, sc = model_scores(c, args.recalibrate)
        for i, l, s in zip(ids, labels, sc):
            lab_by_id[i] = l
            sc_by_id.setdefault(i, []).append(float(s))
    ids = sorted(sc_by_id)
    labels = np.array([lab_by_id[i] for i in ids], dtype=int)
    if args.pool == "mean":
        scores = np.array([np.mean(sc_by_id[i]) for i in ids])
    else:  # noisy-OR: 1 - prod(1 - p)
        scores = np.array([1.0 - np.prod([1.0 - p for p in sc_by_id[i]]) for i in ids])
    n_per = np.array([len(sc_by_id[i]) for i in ids])
    print(f"[pool] {len(ids)} unique images | pos={int(labels.sum())} | models/image: "
          f"min={n_per.min()} max={n_per.max()} (expect = #aug presets)")

    fp = full_pool_metrics(labels, scores)
    bs = bootstrap_ppv_at_90recall(labels, scores, n_bootstrap=args.n, ndbe_multiplier=100, seed=args.seed)
    bs.pop("raw_ppv", None)
    tag = f"ENSEMBLE-{args.pool}{'-RECAL' if args.recalibrate else '-RAW'}"
    print(f"=== {tag}  (IMSY winner full-pool 0.320 / bootstrap ~0.035) ===")
    print("  FULL POOL:", {k: round(v, 4) if isinstance(v, float) else v for k, v in fp.items()})
    print("  BOOTSTRAP 1:100:", {k: round(v, 4) if isinstance(v, float) else v for k, v in bs.items()})
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame({"sample_id": ids, "label": labels, "score": scores}).to_csv(
            os.path.join(args.out, "ensemble_label_score.csv"), index=False)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"tag": tag, "n_models": len(csvs), "full_pool": fp, "bootstrap": bs, "sources": csvs}, f, indent=2)
        print(f"Wrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
