#!/usr/bin/env python3
"""
RARE26 ranking-metric evaluator (DIAGNOSTIC ARTIFACT — keep current).

Replicates the organiser RARE25/RARE26 eval EXACTLY:
  - Primary metric: PPV @ 90% Recall (precision at the threshold where recall >= 0.90),
    read from the precision_recall_curve the organiser way:
        idx = np.where(recalls >= 0.9)[0][-1]; ppv = precisions[idx]
  - Low-prevalence handling: 1000 bootstrap iterations resampling to a FIXED
    100:1 NDBE:NEO prevalence (1:100). Per iter: resample NEO with replacement to
    `min_neoplasia`, NDBE with replacement to `min_neoplasia * ndbe_multiplier`.
    Headline = MEDIAN over the 1000 bootstrapped PPV@90%R; 95% CI = 2.5/97.5 pctl.
  (Faithful to RARE25-Baselines/evaluate.py::bootstrap_evaluation + metrics.py.)

Also reports the FULL-POOL (no resampling) PPV@90%R / AUROC / AUPRC — this is the
"0.320 anchor" the IMSY winner is quoted on.

Usage:
  python bootstrap_ppv.py --csv preds.csv            # csv with columns: label,score
  python bootstrap_ppv.py --csv preds.csv --ratio 100 --n 1000 --seed 0 --out out_dir
  # or programmatically: from bootstrap_ppv import bootstrap_ppv_at_90recall
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def ppv_at_90_recall(y_true, y_scores, min_recall=0.9):
    """Organiser-exact PPV@90%R from the PR curve. Returns np.nan if no point reaches recall>=0.9."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    idx = np.where(recalls >= min_recall)[0]
    if len(idx) == 0:
        return np.nan
    return float(precisions[idx[-1]])


def full_pool_metrics(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    return {
        "PPV@90R_full": ppv_at_90_recall(y_true, y_scores),
        "AUROC_full": float(roc_auc_score(y_true, y_scores)),
        "AUPRC_full": float(average_precision_score(y_true, y_scores)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
    }


def bootstrap_ppv_at_90recall(y_true, y_scores, n_bootstrap=1000, min_neoplasia=1000,
                              ndbe_multiplier=100, seed=0):
    """Returns dict with median/CI of bootstrapped PPV@90R (+AUROC/AUPRC) at fixed 1:`ndbe_multiplier` prevalence."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need both classes present.")
    ppvs, aurocs, auprcs = [], [], []
    for _ in range(n_bootstrap):
        ps = rng.choice(pos_idx, size=min_neoplasia, replace=True)
        ns = rng.choice(neg_idx, size=min_neoplasia * ndbe_multiplier, replace=True)
        idx = np.concatenate([ps, ns])
        yt = y_true[idx]
        ys = y_scores[idx]
        ppvs.append(ppv_at_90_recall(yt, ys))
        try:
            aurocs.append(roc_auc_score(yt, ys))
            auprcs.append(average_precision_score(yt, ys))
        except Exception:
            aurocs.append(np.nan); auprcs.append(np.nan)
    ppvs = np.array(ppvs, dtype=float)
    return {
        "PPV@90R_median": float(np.nanmedian(ppvs)),
        "PPV@90R_CI2.5": float(np.nanpercentile(ppvs, 2.5)),
        "PPV@90R_CI97.5": float(np.nanpercentile(ppvs, 97.5)),
        "PPV@90R_mean": float(np.nanmean(ppvs)),
        "AUROC_median": float(np.nanmedian(aurocs)),
        "AUPRC_median": float(np.nanmedian(auprcs)),
        "n_bootstrap": n_bootstrap,
        "ratio": f"1:{ndbe_multiplier}",
        "raw_ppv": ppvs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: label,score (label in {0,1}).")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--ratio", type=int, default=100, help="NDBE:NEO ratio (default 100 = 1:100).")
    ap.add_argument("--n", type=int, default=1000, help="bootstrap iters")
    ap.add_argument("--min-neoplasia", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y_true = df[args.label_col].values.astype(int)
    y_scores = df[args.score_col].values.astype(float)

    fp = full_pool_metrics(y_true, y_scores)
    bs = bootstrap_ppv_at_90recall(y_true, y_scores, n_bootstrap=args.n,
                                   min_neoplasia=args.min_neoplasia,
                                   ndbe_multiplier=args.ratio, seed=args.seed)
    print("=== FULL POOL (anchor; IMSY winner = 0.320) ===")
    for k, v in fp.items():
        print(f"  {k}: {v}")
    print(f"=== BOOTSTRAP 1:{args.ratio} (RANKING METRIC; IMSY ~0.035) ===")
    for k, v in bs.items():
        if k != "raw_ppv":
            print(f"  {k}: {v}")

    if args.out:
        import os, json
        os.makedirs(args.out, exist_ok=True)
        raw = bs.pop("raw_ppv")
        pd.DataFrame({"ppv_at_90r": raw}).to_csv(os.path.join(args.out, "bootstrap_raw.csv"), index=False)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"full_pool": fp, "bootstrap": bs}, f, indent=2)
        print(f"Wrote {args.out}/summary.json + bootstrap_raw.csv")


if __name__ == "__main__":
    main()
