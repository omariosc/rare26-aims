#!/usr/bin/env python3
"""Confirm the LEAKAGE-FREE fusion candidates (fixed rule, no weight-fitting) + P0 at the
FULL organiser bootstrap (min_neo=1000, 1:100). Settles whether geo/logit-mean fusion is a
CI-separated win over P0 or just noise. (In-sample-weighted + isotonic variants are excluded:
they're fit-on-test artifacts that evaporate under LOFO — see calib_fusion_experiment.py.)"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_fusion_experiment import RN50, VIT, assemble, fuse
from bootstrap_ppv import bootstrap_ppv_at_90recall, full_pool_metrics

allm = {**RN50, **VIT}
ids, labels, folds, S, names = assemble(allm, recal=True)
idx = {m: k for k, m in enumerate(names)}
rn50 = [idx[m] for m in RN50]
all8 = list(range(len(names)))

cands = {
    "P0 (rn50_top1, recal)":       S[idx["rn50_top1"]],
    "RN50-4 arith-mean":           fuse(S[rn50], "mean"),
    "RN50-4 noisy-OR":             fuse(S[rn50], "noisy_or"),
    "RN50-4 GEO-mean":             fuse(S[rn50], "geo"),
    "RN50-4 LOGIT-mean":           fuse(S[rn50], "logit"),
    "ALL-8 GEO-mean":              fuse(S[all8], "geo"),
    "ALL-8 LOGIT-mean":            fuse(S[all8], "logit"),
}
print(f"{len(ids)} imgs, pos={int(labels.sum())} | FULL organiser bootstrap n=1000 min_neo=1000 @1:100\n")
print(f"{'variant':26s} {'full':>6s} {'AUROC':>6s} {'boot-med':>9s}  {'95% CI':>16s}")
res = {}
for name, vec in cands.items():
    fp = full_pool_metrics(labels, vec)
    b = bootstrap_ppv_at_90recall(labels, vec, n_bootstrap=1000, ndbe_multiplier=100, seed=0)
    res[name] = (b["PPV@90R_median"], b["PPV@90R_CI2.5"], b["PPV@90R_CI97.5"])
    print(f"{name:26s} {fp['PPV@90R_full']:.3f} {fp['AUROC_full']:.3f} "
          f"{b['PPV@90R_median']:9.4f}  [{b['PPV@90R_CI2.5']:.3f},{b['PPV@90R_CI97.5']:.3f}]")
p0m, p0lo, p0hi = res["P0 (rn50_top1, recal)"]
print(f"\nP0 median={p0m:.4f} CI[{p0lo:.3f},{p0hi:.3f}]. A CI-separated win needs cand CI2.5 > P0 median.")
for name, (m, lo, hi) in res.items():
    if name.startswith("P0"): continue
    verdict = "WIN (CI-separated)" if lo > p0m else ("higher point-est, CIs OVERLAP" if m > p0m else "not better")
    print(f"  {name:26s} med={m:.4f} vs P0 {p0m:.4f} -> {verdict}")
