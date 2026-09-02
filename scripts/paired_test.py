#!/usr/bin/env python3
"""PAIRED bootstrap test ([[sota-claims-rigor]]): is leakage-free GEO/LOGIT-mean fusion a REAL
win over P0? Marginal-CI overlap is the wrong test for the SAME images — resample ONCE per draw,
score both, take the difference. Report median Δ, 95% Δ-CI, and P(cand > P0). A defensible win =
Δ-CI excludes 0 AND P(superiority) >= 0.95."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_fusion_experiment import RN50, VIT, assemble, fuse
from bootstrap_ppv import ppv_at_90_recall

ids, labels, folds, S, names = assemble({**RN50, **VIT}, recal=True)
idx = {m: k for k, m in enumerate(names)}
rn50 = [idx[m] for m in RN50]; all8 = list(range(len(names)))
P0 = S[idx["rn50_top1"]]
cands = {"RN50-4 GEO": fuse(S[rn50], "geo"), "RN50-4 LOGIT": fuse(S[rn50], "logit"),
         "ALL-8 GEO": fuse(S[all8], "geo"), "ALL-8 LOGIT": fuse(S[all8], "logit")}

rng = np.random.default_rng(0)
pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
MN, RATIO, N = 1000, 100, 1000
print(f"paired bootstrap n={N}, min_neo={MN}, 1:{RATIO} (same resample scores both)\n")
for name, cand in cands.items():
    d = np.empty(N); a = np.empty(N); b = np.empty(N)
    r2 = np.random.default_rng(0)  # identical draws across candidates
    for i in range(N):
        ps = r2.choice(pos, MN, True); ns = r2.choice(neg, MN * RATIO, True)
        ix = np.concatenate([ps, ns]); y = labels[ix]
        pp0 = ppv_at_90_recall(y, P0[ix]); pc = ppv_at_90_recall(y, cand[ix])
        a[i] = pc; b[i] = pp0; d[i] = pc - pp0
    psup = float(np.mean(d > 0)); ties = float(np.mean(d == 0))
    print(f"{name:14s} cand_med={np.median(a):.4f} P0_med={np.median(b):.4f} | "
          f"Δmed={np.median(d):+.4f} Δ95%CI[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] "
          f"P(cand>P0)={psup:.3f} (ties={ties:.2f})")
print("\nDefensible win iff Δ-CI excludes 0 AND P(cand>P0)>=0.95.")
