#!/usr/bin/env python3
"""
RARE26 — LOW-PREVALENCE CALIBRATION + FUSION experiment on the pooled OOF (CPU, no training).

Context: ensemble DIVERSITY is REFUTED (P0-single 0.077 > RN50-20 0.065 > CNN×ViT 0.068 >
DINOv3-20 0.045, all within CI). AUROC 0.94-0.96 => discrimination fine; the 1:100
operating-point PPV@90R is the bottleneck. This script tests whether a CHEAP post-hoc
recipe on the frozen OOF logits beats the single P0 (recal) = 0.077 boot-median @1:100.

KEY FACT (drives the design): PPV@90%R is computed from the PR curve => it depends ONLY on
the RANK-ORDER of scores. Any GLOBAL monotonic transform (temperature, Platt, isotonic,
Venn-Abers, conformal calibration on one homogeneous set) leaves every bootstrap draw's
PPV@90R EXACTLY unchanged. We DEMONSTRATE this (Exp A) then search only rank-CHANGING ops:
  B. weighted fusion (w ∝ per-member PPV@90R / AUPRC), softmax/pow weights
  C. top-k member selection (best-k by per-member metric)
  D. rank-mean / noisy-OR / max / geo-mean / logit-mean fusion
  E. per-member affine prior-recal (psrcal, [100/101,1/101]) THEN fusion (winner recipe)
  F. per-member weights via LEAVE-ONE-FOLD-OUT (leakage-controlled; the honest estimate)
Each scored with the organiser-exact 1000× bootstrap @1:100 (median + 2.5/97.5 CI).
Gate vs P0 0.077: report any variant whose boot-median > 0.077 AND CI2.5 > P0 CI2.5-ish
(non-overlapping-ish, [[sota-claims-rigor]]). Selection-on-scored-set is optimistic ->
Exp F is the defensible one; in-sample B/C/E are UPPER BOUNDS (if they don't win, dead).
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ppv import full_pool_metrics, bootstrap_ppv_at_90recall, ppv_at_90_recall

W = "/scratch/sc20osc/miccai-2026/RARE26/work"
RN50 = {  # member -> list of per-fold OOF csvs (cols: sample_id,target,logits_0,logits_1)
    "rn50_top1": sorted(glob.glob(f"{W}/rn50_first/top1/fold*/**/oof_val_predictions.csv", recursive=True)),
    "rn50_top2": sorted(glob.glob(f"{W}/rn50/top2/fold*/**/oof_val_predictions.csv", recursive=True)),
    "rn50_top3": sorted(glob.glob(f"{W}/rn50/top3/fold*/**/oof_val_predictions.csv", recursive=True)),
    "rn50_top4": sorted(glob.glob(f"{W}/rn50/top4/fold*/**/oof_val_predictions.csv", recursive=True)),
}
VIT = {f"vitl_{a}": sorted(glob.glob(f"{W}/vitl/RARE_dinov3_vitl16/{a}/test_predictions_fold_*.csv"))
       for a in ("top1", "top2", "top3", "top4")}


def _softmax2(l0, l1):
    z = np.stack([l0, l1], 1); z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e[:, 1] / e.sum(1)


def _recal_fold(l0, l1, labels):
    """winner per-model affine prior-recal to [100/101,1/101] -> pos prob."""
    import torch
    from psrcal.calibration import calibrate, AffineCalLogLoss
    preds = np.stack([l0, l1], 1)
    cal, _ = calibrate(trnscores=torch.tensor(preds, dtype=torch.float64),
                       trnlabels=torch.tensor(labels.astype(int)),
                       tstscores=torch.tensor(preds, dtype=torch.float64),
                       calclass=AffineCalLogLoss, bias=True, priors=[100/101, 1/101], quiet=True)
    cal = cal.detach().cpu().numpy()
    return _softmax2(cal[:, 0], cal[:, 1])


def load_member(csvs, recal):
    """Return dict sample_id -> (label, pos_score, fold_idx) pooled over the member's folds."""
    out = {}
    for fi, c in enumerate(csvs):
        df = pd.read_csv(c)
        ids = df["sample_id"].astype(str).values
        y = df["target"].values.astype(int)
        l0 = df["logits_0"].values.astype(float); l1 = df["logits_1"].values.astype(float)
        s = _recal_fold(l0, l1, y) if recal else (_softmax2(l0, l1)
              if not np.allclose(l0, 0) else 1/(1+np.exp(-l1)))
        for i, lab, sc in zip(ids, y, s):
            out[i] = (int(lab), float(sc), fi)
    return out


def assemble(members, recal):
    """-> ids, labels, folds, S[n_members, n]  aligned by sample_id (intersection)."""
    mdicts = {m: load_member(csvs, recal) for m, csvs in members.items()}
    ids = sorted(set.intersection(*[set(d) for d in mdicts.values()]))
    labels = np.array([next(iter(mdicts.values()))[i][0] for i in ids], int)
    folds = np.array([next(iter(mdicts.values()))[i][2] for i in ids], int)
    S = np.stack([[mdicts[m][i][1] for i in ids] for m in members], 0)  # [M,n]
    return ids, labels, folds, S, list(members)


def fast_boot(labels, s, n=1000, min_neo=1000, ratio=100, seed=0):
    """PPV@90R-ONLY bootstrap (organiser-exact math, ~3x faster: no AUROC/AUPRC per draw).
    PPV@90R = precision at the highest-threshold point whose recall>=0.9 (== sklearn
    precision_recall_curve idx[recall>=0.9][-1]). Single argsort + cumsum per draw."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels); s = np.asarray(s, float)
    pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
    P = min_neo; target = 0.9 * P
    ppvs = np.empty(n)
    for b in range(n):
        ps = rng.choice(pos, P, True); ns = rng.choice(neg, P * ratio, True)
        sc = np.concatenate([s[ps], s[ns]])
        y = np.concatenate([np.ones(P), np.zeros(P * ratio)])
        order = np.argsort(-sc, kind="stable")
        cum = np.cumsum(y[order])                 # cumulative TP down the ranking
        k = int(np.searchsorted(cum, target, side="left"))  # first point with recall>=0.9
        k = min(k, len(cum) - 1)
        ppvs[b] = cum[k] / (k + 1.0)              # precision there
    return {"boot_med": float(np.median(ppvs)),
            "ci": [float(np.percentile(ppvs, 2.5)), float(np.percentile(ppvs, 97.5))]}


SWEEP_MIN_NEO = 250   # sweep speed: 1:100 ratio preserved -> median ~invariant (confirm at 1000)


def score(labels, s, n=1000, seed=0):
    fp = full_pool_metrics(labels, s)
    bs = fast_boot(labels, s, n=n, min_neo=SWEEP_MIN_NEO, seed=seed)
    return {"full": round(fp["PPV@90R_full"], 4), "auroc": round(fp["AUROC_full"], 4),
            "boot_med": round(bs["boot_med"], 4),
            "ci": [round(bs["ci"][0], 4), round(bs["ci"][1], 4)]}


def ranknorm(x):  # ranks in [0,1]
    r = np.empty_like(x, float); r[np.argsort(x)] = np.arange(len(x)); return r/(len(x)-1)


def fuse(S, rule, w=None):
    M = S.shape[0]
    if w is None: w = np.ones(M)/M
    w = np.asarray(w, float); w = w/w.sum()
    if rule == "mean":     return (w[:, None]*S).sum(0)
    if rule == "rankmean": return (w[:, None]*np.stack([ranknorm(S[m]) for m in range(M)])).sum(0)
    if rule == "noisy_or": return 1 - np.prod((1 - S*w[:, None]*M)  .clip(1e-9, 1), 0)  # weight-scaled
    if rule == "max":      return S.max(0)
    if rule == "geo":      return np.exp((w[:, None]*np.log(S.clip(1e-9, 1))).sum(0))
    if rule == "logit":    z = np.log(S.clip(1e-6, 1-1e-6)/(1-S.clip(1e-6, 1-1e-6)));  \
                           return 1/(1+np.exp(-(w[:, None]*z).sum(0)))
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=f"{W}/calib_fusion")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {}
    P0 = 0.077  # bar: single P0 RN50-top1 recal boot-median

    def log(tag, labels, s):
        r = score(labels, s, args.n); results[tag] = r
        win = "  <-- BEATS P0" if r["boot_med"] > P0 and r["ci"][0] >= 0.061 else ""
        print(f"  {tag:38s} full={r['full']:.3f} auroc={r['auroc']:.3f} "
              f"boot={r['boot_med']:.4f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]{win}")

    print("\n=== assembling members (RECAL per-fold, winner recipe) ===")
    allm = {**RN50, **VIT}
    ids, labels, folds, S, names = assemble(allm, recal=True)
    print(f"  {len(ids)} images | pos={int(labels.sum())} | members={names}")
    idx = {m: k for k, m in enumerate(names)}

    # per-member individual metrics (for weighting + single-member scan)
    print("\n[Exp0] single-member scan (recal):")
    mem_ppv, mem_auprc = {}, {}
    for m in names:
        r = score(labels, S[idx[m]], args.n); mem_ppv[m] = r["boot_med"]
        from bootstrap_ppv import full_pool_metrics as fpm
        mem_auprc[m] = fpm(labels, S[idx[m]])["AUPRC_full"]
        log(f"single:{m}", labels, S[idx[m]])

    # ---------- Exp A: monotonic-invariance demonstration ----------
    print("\n[ExpA] monotonic calibration on pooled P0 vector (MUST be a no-op for PPV@90R):")
    p0 = S[idx["rn50_top1"]]
    log("A:P0_raw(recal-pooled)", labels, p0)
    # isotonic (monotonic) + temperature (monotonic) + a random strictly-increasing map
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(p0, labels)
    log("A:P0+isotonic", labels, iso.predict(p0))
    log("A:P0+temp0.5", labels, p0**0.5)                       # monotone
    log("A:P0+logit*3", labels, 1/(1+np.exp(-3*np.log(p0.clip(1e-6,1-1e-6)/(1-p0.clip(1e-6,1-1e-6))))))

    # ---------- Exp B/D: fusion rules (equal weight) over member subsets ----------
    subsets = {"RN50-4": list(RN50), "VIT-4": list(VIT), "ALL-8": names,
               "RN50top1+top3": ["rn50_top1", "rn50_top3"]}
    for sub, mem in subsets.items():
        Ss = S[[idx[m] for m in mem]]
        for rule in ("mean", "rankmean", "noisy_or", "max", "geo", "logit"):
            log(f"B:{sub}:{rule}", labels, fuse(Ss, rule))

    # ---------- Exp B: weighted fusion (w ∝ per-member metric), in-sample (UPPER BOUND) ----------
    print("\n[ExpB] weighted fusion — weights from per-member metric (IN-SAMPLE = optimistic):")
    for wname, wsrc in [("wPPV", mem_ppv), ("wAUPRC", mem_auprc)]:
        for sub, mem in subsets.items():
            Ss = S[[idx[m] for m in mem]]
            w = np.array([wsrc[m] for m in mem]);
            if w.sum() == 0: continue
            for rule in ("mean", "rankmean"):
                log(f"B:{sub}:{rule}:{wname}", labels, fuse(Ss, rule, w))
                log(f"B:{sub}:{rule}:{wname}^4", labels, fuse(Ss, rule, w**4))

    # ---------- Exp C: top-k member selection by per-member PPV ----------
    print("\n[ExpC] top-k members by per-member PPV (in-sample = optimistic):")
    order = sorted(names, key=lambda m: -mem_ppv[m])
    for k in (1, 2, 3, 4):
        mem = order[:k]; Ss = S[[idx[m] for m in mem]]
        log(f"C:topk={k}({'+'.join(mem)}):mean", labels, fuse(Ss, "mean"))
        log(f"C:topk={k}:rankmean", labels, fuse(Ss, "rankmean"))

    # ---------- Exp F: LEAVE-ONE-FOLD-OUT weighting (leakage-controlled, DEFENSIBLE) ----------
    print("\n[ExpF] LOFO-weighted fusion (weights from other folds -> honest):")
    for sub, mem in subsets.items():
        Ss = S[[idx[m] for m in mem]]
        for rule in ("mean", "rankmean"):
            fused = np.zeros(len(ids))
            for f in np.unique(folds):
                tr = folds != f; te = folds == f
                # weight each member by its PPV on the TRAIN folds only
                w = np.array([ppv_at_90_recall(labels[tr], Ss[k][tr]) for k in range(len(mem))])
                w = np.nan_to_num(w, nan=0.0)
                if w.sum() == 0: w = np.ones(len(mem))
                fused[te] = fuse(Ss[:, te], rule, w)
            log(f"F:{sub}:{rule}:LOFO-wPPV", labels, fused)

    # summary
    best = max(results.items(), key=lambda kv: kv[1]["boot_med"])

    # ---- CONFIRM P0 + the best (non-invariance) variant at the FULL organiser bootstrap (min_neo=1000) ----
    print("\n[confirm] full organiser bootstrap (min_neo=1000, AUROC/AUPRC incl.):")
    confirm = {}
    best_nonA = max((k for k in results if not k.startswith("A:")),
                    key=lambda k: results[k]["boot_med"])
    to_confirm = {"P0:rn50_top1_recal": S[idx["rn50_top1"]]}
    if best_nonA.startswith("single:"):
        to_confirm[f"BEST:{best_nonA}"] = S[idx[best_nonA.split(":", 1)[1]]]
    for tag, vec in to_confirm.items():
        b = bootstrap_ppv_at_90recall(labels, vec, n_bootstrap=args.n, ndbe_multiplier=100, seed=0)
        confirm[tag] = {"boot_med": round(b["PPV@90R_median"], 4),
                        "ci": [round(b["PPV@90R_CI2.5"], 4), round(b["PPV@90R_CI97.5"], 4)]}
        print(f"  {tag}: boot-median={b['PPV@90R_median']:.4f} "
              f"[{b['PPV@90R_CI2.5']:.3f},{b['PPV@90R_CI97.5']:.3f}]")
    print(f"  (sweep bar via fast_boot min_neo={SWEEP_MIN_NEO}: "
          f"P0={results['single:rn50_top1']['boot_med']}, best({best_nonA})={results[best_nonA]['boot_med']})")
    print(f"\n=== BEST boot-median: {best[0]} = {best[1]['boot_med']:.4f} "
          f"CI{best[1]['ci']} (P0 bar = {P0}) ===")
    beats = {k: v for k, v in results.items()
             if v["boot_med"] > P0 and v["ci"][0] >= 0.061 and not k.startswith("A:")}
    print(f"=== variants beating P0 0.077 with CI2.5>=~0.061: "
          f"{list(beats) if beats else 'NONE'} ===")
    with open(os.path.join(args.out, "results.json"), "w") as fjson:
        json.dump({"P0_bar": P0, "results": results, "best": best[0],
                   "beats_P0": list(beats), "confirm_full_boot": confirm}, fjson, indent=2)
    print(f"Wrote {args.out}/results.json")


if __name__ == "__main__":
    main()
