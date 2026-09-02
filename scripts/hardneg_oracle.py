#!/usr/bin/env python3
"""
RARE26 — HARD-NEGATIVE ORACLE diagnostic (CPU, decisive). Is the FPR@90R gap
(10.9% -> 1.93% needed for PPV 0.320) RECOVERABLE by tail-training, or data-floored?

Steps:
  1. pool P0 (rn50_top1, per-fold affine-recal) OOF -> (label, score, image_path).
  2. threshold @ 90% recall -> FPR@90R, PPV, the set of NDBE FALSE POSITIVES (score>=thr).
  3. handcrafted image stats (glare / blur / brightness / redness / texture / colorfulness ...)
     for every image -> interpretable, model-independent artifact signatures.
  4. cluster the hard-neg FPs (k-means) -> taxonomy; save a montage per cluster.
  5. ORACLE-FPR: threshold @90R is set by the NEO score dist (NOT by NDBE), so removing NDBE
     FPs does NOT move it -> oracle_FPR = (#FP not fixed)/#NDBE. Report FPR after fixing the
     top-k worst clusters + FP-mass concentration.
  6. DISCRIMINABILITY: 5-fold logistic-regression AUC for {FP-NDBE vs true-NEO} and
     {FP-NDBE vs correctly-low NDBE} on the stats. High AUC(FP vs NEO) => a learnable
     artifact signal the model ignores => RECOVERABLE. ~0.5 => FPs are NEO-mimics => floor.
  7. VERDICT heuristic.
"""
import glob, json, os
import numpy as np
import pandas as pd
import cv2

W = "/scratch/sc20osc/miccai-2026/RARE26/work"
OUT = f"{W}/hardneg_oracle"; os.makedirs(OUT, exist_ok=True)


def softmax2(l0, l1):
    z = np.stack([l0, l1], 1); z -= z.max(1, keepdims=True); e = np.exp(z)
    return e[:, 1] / e.sum(1)


def recal_fold(l0, l1, y):
    import torch
    from psrcal.calibration import calibrate, AffineCalLogLoss
    preds = np.stack([l0, l1], 1)
    cal, _ = calibrate(trnscores=torch.tensor(preds, dtype=torch.float64),
                       trnlabels=torch.tensor(y.astype(int)),
                       tstscores=torch.tensor(preds, dtype=torch.float64),
                       calclass=AffineCalLogLoss, bias=True, priors=[100/101, 1/101], quiet=True)
    cal = cal.detach().cpu().numpy(); return softmax2(cal[:, 0], cal[:, 1])


def pool_p0():
    rows = []
    for c in sorted(glob.glob(f"{W}/rn50_first/top1/fold*/**/oof_val_predictions.csv", recursive=True)):
        df = pd.read_csv(c)
        y = df["target"].values.astype(int)
        s = recal_fold(df["logits_0"].values.astype(float), df["logits_1"].values.astype(float), y)
        rows.append(pd.DataFrame({"sample_id": df["sample_id"].astype(str), "label": y,
                                  "score": s, "image_path": df["image_path"]}))
    return pd.concat(rows, ignore_index=True)


def ppv_fpr_at_90(labels, scores):
    order = np.argsort(-scores); ys = labels[order]; ss = scores[order]
    P = labels.sum(); N = len(labels) - P
    cum_tp = np.cumsum(ys); recall = cum_tp / P
    k = int(np.searchsorted(cum_tp, 0.9 * P, side="left")); k = min(k, len(ys) - 1)
    thr = ss[k]
    tp = cum_tp[k]; fp = (k + 1) - tp
    ppv = tp / (k + 1); fpr = fp / N
    return thr, float(ppv), float(fpr), int(fp), int(N)


def img_stats(path):
    im = cv2.imread(path)
    if im is None:
        return None
    im = cv2.resize(im, (224, 224))
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    V, S = hsv[..., 2], hsv[..., 1]
    R, G, B = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    rg = rgb[..., 0] - rgb[..., 1]; yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
    return dict(bright=V.mean()/255, glare=float(np.mean((V > 240) & (S < 40))),
                dark=float(np.mean(V < 40)), blur=float(lap), sat=S.mean()/255,
                redness=float(R/(G+B+1e-6)), colorful=float(colorfulness),
                edge=float(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1)).mean()),
                contrast=float(gray.std()))


def montage(paths, fn, cols=6, cell=140):
    n = min(len(paths), 24); rows = (n + cols - 1) // cols
    canvas = np.full((rows*cell, cols*cell, 3), 30, np.uint8)
    for i, p in enumerate(paths[:n]):
        im = cv2.imread(p)
        if im is None: continue
        im = cv2.resize(im, (cell-4, cell-4)); r, c = i//cols, i % cols
        canvas[r*cell+2:r*cell+2+cell-4, c*cell+2:c*cell+2+cell-4] = im
    cv2.imwrite(fn, canvas)


def main():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    df = pool_p0()
    labels = df["label"].values
    scores = df["score"].values
    thr, ppv, fpr, nfp, nneg = ppv_fpr_at_90(labels, scores)
    print(f"[base] @90%recall: thr={thr:.4f} PPV={ppv:.4f} FPR={fpr:.4f} "
          f"(#NDBE-FP={nfp}/{nneg}); need FPR<=0.0193 for PPV~0.32")

    print("[stats] computing image stats for all", len(df), "images ...")
    feats = []
    for p in df["image_path"]:
        s = img_stats(p); feats.append(s if s else {})
    fdf = pd.DataFrame(feats).fillna(0.0)
    keys = list(fdf.columns)
    df = pd.concat([df.reset_index(drop=True), fdf], axis=1)
    df.to_parquet(f"{OUT}/all_stats.parquet")

    is_fp = (df["label"].values == 0) & (df["score"].values >= thr)
    is_ndbe_ok = (df["label"].values == 0) & (df["score"].values < thr)
    is_neo = df["label"].values == 1
    print(f"[sets] FP-NDBE={is_fp.sum()} | NDBE-correct={is_ndbe_ok.sum()} | NEO={is_neo.sum()}")

    # ---- cluster the FPs ----
    X = StandardScaler().fit_transform(df.loc[is_fp, keys].values)
    K = min(6, max(2, is_fp.sum() // 20))
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    cl = km.labels_
    fp_df = df.loc[is_fp].copy(); fp_df["cluster"] = cl
    print(f"\n[clusters] {K} clusters of {is_fp.sum()} hard-neg FPs (mean stats):")
    order = pd.Series(cl).value_counts()
    taxo = {}
    for cid, cnt in order.items():
        m = fp_df.loc[fp_df.cluster == cid, keys].mean()
        desc = []
        if m["glare"] > 0.02: desc.append("glare/specular")
        if m["blur"] < 60: desc.append("blurry/motion")
        if m["dark"] > 0.15: desc.append("dark/underexposed")
        if m["bright"] > 0.6: desc.append("bright")
        if m["redness"] > 1.15: desc.append("red/inflamed")
        if m["edge"] > 40: desc.append("high-texture/folds")
        if not desc: desc.append("flat/normal-looking")
        taxo[int(cid)] = {"n": int(cnt), "desc": "+".join(desc),
                          "stats": {k: round(float(m[k]), 3) for k in keys}}
        print(f"  cluster {cid}: n={cnt:3d} [{taxo[int(cid)]['desc']}]  "
              f"glare={m['glare']:.3f} blur={m['blur']:.0f} bright={m['bright']:.2f} "
              f"red={m['redness']:.2f} edge={m['edge']:.0f}")
        montage(fp_df.loc[fp_df.cluster == cid, "image_path"].tolist(),
                f"{OUT}/cluster_{cid}_{taxo[int(cid)]['desc'].replace('/','-')}.png")

    # ---- oracle FPR (threshold fixed by NEO, so removing FPs just shrinks numerator) ----
    print(f"\n[oracle] FPR@90R if worst clusters were perfectly fixed (baseline {fpr:.4f}):")
    sizes = order.values; cum = 0
    for i, (cid, cnt) in enumerate(order.items()):
        cum += cnt
        print(f"  fix top-{i+1} clusters ({cum}/{nfp} FPs, {100*cum/nfp:.0f}% mass): "
              f"oracle-FPR={ (nfp-cum)/nneg :.4f}")
    frac_top2 = sizes[:2].sum()/nfp if len(sizes) >= 2 else 1.0
    print(f"  FP-mass concentration: top-1 cluster={sizes[0]/nfp:.0%}, top-2={frac_top2:.0%}")

    # ---- discriminability: can cheap stats separate FP from NEO / from correct-NDBE? ----
    def lr_auc(mask_a, mask_b):
        Xa = df.loc[mask_a, keys].values; Xb = df.loc[mask_b, keys].values
        X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
        X = StandardScaler().fit_transform(X)
        auc = cross_val_score(LogisticRegression(max_iter=1000, class_weight="balanced"),
                              X, y, cv=5, scoring="roc_auc")
        return float(auc.mean()), float(auc.std())
    a1, s1 = lr_auc(is_fp, is_neo)
    a2, s2 = lr_auc(is_fp, is_ndbe_ok)
    print(f"\n[discrim] LR-AUC(FP-NDBE vs true-NEO)  = {a1:.3f}±{s1:.3f}  "
          f"(>0.75 => FPs are visually distinct from NEO => tail-signal exists)")
    print(f"[discrim] LR-AUC(FP-NDBE vs correct-NDBE)= {a2:.3f}±{s2:.3f}  "
          f"(>0.75 => the FPs have a systematic artifact signature)")

    # ---- verdict ----
    recoverable = (frac_top2 >= 0.5 and a2 >= 0.70) or a1 >= 0.72
    verdict = ("RECOVERABLE (artifact-concentrated / tail-signal exists) -> pursue tail-training"
               if recoverable else
               "DATA-FLOOR-LEANING (FPs diffuse & NEO-like) -> Cortex data escalation stands")
    print(f"\n=== VERDICT: {verdict} ===")
    json.dump({"base": {"thr": thr, "ppv": ppv, "fpr": fpr, "nfp": nfp, "nneg": nneg,
                        "fpr_needed_for_0.32": 0.0193},
               "taxonomy": taxo, "concentration_top2": frac_top2,
               "auc_fp_vs_neo": a1, "auc_fp_vs_correctNDBE": a2,
               "recoverable": bool(recoverable), "verdict": verdict},
              open(f"{OUT}/oracle.json", "w"), indent=2)
    print(f"Wrote {OUT}/oracle.json + montages + all_stats.parquet")


if __name__ == "__main__":
    main()
