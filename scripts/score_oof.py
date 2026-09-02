#!/usr/bin/env python3
"""
Score the winner-repo training output on the held-out fold with the RANKING metric.

GLUE: maps the RN50/ViT-L training output -> a (label, score) table -> the
organiser-exact bootstrap PPV@90%R scorer (scripts/bootstrap_ppv.py).

WINNER OUTPUT FORMAT (verified from training/train.py + training/evaluation.py::
create_predictions_dataframe): each run writes
    <output_dir>/checkpoints_<ts>_..._<loss>_..._5fold_cv/oof_val_predictions.csv
with columns: image_path, sample_id, center, target, logits_0, logits_1, fold, split_type.
  - `target`     = ground-truth label (0=ndbe, 1=neo)  -> bootstrap `label`
  - positive score = softmax([logits_0, logits_1])[:, 1] -> bootstrap `score`
(For the 2-logit head this softmax prob is exactly what inference.py pools; for a 1-logit
head logits_1 holds the single logit and logits_0==0 -> we sigmoid(logits_1) instead.)

OPTIONAL --recalibrate applies the winner's affine prior-recal to priors [100/101, 1/101]
(psrcal.AffineCalLogLoss, same as training/recalibrate_files.py) before softmax — this is
the low-prevalence lever and is the score the leaderboard actually sees. We report BOTH raw
and recalibrated so the lever's effect is visible per the §4b diagnostic discipline.

Usage:
  # auto-discover the newest oof_val_predictions.csv under a run dir, score it:
  python score_oof.py --run_dir /scratch/.../RARE26/work/rn50/top1/fold0
  # or point straight at the csv:
  python score_oof.py --oof_csv .../oof_val_predictions.csv --recalibrate
  # multiple OOF csvs (e.g. all folds) pooled into one held-out score:
  python score_oof.py --oof_csv a.csv b.csv c.csv d.csv e.csv --out /scratch/.../score
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

# Import the (already-verified) organiser-exact scorer.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ppv import full_pool_metrics, bootstrap_ppv_at_90recall


def _softmax2(l0, l1):
    z = np.stack([l0, l1], axis=1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def oof_to_label_score(df, recalibrate=False):
    """Return (labels, pos_scores) from one oof_val_predictions.csv DataFrame."""
    labels = df["target"].values.astype(int)
    l0 = df["logits_0"].values.astype(float)
    l1 = df["logits_1"].values.astype(float)
    # 1-logit head convention in create_predictions_dataframe: logits_0 == 0, logits_1 = logit
    one_logit = np.allclose(l0, 0.0)
    if recalibrate:
        try:
            import torch
            from psrcal.calibration import calibrate, AffineCalLogLoss
            preds = np.stack([l0, l1], axis=1)
            cal, (t, b) = calibrate(
                trnscores=torch.tensor(preds, dtype=torch.float64),
                trnlabels=torch.tensor(labels),
                tstscores=torch.tensor(preds, dtype=torch.float64),
                calclass=AffineCalLogLoss, bias=True,
                priors=[100 / 101, 1 / 101], quiet=True,
            )
            cal = cal.detach().cpu().numpy()
            # cal are log-posteriors; positive-class score = exp(col1) (already normalised) or softmax
            scores = _softmax2(cal[:, 0], cal[:, 1])
            return labels, scores
        except Exception as e:
            print(f"[score_oof] WARN recalibration failed ({e}); falling back to raw softmax.")
    if one_logit:
        return labels, 1.0 / (1.0 + np.exp(-l1))
    return labels, _softmax2(l0, l1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", help="Dir to search recursively for oof_val_predictions.csv (newest used).")
    ap.add_argument("--oof_csv", nargs="*", help="One or more oof_val_predictions.csv paths (pooled).")
    ap.add_argument("--recalibrate", action="store_true", help="Apply affine prior-recal to [100/101,1/101] before scoring.")
    ap.add_argument("--ratio", type=int, default=100)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Dir to write label_score.csv + summary.json.")
    args = ap.parse_args()

    csvs = list(args.oof_csv or [])
    if args.run_dir:
        found = sorted(glob.glob(os.path.join(args.run_dir, "**", "oof_val_predictions.csv"), recursive=True),
                       key=os.path.getmtime)
        if not found:
            sys.exit(f"No oof_val_predictions.csv under {args.run_dir}")
        csvs.append(found[-1])  # newest
    if not csvs:
        sys.exit("Provide --run_dir or --oof_csv.")

    all_lab, all_sc = [], []
    for c in csvs:
        df = pd.read_csv(c)
        lab, sc = oof_to_label_score(df, recalibrate=args.recalibrate)
        all_lab.append(lab); all_sc.append(sc)
        print(f"[score_oof] {c}: n={len(df)} pos={int(lab.sum())}")
    labels = np.concatenate(all_lab)
    scores = np.concatenate(all_sc)

    fp = full_pool_metrics(labels, scores)
    bs = bootstrap_ppv_at_90recall(labels, scores, n_bootstrap=args.n,
                                   ndbe_multiplier=args.ratio, seed=args.seed)
    tag = "RECALIBRATED" if args.recalibrate else "RAW"
    print(f"=== FULL POOL ({tag}; IMSY winner full-pool = 0.320) ===")
    for k, v in fp.items():
        print(f"  {k}: {v}")
    print(f"=== BOOTSTRAP 1:{args.ratio} ({tag} RANKING METRIC; IMSY ~0.035) ===")
    for k, v in bs.items():
        if k != "raw_ppv":
            print(f"  {k}: {v}")

    if args.out:
        import json
        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame({"label": labels, "score": scores}).to_csv(
            os.path.join(args.out, "label_score.csv"), index=False)
        bs.pop("raw_ppv", None)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"tag": tag, "full_pool": fp, "bootstrap": bs, "sources": csvs}, f, indent=2)
        print(f"Wrote {args.out}/label_score.csv + summary.json")


if __name__ == "__main__":
    main()
