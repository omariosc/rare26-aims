#!/usr/bin/env python3
"""
Build the RARE26 held-out SELF-VAL split from the 3,095 labeled train images.

The closed val/test are private; we make our OWN held-out split to estimate the
ranking metric (1:100 bootstrap PPV@90%R) locally and multi-seed.

Protocol (matches dossier P0-P4 validation):
  - Expect organiser layout: <data>/center_{1,2}/{ndbe,neo}/*.png  (target ndbe=0, neo=1).
  - PATIENT-DISJOINT where possible: RARE25 filenames don't always expose patient id, so we
    fall back to STRATIFIED-by-(center,class) image-level splitting; if a patient id can be
    parsed it is used to keep a patient on one side (prevents leakage).
  - Produce 5-fold StratifiedKFold (fold col) + a held-out test fold for the headline,
    written as splits/selfval_splits.csv with columns:
        image_path,sample_id,center,class_name,target,patient_id,fold,split
    `split in {train,selfval}` marks the headline held-out fold (fold 0 -> selfval by default).
  - The remaining folds 1..4 are the CV folds for ensemble training.

Usage:
  python make_selfval_split.py --data /scratch/sc20osc/miccai-2026/RARE26/data/train \
                               --out  /scratch/sc20osc/miccai-2026/RARE26/splits --seed 0 --holdout-fold 0
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CLASS_MAP = {"ndbe": 0, "neo": 1}
CENTERS = ["center_1", "center_2"]
PAT_RE = re.compile(r"(?:pat|patient|case)[_-]?(\d+)", re.IGNORECASE)


def collect(data_dir: Path) -> pd.DataFrame:
    rows = []
    for center in CENTERS:
        for cname, target in CLASS_MAP.items():
            cdir = data_dir / center / cname
            if not cdir.exists():
                print(f"WARN missing {cdir}")
                continue
            for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
                for fp in cdir.glob(ext):
                    m = PAT_RE.search(fp.name)
                    pid = m.group(1) if m else None
                    rows.append(dict(
                        image_path=str(fp.relative_to(data_dir)),
                        sample_id=fp.name, center=center, class_name=cname,
                        target=target,
                        patient_id=(f"{center}_{pid}" if pid else f"{center}_{cname}_{fp.stem}"),
                    ))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No images under {data_dir} (expected center_*/{{ndbe,neo}}/*.png). "
                         "Data is still GATED — run scripts/download_data.sh first.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-fold", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = collect(data)
    print(f"Collected {len(df)} imgs | pos(neo)={int(df.target.sum())} neg(ndbe)={int((df.target==0).sum())}")
    print(df.groupby(['center', 'class_name']).size())

    groups = df["patient_id"].values
    has_real_groups = df["patient_id"].nunique() < len(df)  # patients actually parsed
    df["fold"] = -1
    try:
        if has_real_groups:
            skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            it = skf.split(df, df["target"], groups)
            print("Using StratifiedGroupKFold (patient-disjoint).")
        else:
            raise ValueError("no real patient groups")
    except Exception as e:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        it = skf.split(df, df["target"])
        print(f"Using StratifiedKFold (image-level): {e}")

    for fold_id, (_, val_idx) in enumerate(it):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_id

    df["split"] = np.where(df["fold"] == args.holdout_fold, "selfval", "train")
    outcsv = out / "selfval_splits.csv"
    df.to_csv(outcsv, index=False)
    print(f"\nWrote {outcsv}")
    print(df.groupby(['split', 'class_name']).size())
    n_sv_pos = int(((df.split == 'selfval') & (df.target == 1)).sum())
    print(f"Self-val held-out NEO count = {n_sv_pos} "
          f"({'OK for 1:100 bootstrap' if n_sv_pos >= 15 else 'LOW — bootstrap CI will be wide'})")


if __name__ == "__main__":
    main()
