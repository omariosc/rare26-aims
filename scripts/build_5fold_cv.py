#!/usr/bin/env python3
"""
Build the RN50 / ViT-L loader's `5fold_cv.csv` from `selfval_splits.csv`.

WHY THIS EXISTS (verified from data_splitting/split_loader.py::get_fold_data):
  The winner repo's SplitLoader, for split_name == '5fold_cv', masks rows by
      train: df['split'] != f'fold_{fold}'
      test : df['split'] == f'fold_{fold}'
  i.e. it REQUIRES a column literally named `split` whose values are the strings
  `fold_0 .. fold_4`. It also reads `image_path` and `target`.

  But `make_selfval_split.py` writes `selfval_splits.csv` with `split in {train, selfval}`
  (a held-out-fold marker) and the per-row fold id in a SEPARATE integer `fold` column.
  => A bare `ln -s selfval_splits.csv 5fold_cv.csv` is WRONG: the `split` values don't
     match what the loader greps for, so EVERY row lands in `train` and the val set is
     empty. This script does the correct remap: `split <- 'fold_' + str(fold)`.

Output columns (superset is fine; loader only needs image_path,target,split; center kept
for get_split_info): image_path, sample_id, center, class_name, target, split

Usage:
  python build_5fold_cv.py --selfval /scratch/.../RARE26/splits/selfval_splits.csv \
                           --out     /scratch/.../RARE26/splits/5fold_cv.csv
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfval", required=True, help="Path to selfval_splits.csv (from make_selfval_split.py)")
    ap.add_argument("--out", required=True, help="Path to write 5fold_cv.csv (loader format)")
    args = ap.parse_args()

    df = pd.read_csv(args.selfval)
    if "fold" not in df.columns:
        raise SystemExit(f"{args.selfval} has no `fold` column — re-run make_selfval_split.py.")
    # The loader greps split == 'fold_{k}'. Map the integer fold id -> that exact string.
    out = df.copy()
    out["split"] = out["fold"].apply(lambda f: f"fold_{int(f)}")
    # center column: split_loader.get_split_info expects a `center` col; keep what we have.
    keep = [c for c in ["image_path", "sample_id", "center", "class_name", "target", "split"] if c in out.columns]
    out = out[keep]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(out)} rows)")
    print(out.groupby(["split", "target"]).size())
    # sanity: every fold has both classes for a usable val
    for fk in sorted(out["split"].unique()):
        sub = out[out["split"] == fk]
        pos = int((sub["target"] == 1).sum())
        print(f"  {fk}: n={len(sub)} pos(neo)={pos} {'(no NEO in val!)' if pos == 0 else ''}")


if __name__ == "__main__":
    main()
