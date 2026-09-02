# AIMS-R30 (RARE 2026)

Barrett's neoplasia (NEO) vs non-dysplastic Barrett's oesophagus (NDBE) classification, submitted
to the [RARE 2026 challenge](https://rare26.grand-challenge.org/) by team AIMS, University of Leeds.

A 30-member ensemble of LoRA adapters on a single frozen DINOv3 backbone, trained against a
tail-focused objective.

## Method

| Component | Choice |
|---|---|
| Backbone | DINOv3 ViT-L/16, LVD-1689M weights, frozen |
| Adaptation | LoRA, rank 32, alpha 64, applied to every `nn.Linear`, plus a 2-way head |
| Trainable tensors | 194 per member. The backbone is never updated. |
| Ensemble | 30 members (5 CV folds x 6 seeds, 40 to 45), averaged |
| Objective | Partial AUC over the low-FPR region, with hard-negative mining |
| Test-time augmentation | Multi-crop max over 5 crops, plus 336px resolution |

The ranking metric is PPV at 90% recall, which only looks at the extreme tail of the score
distribution. That metric is invariant to monotone rescaling, so adjusting the threshold after
training does nothing to it. Changing the ordering of the tail during training does. So we train a
partial-AUC objective restricted to the low-false-positive region instead of optimising overall
separability and picking a threshold afterwards.

## Results

| Stage | PPV@90recall (internal split) |
|---|---|
| Baseline linear probe | 0.077 |
| + tail-focused training (pAUC + hard negatives) | 0.285 |
| + 3-seed ensembling | 0.351 |

The same container scored PPV@90recall = 0.0200 on the Open Development leaderboard, placing 8th of
66 teams against a leader at 0.0335.

The two sets of numbers are not comparable. The internal split is different and easier, so the table
above should be read only as a relative comparison between stages, not as a leaderboard estimate.
The 0.077 to 0.285 step has separated confidence intervals.

### What did not work

All of these were run as paired comparisons.

* Going from 3 to 10 seeds in the ensemble gave no separable gain. The training data covers few
  centres, so member errors are correlated.
* A mixed CNN and ViT ensemble did not help on either split. ResNet-50 dilutes the average.
* In-domain self-supervised pretraining failed, and so did its non-SSL control. Since the control
  failed too, there was no SSL-specific effect to find.
* Multi-crop max and 336px TTA both passed. A `tta224max` control narrows the cause to the max
  operator combined with the higher resolution.

## Limitations

1. Centre coverage limits this method more than anything else. With few centres, ensemble members
   make correlated errors, so adding diversity does not pay off.
2. The bootstrap CI on PPV@90recall is wider than the spread across the leaderboard, so small gaps
   between teams do not mean much.
3. Early internal cross-validation pooled scores across folds, which inflated some absolute numbers.
   Per-fold, three independent estimators agree. The pooling artefact does not exist at inference,
   where there is one ensemble and one score scale.

## Layout

```
src/inference.py        Grand Challenge entry point
src/rare_model.py       Frozen DINOv3 + LoRA + linear head
docker/Dockerfile       Container definition
docker/contract_test.sh Interface checks run before submission
scripts/                Training, cross-validation, fusion and statistics
```

## Running it

DINOv3 is not included here (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Get it separately:

```bash
git clone https://github.com/facebookresearch/dinov3
export DINOV3_REPO=$PWD/dinov3
```

Then:

```bash
docker build -t aims-r30 -f docker/Dockerfile .
docker run --rm --gpus all --network none \
  -v /path/to/input:/input:ro -v /path/to/output:/output aims-r30
```

Base image is `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`, 5.5 GB. It runs on an A10G inside the
600 s per case limit. The backbone loads with `pretrained=False` and `torch.load`, so there is no
network path at runtime.

### Weights

The 30 member checkpoints come to about 2.6 GB, so they are not in Git. Available on request.

## Licence

MIT, see [LICENSE](LICENSE). Third-party components including DINOv3 stay under their own terms,
described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

Omar Choudhry, <O.Choudhry@leeds.ac.uk>
Artificial Intelligence in Medicine and Surgery Group, School of Computer Science,
University of Leeds, Leeds LS2 9JT, UK.
ORCID [0000-0003-4434-3550](https://orcid.org/0000-0003-4434-3550)

Funded by UKRI EPSRC grant EP/S024336/1.
