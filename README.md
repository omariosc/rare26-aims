# AIMS-R30 — RARE 2026 Challenge submission

Detection of Barrett's neoplasia (NEO) against non-dysplastic Barrett's oesophagus (NDBE),
submitted to the [RARE 2026 challenge](https://rare26.grand-challenge.org/) by team **AIMS**,
University of Leeds.

A tail-trained, 30-member DINOv3 LoRA ensemble built on a single frozen self-supervised backbone.

## Method

| Component | Choice |
|---|---|
| Backbone | DINOv3 ViT-L/16, LVD-1689M weights, **frozen** |
| Adaptation | LoRA adapters, rank 32, alpha 64, on every `nn.Linear`, plus a 2-way head |
| Trainable tensors | 194 per member; the backbone is never updated |
| Ensemble | 30 members = 5 cross-validation folds x 6 seeds (40-45), fused by arithmetic mean |
| Training objective | Partial-AUC restricted to the low-FPR region, with hard-negative mining |
| Inference-time augmentation | Multi-crop max over 5 crops, plus 336-pixel resolution TTA |

### Why the objective is the contribution

The ranking metric is PPV at 90% recall, which depends only on the extreme tail of the score
distribution. Post-hoc threshold calibration cannot change that metric, because it is invariant
to monotone rescaling; re-ordering the tail *during training* can. We therefore optimise a
partial-AUC objective restricted to the low-false-positive region rather than training for
overall separability and thresholding afterwards.

## Results

Reported honestly, and the two columns are **not** comparable to each other.

| Stage | PPV@90recall (internal split) |
|---|---|
| Baseline linear probe | 0.077 |
| **+ tail-focused training (pAUC + hard negatives)** | **0.285** |
| + 3-seed ensembling | 0.351 |

On the **Open Development leaderboard** the same container scored **PPV@90recall = 0.0200**
(8th of 66 teams; leader 0.0335). The internal figures use a different and more favourable
split, so they should not be read as leaderboard-comparable. We were misled by this ourselves
early on and corrected it.

The step from 0.077 to 0.285 is the methodological result, with separated confidence intervals.

### Negative results

All were run as paired designs and are recorded because they constrain what is worth trying next.

- Scaling the ensemble from 3 to 10 seeds gave no separable gain. Members' errors are correlated
  because the training data comes from few centres.
- A CNN x ViT heterogeneous ensemble did not help on either split; ResNet-50 dilutes.
- An in-domain self-supervised pretraining arm failed, **and so did its non-SSL control**, which
  is what shows there was no SSL-specific effect to find.
- Multi-crop-max and 336-resolution TTA did pass, and a `tta224max` control isolates the
  mechanism as the max operator combined with resolution.

## Limitations

1. **Centre coverage is the binding constraint.** Few centres means correlated member errors, so
   diversity levers do not pay. More centres, not more parameters.
2. **Metric variance is large relative to the field.** The bootstrap CI of PPV@90recall is wider
   than the spread of the leaderboard, so small between-team differences should not be
   over-interpreted.
3. Early internal cross-validation pooled scores across folds, which inflated some absolute
   numbers. Per-fold, three independent estimators agree, and the pooling artefact does not exist
   at inference, where there is one ensemble and one score scale.

## Repository layout

```
src/inference.py     Grand Challenge entry point (the submitted container's inference)
src/rare_model.py    Architecture: frozen DINOv3 + LoRA + linear head
docker/Dockerfile    Container definition
docker/contract_test.sh   Interface conformance checks run before submission
scripts/             Training, cross-validation, fusion and statistics utilities
```

## Reproducing

DINOv3 is **not** included here (see `THIRD_PARTY_NOTICES.md`). Obtain it separately:

```bash
git clone https://github.com/facebookresearch/dinov3
export DINOV3_REPO=$PWD/dinov3
```

Build and run the container:

```bash
docker build -t aims-r30 -f docker/Dockerfile .
docker run --rm --gpus all --network none \
  -v /path/to/input:/input:ro -v /path/to/output:/output aims-r30
```

Runtime: base image `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`, 5.5 GB, runs on an A10G
within the 600 s per case limit. The backbone loads with `pretrained=False` plus `torch.load`,
so no network path exists at runtime.

### Weights

The 30 member checkpoints total roughly 2.6 GB and are therefore not in Git. They are available
on request from the contact below.

## Licence

Code in this repository is released under the [MIT Licence](LICENSE). Third-party components,
including DINOv3, remain under their own terms — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

Omar Choudhry — <O.Choudhry@leeds.ac.uk>
Artificial Intelligence in Medicine and Surgery Group, School of Computer Science,
University of Leeds, Leeds LS2 9JT, United Kingdom
ORCID [0000-0003-4434-3550](https://orcid.org/0000-0003-4434-3550)

Funded by UKRI Engineering and Physical Sciences Research Council (EPSRC), grant EP/S024336/1.
