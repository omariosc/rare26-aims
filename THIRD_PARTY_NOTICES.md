# Third-party components

The MIT licence in `LICENSE` covers **the code in this repository only**. It does not
and cannot relicense the third-party components below, which remain under their own terms.

## DINOv3 (Meta Platforms, Inc.)

The backbone is DINOv3 ViT-L/16 with the LVD-1689M pretrained checkpoint. The DINOv3
source and weights are **deliberately not vendored here**, because they are distributed
under the DINOv3 Licence, which is not MIT and does not permit relicensing.

To reproduce, obtain DINOv3 separately from https://github.com/facebookresearch/dinov3
and accept Meta's terms. `src/rare_model.py` expects the repository path in the
`DINOV3_REPO` environment variable (default `/opt/app/dino_repo`) and the checkpoint
path passed as `weight_path`.

## Other dependencies

PyTorch (BSD-3-Clause), SimpleITK (Apache-2.0), NumPy (BSD-3-Clause), and the base image
`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`, each under its own licence.

## Data

No challenge data is included in this repository. The RARE challenge training data is
distributed by the organisers under their own terms and is not ours to redistribute.
