"""RARE26 — AIMS_UK inference container entrypoint.

Grand Challenge algorithm contract (verified against TUE-ARIA/RARE25-Submission @ dbcce3d):
  IN   /input/inputs.json                                          (socket manifest)
       /input/images/stacked-barretts-esophagus-endoscopy/*.tiff   (SimpleITK -> (Z,H,W,C) uint8)
  OUT  /output/stacked-neoplastic-lesion-likelihoods.json          (plain list of Z floats in [0,1])
  RUN  --network none  --gpus all  (only /tmp and /output writable)

Method: ensemble of DINOv3-ViT-L/16 (LVD-1689M) frozen backbones + LoRA(r=32,a=64) +
2-way head, trained with CE + surrogate-PPV@0.9 + pAUC@FPR<=0.02 + hard-negative
mining ("tail training", harness tail_train4.py, inner-val checkpoint selection).
Members = {5 CV folds} x {N seeds}.  Fusion = arithmetic MEAN of the positive-class
probability -- the rule the 2026-07-30 audit selected for the *domain-shifted*
regime, which is what a multi-centre hidden test is.

DELIVERY SAFETY NET (declared, not hidden): a format-valid output of all-0.5 is
written BEFORE the model is touched, so a crash still yields a parseable file
rather than a failed job.  An all-0.5 constant scores at the metric's chance floor
(~1/101 = 0.0099), so a fallback-only run is trivially distinguishable from a real
one on the leaderboard -- it cannot be mistaken for a result.  The log also prints
`*** FALLBACK ONLY ***` in that case.
"""
from pathlib import Path
from glob import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import SimpleITK

INPUT_PATH = Path(os.environ.get("RARE_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("RARE_OUTPUT", "/output"))
RESOURCE_PATH = Path(os.environ.get("RARE_RESOURCES", "/opt/app/resources"))
OUT_FILE = OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json"

# how many ensemble members to actually run; -1 = all found. Used by the CPU contract test.
MAX_MEMBERS = int(os.environ.get("RARE_MAX_MEMBERS", "-1"))
BATCH = int(os.environ.get("RARE_BATCH", "16"))
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def log(*a):
    print(*a, flush=True)


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    Path(location).parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_image_file_as_array(*, location):
    input_files = sorted(
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    if not input_files:
        raise FileNotFoundError(f"no image file under {location}")
    if len(input_files) > 1:
        log(f"[warn] {len(input_files)} image files present; the contract declares one "
            f"Image socket. Using {input_files[0]}")
    result = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(result)


# ------------------------------------------------------------------ preprocessing
def preprocess(stack):
    """(Z,H,W,C) uint8 -> (Z,3,224,224) float32.

    Uses the EXACT eval transform from training (tail_train4.py::DS, train=False):
        PIL.Image.fromarray(...).convert("RGB")
        T.Resize((224,224)) -> T.ToTensor() -> T.Normalize(IMNET)
    Reproducing this through torch.nn.functional.interpolate is NOT bit-identical
    (PIL's bilinear filter differs), and a preprocessing mismatch between training and
    inference is a classic silent accuracy leak -- so we import torchvision/PIL and run
    the same objects.  The pure-numpy path below is only a last-resort fallback and
    says so loudly.
    """
    import torch
    a = np.asarray(stack)
    if a.ndim == 3:                    # a single (H,W,C) image
        a = a[None]
    if a.ndim != 4 or a.shape[-1] not in (1, 3):
        raise ValueError(f"unexpected stack shape {a.shape}")
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    try:
        from PIL import Image
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize(list(IMNET_MEAN), list(IMNET_STD))])
        return torch.stack([tf(Image.fromarray(im.astype(np.uint8)).convert("RGB")) for im in a])
    except Exception as e:            # pragma: no cover -- must never happen in the shipped image
        log(f"*** WARNING: torchvision/PIL unavailable ({e}); falling back to F.interpolate, "
            f"which is NOT the training transform and may cost accuracy ***")
        import torch.nn.functional as F
        x = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        return (x - torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMNET_STD).view(1, 3, 1, 1)


# ------------------------------------------------------------------ the model
def discover_members():
    ms = sorted(glob(str(RESOURCE_PATH / "members" / "*.pt")))
    if MAX_MEMBERS > 0:
        ms = ms[:MAX_MEMBERS]
    return ms


def run_ensemble(stack):
    import torch
    sys.path.insert(0, "/opt/app")
    from rare_model import DinoV3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members = discover_members()
    if not members:
        raise FileNotFoundError(f"no ensemble members under {RESOURCE_PATH/'members'}")
    log(f"[model] device={dev}  members={len(members)}  images={len(stack)}")

    backbone_w = str(RESOURCE_PATH / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    t0 = time.time()
    net = DinoV3(weight_path=backbone_w, lora=True, dino_repo="/opt/app/dino_repo")
    net.to(dev).eval()
    tk = set(net.trainable_keys())
    log(f"[model] backbone+LoRA built in {time.time()-t0:.1f}s  ({len(tk)} trainable tensors)")

    x = preprocess(stack).to(dev)
    use_bf16 = (dev == "cuda")
    acc = np.zeros(len(x), dtype=np.float64)
    used = 0
    for mi, mpath in enumerate(members):
        ck = torch.load(mpath, map_location="cpu", weights_only=False)
        state = ck["state"] if isinstance(ck, dict) and "state" in ck else ck
        keys = set(state.keys())
        if keys != tk:
            raise RuntimeError(
                f"member {os.path.basename(mpath)} key-set mismatch: "
                f"{len(keys - tk)} unexpected / {len(tk - keys)} missing -- refusing to run "
                f"with a partially-reset LoRA (would silently score garbage)")
        net.load_state_dict(state, strict=False)
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), BATCH):
                xb = x[i:i + BATCH]
                if use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = net(xb)
                else:
                    o = net(xb)
                outs.append(o.float().cpu().numpy())
        o = np.concatenate(outs, axis=0)
        p = 1.0 / (1.0 + np.exp(-(o[:, 1] - o[:, 0])))
        acc += p
        used += 1
        if mi == 0 or (mi + 1) % 10 == 0 or mi == len(members) - 1:
            log(f"[model] member {mi+1}/{len(members)}  {os.path.basename(mpath)}  "
                f"({time.time()-t0:.1f}s elapsed)")
    probs = acc / used
    log(f"[model] DONE {used} members in {time.time()-t0:.1f}s  "
        f"p: min={probs.min():.4f} med={np.median(probs):.4f} max={probs.max():.4f}")
    return [float(v) for v in probs]


# ------------------------------------------------------------------ handler
def interface_0_handler():
    stack = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy")
    n = len(stack) if np.asarray(stack).ndim == 4 else 1
    log(f"[io] input stack shape={np.asarray(stack).shape} -> {n} slices")

    # ---- DELIVERY SAFETY NET: format-valid output written before anything can fail
    write_json_file(location=OUT_FILE, content=[0.5] * n)
    log(f"[io] pre-wrote fallback output ({n} x 0.5) to {OUT_FILE}")

    try:
        probs = run_ensemble(stack)
        assert len(probs) == n, f"produced {len(probs)} scores for {n} slices"
        assert all(0.0 <= p <= 1.0 for p in probs), "scores outside [0,1]"
        write_json_file(location=OUT_FILE, content=probs)
        log("[io] wrote REAL model output")
    except Exception:
        log("*** FALLBACK ONLY *** the model failed; the constant-0.5 output stands.")
        log("*** This scores at the metric's chance floor (~0.0099) and is NOT a result. ***")
        traceback.print_exc()
        return 0
    return 0


def _show_torch_cuda_info():
    import torch
    log("=+=" * 10)
    log(f"Torch {torch.__version__}; CUDA available: {(av := torch.cuda.is_available())}")
    if av:
        log(f"\tdevices: {torch.cuda.device_count()}  props: "
            f"{torch.cuda.get_device_properties(torch.cuda.current_device())}")
    log("=+=" * 10)


def run():
    _show_torch_cuda_info()
    key = get_interface_key()
    log(f"[io] interface key = {key}")
    handler = {("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler}[key]
    return handler()


if __name__ == "__main__":
    raise SystemExit(run())
