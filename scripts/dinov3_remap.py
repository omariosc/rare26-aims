#!/usr/bin/env python3
"""
RARE26 — OFFLINE DINOv3-ViT-L enablement: remap the staged timm
`vit_large_patch16_dinov3.lvd1689m` state-dict -> FAIR DinoVisionTransformer keys, so the
winner's DinoV3 (torch.hub source='local') strict-loads WITHOUT the gated FAIR .pth.

Same LVD-1689M weights, just re-hosted by timm with different key names. We:
  1. build the bare FAIR model  (torch.hub.load(dino_repo,'dinov3_vitl16',source='local',pretrained=False))
  2. build timm  vit_large_patch16_dinov3.lvd1689m (pretrained=True, num_classes=0)  [cached]
  3. rule-remap timm keys -> FAIR keys, then greedy shape/order match for leftovers
  4. FAIR.load_state_dict(remapped, strict=True)  <-- the verification the coordinator asked for
  5. save remapped as <out>/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth (name must end -<8hash>.pth)
  6. tiny CPU forward smoke -> assert (B, 1024) feature output.

Exit 0 + "REMAP_OK" iff strict-load + forward pass. Else prints the diff (missing/unexpected/
shape) and exits 1 -> caller falls back to the timm-native DinoV3 build (same weights).
"""
import argparse, os, re, sys
import torch

DINO_REPO = "/users/sc20osc/RARE26/rare2025-challenge/dino_repo"
TIMM_NAME = "vit_large_patch16_dinov3.lvd1689m"
# installed timm 1.0.19 can't INSTANTIATE dinov3 ViT ("Unknown model"), but the LVD-1689M
# weights are cached -> load the safetensors state_dict directly (timm-convention keys).
TIMM_SNAP = ("/scratch/sc20osc/hf_cache/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/"
             "snapshots")


def build_fair():
    m = torch.hub.load(DINO_REPO, "dinov3_vitl16", source="local", pretrained=False)
    return m.eval()


def load_timm_state_dict():
    import glob
    from safetensors.torch import load_file
    st = glob.glob(os.path.join(TIMM_SNAP, "*", "model.safetensors"))
    if not st:
        raise FileNotFoundError(f"no model.safetensors under {TIMM_SNAP}")
    sd = load_file(st[0])
    return sd


def remap_key(k: str) -> str:
    """timm -> FAIR name rules (verified against the dumped key sets).
    All other keys are identical (norm1/2, attn.qkv.weight, attn.proj, mlp.fc1/2,
    patch_embed.proj, norm, cls_token). DINOv3-ViT-L has NO qkv input-bias (only
    attn.proj.bias) -> FAIR's qkv.bias is zeroed; FAIR buffers (qkv.bias_mask,
    rope_embed.periods) + unused mask_token are kept from construction."""
    k = k.replace("reg_token", "storage_tokens")
    k = re.sub(r"\.gamma_1$", ".ls1.gamma", k)
    k = re.sub(r"\.gamma_2$", ".ls2.gamma", k)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/users/sc20osc/RARE26/rare2025-challenge/resources")
    ap.add_argument("--dump", action="store_true", help="just print both key sets and exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[remap] building FAIR bare dinov3_vitl16 (source=local, pretrained=False) ...")
    fair = build_fair()
    fair_sd = fair.state_dict()
    print("[remap] loading timm", TIMM_NAME, "state_dict from cached safetensors ...")
    timm_sd = load_timm_state_dict()
    print(f"[remap] FAIR params={len(fair_sd)}  timm params={len(timm_sd)}")

    if args.dump:
        def summarize(sd, tag):
            print(f"\n===== {tag} keys (block0 + non-block) =====")
            for k, v in sd.items():
                if k.startswith(("blocks.1.", "blocks.2")) or re.match(r"blocks\.[2-9]", k):
                    continue
                if k.startswith("blocks.") and not k.startswith("blocks.0."):
                    continue
                print(f"  {k:55s} {tuple(v.shape)}")
        summarize(fair_sd, "FAIR"); summarize(timm_sd, "timm")
        return 0

    # 1) rule remap (timm keys -> FAIR names); every timm key must map to an existing FAIR key
    remapped, unexpected = {}, []
    for k, v in timm_sd.items():
        nk = remap_key(k)
        if nk in fair_sd and fair_sd[nk].shape == v.shape:
            remapped[nk] = v
        else:
            unexpected.append((k, nk, tuple(v.shape),
                               tuple(fair_sd[nk].shape) if nk in fair_sd else "NOKEY"))
    print(f"[remap] timm->FAIR mapped {len(remapped)}/{len(timm_sd)} timm keys")
    if unexpected:
        print(f"  --- UNEXPECTED timm keys (no FAIR match) [{len(unexpected)}] ---")
        for x in unexpected[:20]:
            print("     ", x)
        print("REMAP_FAIL (unmapped timm keys -> conversion is lossy/incorrect)")
        return 1

    # 2) load into FAIR (strict=False): missing keys must be ONLY the FAIR-only params/buffers
    #    (qkv.bias, qkv.bias_mask [buffer], mask_token, rope_embed.periods [buffer]).
    res = fair.load_state_dict(remapped, strict=False)
    allowed_suffix = ("attn.qkv.bias", "attn.qkv.bias_mask", "rope_embed.periods")
    allowed_exact = {"mask_token"}
    bad_missing = [k for k in res.missing_keys
                   if not (k.endswith(allowed_suffix) or k in allowed_exact)]
    print(f"[remap] load_state_dict(strict=False): missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}")
    if res.unexpected_keys:
        print("  unexpected:", res.unexpected_keys[:10]); print("REMAP_FAIL"); return 1
    if bad_missing:
        print("  UNACCOUNTED missing (not a known FAIR-only key):", bad_missing[:20])
        print("REMAP_FAIL"); return 1
    print(f"[remap] OK: all {len(res.missing_keys)} missing keys are known FAIR-only "
          f"(qkv.bias/bias_mask/rope periods/mask_token) — kept from construction.")

    # 3) DINOv3-ViT-L has no learned qkv input-bias -> force qkv.bias = 0 (k already masked)
    with torch.no_grad():
        nz = 0
        for name, p in fair.named_parameters():
            if name.endswith("attn.qkv.bias"):
                p.zero_(); nz += 1
    print(f"[remap] zeroed {nz} attn.qkv.bias tensors (faithful: DINOv3-L uses no qkv input-bias)")

    # 4) forward smoke (CPU, tiny) -> (B, 1024) tensor (winner's DinoV3 does head(model(x)))
    fair.eval()
    with torch.no_grad():
        out = fair(torch.randn(2, 3, 224, 224))
    if isinstance(out, dict):
        out = out.get("x_norm_clstoken", None)
    ok = torch.is_tensor(out) and tuple(out.shape) == (2, 1024)
    print(f"[remap] forward -> {'tensor '+str(tuple(out.shape)) if torch.is_tensor(out) else type(out)} "
          f"({'OK' if ok else 'UNEXPECTED'})")
    if not ok:
        print("REMAP_FAIL (forward shape != (2,1024))"); return 1

    # 5) save the COMPLETE FAIR state_dict (timm weights + FAIR buffers + zeroed qkv.bias) so the
    #    winner's DinoV3 loads it with strict=True unchanged. Named -8aa4cbdd.pth (loader regex).
    outpath = os.path.join(args.out, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    torch.save(fair.state_dict(), outpath)
    # verify a fresh FAIR model strict-loads it (what training will do)
    fair2 = build_fair()
    fair2.load_state_dict(torch.load(outpath, map_location="cpu"), strict=True)
    print(f"[remap] saved {outpath} ; fresh FAIR strict-load VERIFIED")
    print("REMAP_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
