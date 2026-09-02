"""RARE26 submission — model definition.

Self-contained copy of the training-time architecture
(`/scratch/sc20osc/miccai-2026/RARE26/work/dino_models/models.py`), with ONE
deliberate change for container safety:

  training used `torch.hub.load(dino_repo, 'dinov3_vitl16', source='local',
  weights=<path>)`, which routes the local .pth through
  `torch.hub.load_state_dict_from_url("file://...")` and therefore needs a
  writable TORCH_HOME and copies 1.2 GB on every start.  Here we build the
  identical architecture with `pretrained=False` and `torch.load` the same
  state dict directly -> no hub cache, no URL machinery, nothing that could
  reach for the network under `--network none`.

The resulting module is byte-for-byte the same architecture, so the LoRA/head
checkpoints (`w_inner.pt`) load with strict key matching.
"""
import math
import os
import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.dropout(self.down(x))) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.lora = LoRALayer(self.in_features, linear_layer.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class DinoV3(nn.Module):
    """DINOv3 ViT-L/16 frozen backbone + LoRA(r=32, alpha=64) on every Linear + 2-way head."""

    def __init__(self, weight_path, lora=True, dino_repo=None):
        super().__init__()
        self.weight_path = weight_path
        self.lora = lora
        dino_repo = dino_repo or os.environ.get("DINOV3_REPO", "/opt/app/dino_repo")
        import sys
        if dino_repo not in sys.path:
            sys.path.insert(0, dino_repo)
        from dinov3.hub.backbones import dinov3_vitl16

        # arch only -- pretrained=False guarantees no URL/hub/network path is taken
        self.model = dinov3_vitl16(pretrained=False, weights=weight_path)
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd, strict=True), None
        self.feat_dim = self.model.embed_dim
        if self.lora:
            for p in self.model.parameters():
                p.requires_grad = False
            self._add_lora_layers()
        self.head = torch.nn.Linear(self.feat_dim, 2)

    def forward(self, x):
        return self.head(self.model(x))

    def _add_lora_layers(self, rank=32, alpha=64):
        def _walk(module, path=""):
            for name, child in module.named_children():
                new_path = f"{path}.{name}" if path else name
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, rank, alpha))
                else:
                    _walk(child, new_path)
        _walk(self.model)

    def trainable_keys(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]
