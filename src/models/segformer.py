# src/models/segformer.py
"""
SegFormer wrapper for 2D semantic segmentation.

Uses HuggingFace transformers' SegformerForSemanticSegmentation.
Supports configurable in_channels (e.g., 4 for RGB-D) and num_classes.

Requirements:
    pip install transformers
"""
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.registry import register_model


class _SegFormerBase(nn.Module):
    """
    SegFormer wrapper.

    Config keys:
      - in_channels (int): input channels, default 3
      - num_classes (int): number of output classes, default 40
      - pretrained (bool): load HuggingFace pretrained weights, default True
      - variant (str): model variant, e.g., "b0", "b1", ..., "b5"
    """

    VARIANT_MAP = {
        "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }

    def __init__(self, cfg: DictConfig | Dict[str, Any], variant: str = "b0"):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        log = get_logger()

        in_channels = int(get_config(cfg, "in_channels", 3))
        num_classes = int(get_config(cfg, "num_classes", 40))
        pretrained = bool(get_config(cfg, "pretrained", True))
        variant = str(get_config(cfg, "variant", variant))

        from transformers import SegformerForSemanticSegmentation, SegformerConfig

        pretrained_name = self.VARIANT_MAP.get(variant)

        if pretrained and pretrained_name:
            model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(
                pretrained_name or "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            config.num_labels = num_classes
            model = SegformerForSemanticSegmentation(config)

        # Modify first patch embedding if in_channels != 3
        if in_channels != 3:
            encoder = model.segformer.encoder
            old_proj = encoder.patch_embeddings[0].proj
            new_proj = nn.Conv2d(
                in_channels, old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
            )
            with torch.no_grad():
                c_copy = min(3, in_channels)
                new_proj.weight[:, :c_copy] = old_proj.weight[:, :c_copy]
                if in_channels > 3:
                    nn.init.kaiming_normal_(
                        new_proj.weight[:, 3:],
                        mode="fan_out", nonlinearity="relu",
                    )
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)
            encoder.patch_embeddings[0].proj = new_proj

        self.model = model
        self._num_classes = num_classes

        log.info(
            f"[SegFormer] variant={variant}, in_channels={in_channels}, "
            f"num_classes={num_classes}, pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        # Upsample to original resolution
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits  # [B, num_classes, H, W]


@register_model("segformer_b0")
class SegFormerB0(_SegFormerBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, variant="b0")


@register_model("segformer_b1")
class SegFormerB1(_SegFormerBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, variant="b1")


@register_model("segformer_b2")
class SegFormerB2(_SegFormerBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, variant="b2")


@register_model("segformer_b3")
class SegFormerB3(_SegFormerBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, variant="b3")


@register_model("segformer_b4")
class SegFormerB4(_SegFormerBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, variant="b4")


@register_model("segformer_b5")
class SegFormerB5(_SegFormerBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, variant="b5")


@register_model("segformer")
class SegFormer(SegFormerB0):
    """Alias: defaults to SegFormer-B0."""
    pass
