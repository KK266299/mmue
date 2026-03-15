# src/models/deeplabv3plus.py
"""
DeepLabV3+ wrapper for 2D semantic segmentation.

Uses torchvision's DeepLabV3 with ResNet backbone,
modified to support configurable input channels (e.g., 4 for RGB-D).
"""
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.registry import register_model


class _DeepLabV3PlusBase(nn.Module):
    """
    DeepLabV3+ wrapper that supports:
      - Configurable in_channels (default 3; for 4-ch RGB-D, replaces first conv)
      - Configurable num_classes
      - Optional pretrained backbone (ImageNet)
    """

    def __init__(
        self,
        cfg: DictConfig | Dict[str, Any],
        backbone_fn=deeplabv3_resnet101,
    ):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        log = get_logger()

        in_channels = int(get_config(cfg, "in_channels", 3))
        num_classes = int(get_config(cfg, "num_classes", 40))
        pretrained = bool(get_config(cfg, "pretrained", True))

        # Build torchvision model
        weights = "DEFAULT" if pretrained else None
        model = backbone_fn(weights=None, num_classes=num_classes)

        # If pretrained, load weights first then modify
        if pretrained:
            pretrained_model = backbone_fn(weights=weights, num_classes=21)
            # Copy backbone weights (ignore classifier)
            state = pretrained_model.backbone.state_dict()
            if in_channels != 3:
                # Remove first conv weight since channel count differs
                first_conv_key = None
                for k, v in state.items():
                    if "conv1" in k and "weight" in k and v.ndim == 4:
                        first_conv_key = k
                        break
                if first_conv_key:
                    del state[first_conv_key]
            model.backbone.load_state_dict(state, strict=False)
            del pretrained_model

        # Modify first conv if in_channels != 3
        if in_channels != 3:
            old_conv = model.backbone.conv1
            model.backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize: copy RGB weights, zero-init extra channels
            with torch.no_grad():
                if pretrained:
                    old_weight = backbone_fn(weights="DEFAULT", num_classes=21).backbone.conv1.weight
                    c_copy = min(3, in_channels)
                    model.backbone.conv1.weight[:, :c_copy] = old_weight[:, :c_copy]
                    if in_channels > 3:
                        nn.init.kaiming_normal_(
                            model.backbone.conv1.weight[:, 3:],
                            mode="fan_out", nonlinearity="relu",
                        )

        self.model = model

        log.info(
            f"[DeepLabV3+] backbone={backbone_fn.__name__}, "
            f"in_channels={in_channels}, num_classes={num_classes}, "
            f"pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["out"]  # [B, num_classes, H, W]


@register_model("deeplabv3plus_r50")
class DeepLabV3PlusR50(_DeepLabV3PlusBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, backbone_fn=deeplabv3_resnet50)


@register_model("deeplabv3plus_r101")
class DeepLabV3PlusR101(_DeepLabV3PlusBase):
    def __init__(self, cfg: DictConfig | Dict[str, Any], **kwargs):
        super().__init__(cfg, backbone_fn=deeplabv3_resnet101)


@register_model("deeplabv3plus")
class DeepLabV3Plus(DeepLabV3PlusR101):
    """Alias: defaults to ResNet-101 backbone."""
    pass
