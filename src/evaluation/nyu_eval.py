# file: src/evaluation/nyu_eval.py
"""
Evaluation strategy for NYUDepthv2 2D semantic segmentation.

Metrics:
  - Per-class IoU and mean IoU (mIoU) over 40 classes
  - Pixel accuracy
  - Mean accuracy (per-class)
  - Loss: CrossEntropyLoss
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from ..utils.config import get_config
from ..registry import register_evaluation_strategy


NYU_CLASS_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor_mat",
    "clothes", "ceiling", "books", "fridge", "tv",
    "paper", "towel", "shower_curtain", "box", "whiteboard",
    "person", "night_stand", "toilet", "sink", "lamp",
    "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop",
]


@register_evaluation_strategy("nyu_seg")
class NYUSegmentationEvaluationStrategy:
    """
    Evaluation for NYUDepthv2 2D semantic segmentation (40 classes).

    Assumptions:
      - batch["image"] -> FloatTensor [B, C, H, W]
      - batch["label"] -> LongTensor  [B, H, W] with values 0~39
      - Model outputs:  logits [B, 40, H, W]
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        eval_cfg = get_config(self.config, "evaluation", DictConfig({}))
        self.num_classes = int(get_config(eval_cfg, "num_classes", 40))
        self.ignore_index = int(get_config(eval_cfg, "ignore_index", 255))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        total_loss = 0.0
        n_samples = 0

        # Confusion matrix accumulators
        intersection = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        union = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        correct = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        class_total = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        total_correct = 0
        total_pixels = 0

        pbar = tqdm(data_loader, desc="Evaluate SEG (NYU)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)          # [B, C, H, W]
            y = batch["label"].to(device).long()   # [B, H, W]

            if y.ndim == 4 and y.size(1) == 1:
                y = y[:, 0]

            logits = model(x)                      # [B, num_classes, H, W]

            # Resize logits if spatial dims don't match label
            if logits.shape[2:] != y.shape[1:]:
                logits = F.interpolate(
                    logits, size=y.shape[1:],
                    mode='bilinear', align_corners=False,
                )

            loss = self.loss_fn(logits, y)
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            n_samples += bs

            pred = logits.argmax(dim=1)  # [B, H, W]

            valid = y != self.ignore_index
            pred_valid = pred[valid]
            y_valid = y[valid]

            total_correct += int((pred_valid == y_valid).sum().item())
            total_pixels += int(valid.sum().item())

            for c in range(self.num_classes):
                pred_c = pred_valid == c
                y_c = y_valid == c
                intersection[c] += (pred_c & y_c).sum()
                union[c] += (pred_c | y_c).sum()
                correct[c] += (pred_c & y_c).sum()
                class_total[c] += y_c.sum()

        # Compute metrics
        iou_per_class = torch.zeros(self.num_classes)
        acc_per_class = torch.zeros(self.num_classes)
        valid_classes = 0

        for c in range(self.num_classes):
            if union[c] > 0:
                iou_per_class[c] = float(intersection[c]) / float(union[c])
                valid_classes += 1
            if class_total[c] > 0:
                acc_per_class[c] = float(correct[c]) / float(class_total[c])

        miou = float(iou_per_class.sum() / max(valid_classes, 1))
        pixel_acc = total_correct / max(total_pixels, 1)
        classes_with_gt = (class_total > 0).sum().item()
        mean_acc = float(acc_per_class.sum() / max(classes_with_gt, 1))

        metrics = {
            "loss": float(total_loss / max(1, n_samples)),
            "miou": miou,
            "pixel_acc": pixel_acc,
            "mean_acc": mean_acc,
            "avg_dc": miou,  # alias for compatibility with BraTS hooks
            "jc": miou,      # alias
        }

        return metrics
