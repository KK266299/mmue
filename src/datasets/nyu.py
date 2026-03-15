# file: src/datasets/nyu.py
"""
NYUDepthv2 dataset for 2D RGB-D semantic segmentation.

Follows the same pattern as BraTS19: loads preprocessed H5 files
referenced by per-split CSV files.

H5 structure (written by scripts/prepare_nyu.py):
    - "image": float32, shape (4, H, W)  -> [R, G, B, Depth], range [0, 1]
    - "label": uint8,   shape (H, W)     -> values 0~39 (40 classes)
"""
from __future__ import annotations

import os
from typing import Optional, Callable, Any, List, Union, Dict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig

from ..utils.logger import get_logger
from ..utils.config import require_config, get_config
from ..registry import register_dataset_builder
from .base_builder import BaseDatasetBuilder, BaseUEBuilder
from .transforms import get_seg_transforms


# ======================================================================
#   NYU Depth V2 Dataset
# ======================================================================

class NYUDepthDataset(Dataset):
    """
    NYUDepthv2 2D dataset for semantic segmentation.

    Each split CSV must contain at least:
      - case_id:     sample identifier (e.g. "nyu_0001")
      - volume_path: path to .h5 file (contains image + label)
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
        logger=None,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.csv_path = csv_path
        self.split = str(split).lower()
        self.transform = transform

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[NYU] CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = ["case_id", "volume_path"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"[NYU] CSV missing required column: {c}")

        if len(df) == 0:
            raise ValueError(
                f"[NYU] No samples in CSV: csv_path={csv_path}, split={self.split}"
            )

        self.df = df.reset_index(drop=True)
        self.logger.info(
            f"[NYU] Loaded split='{self.split}' from {csv_path}: {len(self.df)} samples"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        h5_path = row["volume_path"]

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"[NYU] H5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            image_np = f["image"][()]  # (4, H, W), float32, [0, 1]
            label_np = f["label"][()]  # (H, W), uint8

        if image_np.ndim != 3:
            raise ValueError(f"[NYU] image ndim={image_np.ndim}, expected 3 (C,H,W)")
        if label_np.ndim != 2:
            raise ValueError(f"[NYU] label ndim={label_np.ndim}, expected 2 (H,W)")

        image = torch.from_numpy(image_np).float()           # [C, H, W]
        label = torch.from_numpy(label_np.astype(np.int64)).long()  # [H, W]

        if self.transform is not None:
            out = self.transform(image, label)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                image, label = out
            else:
                raise RuntimeError(
                    "[NYU] transform must return (image, label), "
                    f"got type={type(out)}"
                )

        case_id = str(row["case_id"])

        return {
            "image": image,       # [C, H, W]
            "label": label,       # [H, W]
            "case_id": case_id,
            "grade": "",
            "index": int(idx),
            "h5_path": h5_path,
        }


# ======================================================================
#   NYU Builder (inherits BaseDatasetBuilder, same pattern as BraTS)
# ======================================================================

class NYUBuilder(BaseDatasetBuilder):
    """
    NYU dataset builder for 2D semantic segmentation.

    Config:
      dataset:
        name: nyu_seg
        train_csv_path: /.../train.csv
        val_csv_path:   /.../val.csv
        test_csv_path:  /.../test.csv
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        dcfg: DictConfig = require_config(config, "dataset")

        train_csv = require_config(dcfg, "train_csv_path", type_=str)
        val_csv = require_config(dcfg, "val_csv_path", type_=str)
        test_csv = require_config(dcfg, "test_csv_path", type_=str)
        self.csv_paths = {
            "train": train_csv,
            "val": val_csv,
            "test": test_csv,
        }

    def build_dataset(self, split: str, **overrides) -> Dataset:
        split_norm = self._normalize_split(split)

        csv_path = overrides.get("csv_path", self.csv_paths.get(split_norm))
        if csv_path is None:
            raise ValueError(f"[NYU] No CSV path configured for split '{split_norm}'.")

        transform = overrides.get("transform", None)

        if transform is None:
            dcfg: DictConfig = require_config(self.config, "training.data")
            tcfg: DictConfig = get_config(dcfg, "transforms", DictConfig({}))

            normalize = bool(require_config(tcfg, "normalize"))
            geom_aug = bool(require_config(tcfg, "geom_aug"))
            intensity_aug = bool(require_config(tcfg, "intensity_aug"))
            mean = get_config(tcfg, "mean", [0.485, 0.456, 0.406, 0.0])
            std = get_config(tcfg, "std", [0.229, 0.224, 0.225, 1.0])

            crop_size_raw = get_config(tcfg, "crop_size", None)
            crop_size = tuple(crop_size_raw) if crop_size_raw is not None else None

            scale_range_raw = get_config(tcfg, "scale_range", [0.5, 2.0])
            scale_range = tuple(scale_range_raw)

            transform = get_seg_transforms(
                ndim=2,
                split=split_norm,
                normalize=normalize,
                geom_aug=geom_aug,
                intensity_aug=intensity_aug,
                mean=mean,
                std=std,
                crop_size=crop_size,
                scale_range=scale_range,
            )

        ds = NYUDepthDataset(
            csv_path=csv_path,
            split=split_norm,
            transform=transform,
            logger=self.logger,
        )
        return ds


# ======================================================================
#   Registry registration
# ======================================================================

@register_dataset_builder("nyu_seg")
class NYUSegBuilder(NYUBuilder):
    """Segmentation task builder for NYUDepthv2."""
    def __init__(self, config: DictConfig):
        super().__init__(config)


@register_dataset_builder("nyu_ue")
class NYUUEBuilder(BaseUEBuilder):
    """UE task builder for NYUDepthv2 (for future use)."""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._base_builder_name = "nyu_seg"
