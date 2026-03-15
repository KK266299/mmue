"""
NYUDepthv2 preprocessing script.

Reads RGB, Depth, and Label images from the raw NYU dataset,
packs them into per-sample HDF5 files (consistent with BraTS pipeline),
and generates train/val/test CSV split files.

Raw data layout (following Sigma convention):
    NYUDepthv2/
    ├── RGB/        (.jpg)
    ├── Depth/      (.png, single-channel)
    ├── Label/      (.png, values 1~40)
    ├── train.txt
    └── test.txt

Output layout (consistent with BraTS pipeline):
    preproc_root/
    ├── h5/
    │   ├── <name>.h5   (image: float32 [4,H,W], label: uint8 [H,W])
    │   └── ...
    ├── train.csv
    ├── val.csv
    └── test.csv
"""

import os
import argparse
import csv

import cv2
import numpy as np
import yaml
import h5py


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_file_list(txt_path: str):
    """Read a txt file with one filename (no extension) per line."""
    with open(txt_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def load_rgb(path: str) -> np.ndarray:
    """Load RGB image as float32 [H, W, 3] in range [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_depth(path: str) -> np.ndarray:
    """Load depth map as float32 [H, W] in raw scale."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def load_label(path: str, offset: int = -1) -> np.ndarray:
    """Load label as uint8 [H, W], apply offset (1~40 -> 0~39)."""
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(f"Cannot read label image: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    label = label.astype(np.int16) + offset
    label = np.clip(label, 0, 255).astype(np.uint8)
    return label


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to [0, 1] range."""
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth)
    return (depth - d_min) / (d_max - d_min)


def process_sample(
    name: str,
    raw_root: str,
    rgb_folder: str,
    depth_folder: str,
    label_folder: str,
    rgb_fmt: str,
    depth_fmt: str,
    label_fmt: str,
    label_offset: int,
    target_size,
    out_dir: str,
) -> str:
    """Process a single NYU sample: load RGB+Depth+Label, save as H5."""
    rgb_path = os.path.join(raw_root, rgb_folder, name + rgb_fmt)
    depth_path = os.path.join(raw_root, depth_folder, name + depth_fmt)
    label_path = os.path.join(raw_root, label_folder, name + label_fmt)

    rgb = load_rgb(rgb_path)          # [H, W, 3], float32, [0, 1]
    depth = load_depth(depth_path)    # [H, W], float32
    label = load_label(label_path, offset=label_offset)  # [H, W], uint8

    # Normalize depth to [0, 1]
    depth = normalize_depth(depth)

    # Resize if needed
    if target_size is not None:
        th, tw = target_size
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (tw, th), interpolation=cv2.INTER_NEAREST)

    # Stack into 4 channels: [R, G, B, Depth] -> [4, H, W]
    image = np.concatenate(
        [rgb.transpose(2, 0, 1), depth[np.newaxis, :, :]],
        axis=0,
    ).astype(np.float32)

    # Save H5
    h5_path = os.path.join(out_dir, f"{name}.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("image", data=image, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")
        f.attrs["sample_id"] = name
        f.attrs["num_channels"] = 4
        f.attrs["image_shape"] = np.array(image.shape, dtype=np.int32)

    return h5_path


def split_train_val_test(
    names,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Split all names into train / val / test by ratio (0.75 / 0.15 / 0.15)."""
    n = len(names)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)

    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # test gets the remainder
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_names = [names[i] for i in train_idx]
    val_names = [names[i] for i in val_idx]
    test_names = [names[i] for i in test_idx]
    return train_names, val_names, test_names


def write_split_csv(csv_path: str, records: list):
    """Write a split CSV with columns: case_id, grade, volume_path, label_path."""
    fieldnames = ["case_id", "grade", "volume_path", "label_path"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"  -> {csv_path} ({len(records)} samples)")


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]

    raw_root = data_cfg["raw_root"]
    preproc_root = data_cfg["preproc_root"]

    rgb_folder = data_cfg.get("rgb_folder", "RGB")
    depth_folder = data_cfg.get("depth_folder", "Depth")
    label_folder = data_cfg.get("label_folder", "Label")
    rgb_fmt = data_cfg.get("rgb_format", ".jpg")
    depth_fmt = data_cfg.get("depth_format", ".png")
    label_fmt = data_cfg.get("label_format", ".png")

    label_offset = int(data_cfg.get("label_offset", -1))
    target_size = data_cfg.get("target_size", None)
    if target_size is not None:
        target_size = tuple(int(x) for x in target_size)

    train_ratio = float(data_cfg.get("train_ratio", 0.75))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    test_ratio = float(data_cfg.get("test_ratio", 0.15))
    split_seed = int(data_cfg.get("split_seed", 42))
    run_preprocess = bool(data_cfg.get("run_preprocess", True))

    # Read file lists, merge all samples then re-split
    train_list_path = os.path.join(raw_root, data_cfg.get("train_list", "train.txt"))
    test_list_path = os.path.join(raw_root, data_cfg.get("test_list", "test.txt"))

    orig_train_names = read_file_list(train_list_path)
    orig_test_names = read_file_list(test_list_path)
    all_names = orig_train_names + orig_test_names
    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for n in all_names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    all_names = unique_names
    print(f"Found {len(all_names)} total samples (orig train={len(orig_train_names)}, orig test={len(orig_test_names)})")

    # Re-split by 0.75 / 0.15 / 0.15
    train_names, val_names, test_names = split_train_val_test(
        all_names, train_ratio, val_ratio, test_ratio, split_seed
    )
    print(f"Split: train={len(train_names)}, val={len(val_names)}, test={len(test_names)}")

    # Prepare output
    ensure_dir(preproc_root)
    out_h5_dir = os.path.join(preproc_root, "h5")
    ensure_dir(out_h5_dir)

    name_to_h5 = {}

    if run_preprocess:
        for i, name in enumerate(all_names):
            print(f"[{i+1}/{len(all_names)}] Processing {name} ...")
            h5_path = process_sample(
                name=name,
                raw_root=raw_root,
                rgb_folder=rgb_folder,
                depth_folder=depth_folder,
                label_folder=label_folder,
                rgb_fmt=rgb_fmt,
                depth_fmt=depth_fmt,
                label_fmt=label_fmt,
                label_offset=label_offset,
                target_size=target_size,
                out_dir=out_h5_dir,
            )
            name_to_h5[name] = h5_path
    else:
        print("[INFO] run_preprocess=False, skipping H5 generation, using existing files.")
        for name in all_names:
            h5_path = os.path.join(out_h5_dir, f"{name}.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(
                    f"run_preprocess=False but H5 not found: {h5_path}\n"
                    f"Set run_preprocess=true to generate H5 files first."
                )
            name_to_h5[name] = h5_path

    # Build CSV records
    def make_records(names):
        records = []
        for n in names:
            records.append({
                "case_id": n,
                "grade": "",
                "volume_path": name_to_h5[n],
                "label_path": name_to_h5[n],
            })
        return records

    print("\nWriting split CSVs ...")
    write_split_csv(os.path.join(preproc_root, "train.csv"), make_records(train_names))
    write_split_csv(os.path.join(preproc_root, "val.csv"), make_records(val_names))
    write_split_csv(os.path.join(preproc_root, "test.csv"), make_records(test_names))

    print(f"\nDone. H5 files: {out_h5_dir}")
    print(f"CSVs: {preproc_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NYUDepthv2 to HDF5")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)
