#!/bin/bash
# NYUDepthv2 segmentation training script (reference)
#
# Usage:
#   1. First run preprocessing:
#      python scripts/prepare_nyu.py --config scripts/configs/nyu.yaml
#
#   2. Then run training:
#      bash train_nyu_seg.sh

python main.py \
    dataset=nyu \
    task=nyu_seg \
    model=unet \
    model.name=unet \
    training.epochs=200 \
    training.optimizer=adamw \
    training.optimizers.adamw.lr=6e-5 \
    training.optimizers.adamw.weight_decay=0.01 \
    training.gpu_ids=[0] \
    training.batch_size=8 \
    training.eval_batch_size=4 \
    training.num_workers=4 \
    training.model_save_start=50 \
    training.model_save_freq=10 \
    task.run_name=nyu_seg_baseline \
    2>&1 | tee logs/nyu_seg_baseline.log
