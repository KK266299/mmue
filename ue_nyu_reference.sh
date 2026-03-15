#!/bin/bash
# NYUDepthv2 UE generation reference script (for future use)
#
# This script is kept as a template for running UE experiments
# on the NYU dataset, analogous to ue_brats_learnable_final.sh.
#
# Prerequisites:
#   1. Run preprocessing: python scripts/prepare_nyu.py --config scripts/configs/nyu.yaml
#   2. Ensure method config exists for the desired UE algorithm
#
# Usage:
#   bash ue_nyu_reference.sh

python ue_generate.py \
    dataset=nyu \
    task=nyu_ue \
    method=noise_slice_frequence_learnable \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=false \
    ue.algorithm.params.freq_constraint_enabled=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=4 \
    ue.surrogates.s_seg.num_classes=40 \
    task.run_name=nyu_ue_reference \
    2>&1 | tee logs/nyu_ue_reference.log
