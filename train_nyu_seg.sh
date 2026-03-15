#!/bin/bash
# NYUDepthv2 语义分割训练脚本
#
# 使用步骤:
#   1. 先运行预处理:
#      python scripts/prepare_nyu.py --config scripts/configs/nyu.yaml
#
#   2. 安装依赖 (SegFormer 需要):
#      pip install transformers
#
#   3. 选择模型运行训练:
#      bash train_nyu_seg.sh deeplabv3plus   # DeepLabV3+ (ResNet-101)
#      bash train_nyu_seg.sh segformer       # SegFormer-B0
#      bash train_nyu_seg.sh segformer_b2    # SegFormer-B2
#      bash train_nyu_seg.sh                 # 默认: DeepLabV3+

mkdir -p logs

MODEL=${1:-deeplabv3plus}

case $MODEL in
    deeplabv3plus|deeplabv3plus_r101)
        MODEL_NAME="deeplabv3plus"
        RUN_NAME="nyu_deeplabv3plus_r101"
        ;;
    deeplabv3plus_r50)
        MODEL_NAME="deeplabv3plus_r50"
        RUN_NAME="nyu_deeplabv3plus_r50"
        ;;
    segformer|segformer_b0)
        MODEL_NAME="segformer_b0"
        RUN_NAME="nyu_segformer_b0"
        ;;
    segformer_b1)
        MODEL_NAME="segformer_b1"
        RUN_NAME="nyu_segformer_b1"
        ;;
    segformer_b2)
        MODEL_NAME="segformer_b2"
        RUN_NAME="nyu_segformer_b2"
        ;;
    segformer_b3)
        MODEL_NAME="segformer_b3"
        RUN_NAME="nyu_segformer_b3"
        ;;
    segformer_b4)
        MODEL_NAME="segformer_b4"
        RUN_NAME="nyu_segformer_b4"
        ;;
    segformer_b5)
        MODEL_NAME="segformer_b5"
        RUN_NAME="nyu_segformer_b5"
        ;;
    *)
        echo "未知模型: $MODEL"
        echo "可选: deeplabv3plus, deeplabv3plus_r50, segformer, segformer_b0~b5"
        exit 1
        ;;
esac

echo "======================================"
echo "  NYUDepthv2 语义分割训练"
echo "  模型: $MODEL_NAME"
echo "  运行名称: $RUN_NAME"
echo "======================================"

python main.py \
    dataset=nyu \
    task=nyu_seg \
    model.name=$MODEL_NAME \
    model.in_channels=4 \
    model.num_classes=40 \
    model.pretrained=true \
    training.epochs=200 \
    training.optimizer=adamw \
    training.optimizers.adamw.lr=6e-5 \
    training.optimizers.adamw.weight_decay=0.01 \
    training.scheduler.name=cosine \
    training.gpu_ids=[0] \
    training.batch_size=8 \
    training.eval_batch_size=4 \
    training.num_workers=4 \
    training.model_save_start=50 \
    training.model_save_freq=10 \
    task.run_name=$RUN_NAME \
    2>&1 | tee logs/${RUN_NAME}.log
