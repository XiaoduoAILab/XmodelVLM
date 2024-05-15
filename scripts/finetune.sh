#!/bin/bash

echo ">>> Start Instruction Tuning ..."

ITER_NUM="0400000"
GRADIENT_ACCUMULATION_STEPS="1"

if [ -z "$2" ]
  then
    PROJECTOR_TYPE="xdpnet"
  else
    PROJECTOR_TYPE="$2"
fi

if [ -z "$3" ]
  then
    VISION_TOWER_TYPE="clip"
  else
    VISION_TOWER_TYPE="$3"
fi


if [ "${VISION_TOWER_TYPE}" = "clip" ]; then
    VISION_MODEL="models/clip-vit-large-patch14-336/"
elif [ "${VISION_TOWER_TYPE}" = "chinese_clip" ]; then
    VISION_MODEL="models/chinese-clip-vit-large-patch14-336px/"
elif [ "${VISION_TOWER_TYPE}" = "siglip" ]; then
    VISION_MODEL="models/siglip-so400m-patch14-384/"
else
    VISION_MODEL="models/clip-vit-large-patch14-336/"
fi


LANGUAGE_MODEL="models/e_line/xl/iter-${ITER_NUM}/"
OUTPUT_DIR_BASE="xmodelvlm_v0_5_${PROJECTOR_TYPE}_${VISION_TOWER_TYPE}_${GRADIENT_ACCUMULATION_STEPS}_${ITER_NUM}"
OUTPUT_DIR_PT="./checkpoints/${OUTPUT_DIR_BASE}.pretrain/"
OUTPUT_DIR_FT="./checkpoints/${OUTPUT_DIR_BASE}/"


deepspeed --include=localhost:"$1" xmodelvlm/train/train.py \
    --deepspeed scripts/deepspeed/zero3.json \
    --model_name_or_path ${LANGUAGE_MODEL} \
    --version v1 \
    --data_path datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder datasets/LLaVA/playground/data \
    --vision_tower ${VISION_MODEL} \
    --vision_tower_type "${VISION_TOWER_TYPE}" \
    --pretrain_mm_mlp_adapter "${OUTPUT_DIR_PT}"mm_projector.bin \
    --mm_projector_type "${PROJECTOR_TYPE}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR_FT}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --logging_dir "./runs/${OUTPUT_DIR_BASE}/"
