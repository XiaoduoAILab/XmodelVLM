#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")/..";pwd)
export PYTHONPATH=${WORK_DIR}
CHCEKPOINT_PATH="$1"
base_name=$(basename ${CHCEKPOINT_PATH})
OUTPUT_DIR_EVAL="./logs/${base_name}.evaluation"
mkdir -p ${OUTPUT_DIR_EVAL}
CONV_MODE="v1"
PLAYGROUND_DATA="/path/data/eval"
cd ${WORK_DIR}


DATASET_NAME=mmvet
MODEL_GENERATOR=xmodelvlm.eval.model_vqa
DATA_ROOT=${PLAYGROUND_DATA}/mm-vet
SPLIT_NAME=llava-mm-vet
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_mmvet_log.txt 2>&1 &

DATASET_NAME=mme
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_loader
DATA_ROOT=${PLAYGROUND_DATA}/MME
SPLIT_NAME=llava_mme
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_mme_log.txt 2>&1 &

DATASET_NAME=gqa
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_loader
DATA_ROOT=${PLAYGROUND_DATA}/gqa
SPLIT_NAME=llava_gqa_testdev_balanced
CUDA_VISIBLE_DEVICES=6 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_gqa_log.txt 2>&1 &

DATASET_NAME=textvqa
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_loader
DATA_ROOT=${PLAYGROUND_DATA}/textvqa
SPLIT_NAME=llava_textvqa_val_v051_ocr
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_textvqa_log.txt 2>&1 &

DATASET_NAME=pope
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_loader
DATA_ROOT=${PLAYGROUND_DATA}/pope
SPLIT_NAME=llava_pope_test
CUDA_VISIBLE_DEVICES=6 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_pope_log.txt 2>&1 &

DATASET_NAME=vizwiz
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_loader
DATA_ROOT=${PLAYGROUND_DATA}/vizwiz
SPLIT_NAME=llava_test
CUDA_VISIBLE_DEVICES=6 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_vizwiz_log.txt 2>&1 &

DATASET_NAME=mmbench
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_mmbench
DATA_ROOT=${PLAYGROUND_DATA}/mmbench
SPLIT_NAME=mmbench_dev_20230712
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_mmbench_log.txt 2>&1 &

DATASET_NAME=mmbench_cn
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_mmbench
DATA_ROOT=${PLAYGROUND_DATA}/mmbench_cn
SPLIT_NAME=mmbench_dev_cn_20231003
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_mmbench_cn_log.txt 2>&1 &

DATASET_NAME=sqa
MODEL_GENERATOR=xmodelvlm.eval.model_vqa_science
DATA_ROOT=${PLAYGROUND_DATA}/scienceqa
SPLIT_NAME=llava_test_CQM-A
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME} > ./logs/${base_name}_sqa_log.txt 2>&1 &