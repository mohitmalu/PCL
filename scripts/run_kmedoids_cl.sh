#!/bin/bash

DATA="../data/esc-50/esc50_embeddings/"
DIST=30
EMB="ast"
REG="lwf"
LOG_DIR="/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/gautham/pcl/results/esc-50/logs"
MODEL="cnn4"
DATASET="ESC-50"
# EWC_LAMBDA=10000000
LWF_LAMBDA=3

LOG_FILE="${LOG_DIR}/kmedoids_threshold_${DIST}_${EMB}_${REG}_${MODEL}_${DATASET}.log"

nohup python main_kmedoids_cl.py \
    --data ${DATA} \
    --dist_threshold ${DIST} \
    --model_type ${MODEL} \
    --embedding_type ${EMB} \
    --reg ${REG} \
    --dataset ${DATASET} \
    --lambda_lwf ${LWF_LAMBDA} \
    > "${LOG_FILE}" 2>&1 &
