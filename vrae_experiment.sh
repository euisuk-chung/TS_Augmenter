#!/bin/sh

BATCH_SIZE=8
EPOCH=10
LR=5e-5
ACCUMULATION_STEP=1

# N_ENC=6
# N_DEC=6

# run distilBART-6-3
python ./src/kobart/main.py\
    --batch-size=${BATCH_SIZE}\
    --lr=${LR}\
    --epoch=${EPOCH}\
    --gradient-accumulation-step=${ACCUMULATION_STEP}\
    --amp\
    --distributed
#     --distill\
    # --n_enc=${N_ENC}\
    # --n_dec=${N_DEC}\
    
