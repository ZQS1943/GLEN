#!/usr/bin/env bash
export PYTHONPATH=./
python model/predict_sentence.py \
    --path_to_ckpt ./ckpts \
    --bs_TI 64 \
    --bs_TC 64 \
    --bs_TR 4 \
    --k 10