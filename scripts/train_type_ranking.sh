#!/usr/bin/env bash
export PYTHONPATH=./
python model/train_type_ranking.py \
    --output_path ./exp/type_ranking/ \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --train_batch_size 64
