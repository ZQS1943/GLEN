#!/usr/bin/env bash
export PYTHONPATH=./
python model/train_trigger_identifier.py \
    --output_path ./exp/trigger_identifier/ \
    --num_train_epochs 5 \
    --learning_rate 0.00001 \
    --max_context_length 128 \
    --train_batch_size 64