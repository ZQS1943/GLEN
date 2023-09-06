#!/usr/bin/env bash
export PYTHONPATH="$PYTHONPATH:./"
python model/predict_trigger_identifier.py \
    --path_to_model ./exp/trigger_identifier/epoch_4/pytorch_model.bin \
    --output_path ./exp/trigger_identifier/epoch_4 \
    --eval_batch_size 64
