#!/usr/bin/env bash

python model/predict_type_classifier.py \
    --path_to_model ./exp/type_classifier/epoch_1/pytorch_model.bin \
    --output_path ./exp/type_classifier/epoch_1 \
    --max_context_length 512 \
    --train_samples_path no_file \
    --eval_batch_size 64
