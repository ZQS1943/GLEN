#!/usr/bin/env bash

python model/train_type_classifier.py \
    --output_path ./exp/experiments_type_classifier \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --train_batch_size 32 \
    --max_context_length 512 \
    --train_samples_path ./cache/type_ranking_results_with_top_20_events_train_set_kairos_predicted.json
