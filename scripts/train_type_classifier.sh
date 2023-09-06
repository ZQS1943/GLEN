#!/usr/bin/env bash

python model/train_type_classifier.py \
    --output_path ./exp/experiments_type_classifier \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --train_batch_size 32 \
    --max_context_length 512 \
    --k 2
    --TC_train_data_path ./exp/type_ranking/epoch_4/type_ranking_results_of_train_set_with_top_20_events.json
