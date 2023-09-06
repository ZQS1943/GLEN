#!/usr/bin/env bash

round=$1 # 0, 1
data_path=$2 # data path to the trianing data

python model/train_type_classifier.py \
    --output_path ./exp/type_classifier_${round} \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --train_batch_size 32 \
    --max_context_length 512 \
    --k 2 \
    --TC_train_data_path ${data_path}
