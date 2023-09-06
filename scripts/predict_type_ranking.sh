#!/usr/bin/env bash

python model/predict_type_ranking.py \
    --path_to_model ./exp/type_ranking/epoch_4/pytorch_model.bin \
    --output_path ./exp/type_ranking/epoch_4 \
    --eval_batch_size 16 \
    --cand_token_ids_path ./data/data_preprocessed/node_tokenized_ids_64_with_event_tag.pt
