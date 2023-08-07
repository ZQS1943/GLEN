#!/bin/bash

objective=$1  # train/predict
batch_size=$2
eval_batch_size=$3
num_train_epochs=$4
learning_rate=$5
output_dir=$6
data_truncation=$7
loss_type=$8
epoch=$9
model_size=base  # large/base/medium/small/mini/tiny

if [ "${epoch}" = "" ] || [ "${epoch}" = "_" ]
then
    epoch=-1
fi

export PYTHONPATH=.

model_dir="exp/experiments_type_ranking/${output_dir}/bert_${model_size}"
model_ckpt="bert-${model_size}-uncased"

if [ "${objective}" = "train" ] || [ "${objective}" = "finetune" ]
then
  echo "Running type ranking training."

  model_path_arg=""
  if [ "${epoch}" != "-1" ]
  then
    model_path_arg="--path_to_model ${model_dir}/epoch_${epoch}/pytorch_model.bin --path_to_trainer_state ${model_dir}/epoch_${epoch}/training_state.th"
  fi
  cmd="python model/train_type_ranking.py \
    --output_path ${model_dir} \
    ${model_path_arg} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${learning_rate} \
    --wb_name ${output_dir} \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --bert_model ${model_ckpt} \
    --last_epoch ${epoch} \
    --loss_type ${loss_type} \
    --data_truncation ${data_truncation}"
  echo $cmd
  $cmd
fi


if [ "${objective}" = "predict" ]
then
  echo "Running type ranking predicting."

  model_config=${model_dir}/training_params.txt
  model_path=${model_dir}/epoch_${epoch}/pytorch_model.bin
  save_dir=${model_dir}/epoch_${epoch}
  mkdir -p save_dir
  chmod 777 save_dir

  cmd="python model/predict_type_ranking.py \
      --path_to_model ${model_path} \
      --output_path ${save_dir} \
      --eval_batch_size ${eval_batch_size} \
      --cand_token_ids_path data/node_tokenized_ids_64_with_event_tag.pt"
  echo $cmd
  $cmd
fi

