#!/bin/bash
objective=$1  # train/finetune/predict
data=$2
context_length=$3  # 128/20 (smallest)
batch_size=$4
eval_batch_size=$5
w_label_propagation=$6 # True/False
output_dir=$7
epoch=$8
base_data=$9  # if finetune
base_epoch=$10  # if finetune
mention_agg_type=all_avg  # all_avg/fl_avg/fl_linear/fl_mlp/none/none_no_mentions
model_size=base  # large/base/medium/small/mini/tiny
mention_scoring_method=qa_linear  # qa_linear/qa_mlp
chunk_start=0
chunk_end=-1

if [ "${epoch}" = "" ] || [ "${epoch}" = "_" ]
then
    epoch=-1
fi

export PYTHONPATH=.

data_type=${data##*/}
base_data_type=${base_data##*/}
if [ $objective = "finetune" ]
then
    data_type="${data_type}_ft_${base_data_type}_${base_epoch}"
fi
model_dir="exp/experiments_trigger_detector/${output_dir}/${mention_agg_type}_${context_length}_bert_${model_size}_${mention_scoring_method}_${w_label_propagation}"
if [ $objective = "finetune" ] && [ ! -d "${model_dir}/epoch_0" ]
then
    mkdir -p ${model_dir}
    base_model_dir="exp/experiments_trigger_detector/${base_data_type}/${mention_agg_type}_${context_length}_bert_${model_size}_${mention_scoring_method}_${w_label_propagation}"
    cp -rf ${base_model_dir}/epoch_${base_epoch} ${model_dir}/epoch_0
    rm ${model_dir}/epoch_0/training_state.th
    epoch=0
fi

# passed in a full file path at this point
if [ -d "${data}/tokenized" ]
then
  data_path="${data}/tokenized"
elif [ -d "${data}" ]
then
  data_path="${data}"
elif [ -d "all_inference_data/${data}" ]
then
  data_path="all_inference_data/${data}/tokenized"
else
  echo "Data not found: ${data}"
  exit
fi

if [ "${mention_agg_type}" = "none" ]
then
  all_mention_args=""
elif [ "${mention_agg_type}" = "none_no_mentions" ]
then
  all_mention_args="--no_mention_bounds"
else
  all_mention_args="--no_mention_bounds \
    --mention_aggregation_type ${mention_agg_type}"
fi


if [ "${context_length}" = "" ]
then
  context_length="128"
fi

if [ "${model_size}" = "base" ] || [ "${model_size}" = "large" ]
then
  model_ckpt="bert-${model_size}-uncased"
elif [ "${model_size}" = "tiny" ]
then
  model_ckpt="prajjwal1/bert-tiny"
else
  model_ckpt="/checkpoint/belindali/BERT/${model_size}"
fi


if [ "${w_label_propagation}" = "True" ]
then
  w_label_propagation="--with_label_propagation"
else
  w_label_propagation=""
fi

if [ "${epoch}" = "" ]
then
  epoch=-1
fi


echo $3

if [ "${objective}" = "train" ] || [ "${objective}" = "finetune" ]
then
  echo "Running ${mention_agg_type} biencoder training on ${data} dataset."
  distribute_train_samples_arg=""
  if [ "${data_type}" != "wiki_all_ents" ]
  then
    distribute_train_samples_arg="--dont_distribute_train_samples"
  fi

  model_path_arg=""
  if [ "${epoch}" != "-1" ]
  then
    model_path_arg="--path_to_model ${model_dir}/epoch_${epoch}/pytorch_model.bin --path_to_trainer_state ${model_dir}/epoch_${epoch}/training_state.th"
  fi
  cmd="python model/train_trigger_detector.py \
    --output_path ${model_dir} \
    ${model_path_arg} \
    --data_path ${data_path} \
    --num_train_epochs 5 \
    --learning_rate 0.00001 \
    --max_context_length ${context_length} \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --bert_model ${model_ckpt} \
    --mention_scoring_method ${mention_scoring_method} \
    --last_epoch ${epoch} \
    ${w_label_propagation} \
    ${all_mention_args} --get_losses ${distribute_train_samples_arg}" 
  echo ${w_label_propagation}
  echo $cmd
  $cmd
fi


if [ "${objective}" = "predict" ]
then
  echo ${data_type}

  model_config=${model_dir}/training_params.txt
  model_path=${model_dir}/epoch_${epoch}/pytorch_model.bin
  save_dir=${model_dir}/epoch_${epoch}
  mkdir -p save_dir
  chmod 777 save_dir

  cmd="python model/predict_trigger_detector.py \
      --path_to_model ${model_path} \
      --output_path ${save_dir} \
      --eval_batch_size ${eval_batch_size} \
      --prediction_data dev \
      --data_path ${data_path} \
      ${all_mention_args}"
  echo $cmd
  $cmd
fi

