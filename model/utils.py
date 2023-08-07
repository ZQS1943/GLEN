import os
import io
import sys
import json
import torch
import logging
import numpy as np
from collections import OrderedDict
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset

from model.encoder import EncoderRanker
from model.data_process import process_mention_data, process_mention_data_yes_no
from model.params import xpo_used
  

def encode_all_candidates(params, model, device, one_vec=False, only_used_xpo=False):
    # eval_batch_size = params["eval_batch_size"]
    eval_batch_size = 128
    candidate_token_ids = torch.load(params["cand_token_ids_path"])
    if only_used_xpo:
        cand_index = torch.tensor([i for i in range(len(candidate_token_ids))])
        cand_mask = torch.tensor(xpo_used, dtype=torch.bool)
        used_cand = cand_index[cand_mask]
        candidate_token_ids_list = torch.tensor([candidate_token_ids[int(i)] for i in used_cand], dtype=torch.long)
        used_cand = used_cand.to(device)
    else:
        used_cand = None
        candidate_token_ids_list = torch.tensor([candidate_token_ids[i] for i in range(len(candidate_token_ids))], dtype=torch.long)
    print(f"predicting {len(candidate_token_ids_list)} cand vecs...")
    # assert 1==0
    all_cands = TensorDataset(candidate_token_ids_list)
    all_cands_loder = DataLoader(all_cands, batch_size=eval_batch_size, shuffle=False)
    cand_encs = []
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(all_cands_loder, desc="Candidates")):
            batch = batch[0].to(device)
            result = model.encode_candidate(batch, one_vec=one_vec)
            result = result.detach()
            cand_encs.append(result)
    cand_encs = torch.cat(cand_encs, 0)
    return cand_encs, used_cand

def read_dataset(dataset_name, params, tokenizer, logger = None, truncate = -1, has_true_label=False, add_sent_event_token=False, yes_no_format=False, seed_round=False):
    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(params['data_path'], file_name)

    samples = []
    with io.open(txt_file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            samples.append(json.loads(line.strip()))

    # if 'debug' in params['output_path']:
    #     samples = samples[:100]
    if yes_no_format:
        processed_data = process_mention_data_yes_no(
            samples=samples,  # use subset of valid data
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            add_mention_bounds=(not params["no_mention_bounds"]),
            candidate_token_ids=None,
            params=params,
            truncate = truncate, 
            has_true_label = has_true_label,
            add_sent_event_token = add_sent_event_token,
            seed_round=seed_round
        )
        return processed_data

    processed_data = process_mention_data(
        samples=samples,  # use subset of valid data
        tokenizer=tokenizer,
        max_context_length=params["max_context_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        add_mention_bounds=(not params["no_mention_bounds"]),
        candidate_token_ids=None,
        params=params,
        truncate = truncate, 
        has_true_label = has_true_label,
        add_sent_event_token = add_sent_event_token 
    )
    return processed_data


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def accuracy_multiple_labels(logits, labels):
    out_label = np.argmax(logits, axis=1)
    # print(out_label, logits, labels)
    cnt = 0
    for i, label in enumerate(out_label):
        if labels[i][label] == 1:
            cnt += 1
    return cnt

def accuracy_multiple_labels_label_idx(logits, label_idx):
    out_label = np.argmax(logits, axis=1)
    cnt = 0
    results = []
    num_wo_other = 0
    num_correct_wo_other = 0
    for t, label_set in zip(out_label, label_idx):
        # print(label_set)
        if t in label_set:
            cnt += 1
            results.append(1)
            if 0 not in label_set:
                num_correct_wo_other += 1
        else:
            results.append(0)
        
        if 0 not in label_set:
            num_wo_other += 1

    # assert 1==0
    return cnt, results, out_label, num_wo_other, num_correct_wo_other



def accuracy_label_idx(logits, label_idx, label_mask):
    out_label = np.argmax(logits, axis=1)
    cnt = 0
    results = []
    total_results = []
    num_wo_other = 0
    num_correct_wo_other = 0
    cur_label = 0
    for label, l_mask in zip(label_idx, label_mask):
        tmp_results = []
        label = label[l_mask]
        for l in label:
            tmp_correct = int(out_label[cur_label] == l)
            total_results.append(tmp_correct)
            cnt += tmp_correct
            
            if l != 0:
                num_correct_wo_other += tmp_correct
                tmp_results.append(tmp_correct)
                num_wo_other += 1
            
            cur_label += 1
        results.append(tmp_results)
    assert cur_label == len(out_label)
    # assert 1==0
    return cnt, results,total_results, out_label, num_wo_other, num_correct_wo_other

def get_predicted_mention_bounds(logits, bounds):
    predicted_bounds = []
    predicted_logits = []
    for b_logits, b_bounds in zip(logits, bounds):
        predicted_bounds.append(b_bounds[b_logits>0.5])
        predicted_logits.append(b_logits[b_logits>0.5])
    return predicted_bounds, predicted_logits

def eval_trigger_detection(predicted_mention_bounds, mention_idx, mention_idx_mask, roleset_ids, roleset_detail):
    predict_num_list = []
    correct_num_list = []
    gold_num_list = []
    for predicted, gold, gold_mask, rolesets in zip(predicted_mention_bounds, mention_idx, mention_idx_mask, roleset_ids):
        tmp_predict_num = 0
        tmp_correct_num = 0
        tmp_gold_num = 0

        gold = gold[gold_mask]
        rolesets = rolesets[gold_mask]
        for bound, roleset in zip(gold, rolesets):
            tmp_gold_num += 1
            roleset = int(roleset)
            roleset_detail[roleset]['gold_num'] += 1
            if bound in predicted:
                tmp_correct_num += 1
                roleset_detail[roleset]['correct_num'] += 1
        tmp_predict_num += len(predicted)

        predict_num_list.append(tmp_predict_num)
        correct_num_list.append(tmp_correct_num)
        gold_num_list.append(tmp_gold_num)


    return predict_num_list, correct_num_list , gold_num_list, roleset_detail

def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = "".join(key.split(".module"))
        new_state_dict[name] = value
    return new_state_dict


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model.state_dict(), output_model_file)
    # model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def get_logger(output_dir=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('UniversalED')
    logger.setLevel(10)
    return logger


def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


def get_biencoder(parameters):
    return EncoderRanker(parameters)
    