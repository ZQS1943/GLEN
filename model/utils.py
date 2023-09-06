import os
import io
import json
import torch
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from data_process import process_mention_data, process_mention_data_TC
from model.constants import xpo_used


def padding_with_multiple_dim(data, pad_idx=-1, dtype=torch.long):
    tmp_data = data
    max_length = [1]
    max_length.append(max(len(x) for x in tmp_data))
    def continue_walking(tmp_data):
        for item in tmp_data:
            if len(item) and isinstance(item[0], list):
                return True
        return False
    while continue_walking(tmp_data):
        tmp_tmp_data = []
        for item in tmp_data:
            tmp_tmp_data.extend(item)
        tmp_data = tmp_tmp_data
        max_length.append(max(len(x) for x in tmp_data))
    # print(f"padding: {max_length}")

    def padding(data, max_length):
        padded_data = []
        padding_mask = []
        if len(max_length) > 1:
            for item in data:
                tmp_data, tmp_mask = padding(item, max_length[1:])
                padded_data.append(tmp_data)
                padding_mask.append(tmp_mask)
            for _ in range(max_length[0] - len(data)):
                tmp_data, tmp_mask = padding([], max_length[1:])
                padded_data.append(tmp_data)
                padding_mask.append(tmp_mask)
        else:
            if len(max_length) == 1 and pad_idx != -1:
                if len(data):
                    padded_data = data
                    padding_mask = [1]
                else:
                    padded_data = pad_idx
                    padding_mask = [0]
            else:
                padded_data = data + [pad_idx for _ in range(max_length[0] - len(data))]
                padding_mask = [1 for _ in data] + [0 for _ in range(max_length[0] - len(data))]       
        return padded_data, padding_mask
            
    padded_data, padding_mask = padding(data, max_length)
    padded_data = torch.tensor(padded_data, dtype=dtype)
    padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
    if pad_idx != -1:
        padding_mask = padding_mask.squeeze(-1)
    # print(padded_data.shape, padding_mask.shape)
    return padded_data, padding_mask



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

def read_dataset(dataset_name, params, tokenizer, truncate = -1, has_true_label=False, add_sent_event_token=False, yes_no_format=False, seed_round=False):
    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(params['data_path'], file_name)

    samples = []
    with io.open(txt_file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            samples.append(json.loads(line.strip()))

    # if 'debug' in params['output_path']:
    #     samples = samples[:100]
    if yes_no_format:
        processed_data = process_mention_data_TC(
            samples=samples,  # use subset of valid data
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
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
        params=params,
        truncate = truncate, 
        has_true_label = has_true_label,
        add_sent_event_token = add_sent_event_token 
    )
    return processed_data


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


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    torch.save(model.state_dict(), output_model_file)
    tokenizer.save_vocabulary(output_dir)


def write_to_file(path, string, mode="w"):
    with open(path, mode) as f:
        f.write(string)
