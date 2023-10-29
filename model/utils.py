import os
import io
import json
import torch
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from model.data_process import process_mention_data, process_data_TC_predict_w_sentence
from model.constants import xpo_used, id2node_detail, node_relation


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
    print(f"Predicting {len(candidate_token_ids_list)} Events ...")
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
    print("Data Loading ...")
    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(params['data_path'], file_name)

    samples = []
    with io.open(txt_file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            samples.append(json.loads(line.strip()))

    # if 'debug' in params['output_path']:
    #     samples = samples[:100]
    if yes_no_format:
        processed_data = process_data_TC_predict_w_sentence(
            samples=samples,  # use subset of valid data
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
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

def evaluate_final_score(predict_samples, params, eval_on_gold = False):
    c_m_p_threshold = params['c_m_p_threshold']
    num_TD_correct = 0
    num_TD_gold = 0
    num_TD_predict = 0

    num_TC_correct = 0
    num_TC_gold = 0
    num_TC_predict = 0

    test_results = []
    for item in tqdm(predict_samples):
        tmp_item = {}
        tmp_item['sent_id'] = item['data_id']
        tokens = item['context']['tokens']
        tmp_item['sentence'] = ' '.join(tokens[1:-1]).replace(' ##', '')
        tmp_item['events'] = {}
        predicted_results = []
        if eval_on_gold:
            trigger_iter = enumerate(item['context']['mention_idxs'])
        else:
            trigger_iter = enumerate(item['predicted_triggers'])
        for event_id, predicted_trigger in trigger_iter:
            predicted_types = sorted(item['TC_results'][event_id], key=lambda x:x[0], reverse=True)
            node_to_prob = {node:prob for prob, node in predicted_types}
            top_1_node = predicted_types[0][1]
            for node in node_relation[top_1_node]['parents']:
                if node in node_to_prob and predicted_types[0][0] - node_to_prob[node] < c_m_p_threshold:
                    predicted_results.append((predicted_trigger, node))
                    break
            else:
                predicted_results.append((predicted_trigger, top_1_node))    
        gold_results = list(zip(item['context']['mention_idxs'], item['true_label']))

        def add_to_results(results, key_name = 'pred'):
            for trigger_id, e_type in results:
                trigger_word = tokens[trigger_id[0]: trigger_id[1] + 1]
                trigger_word = ' '.join(trigger_word).replace(' ##', '')

                trigger_id = f'{trigger_word}({trigger_id[0]}-{trigger_id[1] + 1})'
                if trigger_id not in tmp_item['events']:
                    tmp_item['events'][trigger_id] = {}
                name, des, _ = id2node_detail[e_type]
                tmp_item['events'][trigger_id][key_name] = f'{name}: {des}'

        add_to_results(predicted_results, 'pred')
        add_to_results(gold_results, 'gold')

        gold_trigger = set(tuple(x[0]) for x in gold_results)
        assert len(gold_trigger) == len(gold_results)
        
        for p_trigger, p_etype in predicted_results:
            for g_trigger, g_etype in gold_results:
                if g_trigger == p_trigger:
                    num_TD_correct += 1
                    num_TC_gold += 1
                    num_TC_predict += 1
                    if p_etype == g_etype:
                        num_TC_correct += 1
                    break
        num_TD_predict += len(predicted_results)
        num_TD_gold += len(gold_results)
        
        test_results.append(tmp_item)

    scores = {}
    scores['num_TI_correct'] = num_TD_correct
    scores['num_TI_gold'] = num_TD_gold
    scores['num_TI_predict'] = num_TD_predict
    scores['num_TC_correct'] = num_TC_correct

    scores["TI_prec"] = num_TD_correct/num_TD_predict
    scores["TI_recall"] = num_TD_correct/num_TD_gold
    scores["TI_F1"] = 2 * scores["TI_prec"] * scores["TI_recall"] / (scores["TI_prec"] + scores["TI_recall"])
    scores["TC_accuracy"] = num_TC_correct/num_TD_correct
    scores["TC_prec"] = num_TC_correct/num_TD_predict
    scores["TC_recall"] = num_TC_correct/num_TD_gold
    scores["TC_F1"] = 2 * scores["TC_prec"] * scores["TC_recall"] / (scores["TC_prec"] + scores["TC_recall"])
    print(json.dumps(scores, indent=True))

def hit_k(predict_samples, eval_on_gold = False):
    matched_trigger = 0
    hit_at_k_cnt = {x:0 for x in [1,2,5,10]}
    hit_at_k_cnt_in_top_10 = {x:0 for x in [1,2,5,10]}
    matched_trigger_in_top_10 = 0

    for item in tqdm(predict_samples):
        tmp_item = {}
        tmp_item['sent_id'] = item['data_id']
        tokens = item['context']['tokens']
        tmp_item['sentence'] = ' '.join(tokens[1:-1]).replace(' ##', '')
        tmp_item['events'] = {}

        gold_results = list(zip(item['context']['mention_idxs'], item['true_label']))
        if eval_on_gold:
            trigger_iter = enumerate(item['context']['mention_idxs'])
        else:
            trigger_iter = enumerate(item['predicted_triggers'])
        for event_id, predicted_trigger in trigger_iter:
            predicted_types = sorted(item['TC_results'][int(event_id)], key=lambda x:x[0], reverse=True)
            for gold_trigger, gold_type in gold_results:
                if gold_trigger == predicted_trigger:
                    matched_trigger += 1
                    in_top_10 = gold_type in [x[1] for x in predicted_types]
                    if in_top_10:
                        matched_trigger_in_top_10 += 1
                    for k in hit_at_k_cnt:
                        k_types = [x[1] for x in predicted_types[:k]]
                        if gold_type in k_types:
                            hit_at_k_cnt[k] += 1
                            if in_top_10:
                                hit_at_k_cnt_in_top_10[k] += 1
                    break
            


    scores = {}
    scores['matched_trigger'] = matched_trigger
    scores['matched_trigger_in_top_10'] = matched_trigger_in_top_10
    for k in hit_at_k_cnt:
        scores[f"Hit@{k}"] = hit_at_k_cnt[k]/matched_trigger
    for k in hit_at_k_cnt_in_top_10:
        scores[f"Hit@{k}_in_top_10"] = hit_at_k_cnt_in_top_10[k]/matched_trigger_in_top_10
    print(json.dumps(scores, indent=True))

