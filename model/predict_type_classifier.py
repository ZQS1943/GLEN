import sys
sys.path.append('./')
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from encoder import TypeClassification
import utils as utils
from params import parse_arguments, id2node_detail
import pickle as pkl

from model.dataset import TCdataset, collate_fn_TC

def evaluate(event_trigger_matcher, test_dataloader, device, params, out_name = ''):
    output_path =f"{params['output_path']}/bts_predict_scores_{out_name}.pkl"
    results = []
    gold_labels = []
    case_ids = []
    with torch.no_grad():
        event_trigger_matcher.eval()
        for batch in tqdm(test_dataloader):
            data_id,event_idx,event_id,input_ids,labels, mask_token_mask = batch
            input_ids = input_ids.to(device)
            # labels = labels.to(device)
            mask_token_mask = mask_token_mask.to(device)
            yes_scores,_ = event_trigger_matcher(input_ids, mask_token_mask, labels=labels, return_loss=False)
            yes_scores = yes_scores.detach().cpu()
            results.append(yes_scores)
            gold_labels.append(labels)
            case_ids.extend(list(zip(data_id, event_idx)))
            # print(yes_no_scores, yes_no_scores.shape)
            # break
    
    results = torch.cat(results)
    gold_labels = torch.cat(gold_labels)
    results_dict = defaultdict(list)
    for score, label, case_id in zip(results, gold_labels, case_ids):
        results_dict[case_id].append((score, label))
    
    with open(output_path, 'wb') as f:
        pkl.dump(results_dict, f)
    # assert 1==0

    correct_num = 0
    for case_id in results_dict:
        scores = torch.tensor([x[0] for x in results_dict[case_id]])
        labels = [x[1] for x in results_dict[case_id]]
        predicted_idx = torch.argmax(scores)
        # print(case_id)
        # print(f"scores: {scores}, arg_max: {int(predicted_idx)}, correct:{int(labels[predicted_idx])}")
        if labels[predicted_idx] == 1:
            correct_num += 1
    print(f'accuracy of label selection: {correct_num/len(results_dict)}, #cases: {len(results_dict)}')
    return correct_num/len(results_dict)
    # print(results, results.shape)
    # print(gold_labels, gold_labels.shape)
    for th in torch.linspace(0.5, 1.0, 20):
        print('*'*10)
        print(f"threshold of yes score: {th}")
        yes_label = torch.where(results >= th, 1, 0)
        # print(yes_label)
        # print(yes_label == gold_labels)
        correct_num = torch.sum(yes_label == gold_labels)
        # print(correct_num)
        print(f"accuracy: {float(correct_num/len(yes_label))}; total num: {len(yes_label)}")
    assert 1==0

def predict_train_set(event_trigger_matcher, test_dataloader, device, params, events_list):
    output_path =f"{params['output_path']}/bts_prediction_results_on_trainset.jsonl"
    results = []
    case_ids = []
    with torch.no_grad():
        event_trigger_matcher.eval()
        for batch in tqdm(test_dataloader):
            data_id,event_idx,event_id,input_ids, _ , mask_token_mask = batch
            # print(batch)
            input_ids = input_ids.to(device)
            # labels = labels.to(device)
            mask_token_mask = mask_token_mask.to(device)
            yes_scores,_ = event_trigger_matcher(input_ids, mask_token_mask,return_loss=False)
            yes_scores = yes_scores.detach().cpu()
            results.append(yes_scores)
            case_ids.extend(list(zip(data_id, event_idx)))
            # print(yes_no_scores, yes_no_scores.shape)
            # break
    
    results = torch.cat(results)
    results_dict = defaultdict(list)
    for score, case_id in zip(results, case_ids):
        results_dict[case_id].append(float(score))
    # print(results_dict)
    with open(output_path, 'w') as f:
        for key in events_list:
            f.write(json.dumps({'sent_id': key[0], 'event_id': key[1], 'scores': results_dict[key]}) + '\n')

def get_train_dataloder(params):
    with open(params['TC_train_data_path'], 'r') as f:
        train_samples = json.load(f)
    if params['data_truncation'] != -1:
        train_samples = train_samples[:params['data_truncation']]
    processed_train_samples = []
    events_list = []
    max_context_length = params['max_context_length']
    prefix_template = f"⟨type⟩ is defined as ⟨definition⟩."
    suffix_template = f"Does ⟨trigger⟩ indicate a ⟨type⟩ event? [MASK]"
    cnt_events = 0
    for item in tqdm(train_samples):
        for eid, (trigger, candidate_set) in enumerate(zip(item['context']['mention_idxs'], item['label_idx'])):
            if len(candidate_set) <= 1:
                continue
            cnt_events += 1
            events_list.append((item['data_id'], eid))
            for node in candidate_set:
                name, des, _ = id2node_detail[node]
                # print(node, name, des)
                if des is None:
                    des = ''
                trigger_words = ' '.join(item['context']['tokens'][trigger[0]:trigger[1] + 1]).replace(' ##', '')
                prefix = prefix_template.replace('⟨type⟩', name).replace('⟨definition⟩', des)
                # print(prefix)
                suffix = suffix_template.replace('⟨trigger⟩', trigger_words).replace('⟨type⟩', name)
                # print(suffix)
                prefix_id = tokenizer.encode(prefix)
                suffix_id = tokenizer.encode(suffix)
                input_ids = prefix_id + item['context']['original_input'] + suffix_id
                # print(input_ids)
                mask_token_id = len(input_ids)
                if len(input_ids) > max_context_length - 2:
                    print(input_ids)
                    assert 1==0
                input_ids = [101] + input_ids + [102] + [0]*(max_context_length - 2 - len(input_ids))
                assert len(input_ids) == max_context_length
                mask_token_mask = [0]*max_context_length
                mask_token_mask[mask_token_id] = 1
                processed_train_samples.append({
                    'id': item['data_id'],
                    'event_idx': eid,
                    'event_id': node,
                    'input_ids': input_ids,
                    'label': -1,
                    'mask_token_mask':mask_token_mask
                })
    print(f"get {len(processed_train_samples)} training data from {cnt_events} events")

    predict_set = TCdataset(processed_train_samples)
    predcit_dataloader = DataLoader(predict_set, batch_size=params["eval_batch_size"], shuffle=False, collate_fn=collate_fn_TC)
    return predcit_dataloader, events_list

def get_predict_dataloder(params, predict_samples, k = 10, eval_on_gold=False):
    if params['data_truncation'] != -1:
        predict_samples = predict_samples[:params['data_truncation']]
    processed_predict_samples = []
    max_context_length = params['max_context_length']
    prefix_template = f"⟨type⟩ is defined as ⟨definition⟩."
    suffix_template = f"Does ⟨trigger⟩ indicate a ⟨type⟩ event? [MASK]"
    cnt_events = 0
    for item_id, item in tqdm(enumerate(predict_samples)):
        types_for_sentence = item['top_20_events'][:k]
        if eval_on_gold:
            trigger_iter = enumerate(item['context']['mention_idxs'])
        else:
            trigger_iter = enumerate(item['predicted_triggers'])
        for eid, trigger in trigger_iter:
            cnt_events += 1
            trigger_words = ' '.join(item['context']['tokens'][trigger[0]:trigger[1] + 1]).replace(' ##', '')
            for node in types_for_sentence:
                name, des, _ = id2node_detail[node]
                if des is None:
                    des = ''
                prefix = prefix_template.replace('⟨type⟩', name).replace('⟨definition⟩', des)
                suffix = suffix_template.replace('⟨trigger⟩', trigger_words).replace('⟨type⟩', name)
                prefix_id = tokenizer.encode(prefix)
                suffix_id = tokenizer.encode(suffix)
                input_ids = prefix_id + item['context']['original_input'] + suffix_id
                mask_token_id = len(input_ids)
                if len(input_ids) > max_context_length - 2:
                    print(input_ids)
                    assert 1==0
                input_ids = [101] + input_ids + [102] + [0]*(max_context_length - 2 - len(input_ids))
                assert len(input_ids) == max_context_length
                mask_token_mask = [0]*max_context_length
                mask_token_mask[mask_token_id] = 1
                processed_predict_samples.append({
                    'id': item_id,
                    'event_idx': eid,
                    'event_id': node,
                    'input_ids': input_ids,
                    'label': -1,
                    'mask_token_mask':mask_token_mask
                })
    print(f"get {len(processed_predict_samples)} predicting data from {cnt_events} events")

    predict_set = TCdataset(processed_predict_samples)
    predcit_dataloader = DataLoader(predict_set, batch_size=params["eval_batch_size"], shuffle=False, collate_fn=collate_fn_TC)
    return predcit_dataloader

def evaluate_final_score(event_trigger_matcher, predict_samples, test_dataloader, device, params, eval_on_gold=False):
    results = []
    case_ids = []
    event_ids = []
    with torch.no_grad():
        event_trigger_matcher.eval()
        for batch in tqdm(test_dataloader):
            data_id,event_idx,event_id,input_ids,labels, mask_token_mask = batch
            input_ids = input_ids.to(device)
            # labels = labels.to(device)
            mask_token_mask = mask_token_mask.to(device)
            yes_scores,_ = event_trigger_matcher(input_ids, mask_token_mask, labels=labels, return_loss=False)
            yes_scores = yes_scores.detach().cpu()
            results.append(yes_scores)
            event_ids.extend(event_id)
            case_ids.extend(list(zip(data_id, event_idx)))
            # print(yes_no_scores, yes_no_scores.shape)
            # break
    
    results = torch.cat(results)
    results_dict = defaultdict(list)
    for score, label, case_id in zip(results, event_ids, case_ids):
        results_dict[case_id].append((float(score), label))
    
    for case_id in results_dict:
        sentence_id, event_id = case_id
        if 'tc_types' not in predict_samples[sentence_id]:
            predict_samples[sentence_id]['tc_types'] = {}
        predict_samples[sentence_id]['tc_types'][event_id] = results_dict[case_id]
    
    output_file_name = f"{params['output_path']}/TC_and_TR_and_TD_results_annotated_test_no_other.json"
    if eval_on_gold:
        output_file_name = f"{params['output_path']}/TC_and_TR_and_TD_results_annotated_test_no_other_eval_on_gold.json"
    with open(output_file_name, 'w') as f:
        f.write(json.dumps(predict_samples))
    
    return results_dict


if __name__ == "__main__":
    params = parse_arguments()

    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Init model
    event_trigger_matcher = TypeClassification(params)
    tokenizer = event_trigger_matcher.tokenizer
    device = event_trigger_matcher.device

    # evaluate on dev set
    # dev_samples = utils.read_dataset("annotated_dev_set", params, tokenizer, yes_no_format = True)
    # dev_set = TCdataset(dev_samples)
    # dev_dataloader = DataLoader(dev_set, batch_size=params["eval_batch_size"], shuffle=False, collate_fn=collate_fn_TC)
    # evaluate(event_trigger_matcher, dev_dataloader, device, params, out_name = 'dev_set')

    # evaluate on test set
    # test_samples = utils.read_dataset("annotated_test_set", params, tokenizer, yes_no_format = True)
    # test_set = TCdataset(test_samples)
    # test_dataloader = DataLoader(test_set, batch_size=params["eval_batch_size"], shuffle=False, collate_fn=collate_fn_TC)
    # evaluate(event_trigger_matcher, test_dataloader, device, params)

    # prediction of train set
    # predict_dataloder, events_list = get_train_dataloder(params)
    # predict_train_set(event_trigger_matcher, predict_dataloder, device, params, events_list)

    # evaluate on annotated test set based on TR and TD results
    eval_on_gold = True
    # eval_on_gold = False
    with open('./exp/experiments_type_ranking/se_id_new_loss_new_ontology/bert_base/epoch_4/TR_and_TD_results_annotated_test_no_other.json', 'r') as f:
        predict_samples = json.load(f)
    predict_dataloder = get_predict_dataloder(params, predict_samples, eval_on_gold=eval_on_gold)
    results_dict = evaluate_final_score(event_trigger_matcher, predict_samples, predict_dataloder, device, params, eval_on_gold=eval_on_gold)
    

    # get scores
    # get_final_score(results_dict, predict_samples)

