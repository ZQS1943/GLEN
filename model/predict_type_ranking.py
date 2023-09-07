import sys
sys.path.append('./')
import torch
import json
import os
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.encoder import TypeRanking
from model.params import parse_arguments
from model.dataset import TITRdataset, collate_fn_TR_Predict
from model.utils import encode_all_candidates, read_dataset

def predict_top_k(output_file, type_ranking, device, predict_dataloader, cand_encs, predict_samples, used_cand, k = 20):
    results = {}
    with torch.no_grad():
        type_ranking.eval()
        for step, batch in enumerate(tqdm(predict_dataloader, desc="Prediction")):
            context_vecs, data_index= batch
            context_vecs = context_vecs.to(device)

            _, logits = type_ranking(
                context_vecs, None,
                cand_encs=cand_encs,
            )
            
            top_k = torch.topk(logits, k, dim=1)
            converted_indices = used_cand[top_k.indices]
            
            for idx, indices in zip(data_index, converted_indices):
                predict_samples[idx][f'top_events'] = indices.tolist()

    with open(output_file, 'w') as f:
        f.write(json.dumps(predict_samples))    
    return results


def predict_all_cands(type_ranking, device, predict_dataloader, cand_encs,  used_cand):
    used_cand = used_cand.to(device)
    results = {}

    k_list = [1,10,20,50,100]
    k_dict = {k: {'total_num':0, 'correct_num': 0} for k in k_list}
    with torch.no_grad():
        type_ranking.eval()
        for step, batch in enumerate(tqdm(predict_dataloader, desc="Prediction")):
            context_vecs, _, label, _, _, _, true_label = batch
            context_vecs = context_vecs.to(device)
            
            _, logits = type_ranking(
                context_vecs, _,
                cand_encs=cand_encs,
                return_loss=False,
            )
            
            for k in k_list:
                total_num = 0
                correct_num = 0
                top_k = torch.topk(logits, k, dim=1)
                # print(used_cand)
                # print(top_k.indices)
                converted_indices = used_cand[top_k.indices]
                for label_list, indices in zip(true_label, converted_indices):
                    for label in label_list:
                        total_num += 1
                        if label in indices:
                            correct_num += 1
                k_dict[k]['total_num'] += total_num
                k_dict[k]['correct_num'] += correct_num

    for k in k_list:
        print(f"Top {k} ratio: {k_dict[k]['correct_num']/k_dict[k]['total_num']}")

    return results


if __name__ == "__main__":
    print("-- Type Ranking: Predict --")
    params = parse_arguments()
    
    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Init model
    type_ranking = TypeRanking(params)
    tokenizer = type_ranking.tokenizer
    device = type_ranking.device

    eval_batch_size = params["eval_batch_size"]

    if params['predict_set'] == 'train_set':
        predict_samples = read_dataset("train", params, tokenizer, add_sent_event_token=True)
        output_file = os.path.join(params['output_path'], f'train_data_for_TC.json')
        predict_set = TITRdataset(predict_samples, with_sent_tag=True, only_sen_w_events=True)
        predict_samples = predict_set.data

    
    # params["max_context_length"] = 512
    # evaluate on test set
    # predict_samples = read_dataset("annotated_test_set", params, tokenizer, add_sent_event_token=True, has_true_label=True)
    # predict_set = TITRdataset(predict_samples, with_sent_tag=True)

    # get train data for type classification
    # k = 20
    # predict_samples = read_dataset("train", params, tokenizer, add_sent_event_token=True)
    # output_file = f'./cache/type_ranking_results_with_top_{k}_events_train_set_kairos.json'
    # predict_set = TITRdataset(predict_samples, with_sent_tag=True, only_sen_w_events=True)
    # predict_samples = predict_set.data
    
    
    # predict based on results of trigger detector
    # k=20
    # with open('./exp/experiments_trigger_identifier/wo_other_new_ontology/all_avg_128_bert_base_qa_linear_False/epoch_4/TD_results_test_annotated_no_other.json', 'r') as f:
    #     predict_samples = json.load(f)
    # predict_set = TITRdataset(predict_samples, with_sent_tag=True)
    # output_file = f"{params['output_path']}/TR_and_TD_results_annotated_test_no_other.json"
    
    # output_file = f'./cache/processed_mention_data_with_top_{k}_events_annotated_dev_set.json'
    # output_file = f'./cache/processed_mention_data_with_top_{k}_events_annotated_test_set.json'
    
    # output_file = f'./cache/TR_and_TD_results_annotated_test_no_other.json'

    # predict_set = TITRdataset(predict_samples, with_sent_tag=True)    
    predict_dataloader = DataLoader(predict_set, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn_TR_Predict)

    cand_encs, used_cand = encode_all_candidates(params, type_ranking, device, only_used_xpo=True)

    # type_classification_results = predict_all_cands(type_ranking, device, predict_dataloader, cand_encs, used_cand)
    
    type_classification_results = predict_top_k(output_file, type_ranking, device, predict_dataloader, cand_encs, predict_samples, used_cand, k = params['k'])