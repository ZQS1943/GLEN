import sys
sys.path.append('./')
import os
import torch
import json
import random
import time
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from model.optimizer import get_optimizer
from model.params import parse_arguments
from model.utils import read_dataset, write_to_file, save_model
from model.constants import id2node_detail
from model.encoder import TypeClassification
from model.dataset import TCdataset, collate_fn_TC

def training(params):
    print("-- Type Classifier: Train --")
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    
    # Init model
    event_trigger_matcher = TypeClassification(params)
    tokenizer = event_trigger_matcher.tokenizer
    device = event_trigger_matcher.device
    n_gpu = event_trigger_matcher.n_gpu

    print("Data Loading ...")
    print(f"load training data from {params['train_samples_path']}")
    with open(params['train_samples_path'], 'r') as f:
        train_samples = json.load(f)
    if params['data_truncation'] != -1:
        train_samples = train_samples[:params['data_truncation']]
    
    processed_train_samples = []
    max_context_length = params['max_context_length']
    prefix_template = f"⟨type⟩ is defined as ⟨definition⟩."
    suffix_template = f"Does ⟨trigger⟩ indicate a ⟨type⟩ event? [MASK]"
    cnt_events = 0
    cnt_one_cand = 0
    cnt_predicted = 0
    for item in tqdm(train_samples):
        for eid, (trigger, candidate_set) in enumerate(zip(item['context']['mention_idxs'], item['label_idx'])):
            if len(candidate_set) != 1:
                if 'predicted_labels_by_bts' in item and str(eid) in item['predicted_labels_by_bts']:
                    gt_node = item['predicted_labels_by_bts'][str(eid)]
                    cnt_predicted += 1
                else:
                    continue
            else:
                gt_node = candidate_set[0]
                cnt_one_cand += 1
            cnt_events += 1
            node_list = item['top_20_events'][:2]
            if gt_node not in node_list:
                node_list.append(gt_node)
            # print(gt_node, node_list)
            for node in node_list:
                label = 0
                if node == gt_node:
                    label = 1
                # print(node, id2node_detail[node])
                name, des, _ = id2node_detail[node]
                if des is None:
                    des = ''
                trigger_words = ' '.join(item['context']['tokens'][trigger[0]:trigger[1] + 1]).replace(' ##', '')
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
                processed_train_samples.append({
                    'id': item['data_id'],
                    'event_idx': eid,
                    'event_id': node,
                    'input_ids': input_ids,
                    'label': label,
                    'mask_token_mask':mask_token_mask
                })
    print(f"get {len(processed_train_samples)} training data (one candidated: {cnt_one_cand} + predicted: {cnt_predicted}) from {cnt_events} events")


    
    test_samples = read_dataset("annotated_test_set", params, tokenizer, yes_no_format = True, truncate=params['data_truncation'])

    
    grad_acc_steps = params["gradient_accumulation_steps"]
    train_batch_size = params["train_batch_size"] // grad_acc_steps

    test_set = TCdataset(test_samples)
    train_set = TCdataset(processed_train_samples)
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn_TC)
    

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if event_trigger_matcher.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    time_start = time.time()
    write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )   

    num_train_epochs = params["num_train_epochs"]
    t_total = len(train_dataloader) // grad_acc_steps * num_train_epochs
    optimizer, scheduler = get_optimizer(event_trigger_matcher, t_total, params)

    print("Model Training ...")
    event_trigger_matcher.train() 
    for epoch_idx in trange(1, int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        event_trigger_matcher.train()
        pbar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            data_id,event_idx,event_id,input_ids,labels, mask_token_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask_token_mask = mask_token_mask.to(device)
            _, loss = event_trigger_matcher(input_ids, mask_token_mask, labels=labels)

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps
            tr_loss += loss.item()
            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                pbar.update(grad_acc_steps)
                pbar.set_postfix({'loss': float(tr_loss/grad_acc_steps)})
                tr_loss = 0

                torch.nn.utils.clip_grad_norm_(
                    event_trigger_matcher.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
        
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        save_model(event_trigger_matcher, tokenizer, epoch_output_folder_path)


    execution_time = (time.time() - time_start) / 60
    write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )


if __name__ == "__main__":
    params = parse_arguments()
    training(params)

