import os
import torch
import json
import random
import time
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from torch.utils.data import DataLoader
from encoder import EventTriggerMatchingYN
import utils as utils
from params import EDParser, id2node
import wandb
import pickle as pkl

from model.optimizer import get_optimizer
from model.dataset import YNdataset, collate_fn_YN

def evaluate(event_trigger_matcher, test_dataloader, device):
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
            yes_scores,_ = event_trigger_matcher(input_ids,labels, mask_token_mask,return_loss=False)
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
    
    with open('./cache/bts_predict_scores.pkl', 'wb') as f:
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

def training(params):
    wandb.init(project="universal_ED_ETM_BST", name=f"{params['wb_name']}-bs={params['train_batch_size']}-lr={params['learning_rate']}")
    wandb.config = params 
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("accuracy_on_test", summary="max")

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])
    
    
    # Init model
    event_trigger_matcher = EventTriggerMatchingYN(params)
    tokenizer = event_trigger_matcher.tokenizer
    device = event_trigger_matcher.device
    n_gpu = event_trigger_matcher.n_gpu


    # train_samples = utils.read_dataset("train", params, tokenizer, logger = logger, yes_no_format = True, truncate=params['data_truncation'], seed_round=True)
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
                # print(node, id2node[node])
                name, des, _ = id2node[node]
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
    # print(processed_train_samples)
    print(f"get {len(processed_train_samples)} training data (one candidated: {cnt_one_cand} + predicted: {cnt_predicted}) from {cnt_events} events")
    # assert 1==0

    
    test_samples = utils.read_dataset("annotated_test_set", params, tokenizer, logger = logger, yes_no_format = True, truncate=params['data_truncation'])

    
    params["train_batch_size"] = params["train_batch_size"] // params["gradient_accumulation_steps"]
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    test_set = YNdataset(test_samples)
    test_dataloader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn_YN)
    train_set = YNdataset(processed_train_samples)
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn_YN)
    

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if event_trigger_matcher.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    logger.info("Starting processing data")

    time_start = time.time()
    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    

    num_train_epochs = params["num_train_epochs"]
    t_total = len(train_dataloader) // grad_acc_steps * num_train_epochs
    optimizer, scheduler = get_optimizer(event_trigger_matcher, t_total, params)

    trainer_path = params.get("path_to_trainer_state", None)
    if trainer_path is not None and os.path.exists(trainer_path):
        training_state = torch.load(trainer_path)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        logger.info("Loaded saved training state")

    event_trigger_matcher.train()
    wandb.watch(event_trigger_matcher)

    logger.info("Num samples per batch : %d" % (train_batch_size * params["gradient_accumulation_steps"]))
    
    
    # accuracy = evaluate(event_trigger_matcher, test_dataloader, device)
    # wandb.log({"accuracy_on_test": float(accuracy)})
    for epoch_idx in trange(params["last_epoch"] + 1, int(num_train_epochs), desc="Epoch"):
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
                wandb.log({"loss": float(tr_loss/grad_acc_steps)})
                tr_loss = 0

                torch.nn.utils.clip_grad_norm_(
                    event_trigger_matcher.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
        # accuracy = evaluate(event_trigger_matcher, test_dataloader, device)
        # wandb.log({"accuracy_on_test": float(accuracy)})
        
        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(event_trigger_matcher, tokenizer, epoch_output_folder_path)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, os.path.join(epoch_output_folder_path, "training_state.th"))

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))


if __name__ == "__main__":
    parser = EDParser(add_model_args=True)
    parser.add_training_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    
    
    training(params)

