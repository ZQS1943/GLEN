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
from model.encoder import TypeClassifier
from model.dataset import TCdataset, collate_fn_TC
from model.data_process import process_data_TC_w_token_id

def training(params):
    print("-- Type Classifier: Train --")
    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    # Init model
    type_classifier = TypeClassifier(params)
    tokenizer = type_classifier.tokenizer
    device = type_classifier.device

    train_data_path = params['TC_train_data_path']
    print(f"Data Loading from {train_data_path} ...")
    with open(train_data_path, 'r') as f:
        train_samples = json.load(f)
    if params['data_truncation'] != -1:
        train_samples = train_samples[:params['data_truncation']]
    
    processed_train_samples = process_data_TC_w_token_id(params, train_samples, id2node_detail, tokenizer)

    
    grad_acc_steps = params["gradient_accumulation_steps"]
    train_batch_size = params["train_batch_size"] // grad_acc_steps

    train_set = TCdataset(processed_train_samples)
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn_TC)
    

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if type_classifier.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    time_start = time.time()
    write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )   

    num_train_epochs = params["num_train_epochs"]
    t_total = len(train_dataloader) // grad_acc_steps * num_train_epochs
    optimizer, scheduler = get_optimizer(type_classifier, t_total, params)

    print("Model Training ...")
    type_classifier.train() 
    for epoch_idx in trange(1, int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        type_classifier.train()
        pbar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            _,_,_,input_ids,labels, mask_token_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask_token_mask = mask_token_mask.to(device)
            _, loss = type_classifier(input_ids, mask_token_mask, labels=labels)

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps
            tr_loss += loss.item()
            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                pbar.update(grad_acc_steps)
                pbar.set_postfix({'loss': float(tr_loss/grad_acc_steps)})
                tr_loss = 0

                torch.nn.utils.clip_grad_norm_(
                    type_classifier.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
        
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        save_model(type_classifier, tokenizer, epoch_output_folder_path)


    execution_time = (time.time() - time_start) / 60
    write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )


if __name__ == "__main__":
    params = parse_arguments()
    training(params)

