import os
import torch
import random
import time
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from .optimizer import get_optimizer
from .params import parse_arguments
from .utils import read_dataset, write_to_file, save_model
from .encoder import TypeRanking
from .dataset import TITRdataset, collate_fn_TR_Train

def main(params):
    print("-- Type Ranking: Train --")
    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    
    # Init model
    type_ranking = TypeRanking(params)
    tokenizer = type_ranking.tokenizer
    device = type_ranking.device

    grad_acc_steps = params["gradient_accumulation_steps"]
    train_batch_size = params["train_batch_size"] // grad_acc_steps

    train_samples = read_dataset("train", params, tokenizer, add_sent_event_token=True, truncate=params['data_truncation'])
    train_set = TITRdataset(train_samples, only_sen_w_events=True)
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn_TR_Train)

    write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    num_train_epochs = params["num_train_epochs"]
    t_total = len(train_dataloader) // grad_acc_steps * num_train_epochs
    optimizer, scheduler = get_optimizer(type_ranking, t_total, params)
    
    print("Model Training ...")
    type_ranking.train()
    time_start = time.time()
    for epoch_idx in trange(0, num_train_epochs, desc="Epoch"):
        tr_loss = 0
        pbar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            context_vecs, event_type_vecs, index, event_indexer, candidate_label_sets, negative_smaples = batch
            if context_vecs is None:
                continue

            context_vecs = context_vecs.to(device)
            event_type_vecs = event_type_vecs.to(device)
            candidate_label_sets = [[x.to(device) for x in cand_set] for cand_set in candidate_label_sets]
            negative_smaples = [x.to(device) for x in negative_smaples]
            loss, _ = type_ranking(
                context_vecs, event_type_vecs,
                candidate_label_sets=candidate_label_sets,
                negative_smaples=negative_smaples,
            )


            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()
            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                pbar.update(grad_acc_steps)
                pbar.set_postfix({'loss': float(tr_loss / grad_acc_steps)})
                tr_loss = 0            
                torch.nn.utils.clip_grad_norm_(
                    type_ranking.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        save_model(type_ranking, tokenizer, epoch_output_folder_path)


    execution_time = (time.time() - time_start) / 60
    write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )



if __name__ == "__main__":
    params = parse_arguments()
    main(params)
