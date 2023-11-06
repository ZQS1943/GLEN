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
from .encoder import TriggerIdentifier
from .dataset import TITRdataset, collate_fn_TI

def main(params):
    print("-- Trigger Identifier: Train --")
    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    # Init model
    trigger_identifier = TriggerIdentifier(params)
    tokenizer = trigger_identifier.tokenizer
    device = trigger_identifier.device

    grad_acc_steps = params["gradient_accumulation_steps"]
    train_batch_size = params["train_batch_size"] // grad_acc_steps

    train_samples = read_dataset("train", params, tokenizer, truncate=params['data_truncation'])
    train_set = TITRdataset(train_samples)
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn_TI)

    write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    num_train_epochs = params["num_train_epochs"]
    t_total = len(train_dataloader) // grad_acc_steps * num_train_epochs
    optimizer, scheduler = get_optimizer(trigger_identifier, t_total, params)

    print("Model Training ...")
    trigger_identifier.train()
    time_start = time.time()
    for epoch_idx in trange(0, num_train_epochs, desc="Epoch"):
        tr_loss = 0
        pbar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            context_input, mention_label, mention_label_mask, index = batch

            context_input = context_input.to(device)
            mention_label = mention_label.to(device)
            mention_label_mask = mention_label_mask.to(device)
          
            loss, _, _  = trigger_identifier(
                context_input,
                gold_mention_bounds=mention_label,
                gold_mention_bounds_mask=mention_label_mask,
                return_loss=True,
            )


            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                pbar.update(params["print_interval"] * grad_acc_steps)
                pbar.set_postfix({'loss': float(tr_loss / (params["print_interval"] * grad_acc_steps))})
                tr_loss = 0

            loss.backward()
            

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trigger_identifier.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        

        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        save_model(trigger_identifier, tokenizer, epoch_output_folder_path)


    execution_time = (time.time() - time_start) / 60
    write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )


if __name__ == "__main__":
    params = parse_arguments()
    main(params)