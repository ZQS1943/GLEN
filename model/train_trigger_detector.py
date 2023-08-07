import os
import torch
import random
import time
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from encoder import TriggerLocalizer
import utils as utils
from params import EDParser
import wandb

from model.optimizer import get_optimizer
from model.dataset import UniEDdataset, collate_fn_TD

def main(params):
    wandb.init(project="universal_ED_TD")
    wandb.config = params    
    wandb.define_metric("loss", summary="min")

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])
    
    
    # Init model
    trigger_detector = TriggerLocalizer(params)
    tokenizer = trigger_detector.tokenizer
    device = trigger_detector.device

    params["train_batch_size"] = params["train_batch_size"] // params["gradient_accumulation_steps"]
    train_batch_size = params["train_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    train_samples = utils.read_dataset("train", params, tokenizer, logger = logger)
    train_set = UniEDdataset(train_samples)
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn_TD)

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    num_train_epochs = params["num_train_epochs"]
    t_total = len(train_dataloader) // grad_acc_steps * num_train_epochs
    optimizer, scheduler = get_optimizer(trigger_detector, t_total, params)

    trainer_path = params.get("path_to_trainer_state", None)
    if trainer_path is not None and os.path.exists(trainer_path):
        training_state = torch.load(trainer_path)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        logger.info("Loaded saved training state")

    trigger_detector.train()
    wandb.watch(trigger_detector)
    
    time_start = time.time()
    for epoch_idx in trange(params["last_epoch"] + 1, int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        pbar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            context_input, mention_label, mention_label_mask, index = batch
            # batch = tuple(t.to(device) for t in batch)
            context_input = context_input.to(device)
            mention_label = mention_label.to(device)
            mention_label_mask = mention_label_mask.to(device)
          
            loss, _, _  = trigger_detector(
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
            
            wandb.log({"loss": loss})

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trigger_detector.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(trigger_detector, tokenizer, epoch_output_folder_path)
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

    
    wandb.finish()

if __name__ == "__main__":
    parser = EDParser(add_model_args=True)
    parser.add_training_args()

    args = parser.parse_args()
    params = args.__dict__
    main(params)