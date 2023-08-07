import os
import torch
import random
import time
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from encoder import TypeRanking
import utils as utils
from params import EDParser
import wandb

from model.dataset import UniEDdataset, collate_fn_TC, collate_fn_TR
from model.optimizer import get_optimizer

def main(params):
    wandb.init(project="universal_ED_TR", name=f"{params['wb_name']}-bs={params['train_batch_size']}-lr={params['learning_rate']}")
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
    type_classifier = TypeRanking(params)
    tokenizer = type_classifier.tokenizer
    device = type_classifier.device

    params["train_batch_size"] = params["train_batch_size"] // params["gradient_accumulation_steps"]
    train_batch_size = params["train_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    
    logger.info("Starting processing data")

    train_samples = utils.read_dataset("train", params, tokenizer, logger = logger, add_sent_event_token=True, truncate=params['data_truncation'])
    train_set = UniEDdataset(train_samples, only_sen_w_events=True)
    if params['loss_type'] == 'margin_loss':
        collate_function = collate_fn_TC
    elif params['loss_type'] == 'new_loss':
        collate_function = collate_fn_TR
    else:
        raise NotImplementedError
    train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, collate_fn=collate_function)

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    num_train_epochs = params["num_train_epochs"]
    t_total = len(train_dataloader) // grad_acc_steps * num_train_epochs
    optimizer, scheduler = get_optimizer(type_classifier, t_total, params)
    

    trainer_path = params.get("path_to_trainer_state", None)
    if trainer_path is not None and os.path.exists(trainer_path):
        training_state = torch.load(trainer_path)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        logger.info("Loaded saved training state")

    type_classifier.train()
    wandb.watch(type_classifier)
    
    time_start = time.time()
    for epoch_idx in trange(params["last_epoch"] + 1, int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        pbar = tqdm(total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            if params['loss_type'] == 'margin_loss':
                context_vecs, event_type_vecs, label, index, event_indexer, margin_label, true_label = batch
                # if margin_label.size()[0] == 0:
                #     continue
                # batch = tuple(t.to(device) for t in batch)
                context_vecs = context_vecs.to(device)
                event_type_vecs = event_type_vecs.to(device)
                label = label.to(device)
                margin_label = margin_label.to(device)
                loss, _ = type_classifier(
                    context_vecs, event_type_vecs,
                    label=label,
                    margin_label=margin_label,
                    return_loss=True,
                )
            elif params['loss_type'] == 'new_loss':
                context_vecs, event_type_vecs, index, event_indexer, candidate_label_sets, negative_smaples = batch
                if context_vecs is None:
                    continue

                context_vecs = context_vecs.to(device)
                event_type_vecs = event_type_vecs.to(device)
                candidate_label_sets = [[x.to(device) for x in cand_set] for cand_set in candidate_label_sets]
                negative_smaples = [x.to(device) for x in negative_smaples]
                loss, _ = type_classifier.forward_new_loss(
                    context_vecs, event_type_vecs,
                    candidate_label_sets=candidate_label_sets,
                    negative_smaples=negative_smaples,
                    return_loss=True,
                )
            else:
                raise NotImplementedError


            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()
            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                pbar.update(grad_acc_steps)
                pbar.set_postfix({'loss': float(tr_loss / grad_acc_steps)})
                wandb.log({"loss": tr_loss})
                tr_loss = 0            
                torch.nn.utils.clip_grad_norm_(
                    type_classifier.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(type_classifier, tokenizer, epoch_output_folder_path)
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
    main(params)
