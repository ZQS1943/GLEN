import torch
from transformers import get_linear_schedule_with_warmup

def get_optimizer(model, t_total, args):
    no_decay = ["bias", "LayerNorm.weight"]
    params = model.named_parameters()

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": args['learning_rate']
        },
        {   "params": [p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args['learning_rate']
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)

    return optimizer, scheduler
