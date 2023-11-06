import torch
import json
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from encoder import TriggerIdentifier
from collections import defaultdict

from .params import parse_arguments
from .dataset import TITRdataset, collate_fn_TI
from .utils import get_predicted_mention_bounds, read_dataset

def evaluate(params, trigger_identifier, device, predict_samples, predict_dataloader, predict_set):
    results = {}
    predict_num = correct_num = gold_num = 0

    domain_detail = defaultdict(lambda: {'gold_num':0, 'correct_num':0, 'predict_num': 0})
    trigger_identifier.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(predict_dataloader, desc="Prediction")):
            context_input, mention_label, mention_label_mask, data_index = batch
            context_input = context_input.to(device)
            
            _, mention_logits, mention_bounds  = trigger_identifier(
                context_input,
                return_loss=False,
            )

            mention_logits = torch.sigmoid(mention_logits).cpu().numpy()
            mention_bounds = mention_bounds.cpu().numpy()

            predicted_mention_bounds, predicted_logits = get_predicted_mention_bounds(mention_logits, mention_bounds)

            correct_num_list = []
            gold_num_list = []
            predict_num_list = []
            for predicted_mention, gt_mention, gt_mention_mask in zip(predicted_mention_bounds, mention_label, mention_label_mask):
                predicted_mention = predicted_mention.tolist()
                gt_mention = gt_mention[gt_mention_mask].tolist()
                predict_num_list.append(len(predicted_mention))
                gold_num_list.append(len(gt_mention))
                cnt = 0
                for mention in predicted_mention:
                    if any(mention == m for m in gt_mention):
                        cnt += 1
                correct_num_list.append(cnt)
        
            for idx, predicted_mention in zip(data_index, predicted_mention_bounds):
                predicted_mention = predicted_mention.tolist()
                predict_samples[idx][f'predicted_triggers'] = predicted_mention

            for i, idx in enumerate(data_index):
                cur_data = predict_set.data[idx]

                domain = cur_data['domain']
                domain_detail[domain]['correct_num'] += correct_num_list[i]
                domain_detail[domain]['gold_num'] += gold_num_list[i]
                domain_detail[domain]['predict_num'] += predict_num_list[i]

            predict_num += sum(predict_num_list)
            correct_num += sum(correct_num_list)
            gold_num += sum(gold_num_list)
    
    with open(f"{params['output_path']}/TI_result.json", 'w') as f:
        f.write(json.dumps(predict_samples))
            
    p = correct_num / predict_num
    r = correct_num / gold_num
    f1 = 2 * p * r / (p + r)

    results["Precision"] = p
    results["Recall"] = r
    results["F1"] = f1
    print("Total, %d - P R F1: %.3f %.3f %.3f" % (gold_num, p, r, f1))

    predict_num = correct_num = gold_num = 0
    
    for domain in domain_detail:
        if domain != 'propbank':
            predict_num += domain_detail[domain]['predict_num']
            correct_num += domain_detail[domain]['correct_num']
            gold_num += domain_detail[domain]['gold_num']
        p = domain_detail[domain]['correct_num'] / domain_detail[domain]['predict_num']
        r = domain_detail[domain]['correct_num'] / domain_detail[domain]['gold_num']
        f1 = 2 * p * r / (p + r)
        print("%s, %d - P R F1: %.3f %.3f %.3f" % (domain, domain_detail[domain]['gold_num'], p, r, f1))

    p = correct_num / predict_num
    r = correct_num / gold_num
    f1 = 2 * p * r / (p + r)

    print("Total w/o propbank, %d - P R F1: %.3f %.3f %.3f" % (gold_num, p, r, f1))
    
    return results

if __name__ == "__main__":
    print("-- Trigger Identifier: Predict --")
    params = parse_arguments()

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Init model
    trigger_identifier = TriggerIdentifier(params)
    tokenizer = trigger_identifier.tokenizer
    device = trigger_identifier.device
    
    eval_batch_size = params["eval_batch_size"]

    predict_samples = read_dataset("test_annotated", params, tokenizer, has_true_label=True, add_sent_event_token=True)

    predict_set = TITRdataset(predict_samples)
    predict_dataloader = DataLoader(predict_set, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn_TI)

    evaluate(params, trigger_identifier, device, predict_samples, predict_dataloader, predict_set)    
