import torch
import json
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from encoder import TriggerLocalizer
import utils as utils
from params import EDParser
from collections import defaultdict

from model.dataset import UniEDdataset, collate_fn_TD
from model.utils import get_predicted_mention_bounds
from model.params import id2roleset

def eval_mention_detection(params, trigger_detector, device, predict_samples, predict_dataloader, silent=True):
    results = {}
    predict_num = correct_num = gold_num = 0
    predicted_data = []

    roleset_detail = defaultdict(lambda: {'gold_num':0, 'correct_num':0})
    domain_detail = defaultdict(lambda: {'gold_num':0, 'correct_num':0, 'predict_num': 0})
    trigger_detector.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(predict_dataloader, desc="Prediction")):
            context_input, mention_label, mention_label_mask, data_index = batch
            context_input = context_input.to(device)
            
            _, mention_logits, mention_bounds  = trigger_detector(
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
                    # print(mention, gt_mention)
                    if any(mention == m for m in gt_mention):
                        cnt += 1
                correct_num_list.append(cnt)
                # print(predicted_mention,gt_mention, cnt, len(predicted_mention), len(gt_mention))
            #     print(predicted_mention, gt_mention, l )
            # assert 1==0

            # predict_num_list, correct_num_list , gold_num_list, roleset_detail= eval_trigger_detection(predicted_mention_bounds, mention_idx, mention_idx_mask, roleset_ids, roleset_detail)
        
            for idx, predicted_mention in zip(data_index, predicted_mention_bounds):
                predicted_mention = predicted_mention.tolist()
                predict_samples[idx][f'predicted_triggers'] = predicted_mention

            if not silent:
                for i, idx in enumerate(data_index):
                    cur_data = predict_set.data[idx]
                    # context = tokenizer.convert_ids_to_tokens(context_input[i])

                    domain = cur_data['domain']
                    domain_detail[domain]['correct_num'] += correct_num_list[i]
                    domain_detail[domain]['gold_num'] += gold_num_list[i]
                    domain_detail[domain]['predict_num'] += predict_num_list[i]

                    tokens = cur_data['context']['tokens']
                    tmp_item = {
                        'data_id': cur_data['data_id'],
                        'tokens': tokens,
                        'trigger_bound': cur_data['context']['mention_idxs'],
                        'trigger': [' '.join(tokens[mention[0]: mention[1] + 1]) for mention in cur_data['context']['mention_idxs']],
                        'roleset': [id2roleset[x] for x in cur_data['roleset_ids']],
                        'predicted_bounds': predicted_mention_bounds[i].tolist(),
                        'predicted_trigger': [' '.join(tokens[mention[0]: mention[1] + 1]) for mention in predicted_mention_bounds[i]],
                        'predicted_logits': predicted_logits[i].tolist()
                    }
                    # if domain == 'anc' and correct_num_list[i] != gold_num_list[i]:
                    predicted_data.append(tmp_item)

            predict_num += sum(predict_num_list)
            correct_num += sum(correct_num_list)
            gold_num += sum(gold_num_list)
    
    with open(f"{params['output_path']}/TD_results_test_annotated_no_other.json", 'w') as f:
        f.write(json.dumps(predict_samples))
            
    p = correct_num / predict_num
    r = correct_num / gold_num
    f1 = 2 * p * r / (p + r)

    if not silent:
        output_path =f"{params['output_path']}/trigger_detection_predicted_result.json"
        with open(output_path, 'w') as f:
            f.write(json.dumps(predicted_data, indent=True))

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
    
    results["roleset_detail"] = roleset_detail
    return results

if __name__ == "__main__":
    parser = EDParser(add_model_args=True)
    parser.add_training_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__

    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Init model
    trigger_detector = TriggerLocalizer(params)
    tokenizer = trigger_detector.tokenizer
    device = trigger_detector.device
    eval_batch_size = params["eval_batch_size"]


    # predict_samples = utils.read_dataset("test", params, tokenizer)
    # predict_samples = utils.read_dataset("sample", params, tokenizer, has_true_label=True)
    predict_samples = utils.read_dataset("annotated_test_set", params, tokenizer, has_true_label=True, add_sent_event_token=True)
    # print(predict_samples[:10])
    # assert 1==0

    predict_set = UniEDdataset(predict_samples)
    predict_dataloader = DataLoader(predict_set, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn_TD)
    type_classification_results = eval_mention_detection(params, trigger_detector, device, predict_samples, predict_dataloader, silent=False)    
