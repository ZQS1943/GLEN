from transformers import BertTokenizerFast
from model.encoder import TriggerIdentifier, TypeRanking, TypeClassifier
from model.dataset import GLENDataset, collate_fn_TD, collate_fn_TR, get_tc_dataloder, id2node, xpo_used, is_ldc_event
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm

def sentence_tokenization(sentence_list, tokenizer, max_length=512):
    data = []
    for sent_id, sentence in enumerate(sentence_list):
        tokens = tokenizer(sentence, return_offsets_mapping=True, max_length=max_length, padding=False, truncation=True)
        tokenized_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'])
        original_tokenized_text_ids = tokens['input_ids']
        tokenized_text_ids = original_tokenized_text_ids + [0]*(max_length - len(original_tokenized_text_ids))
        assert len(tokenized_text_ids) == max_length
        
        data.append({
            'sen_id': sent_id,
            'sentence': sentence,
            'original_tokenized_text_ids': original_tokenized_text_ids,
            'tokenized_text_ids': tokenized_text_ids,
            'token_offsets': tokens['offset_mapping'],
            'tokenized_tokens': tokenized_tokens
        })
    return data

def get_predicted_mention_bounds(logits, bounds):
    predicted_bounds = []
    predicted_logits = []
    for b_logits, b_bounds in zip(logits, bounds):
        predicted_bounds.append(b_bounds[b_logits>0.5])
        predicted_logits.append(b_logits[b_logits>0.5])
    return predicted_bounds, predicted_logits


def encode_all_candidates(params, model, device, one_vec=False, only_used_xpo=False):
    # eval_batch_size = params["eval_batch_size"]
    eval_batch_size = 128
    candidate_token_ids = torch.load(params["cand_token_ids_path"])
    if only_used_xpo:
        cand_index = torch.tensor([i for i in range(len(candidate_token_ids))])
        cand_mask = torch.tensor(xpo_used, dtype=torch.bool)
        used_cand = cand_index[cand_mask]
        candidate_token_ids_list = torch.tensor([candidate_token_ids[int(i)] for i in used_cand], dtype=torch.long)
    else:
        used_cand = None
        candidate_token_ids_list = torch.tensor([candidate_token_ids[i] for i in range(len(candidate_token_ids))], dtype=torch.long)
    print(f"predicting {len(candidate_token_ids_list)} cand vecs...")
    # assert 1==0
    all_cands = TensorDataset(candidate_token_ids_list)
    all_cands_loder = DataLoader(all_cands, batch_size=eval_batch_size, shuffle=False)
    cand_encs = []
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(all_cands_loder, desc="Candidates")):
            batch = batch[0].to(device)
            result = model.encode_candidate(batch, one_vec=one_vec)
            result = result.detach()
            cand_encs.append(result)
    cand_encs = torch.cat(cand_encs, 0)
    return cand_encs, used_cand


def predict(sentence_list):
    params = {
        'bert_model': 'bert-base-uncased',
        'path_to_trigger_detector': './ckpts/trigger_detector.bin',
        'path_to_type_ranking': './ckpts/type_ranker.bin',
        'path_to_type_classifier': './ckpts/type_classifier.bin',
        'bs_td': 64,
        'bs_tr': 4,
        'bs_tc': 64,
        'cand_token_ids_path': './data/node_tokenized_ids_64_with_event_tag.pt',
        'max_mention_length':10,
        'type_ranking_k': 10,

    }
    tokenizer = BertTokenizerFast.from_pretrained(params['bert_model'])

    parsed_data = sentence_tokenization(sentence_list, tokenizer, max_length=512)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # trigger detecting
    predict_set = GLENDataset(parsed_data)
    predict_dataloader_TD = DataLoader(predict_set, batch_size=params['bs_td'], shuffle=False, collate_fn=collate_fn_TD)
    trigger_detector = TriggerLocalizer(params)
    trigger_detector = trigger_detector.to(device)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(predict_dataloader_TD, desc="Trigger Detecting")):
            context_input, data_index = batch
            context_input = context_input.to(device)
            
            _, mention_logits, mention_bounds  = trigger_detector(
                context_input,
                return_loss=False,
            )

            mention_logits = torch.sigmoid(mention_logits).cpu().numpy()
            mention_bounds = mention_bounds.cpu().numpy()
            
            predicted_mention_bounds, predicted_logits = get_predicted_mention_bounds(mention_logits, mention_bounds)

            for idx, predicted_bounds, _predicted_logits in zip(data_index, predicted_mention_bounds, predicted_logits):
                predict_set.data[idx]['predicted_triggers'] = predicted_bounds
                predict_set.data[idx]['predicted_logits'] = _predicted_logits
                

    # type ranking
    type_ranker = TypeRanking(params)
    type_ranker = type_ranker.to(device)
    predict_dataloader_TR = DataLoader(predict_set, batch_size=params['bs_tr'], shuffle=False, collate_fn=collate_fn_TR)
    cand_encs, used_cand = encode_all_candidates(params, type_ranker, device, only_used_xpo=True)
    k = params['type_ranking_k']
    with torch.no_grad():
        type_ranker.eval()
        for step, batch in enumerate(tqdm(predict_dataloader_TR, desc='Type Ranking')):
            context_vecs, data_index = batch
            context_vecs = context_vecs.to(device)
            _, logits = type_ranker.forward_new_loss(
                context_vecs,
                cand_encs=cand_encs,
                return_loss=False,
            )
            
            top_k = torch.topk(logits, k, dim=1)
            converted_indices = used_cand[top_k.indices]
            for idx, indices in zip(data_index, converted_indices):
                predict_set[idx][f'top_{k}_events'] = indices.tolist()

    # type classification
    predicted_items = predict_set.data
    for i in range(len(predicted_items)):
        predicted_items[i]['tc_scores'] = [{} for _ in predicted_items[i]['predicted_triggers']]
    type_classifier = EventTriggerMatchingYN(params, device)
    type_classifier = type_classifier.to(device)
    tc_dataloder = get_tc_dataloder(params, predicted_items, type_classifier.tokenizer,k=k)
    with torch.no_grad():
        type_classifier.eval()
        for batch in tqdm(tc_dataloder, desc='Type Classification'):
            data_id,event_idx,event_id,input_ids, mask_token_mask = batch
            input_ids = input_ids.to(device)
            mask_token_mask = mask_token_mask.to(device)
            yes_scores,_ = type_classifier(input_ids, mask_token_mask, return_loss=False)
            yes_scores = yes_scores.detach().cpu().tolist()
            for _data_id, _event_idx, _event_id, _score in zip(data_id, event_idx, event_id, yes_scores):
                predicted_items[_data_id]['tc_scores'][_event_idx][_event_id] = _score

    # summarize results
    for i, item in enumerate(predicted_items):
        predicted_items[i]['predicted_mentions'] = []
        for trigger_idx, trigger in enumerate(item['predicted_triggers']):
            tmp_event = {}
            offset = [item['token_offsets'][trigger[0]][0],item['token_offsets'][trigger[1]][1]]
            tmp_event['sentence_offset'] = offset
            tmp_event['trigger_words'] = item['sentence'][offset[0]:offset[1]]
            tmp_event['trigger_confidence'] = float(item['predicted_logits'][trigger_idx])
            sorted_scores = sorted(item['tc_scores'][trigger_idx].items(), key=lambda x:x[1], reverse=True)
            
            ###
            print('*'*10)
            print(tmp_event)
            for event_id, event_logit in sorted_scores:
                name, des, node = id2node[event_id]
                print({
                    'event_id': event_id,
                    'xpo_node': node,
                    'name': name,
                    'description': des,
                    'is_ldc_event': is_ldc_event[event_id],
                    'event_logit': event_logit
                })
            ###
            
            
            
            
            event_id, event_logit  = sorted_scores[0]
            if event_id == 1528:
                event_id = 94
            if event_id in [506, 669, 925, 1573, 2108, 2636, 2919, 4409, 4697]:
                event_id = 96
            name, des, node = id2node[event_id]
            tmp_event['event_type'] = {
                'xpo_node': node,
                'name': name,
                'description': des,
            }
            tmp_event['event_confidence'] = float(event_logit)
            predicted_items[i]['predicted_mentions'].append(tmp_event)
        del predicted_items[i]['tc_scores']
        del predicted_items[i]['top_10_events']
        del predicted_items[i]['predicted_logits']
        del predicted_items[i]['predicted_triggers']
        del predicted_items[i]['token_offsets']
        del predicted_items[i]['tokenized_tokens']
        del predicted_items[i]['index']
        del predicted_items[i]['tokenized_text_ids']
        del predicted_items[i]['original_tokenized_text_ids']

    return predicted_items

if __name__ == '__main__':
    # sample test 
    print('start')

    sen_list = ['One Air Force technician died and 21 others were injured .']
    print(predict(sen_list))






