from transformers import BertTokenizerFast

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from model.params import parse_arguments
from model.data_process import process_data_TC_predict
from model.utils import encode_all_candidates, get_predicted_mention_bounds
from model.constants import id2node_detail
from model.encoder import TriggerIdentifier, TypeRanking, TypeClassifier
from model.dataset import SimpleDataset, collate_fn_TI_sen, collate_fn_TR_sen, collate_fn_TC_sen


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



def predict(sentence_list, params):
    tokenizer = BertTokenizerFast.from_pretrained(params['bert_model'])
    predict_samples = sentence_tokenization(sentence_list, tokenizer, max_length=512)
    device = "cuda"

    # trigger identification
    predict_set = SimpleDataset(predict_samples)
    TI_dataloader = DataLoader(predict_set, batch_size=params['bs_TI'], shuffle=False, collate_fn=collate_fn_TI_sen)
    trigger_identifier = TriggerIdentifier(params).to(device).eval()

    with torch.no_grad():
        for batch in tqdm(TI_dataloader, desc="Trigger Identificaiton"):
            context_input, data_index = batch
            context_input = context_input.to(device)
            
            _, mention_logits, mention_bounds  = trigger_identifier(
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
    type_ranking = TypeRanking(params).to(device).eval()
    TR_dataloader = DataLoader(predict_set, batch_size=params['bs_TR'], shuffle=False, collate_fn=collate_fn_TR_sen)
    cand_encs, used_cand = encode_all_candidates(params, type_ranking, device, only_used_xpo=True)
    k = params['k']
    with torch.no_grad():
        for batch in tqdm(TR_dataloader, desc='Type Ranking'):
            context_vecs, data_index = batch
            context_vecs = context_vecs.to(device)
            _, logits = type_ranking(
                context_vecs, None,
                cand_encs=cand_encs,
            )
            
            top_k = torch.topk(logits, k, dim=1)
            converted_indices = used_cand[top_k.indices]
            for idx, indices in zip(data_index, converted_indices):
                predict_set.data[idx][f'top_events'] = indices.tolist()

    # type classification
    predicted_items = predict_set.data
    for i in range(len(predicted_items)):
        predicted_items[i]['TC_scores'] = [{} for _ in predicted_items[i]['predicted_triggers']]
    type_classifier = TypeClassifier(params, device).to(device).eval()
    processed_samples = process_data_TC_predict(params, predicted_items, tokenizer, eval_on_gold=False)
    predict_set = SimpleDataset(processed_samples)
    tc_dataloder = DataLoader(predict_set, batch_size=params['bs_TC'], shuffle=False, collate_fn=collate_fn_TC_sen)
    with torch.no_grad():
        for batch in tqdm(tc_dataloder, desc='Type Classification'):
            data_id,event_idx,event_id,input_ids, mask_token_mask = batch
            input_ids = input_ids.to(device)
            mask_token_mask = mask_token_mask.to(device)
            yes_scores,_ = type_classifier(input_ids, mask_token_mask, return_loss=False)
            yes_scores = yes_scores.detach().cpu().tolist()
            for _data_id, _event_idx, _event_id, _score in zip(data_id, event_idx, event_id, yes_scores):
                predicted_items[_data_id]['TC_scores'][_event_idx][_event_id] = _score

    # summarize results
    for i, item in enumerate(predicted_items):
        predicted_items[i]['predicted_mentions'] = []
        for trigger_idx, trigger in enumerate(item['predicted_triggers']):
            tmp_event = {}
            offset = [item['token_offsets'][trigger[0]][0],item['token_offsets'][trigger[1]][1]]
            tmp_event['sentence_offset'] = offset
            tmp_event['trigger_words'] = item['sentence'][offset[0]:offset[1]]
            tmp_event['trigger_confidence'] = float(item['predicted_logits'][trigger_idx])
            sorted_scores = sorted(item['TC_scores'][trigger_idx].items(), key=lambda x:x[1], reverse=True)

            event_id, event_logit  = sorted_scores[0]
            name, des, node = id2node_detail[event_id]
            tmp_event['event_type'] = {
                'xpo_node': node,
                'name': name,
                'description': des,
            }
            tmp_event['event_confidence'] = float(event_logit)
            predicted_items[i]['predicted_mentions'].append(tmp_event)
        del predicted_items[i]['TC_scores']
        del predicted_items[i]['top_events']
        del predicted_items[i]['predicted_logits']
        del predicted_items[i]['predicted_triggers']
        del predicted_items[i]['token_offsets']
        del predicted_items[i]['tokenized_tokens']
        del predicted_items[i]['index']
        del predicted_items[i]['tokenized_text_ids']
        del predicted_items[i]['original_tokenized_text_ids']

    return predicted_items

if __name__ == '__main__':
    params = parse_arguments()
    sen_list = ['One Air Force technician died and 21 others were injured .']
    print(f"start predicting {len(sen_list)} sentence")
    print(predict(sen_list, params))






