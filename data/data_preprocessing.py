from math import floor
from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizerFast
import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
from collections import defaultdict
import random
from utils import node_to_id, xpo, mapping_dict_detail
random.seed(42)
# from ..model.params import ENT_TITLE_TAG, EVENT_TAG

EVENT_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"

def process_one_data(item, mapping_dict, xpo_nodes, tokenizer1, tokenizer2, relation_dict = None, labeled=False, no_other=False):
    data =  {
        'id': item['id'],
        'text': item['sentence'],
        'triggers': ['']
    }
    # if item['id'] == 'ontonotes/bn/voa/00/voa_0022_3':
    #     print(item)

    if no_other:
        no_other_events = []
        for event in item['events']:
            if event['pb_roleset'] in mapping_dict:
                no_other_events.append(event)
        item['events'] = no_other_events

    # if item['id'] == 'ontonotes/bn/voa/00/voa_0022_3':
    #     print(item)

    def convert_tokenoff2charoff(event,sen):
        # print(event)
        # print(sen)
        # assert 1==0
        tokens = sen.split(' ')
        # assert tokens[event['offset'][0]] == event['trigger'][0]
        # only implement single word
        start = sum(len(token) + 1 for token in tokens[:event['offset'][0]])
        end = sum(len(token) + 1 for token in tokens[:event['offset'][-1] + 1]) - 1
        # print(sen[start: end], event['trigger'], sep=')')
        return [start, end]
    data['triggers'] = [' '.join(x['trigger']) for x in item['events']]
    data['mentions'] = [convert_tokenoff2charoff(x, item['sentence']) for x in item['events']]

    def get_tokenized_trigger_idxs(trigger_offset, token_offsets):
        # print(trigger_offset, token_offsets)
        # assert 1==0
        start = 0
        end = 0
        for i, token in enumerate(token_offsets):
            if token[0] <= trigger_offset[0]:
                start = i
            if token[1] > trigger_offset[1]:
                end = i
                break
        else:
            end = len(token_offsets)
        return [start, end]


    # tokens1 = tokenizer1.tokenize(item['sentence'])
    # token_ids1 = tokenizer1.convert_tokens_to_ids(tokens1)
    tokens2 = tokenizer2(item['sentence'], return_offsets_mapping=True)
    # print(token_ids1, tokens2)
    # assert 1==0
    
    data['tokenized_text_ids'] = tokens2['input_ids'][1:-1]
    data['tokenized_trigger_idxs'] = [get_tokenized_trigger_idxs(trigger, tokens2['offset_mapping'][1:-1]) for trigger in data['mentions']]

    def get_nodes_info(roleset, key):
        if roleset in mapping_dict:
            return [node[key] for node in mapping_dict[roleset]]
        else:
            # other event types
            if no_other:
                raise ValueError('no other type')
            if key == 'node_id':
                return [0]
            elif key == 'node_code':
                return ['']
            elif key == 'node_name':
                return ['others']
            elif key == 'node_description':
                return ['other event types']

    def get_true_label(event):
        if 'xpo_label' in event:
            return node_to_id[event['xpo_label']]
        elif len(mapping_dict[event['pb_roleset']]) == 1:
            return node_to_id[mapping_dict[event['pb_roleset']][0]['node_code']]
        else:
            return 0

    
    
    data['label_ids'] = [get_nodes_info(event['pb_roleset'], 'node_id') for event in item['events']]    
    
    data['xpo_ids'] = [get_nodes_info(event['pb_roleset'], 'node_code') for event in item['events']]
    data['xpo_titles'] = [get_nodes_info(event['pb_roleset'], 'node_name') for event in item['events']]
    data['label'] = [get_nodes_info(event['pb_roleset'], 'node_description') for event in item['events']]
    data['rolesets'] = [event['pb_roleset'] for event in item['events']]

    if labeled:
        data['true_label'] = [get_true_label(event) for event in item['events']] 
        

    data['neighbor_nodes'] = [[list(relation_dict[label]) for label in _ if relation_dict] for _ in data['label_ids'] ]

    # if data['id'] == 'ontonotes/bn/voa/00/voa_0022_3':
    #     print(data)
    #     assert 1==0

    return data

def preprocess(data, mapping_dict, xpo_nodes, relation_dict = None, labeled = False, no_other = False):
    processed_data = []
    tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer2 = BertTokenizerFast.from_pretrained('bert-base-uncased')
    for item in tqdm(data):
        processed_data.append(process_one_data(item, mapping_dict, xpo_nodes, tokenizer1, tokenizer2, relation_dict = relation_dict, labeled = labeled, no_other = no_other))
    return processed_data


def split_and_save(data):
    random.shuffle(data)
    def write_data(name, dataset):
        with open(f'./data/tokenized/{name}.jsonl', 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
    l = len(data)
    write_data('train', data[:floor(l*0.8)])
    write_data('test', data[floor(l*0.8):floor(l*0.9)])
    write_data('dev', data[floor(l*0.9):])


def get_node_tokenized_ids(xpo, max_seq_length=64, with_event_tag=False, output_file = './data/node_tokenized_ids_<max_seq_length>.pt'):
    output_file = output_file.replace('<max_seq_length>', str(max_seq_length))
    # get tokenized ids for the candidate xpo nodes
    cand_node_tokenized_ids = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    max_l = 0

    title_tokens = tokenizer.tokenize('others')
    cand_tokens = tokenizer.tokenize('other event types')
    cand_tokens = [cls_token] + title_tokens + [ENT_TITLE_TAG] + cand_tokens + [sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length
    cand_node_tokenized_ids[0] = input_ids

    for i, x in tqdm(enumerate(xpo)):
        if 'wd_description' in xpo[x]:
            cand_tokens = tokenizer.tokenize(xpo[x]['wd_description'])
            # print(cand_tokens)
        else:
            cand_tokens = []
            print(f"{x} no description")
        title_tokens = tokenizer.tokenize(xpo[x]['name'])
        if with_event_tag:
            cand_tokens = [EVENT_TAG] + title_tokens + [ENT_TITLE_TAG] + cand_tokens 
        else:
            cand_tokens = title_tokens + [ENT_TITLE_TAG] + cand_tokens
        max_l = max(max_l, len(cand_tokens))

        cand_tokens = cand_tokens[: max_seq_length - 2]
        cand_tokens = [cls_token] + cand_tokens + [sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length

        cand_node_tokenized_ids[i + 1] = input_ids

    
        # print(cand_tokens)
        # print(input_ids)
    # print(max_l)
    # print(cand_node_tokenized_ids)
    
    torch.save(cand_node_tokenized_ids, output_file)

    return 0
    
def get_node_relations(xpo):
    wd_node_2_xpo_node = {xpo[node]['wd_node']:node for node in xpo}
    node_2_node_id = {node:i+1 for i, node in enumerate(xpo)}
    # print(wd_node_2_xpo_node)

    relation_dict = defaultdict(set)

    for node in xpo:
        relation_list = []
        # print(xpo[node])
        
        # assert 1==0
        if 'overlay_parents' in xpo[node]:
            for parent in xpo[node]['overlay_parents']:
                if parent['wd_node'] not in wd_node_2_xpo_node:
                    # print(f"{parent['wd_node']} not in xpo nodes")
                    continue
                c = node_2_node_id[node]
                p = node_2_node_id[wd_node_2_xpo_node[parent['wd_node']]]
                relation_dict[c].add((p, 'parent'))
                relation_dict[p].add((c, 'child'))
        if 'similar_nodes' in xpo[node]:
            for similar_node in xpo[node]['similar_nodes']:
                if similar_node['wd_node'] not in wd_node_2_xpo_node:
                    # print(f"{similar_node['wd_node']} not in xpo nodes")
                    continue
                n1 = node_2_node_id[node]
                n2 = node_2_node_id[wd_node_2_xpo_node[similar_node['wd_node']]]
                relation_dict[n1].add((n2, 'similar'))
                relation_dict[n2].add((n1, 'similar'))
        
    return relation_dict


if __name__=='__main__':

    get_node_tokenized_ids(xpo)
    get_node_tokenized_ids(xpo, with_event_tag=True, output_file='./data/node_tokenized_ids_<max_seq_length>_with_event_tag.pt')

    relation_dict = get_node_relations(xpo)


    # include other type
    # for stage in ['train', 'dev', 'test']:
    #     with open(f'./data/data_split/{stage}.json', 'r') as f:
    #         data = json.load(f)
    #     processed_data = preprocess(data, mapping_dict_detail, xpo, relation_dict=relation_dict, labeled = False, no_other = False)
    #     with open(f'./data/tokenized_final/{stage}.jsonl', 'w') as f:
    #         for item in processed_data:
    #             f.write(json.dumps(item) + '\n')

    # for stage in ['annotated_dev_set', 'annotated_test_set']:
    #     with open(f'./data/data_split/{stage}.json', 'r') as f:
    #         data = json.load(f)
    #     processed_data = preprocess(data, mapping_dict_detail, xpo, relation_dict=relation_dict, labeled = True, no_other = False)
    #     with open(f'./data/tokenized_final/{stage}.jsonl', 'w') as f:
    #         for item in processed_data:
    #             f.write(json.dumps(item) + '\n')

    # exclude other type

    for stage in ['annotated_test_set', 'annotated_dev_set']:
        with open(f'./data/data_split/{stage}.json', 'r') as f:
            data = json.load(f)
        processed_data = preprocess(data, mapping_dict_detail, xpo, relation_dict=relation_dict, labeled = True, no_other = True)
        with open(f'./data/tokenized_final_no_other/{stage}.jsonl', 'w') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')

    for stage in ['train', 'dev', 'test']:
        with open(f'./data/data_split/{stage}.json', 'r') as f:
            data = json.load(f)
        processed_data = preprocess(data, mapping_dict_detail, xpo, relation_dict=relation_dict, labeled = False, no_other = True)
        with open(f'./data/tokenized_final_no_other/{stage}.jsonl', 'w') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')

    

    