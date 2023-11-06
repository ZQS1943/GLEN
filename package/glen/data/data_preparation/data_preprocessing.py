from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizerFast
import json
from tqdm import tqdm
import torch
from collections import defaultdict
import os
import pkg_resources

EVENT_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"

def read_xpo():
    resource_path = 'resources/xpo_glen.json'
    resource_package = 'glen'
    resource_stream = pkg_resources.resource_stream(resource_package, resource_path)


    with resource_stream as f:
        xpo = json.load(f)
    node2id = {x:i + 1 for i, x in enumerate(xpo)}
    node2id['other'] = 0
    id2node = {node2id[k]:k for k in node2id}

    id2node_detail = {i + 1:(xpo[x]['name'], xpo[x]['wd_description'], x) for i, x in enumerate(xpo)}
    id2node[0] = ('others', 'other event types', 'other')

    roleset_list = []
    for node in xpo:
        for roleset in xpo[node]['pb_roleset']:
            if roleset not in roleset_list:
                roleset_list.append(roleset)
    roleset2id = {x:i + 1 for i, x in enumerate(roleset_list)}
    roleset2id['other'] = 0
    id2roleset = {roleset2id[k]:k for k in roleset2id}

    xpo_used = [0]
    for node in xpo:
        if 'removing_reason' in xpo[node] and len(xpo[node]['removing_reason']):
            xpo_used.append(0)
            continue
        if len(xpo[node]['pb_roleset']):
            xpo_used.append(1)
        else:
            xpo_used.append(0)

    return xpo, xpo_used, node2id, id2node, roleset2id, id2roleset, id2node_detail

def process_one_data(item, mapping_dict, xpo_nodes, tokenizer, node2id = None, labeled=False, no_other=False):
    data =  {
        'id': item['id'],
        'text': item['sentence'],
        'triggers': ['']
    }

    if no_other:
        no_other_events = []
        for event in item['events']:
            if event['pb_roleset'] in mapping_dict:
                no_other_events.append(event)
        item['events'] = no_other_events


    def convert_tokenoff2charoff(event,sen):
        tokens = sen.split(' ')
        start = sum(len(token) + 1 for token in tokens[:event['offset'][0]])
        end = sum(len(token) + 1 for token in tokens[:event['offset'][-1] + 1]) - 1
        return [start, end]
    data['triggers'] = [' '.join(x['trigger']) for x in item['events']]
    data['mentions'] = [convert_tokenoff2charoff(x, item['sentence']) for x in item['events']]

    def get_tokenized_trigger_idxs(trigger_offset, token_offsets):

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

    tokens = tokenizer(item['sentence'], return_offsets_mapping=True)
    
    data['tokenized_text_ids'] = tokens['input_ids'][1:-1]
    data['tokenized_trigger_idxs'] = [get_tokenized_trigger_idxs(trigger, tokens['offset_mapping'][1:-1]) for trigger in data['mentions']]

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
            return node2id[event['xpo_label']]
        elif len(mapping_dict[event['pb_roleset']]) == 1:
            return node2id[mapping_dict[event['pb_roleset']][0]['node_code']]
        else:
            return 0

    
    
    data['label_ids'] = [get_nodes_info(event['pb_roleset'], 'node_id') for event in item['events']]    
    
    data['xpo_ids'] = [get_nodes_info(event['pb_roleset'], 'node_code') for event in item['events']]
    data['xpo_titles'] = [get_nodes_info(event['pb_roleset'], 'node_name') for event in item['events']]
    data['label'] = [get_nodes_info(event['pb_roleset'], 'node_description') for event in item['events']]
    data['rolesets'] = [event['pb_roleset'] for event in item['events']]

    if labeled:
        data['true_label'] = [get_true_label(event) for event in item['events']] 
        
    return data

def preprocess(data, mapping_dict, xpo_nodes, node2id = None, labeled = False, no_other = False):
    processed_data = []
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    for item in tqdm(data):
        processed_data.append(process_one_data(item, mapping_dict, xpo_nodes, tokenizer, node2id = node2id, labeled = labeled, no_other = no_other))
    return processed_data


def get_node_tokenized_ids(xpo, max_seq_length=64, with_event_tag = False, output_file = None):
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
    
    torch.save(cand_node_tokenized_ids, output_file)

    return 0

def get_node_id(node2id):
    roleset2nodes = defaultdict(set)
    for node in xpo:
        for roleset in xpo[node]['pb_roleset']:
            roleset2nodes[roleset].add(node)

    roleset2nodes_detail = {}
    for roleset in roleset2nodes:
        nodes_list = []
        for node in roleset2nodes[roleset]:
            nodes_list.append({
                'node_code': node,
                'node_name': xpo[node]['name'],
                'node_id': node2id[node],
                'node_description': xpo[node]['wd_description']
            })
        roleset2nodes_detail[roleset] = nodes_list
    return roleset2nodes_detail

if __name__=='__main__':
    print("Data preprocessing...")

    # read ontology
    xpo, xpo_used, node2id, id2node, roleset2id, id2roleset, id2node_detail = read_xpo()

    if not os.path.exists("./data/data_preprocessed"):
        os.makedirs("./data/data_preprocessed")
        print("Created: data_preprocessed")

    # get dict: roleset -> xpo node list
    roleset2nodes_detail = get_node_id(node2id) 

    # get tokenized ids for each even node
    get_node_tokenized_ids(xpo, with_event_tag=False, output_file = './data/data_preprocessed/node_tokenized_ids_<max_seq_length>.pt')
    get_node_tokenized_ids(xpo, with_event_tag=True, output_file='./data/data_preprocessed/node_tokenized_ids_<max_seq_length>_with_event_tag.pt')
    print("Toeknized: event nodes")

    # preprocess dev data
    print("Processing: dev data...")
    with open(f'./data/data_split/dev_annotated.json', 'r') as f:
        data = json.load(f)
    processed_data = preprocess(data, roleset2nodes_detail, xpo, node2id = node2id, labeled = True, no_other = True)
    with open(f'./data/data_preprocessed/dev_annotated.jsonl', 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')

    # preprocess test data
    print("Processing: test data...")
    with open(f'./data/data_split/test_annotated.json', 'r') as f:
        data = json.load(f)
    processed_data = preprocess(data, roleset2nodes_detail, xpo, node2id = node2id, labeled = True, no_other = True)
    with open(f'./data/data_preprocessed/test_annotated.jsonl', 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    # preprocess train data
    print("Processing: train data...")
    with open(f'./data/data_split/train.json', 'r') as f:
        data = json.load(f)
    processed_data = preprocess(data, roleset2nodes_detail, xpo, node2id = node2id, labeled = False, no_other = True)
    with open(f'./data/data_preprocessed/train.jsonl', 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    