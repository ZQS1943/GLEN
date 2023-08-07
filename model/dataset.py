from torch.utils.data.dataset import Dataset
from torch.nn.functional import softmax
import torch
from typing import List
import torch.nn.functional as F
from model.params import get_node_relation, xpo, node2id
import json
import copy
from tqdm import tqdm
import pickle as pkl

MAX_CAND_LEN=64
MAX_LEN=128
class LPDataset(Dataset):
    def __init__(self, dataset, params, k=10, embedding=False, data_truncate=None):
        self.dataset=dataset
        print(f'processing {len(dataset)} data')
        self.data = []
        self.embedding=embedding
        self.candidate_token_ids = torch.load(params["cand_token_ids_path"])
        if embedding:
            self.candidate_embedding = torch.load(params["cand_embedding_path"])
        self.node_relation = get_node_relation(xpo, node2id)
        for data_idx, item in tqdm(enumerate(dataset)):
            for idx,(mention, cand_nodes, true_node) in enumerate(zip(item['context']['mention_idxs'], item['label_idx'], item['true_label'])):
                if len(cand_nodes) <= 1:
                    continue
                if true_node not in cand_nodes:
                    print(f"data {item['data_id']} true label not in cand_nodes")
                    continue
                label = cand_nodes.index(true_node)
                graph, all_event_nodes = self.build_graph(cand_nodes)
                
                cur_sen = item['context']['original_input']
                cur_sen = cur_sen[:mention[0] - 1] + [5] + cur_sen[mention[0] - 1: mention[1]] + [5] + cur_sen[mention[1]:]
                
                new_input = []
                for l in all_event_nodes:
                    tmp_input = [101] + cur_sen + [4] + self.candidate_token_ids[l][1:]
                    tmp_input = tmp_input[:MAX_LEN]
                    tmp_input = tmp_input + [0] * (MAX_LEN - len(tmp_input))
                    new_input.append(tmp_input)

                # if len(all_event_nodes) >= 100:
                #     print(item, all_event_nodes)
                #     assert 1==0

                self.data.append({
                    'concatenated_input': torch.tensor(new_input),
                    'cur_nodes': cand_nodes,
                    'graph': graph,
                    'label': label,
                    'index': (data_idx, idx),
                })  
                # print(item, self.data) 
                # assert 1==0
        if data_truncate is not None:
            self.data = self.data[:data_truncate]
        print(f'got {len(self.data)} training data for label propagation')
        # assert 1==0

    def build_graph(self, cand_nodes:List[int]):
        """
        build a graph given a event mention:
            nodes: 
                events: top k nodes (must include ground-truth node) + their neighbors
                sen_event: each node maps to a node in top k nodes (must include ground-truth node)
            edges:
                relation between events: parent_child, sen_event_to_event
        """
        src_ids = []
        dst_ids = []
        
        all_event_nodes = set()        
        edge_list = []
        edge_type = []
        for node in cand_nodes:
            edge_list.append((node, node))
            edge_type.append(0)
            for child in self.node_relation[node]['child']:
                all_event_nodes.add(child)
                edge_list.append((child, node))
                edge_type.append(1)
            for parent in self.node_relation[node]['parents']:
                all_event_nodes.add(parent)
                edge_list.append((parent, node))
                edge_type.append(2)
        all_event_nodes = all_event_nodes.difference(set(cand_nodes))
        all_event_nodes = cand_nodes + list(all_event_nodes)
        
        node2id = {node:i for i, node in enumerate(all_event_nodes)}

        src_ids.extend([node2id[x[0]] for x in edge_list])
        dst_ids.extend([node2id[x[1]] for x in edge_list])

        graph_data = {('node', 'edge', 'node'):(torch.tensor(src_ids), torch.tensor(dst_ids))}

        g = dgl.heterograph(graph_data)
        g.ndata['is_cand'] = torch.tensor([True]*len(cand_nodes) + [False] * (len(all_event_nodes) - len(cand_nodes)))

        if self.embedding:
            g.ndata['embedding'] = self.candidate_embedding[all_event_nodes].cpu()
        else:
            g.ndata['tokenized_ids'] = torch.tensor([self.candidate_token_ids[x] for x in all_event_nodes])

        g.ndata['xpo_node'] = torch.tensor(all_event_nodes)

        g.edata['type_int'] = torch.tensor(edge_type)
        return g, all_event_nodes

    def get_label_probs(self, event_trigger_matcher):

        with open('./cache/scores_dict_all_dev.pickle', 'rb') as f:
            scores_dict = pkl.load(f)
        for idx in tqdm(range(len(self.data))):
            self.data[idx]['graph'].ndata['scores'] = scores_dict[self.data[idx]['index']]
        return 0

        print(f"get the label probabilities of {len(self.data)} data")
        device = event_trigger_matcher.device
        event_trigger_matcher.eval()
        scores_dict = {}
        for idx in tqdm(range(len(self.data))):
            item = self.data[idx]

            concatenated_input = item['concatenated_input']
            concatenated_input = concatenated_input.to(device)

            # # TODO: repeat step between different round of label propagation, can be improved
            # node_list = item['graph'].ndata['xpo_node']
            # is_sent = item['graph'].ndata['is_sent']
            # node_list = node_list[~is_sent]
            # node_embedding = self.candidate_embedding[node_list]

            concatenated_input = item['concatenated_input'].to(device)
            with torch.no_grad():
                _, scores = event_trigger_matcher(
                concatenated_input,
                return_loss=False
                )
                scores = scores.detach().cpu()
            self.data[idx]['graph'].ndata['scores'] = scores
            scores_dict[self.data[idx]['index']] = scores

        with open('./cache/scores_dict_all_dev.pickle', 'wb') as f:
            pkl.dump(scores_dict, f)
            # assert 1==0
            # embedding = torch.cat((node_embedding, embeddings),0)
            # self.data[idx]['graph'].ndata['embedding'] = embedding.to('cpu')
    
    def __getitem__(self, index):
        sample = self.data[index]
        return sample


    def __len__(self):
        return len(self.data)

class YNdataset(Dataset):
    def __init__(self, dataset):
        self.data=dataset

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
   

class NSdataset(Dataset):
    def __init__(self, dataset, params):
        self.dataset=dataset
        self.data = []
        self.candidate_token_ids = torch.load(params["cand_token_ids_path"])
        for data_idx, item in tqdm(enumerate(dataset)):
            for idx, mention in enumerate(item['predicted_triggers']):
                cur_sen = item['context']['ids']
                cur_sen = cur_sen[:mention[0]] + [5] + cur_sen[mention[0]: mention[1] + 1] + [5] + cur_sen[mention[1] + 1:]
                cur_sen = cur_sen[:MAX_LEN]

                trigger_tokens = [i for i in range(mention[0] + 1, mention[1] + 2)]
                
                
                # new_input = []
                # for l in cur_label:
                #     tmp_input = [101] + cur_sen + [4] + self.candidate_token_ids[l][1:]
                #     tmp_input = tmp_input[:MAX_LEN]
                #     tmp_input = tmp_input + [0] * (MAX_LEN - len(tmp_input))
                #     new_input.append(tmp_input)

                self.data.append({
                    'concatenated_input': cur_sen,
                    'index': (data_idx, idx),
                    'trigger_tokens': trigger_tokens
                })
       
    def __getitem__(self, index):
        sample = self.data[index]
        return sample


    def __len__(self):
        return len(self.data)

class ETMdataset(Dataset):
    def __init__(self, dataset, params,k=10,predict=False, has_true_label=True, use_true_label=False):
        self.dataset=dataset
        self.data = []
        self.candidate_token_ids = torch.load(params["cand_token_ids_path"])
        for data_idx, item in tqdm(enumerate(dataset)):
            if use_true_label:
                for idx,(mention, label) in enumerate(zip(item['context']['mention_idxs'], item['true_label'])):
                    cur_label = [label]
                    for l in item[f'top_100_events'][:k + 1]:
                        if l != label:
                            cur_label.append(l)
                    cur_label = cur_label[:k]

                    cur_sen = item['context']['original_input']
                    cur_sen = cur_sen[:mention[0] - 1] + [5] + cur_sen[mention[0] - 1: mention[1]] + [5] + cur_sen[mention[1]:]
                    
                    new_input = []
                    for l in cur_label:
                        tmp_input = [101] + cur_sen + [4] + self.candidate_token_ids[l][1:]
                        tmp_input = tmp_input[:MAX_LEN]
                        tmp_input = tmp_input + [0] * (MAX_LEN - len(tmp_input))
                        new_input.append(tmp_input)
                    new_label = 0

                    self.data.append({
                        'concatenated_input': new_input,
                        'label': new_label,
                        'index': (data_idx, idx),
                        'cur_label': cur_label
                    })
                continue

            if predict:
                for idx, mention in enumerate(item['predicted_triggers']):
                    cur_label = item[f'top_100_events']
                    cur_label = cur_label[:k]
                    
                    cur_sen = item['context']['original_input']
                    cur_sen = cur_sen[:mention[0] - 1] + [5] + cur_sen[mention[0] - 1: mention[1]] + [5] + cur_sen[mention[1]:]
                    
                    new_input = []
                    for l in cur_label:
                        tmp_input = [101] + cur_sen + [4] + self.candidate_token_ids[l][1:]
                        tmp_input = tmp_input[:MAX_LEN]
                        tmp_input = tmp_input + [0] * (MAX_LEN - len(tmp_input))
                        new_input.append(tmp_input)

                    self.data.append({
                        'concatenated_input': new_input,
                        'index': (data_idx, idx),
                        'cur_label': cur_label
                    })
            else:
                for idx,(mention, label) in enumerate(zip(item['context']['mention_idxs'], item['label_idx'])):
                    cur_label = copy.deepcopy(label)
                    for l in item[f'top_20_events']:
                        if l not in cur_label:
                            cur_label.append(l)
                    cur_label = cur_label[:k]

                    cur_sen = item['context']['original_input']
                    cur_sen = cur_sen[:mention[0] - 1] + [5] + cur_sen[mention[0] - 1: mention[1]] + [5] + cur_sen[mention[1]:]
                    
                    new_input = []
                    for l in cur_label:
                        tmp_input = [101] + cur_sen + [4] + self.candidate_token_ids[l][1:]
                        tmp_input = tmp_input[:MAX_LEN]
                        tmp_input = tmp_input + [0] * (MAX_LEN - len(tmp_input))
                        new_input.append(tmp_input)
                    new_label = list(range(len(label))) + [-1] * (k - len(label))

                    true_label_idx = 0
                    if has_true_label:
                        true_label = item['true_label'][idx]
                        true_label_idx = cur_label.index(true_label)
                    
                    self.data.append({
                        'concatenated_input': new_input,
                        'label': new_label,
                        'index': (data_idx, idx),
                        'true_label_idx': true_label_idx,
                        'cur_label': cur_label
                    })

    def __getitem__(self, index):
        sample = self.data[index]
        return sample


    def __len__(self):
        return len(self.data)


class TLdataset(Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
        self.data = []
        for idx, item in enumerate(dataset):
            for label, label_vec, mention in item['label_to_mention']:
                new_input = [101] + item['context']['original_input'] + [4] + label_vec[1:]
                new_input = new_input[:MAX_LEN]
                new_input = new_input + [0] * (MAX_LEN - len(new_input))
                self.data.append({
                    'concatenated_input': new_input,
                    'mention_label': mention,
                    'index': idx,
                    'label': label,
                })
            

    def __getitem__(self, index):
        sample = self.data[index]
        return sample


    def __len__(self):
        return len(self.data)



def padding_with_multiple_dim(data, pad_idx=-1, dtype=torch.long):
    tmp_data = data
    max_length = [1]
    max_length.append(max(len(x) for x in tmp_data))
    def continue_walking(tmp_data):
        for item in tmp_data:
            if len(item) and isinstance(item[0], list):
                return True
        return False
    while continue_walking(tmp_data):
        tmp_tmp_data = []
        for item in tmp_data:
            tmp_tmp_data.extend(item)
        tmp_data = tmp_tmp_data
        max_length.append(max(len(x) for x in tmp_data))
    # print(f"padding: {max_length}")

    def padding(data, max_length):
        padded_data = []
        padding_mask = []
        if len(max_length) > 1:
            for item in data:
                tmp_data, tmp_mask = padding(item, max_length[1:])
                padded_data.append(tmp_data)
                padding_mask.append(tmp_mask)
            for _ in range(max_length[0] - len(data)):
                tmp_data, tmp_mask = padding([], max_length[1:])
                padded_data.append(tmp_data)
                padding_mask.append(tmp_mask)
        else:
            if len(max_length) == 1 and pad_idx != -1:
                if len(data):
                    padded_data = data
                    padding_mask = [1]
                else:
                    padded_data = pad_idx
                    padding_mask = [0]
            else:
                padded_data = data + [pad_idx for _ in range(max_length[0] - len(data))]
                padding_mask = [1 for _ in data] + [0 for _ in range(max_length[0] - len(data))]       
        return padded_data, padding_mask
            
    padded_data, padding_mask = padding(data, max_length)
    padded_data = torch.tensor(padded_data, dtype=dtype)
    padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
    if pad_idx != -1:
        padding_mask = padding_mask.squeeze(-1)
    # print(padded_data.shape, padding_mask.shape)
    return padded_data, padding_mask

class UniEDdataset(Dataset):
    def __init__(self, dataset, only_sen_w_events=False, with_sent_tag=False):
        if only_sen_w_events:
            self.data=[item for item in dataset if len(item['context']['mention_idxs'])]
        else:
            self.data = dataset

        self.with_sent_tag = with_sent_tag

    def __getitem__(self, index):
        sample = self.data[index]
        if self.with_sent_tag:
            context_vecs = sample['context']['ids_with_sent_tag']
        else:
            context_vecs = sample['context']['ids']
        results = {
            'index': index,
            'context_no_pad': sample['context']['original_input'],
            'context_vecs': context_vecs,
            'cand_vecs': sample['label']['ids'], 
            'label_idx': sample['label_idx'], 
            'label_probability': sample['label_probability'], 
            # 'neighbor_tokens_vecs': None, 
            # 'relation_type_vecs': None, 
            # 'relation_type_mask': None, 
            'mention_idx_vecs': sample['context']['mention_idxs'], 
            'roleset_ids': sample['roleset_ids'],
            'true_label': sample['true_label'],
            'label_to_mention': sample['label_to_mention'],
            'original_input': sample['context']['original_input']
            # 'mention_idx_mask': None, 
        }
        return results


    def __len__(self):
        return len(self.data)

    def label_propagation_for_sentence(self, data_index, logits, alpha = 0.5,silent=True):
        for idx, index in enumerate(data_index):
            # print("*"*10)
            # for each mention in each case
            # print(self.data[index])
            for i, (l_id, l_probability, n_nodes, n_relations) in enumerate(zip(self.data[index]['label_idx'], self.data[index]['label_probability'], self.data[index]['neighbor_nodes'], self.data[index]['neighbor_relations'])):
                if len(l_id) == 1:
                    continue

                def get_new_prob(l,p,n,n_r):
                    if len(n):
                        result = logits[idx][l] + alpha / len(n) * sum(logits[idx][n_id] for n_id in n)
                        if not silent:
                            print(l,float(logits[idx][l]))
                            for n_id in n:
                                print('\t', n_id, float(logits[idx][n_id]))
                            print(result)
                        return result
                    return logits[idx][l]

                new_l_probability = torch.tensor([get_new_prob(l,p,n,n_r) for l,p,n,n_r in zip(l_id, l_probability, n_nodes,  n_relations)])
                
                # print(l_id, l_probability, n_nodes, n_relations)
                # print(new_l_probability)
                new_l_probability = softmax(new_l_probability)
                if not silent:
                    print(new_l_probability)
                # origin = len(self.data[index]['label_probability'][i])
                self.data[index]['label_probability'][i] = new_l_probability.tolist()

 
    
    def write_tmp_data(self):
        
        with open('./html_better_with_lp/sentences.json', 'w') as f:
            f.write(json.dumps(self.tmp_data, indent=True))
        assert 1==0
                    

                    


    def label_propagation(self, data_index, logits, alpha = 0.5,silent=True):
        mention_logits_p = 0
        for index in data_index:
            # print("*"*10)
            # for each mention in each case
            # print(self.data[index])
            for i, (l_id, l_probability, n_nodes, n_relations) in enumerate(zip(self.data[index]['label_idx'], self.data[index]['label_probability'], self.data[index]['neighbor_nodes'], self.data[index]['neighbor_relations'])):
                if len(l_id) == 1:
                    mention_logits_p += 1
                    continue

                def get_new_prob(l,p,n,n_r):
                    if len(n):
                        result = logits[mention_logits_p][l] + alpha / len(n) * sum(logits[mention_logits_p][n_id] for n_id in n)
                        if not silent:
                            print(l,float(logits[mention_logits_p][l]))
                            for n_id in n:
                                print('\t', n_id, float(logits[mention_logits_p][n_id]))
                            print(result)
                        return result
                    return logits[mention_logits_p][l]

                new_l_probability = torch.tensor([get_new_prob(l,p,n,n_r) for l,p,n,n_r in zip(l_id, l_probability, n_nodes, n_relations)])
                
                # print(l_id, l_probability, n_nodes, n_relations)
                # print(new_l_probability)
                new_l_probability = softmax(new_l_probability)
                if not silent:
                    print(new_l_probability)
                # origin = len(self.data[index]['label_probability'][i])
                self.data[index]['label_probability'][i] = new_l_probability.tolist()
                # after = len(self.data[index]['label_probability'][i])
                # if origin != after:
                #     print(origin, after)
                #     print(l_probability)
                #     print(new_l_probability)
                #     assert 1==0
                mention_logits_p += 1

def collate_fn_uni_ED_new_type(batch):
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    context_vecs = torch.tensor([x['context_vecs'] for x in batch], dtype=torch.long)

    event_type_vecs = []
    event_indexer = {}
    num_mentions = 0
    for x in batch:
        num_mentions += len(x['label_idx'])
        for men_idx, mention in enumerate(x['label_idx']):
            for et_idx, et in enumerate(mention):
                if et not in event_indexer:
                    event_indexer[et] = len(event_type_vecs)
                    event_type_vecs.append(x['cand_vecs'][men_idx][et_idx])

    event_type_vecs = torch.tensor(event_type_vecs, dtype=torch.long)

    label = torch.zeros(num_mentions, len(event_type_vecs))
    cur_mention = 0
    for sen_id, x in enumerate(batch):
        for label_idx, label_p in zip(x['label_idx'], x['label_probability']):
            for l_idx, l_p in zip(label_idx, label_p):
                label[cur_mention][event_indexer[l_idx]] = l_p
            cur_mention += 1
    
    # print(f"label.shape: {label.shape}")

    mention_idx_vecs, mention_idx_mask = padding_with_multiple_dim([x['mention_idx_vecs'] for x in batch], pad_idx = [0,1])

    true_label = None
    if batch[0]['true_label'] is not None:
        true_label, _ = padding_with_multiple_dim([x['true_label'] for x in batch])
    
    # cand_vecs, cand_mask = padding_with_multiple_dim([x['cand_vecs'] for x in batch], pad_idx = [0 for _ in range(MAX_CAND_LEN)])
    # label_idx, label_idx_mask = padding_with_multiple_dim([x['label_idx'] for x in batch])
    roleset_ids, roleset_mask = padding_with_multiple_dim([x['roleset_ids'] for x in batch])
    # label_probability, _ = padding_with_multiple_dim([x['label_probability'] for x in batch], dtype=torch.float)
    # true_label = None
    # return (context_vecs, cand_vecs, label_idx, cand_mask, label_probability, index, roleset_ids, roleset_mask, mention_idx_vecs, mention_idx_mask)
    return (index, context_vecs, event_type_vecs, event_indexer, label, true_label, roleset_ids, mention_idx_vecs, mention_idx_mask)

def collate_fn_YN_Train(batch):
    data_id = [x['id'] for x in batch]
    event_idx = [x['event_idx'] for x in batch]
    event_id = [x['event_id'] for x in batch]
    input_ids = torch.tensor([x['input_ids'] for x in batch], dtype=torch.long)
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
    mask_token_mask = torch.tensor([x['mask_token_mask'] for x in batch], dtype=torch.bool)
    return data_id,event_idx,event_id,input_ids,labels, mask_token_mask


def collate_fn_YN_new_loss(batch):
    return batch[0]

def collate_fn_YN(batch):
    data_id = [x['id'] for x in batch]
    event_idx = [x['event_idx'] for x in batch]
    event_id = [x['event_id'] for x in batch]
    input_ids = torch.tensor([x['input_ids'] for x in batch], dtype=torch.long)
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.float)
    mask_token_mask = torch.tensor([x['mask_token_mask'] for x in batch], dtype=torch.bool)
    return data_id,event_idx,event_id,input_ids,labels, mask_token_mask

def collate_fn_uni_ED(batch):
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    context_vecs = torch.tensor([x['context_vecs'] for x in batch], dtype=torch.long)
    mention_idx_vecs, mention_idx_mask = padding_with_multiple_dim([x['mention_idx_vecs'] for x in batch], pad_idx = [0,1])
    cand_vecs, cand_mask = padding_with_multiple_dim([x['cand_vecs'] for x in batch], pad_idx = [0 for _ in range(MAX_CAND_LEN)])
    label_idx, label_idx_mask = padding_with_multiple_dim([x['label_idx'] for x in batch])
    roleset_ids, roleset_mask = padding_with_multiple_dim([x['roleset_ids'] for x in batch])
    label_probability, _ = padding_with_multiple_dim([x['label_probability'] for x in batch], dtype=torch.float)
    true_label = None
    return (context_vecs, cand_vecs, label_idx, cand_mask, label_probability, index, roleset_ids, roleset_mask, mention_idx_vecs, mention_idx_mask)
    # true_label, _ = padding_with_multiple_dim([x['true_label'] for x in batch])
    # return (context_vecs, cand_vecs, label_idx, cand_mask, label_probability, index, roleset_ids, roleset_mask, true_label, mention_idx_vecs, mention_idx_mask)

def collate_fn_WP(batch):
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    context_vecs = torch.tensor([x['context_vecs'] for x in batch], dtype=torch.long)

    event_type_vecs = []
    event_indexer = {}
    margin_label = []
    for x in batch:
        tmp_label = set()
        for men_idx, mention in enumerate(x['label_idx']):
            for et_idx, et in enumerate(mention):
                if et not in event_indexer:
                    event_indexer[et] = len(event_type_vecs)
                    event_type_vecs.append(x['cand_vecs'][men_idx][et_idx])
                tmp_label.add(event_indexer[et])
        margin_label.append(list(tmp_label))

    event_type_vecs = torch.tensor(event_type_vecs, dtype=torch.long)

    label = torch.zeros(len(context_vecs), len(event_type_vecs))
    for sen_id, x in enumerate(batch):
        for label_idx, label_p in zip(x['label_idx'], x['label_probability']):
            for l_idx, l_p in zip(label_idx, label_p):
                label[sen_id][event_indexer[l_idx]] = l_p

    margin_label, margin_label_mask = padding_with_multiple_dim(margin_label)
    margin_label = F.pad(margin_label, (0,len(event_indexer) - margin_label.size(1)), "constant", -1)
    # print(batch)
    # print(context_vecs, context_vecs.shape)
    # print(event_type_vecs, event_type_vecs.shape)
    # print(label, label.shape)
    # print(margin_label, margin_label.shape)
    # assert 1==0
    true_label = [x['true_label'] for x in batch]
    original_input = [x['original_input'] for x in batch]

    mention_label = [x['mention_idx_vecs'] for x in batch]
    return (context_vecs, event_type_vecs, label, index, event_indexer, margin_label, true_label, original_input, mention_label)


def collate_fn_TR(batch):
    if len(batch) == 1:
        return (None, None, None, None, None, None)
    # print(batch)
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    context_vecs = torch.tensor([x['context_vecs'] for x in batch], dtype=torch.long)

    event_type_vecs = []
    event_indexer = {}
    candidate_label_sets = []
    for x in batch:
        tmp_candidate_sets = []
        for men_idx, mention in enumerate(x['label_idx']):
            tmp_candidate_set = []
            for et_idx, et in enumerate(mention):
                if et not in event_indexer:
                    event_indexer[et] = len(event_type_vecs)
                    event_type_vecs.append(x['cand_vecs'][men_idx][et_idx])
                tmp_candidate_set.append(event_indexer[et])
            tmp_candidate_sets.append(tuple(tmp_candidate_set))
        tmp_candidate_sets = set(tmp_candidate_sets)
        candidate_label_sets.append(list(tmp_candidate_sets))

    negative_smaples = []
    for candidate_sets in candidate_label_sets:
        tmp_nega = [i for i in range(len(event_type_vecs)) if not any(i in cand_set for cand_set in candidate_sets)]
        negative_smaples.append(tmp_nega)
        
    event_type_vecs = torch.tensor(event_type_vecs, dtype=torch.long)

    def convert_to_mask(cand_set):
        cand_mask = torch.tensor([False]*len(event_indexer), dtype=torch.bool)
        # print(cand_set)
        cand_mask[torch.tensor(cand_set)] = True
        return cand_mask
    candidate_label_sets = [[convert_to_mask(cand_set) for cand_set in cand_sets] for cand_sets in candidate_label_sets]

    negative_smaples = [convert_to_mask(tmp) for tmp in negative_smaples]

    return (context_vecs, event_type_vecs, index, event_indexer, candidate_label_sets, negative_smaples)



def collate_fn_TC(batch):
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    context_vecs = torch.tensor([x['context_vecs'] for x in batch], dtype=torch.long)

    event_type_vecs = []
    event_indexer = {}
    margin_label = []
    for x in batch:
        tmp_label = set()
        for men_idx, mention in enumerate(x['label_idx']):
            for et_idx, et in enumerate(mention):
                if et not in event_indexer:
                    event_indexer[et] = len(event_type_vecs)
                    event_type_vecs.append(x['cand_vecs'][men_idx][et_idx])
                tmp_label.add(event_indexer[et])
        margin_label.append(list(tmp_label))

    event_type_vecs = torch.tensor(event_type_vecs, dtype=torch.long)

    label = torch.zeros(len(context_vecs), len(event_type_vecs))
    for sen_id, x in enumerate(batch):
        for label_idx, label_p in zip(x['label_idx'], x['label_probability']):
            for l_idx, l_p in zip(label_idx, label_p):
                label[sen_id][event_indexer[l_idx]] = l_p

    margin_label, margin_label_mask = padding_with_multiple_dim(margin_label)
    margin_label = F.pad(margin_label, (0,len(event_indexer) - margin_label.size(1)), "constant", -1)
    # print(batch)
    # print(context_vecs, context_vecs.shape)
    # print(event_type_vecs, event_type_vecs.shape)
    # print(label, label.shape)
    # print(margin_label, margin_label.shape)
    # assert 1==0
    true_label = [x['true_label'] for x in batch]
    return (context_vecs, event_type_vecs, label, index, event_indexer, margin_label, true_label)

def collate_fn_TL(batch):
    # print(batch)
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    label = torch.tensor([x['label'] for x in batch], dtype=torch.long)
    concatenated_input = torch.tensor([x['concatenated_input'] for x in batch], dtype=torch.long)

    mention_label, mention_label_mask = padding_with_multiple_dim([x['mention_label'] for x in batch], pad_idx = [0,1])
    # print(concatenated_input, mention_label, mention_label_mask, index)
    # assert 1==0

    return (concatenated_input, mention_label, mention_label_mask, index, label)

    

def collate_fn_TD(batch):
    # print(batch)
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    context_input = torch.tensor([x['context_vecs'] for x in batch], dtype=torch.long)

    mention_label, mention_label_mask = padding_with_multiple_dim([x['mention_idx_vecs'] for x in batch], pad_idx = [0,1])


    return (context_input, mention_label, mention_label_mask, index)

def collate_fn_NS(batch):
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    concatenated_input = torch.tensor([x['concatenated_input'] for x in batch], dtype=torch.long)
    trigger_tokens, trigger_tokens_mask = padding_with_multiple_dim([x['trigger_tokens'] for x in batch])
    return index, concatenated_input, trigger_tokens, trigger_tokens_mask
    

def collate_fn_ETM(batch):
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    
    
    concatenated_input = torch.tensor([x['concatenated_input'] for x in batch], dtype=torch.long)
    cur_label = torch.tensor([x['cur_label'] for x in batch], dtype=torch.long)
    label = None
    if 'label' in batch[0]:
        label = torch.tensor([x['label'] for x in batch], dtype=torch.long)
    true_label_idx = None
    if 'true_label_idx' in batch[0]:
        true_label_idx = torch.tensor([x['true_label_idx'] for x in batch], dtype=torch.long)
    return (concatenated_input, label, index, true_label_idx, cur_label)

class Canddataset(Dataset):
    def __init__(self, dataset):
        self.data=dataset

    def __getitem__(self, index):
        sample = self.data[index]
        results = {
            'id': sample[0],
            'cand_vecs': sample[1]
        }
        return results

    def __len__(self):
        return len(self.data)

def collate_fn_cand(batch):
    context_vecs = torch.tensor([x['cand_vecs'] for x in batch], dtype=torch.long)
    return context_vecs