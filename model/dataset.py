from torch.utils.data.dataset import Dataset
import torch
import torch.nn.functional as F
import json

from model.utils import padding_with_multiple_dim

class TCdataset(Dataset):
    def __init__(self, dataset):
        self.data=dataset

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TITRdataset(Dataset):
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
            'mention_idx_vecs': sample['context']['mention_idxs'], 
            'roleset_ids': sample['roleset_ids'],
            'true_label': sample['true_label'],
            'label_to_mention': sample['label_to_mention'],
            'original_input': sample['context']['original_input']
        }
        return results


    def __len__(self):
        return len(self.data)     


def collate_fn_TI(batch):
    index = torch.tensor([x['index'] for x in batch], dtype=torch.long)
    context_input = torch.tensor([x['context_vecs'] for x in batch], dtype=torch.long)

    mention_label, mention_label_mask = padding_with_multiple_dim([x['mention_idx_vecs'] for x in batch], pad_idx = [0,1])

    return (context_input, mention_label, mention_label_mask, index)

def collate_fn_TR_Train(batch):
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
    for x in batch:
        tmp_label = set()
        for men_idx, mention in enumerate(x['label_idx']):
            for et_idx, et in enumerate(mention):
                if et not in event_indexer:
                    event_indexer[et] = len(event_type_vecs)
                    event_type_vecs.append(x['cand_vecs'][men_idx][et_idx])
                tmp_label.add(event_indexer[et])

    event_type_vecs = torch.tensor(event_type_vecs, dtype=torch.long)

    label = torch.zeros(len(context_vecs), len(event_type_vecs))
    for sen_id, x in enumerate(batch):
        for label_idx, label_p in zip(x['label_idx'], x['label_probability']):
            for l_idx, l_p in zip(label_idx, label_p):
                label[sen_id][event_indexer[l_idx]] = l_p

    true_label = [x['true_label'] for x in batch]
    return (context_vecs, event_type_vecs, label, index, event_indexer, true_label)



def collate_fn_TC(batch):
    data_id = [x['id'] for x in batch]
    event_idx = [x['event_idx'] for x in batch]
    event_id = [x['event_id'] for x in batch]
    input_ids = torch.tensor([x['input_ids'] for x in batch], dtype=torch.long)
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.float)
    mask_token_mask = torch.tensor([x['mask_token_mask'] for x in batch], dtype=torch.bool)
    return data_id,event_idx,event_id,input_ids,labels, mask_token_mask
