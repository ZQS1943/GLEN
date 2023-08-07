import os
import torch
from collections import defaultdict
from model.params import relation2idx, roleset2id, SENT_TAG, EVENT_TAG
from tqdm import tqdm

def sort_mentions(
    lst, sort_map=None,
):
    """
    sort_map: {orig_idx: idx in new "sorted" array}
    """
    new_lst = [0 for _ in range(len(lst))]
    for i in range(len(lst)):
        new_lst[sort_map[i]] = lst[i]
    return new_lst


def do_sort(
    sample, orig_idx_to_sort_idx,
):    
    sample['mentions'] = sort_mentions(sample['mentions'], orig_idx_to_sort_idx)
    sample['label_ids'] = sort_mentions(sample['label_ids'], orig_idx_to_sort_idx)
    sample['triggers'] = sort_mentions(sample['triggers'], orig_idx_to_sort_idx)
    sample['xpo_ids'] = sort_mentions(sample['xpo_ids'], orig_idx_to_sort_idx)
    sample['xpo_titles'] = sort_mentions(sample['xpo_titles'], orig_idx_to_sort_idx)
    sample['label'] = sort_mentions(sample['label'], orig_idx_to_sort_idx)
    sample['neighbor_nodes'] = sort_mentions(sample['neighbor_nodes'], orig_idx_to_sort_idx)
    sample['rolesets'] = sort_mentions(sample['rolesets'], orig_idx_to_sort_idx)
    if 'true_label' in sample:
        sample['true_label'] = sort_mentions(sample['true_label'], orig_idx_to_sort_idx)
    


def get_context_representation_multiple_mentions_idxs(
    sample, tokenizer, max_seq_length, add_sent_event_token=False,sent_tag_id = 1
):
    '''
    Also cuts out mentions beyond that context window

    ASSUMES MENTION_IDXS ARE SORTED!!!!

    Returns:
        List of mention bounds that are [inclusive, exclusive) (make both inclusive later)
        NOTE: 2nd index of mention bound may be outside of max_seq_length-range (must deal with later)
    '''
    num_tokens = 2 # [CLS] [SEP]
    mention_idxs = sample["tokenized_trigger_idxs"]
    input_ids = sample["tokenized_text_ids"]

    # sort mentions / entities / everything associated
    # [[orig_index, [start, end]], ....] --> sort by start, then end
    sort_tuples = [[i[0], i[1]] for i in sorted(enumerate(mention_idxs), key=lambda x:(x[1][0], x[1][1]))]
    if [tup[1] for tup in sort_tuples] != mention_idxs:
        orig_idx_to_sort_idx = {itm[0]: i for i, itm in enumerate(sort_tuples)}
        assert [tup[1] for tup in sort_tuples] == sort_mentions(mention_idxs, orig_idx_to_sort_idx)
        mention_idxs = [tup[1] for tup in sort_tuples]
        sample['tokenized_trigger_idxs'] = mention_idxs
        do_sort(sample, orig_idx_to_sort_idx)
        # TODO SORT EVERYTHING

    # fit leftmost mention, then all of the others that can reasonably fit...
    if len(mention_idxs):
        all_mention_spans_range = [mention_idxs[0][0], mention_idxs[-1][1]]
    else:
        all_mention_spans_range = [0,0]
    while all_mention_spans_range[1] - all_mention_spans_range[0] + num_tokens > max_seq_length:
        if len(mention_idxs) == 1:
            # don't cut further
            assert mention_idxs[0][1] - mention_idxs[0][0] + num_tokens > max_seq_length
            # truncate mention
            mention_idxs[0][1] = max_seq_length + mention_idxs[0][0] - num_tokens
        else:
            # cut last mention
            mention_idxs = mention_idxs[:len(mention_idxs) - 1]
        all_mention_spans_range = [mention_idxs[0][0], mention_idxs[-1][1]]
    
    context_left = input_ids[:all_mention_spans_range[0]]
    all_mention_tokens = input_ids[all_mention_spans_range[0]:all_mention_spans_range[1]]
    context_right = input_ids[all_mention_spans_range[1]:]

    left_quota = (max_seq_length - len(all_mention_tokens)) // num_tokens - 1
    right_quota = max_seq_length - len(all_mention_tokens) - left_quota - num_tokens
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:  # tokens left to add <= quota ON THE LEFT
        if right_add > right_quota:  # add remaining quota to right quota
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:  # tokens left to add <= quota ON THE RIGHT
            left_quota += right_quota - right_add  # add remaining quota to left quota

    if left_quota <= 0:
        left_quota = -len(context_left)  # cut entire list (context_left = [])
    if right_quota <= 0:
        right_quota = 0  # cut entire list (context_right = [])
    input_ids_window = context_left[-left_quota:] + all_mention_tokens + context_right[:right_quota]

    # shift mention_idxs
    if len(input_ids) <= max_seq_length - num_tokens:
        try:
            assert input_ids == input_ids_window
        except:
            print(input_ids)
            print(input_ids_window)
            assert 1==0
            import pdb
            pdb.set_trace()
    else:
        assert input_ids != input_ids_window
        cut_from_left = len(context_left) - len(context_left[-left_quota:])
        if cut_from_left > 0:
            # must shift mention_idxs
            for c in range(len(mention_idxs)):
                mention_idxs[c] = [
                    mention_idxs[c][0] - cut_from_left, mention_idxs[c][1] - cut_from_left,
                ]
    
    input_ids_window_padded = [101] + input_ids_window + [102]
    tokens = tokenizer.convert_ids_to_tokens(input_ids_window_padded)

    # +1 for CLS token
    mention_idxs = [[mention[0]+(num_tokens - 1), mention[1]+(num_tokens - 1)] for mention in mention_idxs]

    # input_ids = tokenizer.convert_tokens_to_ids(input_ids_window)
    padding = [0] * (max_seq_length - len(input_ids_window_padded))
    input_ids_window_padded += padding
    # print(input_ids_window_padded)
    # print(len(input_ids_window_padded), max_seq_length)
    if add_sent_event_token:
        if len(input_ids_window) + 3 <= max_seq_length:
            input_ids_with_sent_tag = [101] + [sent_tag_id] + input_ids_window + [102]
        else:
            input_ids_with_sent_tag = [101] + [sent_tag_id] + input_ids_window[:-1] + [102]
        input_ids_with_sent_tag = input_ids_with_sent_tag + [0] * (max_seq_length - len(input_ids_with_sent_tag))
    else:
        input_ids_with_sent_tag = None
    assert len(input_ids_window_padded) == max_seq_length

    return {
        "tokens": tokens,
        "ids": input_ids_window_padded,
        "mention_idxs": mention_idxs,
        "original_input": input_ids_window,
        "ids_with_sent_tag": input_ids_with_sent_tag
        # "pruned_ents": [1 for i in range(len(all_mentions)) if i < len(mention_idxs) else 0],  # pruned last N entities, TODO change if changed
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    silent,
    debug=False,
    logger=None,
    add_mention_bounds=True,
    saved_context_dir=None,
    candidate_token_ids=None,
    params=None,
    truncate=-1, 
    has_true_label = False,
    add_sent_event_token = False
):
    '''
    Returns /inclusive/ bounds
    '''
    extra_ret_values = {}
    print(f"saved_context_dir:{saved_context_dir}")
    if saved_context_dir is not None and os.path.exists(os.path.join(saved_context_dir, "tensor_tuple.pt")):
        data = torch.load(os.path.join(saved_context_dir, "data.pt"))
        tensor_data_tuple = torch.load(os.path.join(saved_context_dir, "tensor_tuple.pt"))
        return data, tensor_data_tuple, extra_ret_values

    if candidate_token_ids is None and not debug:
        candidate_token_ids = torch.load(params["cand_token_ids_path"])
        if logger: logger.info("Loaded saved entities info")
        extra_ret_values["candidate_token_ids"] = candidate_token_ids

    processed_samples = []

    sent_tag_id, event_tag_id = tokenizer.convert_tokens_to_ids([SENT_TAG, EVENT_TAG])

    if truncate != -1:
        samples = samples[:truncate]
    if silent:	
        iter_ = samples
    else:	
        iter_ = tqdm(samples)
    

    for idx, sample in enumerate(iter_):
        assert not add_mention_bounds, "Adding mention bounds, but we have multiple entities per example"
        # print(sample)
        context_tokens = get_context_representation_multiple_mentions_idxs(
            sample, tokenizer, max_context_length, add_sent_event_token = add_sent_event_token,
            sent_tag_id = sent_tag_id
        )
        # if len(context_tokens['mention_idxs']):
        #     print(sample)
        #     print(context_tokens)
        #     assert 1==0
        # if len(context_tokens["mention_idxs"]) == 115:
        #     print(sample)
        #     assert 1==0

        for i in range(len(context_tokens["mention_idxs"])):
            context_tokens["mention_idxs"][i][1] -= 1  # make bounds inclusive

        label_ids = sample["label_ids"]

        def get_roleset_id(x):
            return roleset2id[x] if x in roleset2id else 0
        roleset_ids = [get_roleset_id(roleset) for roleset in sample['rolesets']]
        neighbor_nodes = sample["neighbor_nodes"]


        # remove those that got pruned off
        if len(label_ids) > len(context_tokens['mention_idxs']):
            label_ids = label_ids[:len(context_tokens['mention_idxs'])]
            roleset_ids = roleset_ids[:len(context_tokens['mention_idxs'])]

        def get_label_token_ids(label_id):
            token_ids = candidate_token_ids[label_id]
            if add_sent_event_token:
                tmp = token_ids[:1] + [event_tag_id] + token_ids[1:]
                d = tmp[:len(token_ids)]
            return token_ids


        token_ids = [[get_label_token_ids(label_id) for label_id in _] for _ in label_ids]
        label_tokens = {
            "tokens": "",
            "ids": token_ids,
        }

        label_probability = [[1/len(_) for label in _] for _ in label_ids]
        # label_probability = [[1 for label in _] for _ in label_ids]

        n_nodes = [[[neighbor[0] for neighbor in label] for label in mention] for mention in neighbor_nodes]
        n_relations = [[[relation2idx[neighbor[1]] for neighbor in label] for label in mention] for mention in neighbor_nodes]

        domain = ''
        if 'propbank_' in sample['id']:
            domain = 'propbank'
        elif 'AMR_' in sample['id']:
            domain = 'AMR'
        elif 'ontonotes/' in sample['id']:
            domain = 'ontonotes'
        elif 'google/questionbank' in sample['id']:
            domain = 'questionbank'
        else:
            domain = 'anc'

        true_label = None
        if has_true_label:
            true_label = sample['true_label']

        
        label_to_mention = defaultdict(list)
        if has_true_label:
            for mention, l in zip(context_tokens['mention_idxs'], true_label):
                label_to_mention[l].append(mention)
        else:
            for mention, label in zip(context_tokens['mention_idxs'], label_ids):
                for l in label:
                    label_to_mention[l].append(mention)
        label_to_mention = [(x, candidate_token_ids[x], label_to_mention[x]) for x in label_to_mention]
        
        record = {
            "data_id": sample['id'],
            'domain': domain,
            "context": context_tokens,#这里加sent token
            "label": label_tokens,
            "label_idx": label_ids,
            "true_label": true_label,
            "roleset_ids": roleset_ids,
            "label_probability": label_probability,
            "neighbor_nodes": n_nodes,
            "neighbor_relations": n_relations,
            'label_to_mention': label_to_mention,
        }
        # print(record)
        
        # print (record['label_idx'], record['true_label'])
        processed_samples.append(record)

        # if record['context']['tokens'] == ['[CLS]', 'john', 'killed', 'mary', 'with', 'a', 'lead', 'pipe', ',', 'in', 'the', 'conservatory', '.', '[SEP]']:
        #     processed_samples.append(record)
    # assert 12==0
    return processed_samples


def process_mention_data_yes_no(
    samples,
    tokenizer,
    max_context_length,
    silent,
    debug=False,
    logger=None,
    add_mention_bounds=True,
    saved_context_dir=None,
    candidate_token_ids=None,
    params=None,
    truncate=-1, 
    has_true_label = False,
    add_sent_event_token = False,
    seed_round=False
):
    '''
    Returns /inclusive/ bounds
    '''
    extra_ret_values = {}
    print(f"saved_context_dir:{saved_context_dir}")
    if saved_context_dir is not None and os.path.exists(os.path.join(saved_context_dir, "tensor_tuple.pt")):
        data = torch.load(os.path.join(saved_context_dir, "data.pt"))
        tensor_data_tuple = torch.load(os.path.join(saved_context_dir, "tensor_tuple.pt"))
        return data, tensor_data_tuple, extra_ret_values

    candidate_token_ids = torch.load(params["cand_token_ids_path"])

    processed_samples = []

    if truncate != -1:
        samples = samples[:truncate]

    if silent:	
        iter_ = samples
    else:	
        iter_ = tqdm(samples)
    
    template = f"⟨type⟩ is defined as ⟨definition⟩. ⟨sentence⟩ Does ⟨trigger⟩ indicate a ⟨type⟩ event? [MASK]"

    for idx, sample in enumerate(iter_):
        sen = sample['text']
        for eid, (mention, label_ids, label_titles, label_descriptions, true_label) in enumerate(zip(sample['mentions'], sample['label_ids'], sample['xpo_titles'], sample['label'], sample['true_label'])):
            # sen_w_trg = sen[:mention[0]] + TRG_TAG + ' ' + sen[mention[0]:mention[1]] + ' ' + TRG_TAG + sen[mention[1]:]
            sen_w_trg = sen
            trigger_word = sen[mention[0]:mention[1]]
            if len(label_ids) == 1:
                continue
            for l_id, l_title, l_des in zip(label_ids, label_titles, label_descriptions):
                context = template.replace('⟨type⟩', l_title).replace('⟨definition⟩', l_des).replace('⟨sentence⟩', sen_w_trg).replace('⟨trigger⟩', trigger_word)
                input_ids = tokenizer.encode(context)
                mask_token_id = len(input_ids)
                label = 0
                if l_id == true_label:
                    label = 1
                if len(input_ids) > max_context_length - 2:
                    print(input_ids)
                    assert 1==0
                input_ids = [101] + input_ids + [102] + [0]*(max_context_length - 2 - len(input_ids))
                assert len(input_ids) == max_context_length
                mask_token_mask = [0]*max_context_length
                mask_token_mask[mask_token_id] = 1
                # print(tokenizer.convert_ids_to_tokens(input_ids))
                # print(tokenizer.convert_ids_to_tokens(input_ids)[mask_token_id])
                processed_samples.append({
                    'id': sample['id'],
                    'event_idx': eid,
                    'event_id': l_id,
                    'input_ids': input_ids,
                    'label': label,
                    'mask_token_mask':mask_token_mask
                })
    
    # assert 12==0
    return processed_samples


