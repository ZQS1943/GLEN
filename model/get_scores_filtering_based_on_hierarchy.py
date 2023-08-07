import json
from model.params import id2node, node_relation, roleset2id
from tqdm import tqdm
import torch
from data.utils import mapping_dict

threshold = 0
c_m_p_threshold = 0.065
print(threshold)

# input_file = './exp/experiments_type_ranking/all_data_new_ontology_step_2/bert_base/epoch_1/TC_and_TR_and_TD_results_annotated_test_no_other_eval_on_gold.json'
# input_file = './exp/experiments_type_ranking/all_data_new_ontology/bert_base/epoch_1/TC_and_TR_and_TD_results_annotated_test_no_other_eval_on_gold.json'
input_file = './exp/experiments_type_ranking/all_data_new_ontology_step_2/bert_base/epoch_1/TC_and_TR_and_TD_results_annotated_test_no_other.json'
# output_file = './exp/experiments_type_ranking/all_data_new_ontology_step_2/bert_base/epoch_1/test_final_results_eval_on_gold.json'
eval_on_gold = False

with open(input_file, 'r') as f:
    predict_samples = json.load(f)
# with open('./exp/experiments_type_ranking/all_data_new_ontology/bert_base/epoch_1/TC_and_TR_and_TD_results_annotated_test_no_other.json', 'r') as f:
#     predict_samples = json.load(f)

def send_to_zoey():
    test_results = []
    for item in tqdm(predict_samples):
        # if 'tc_types' not in item:
        #     continue
        tmp_item = {}
        tmp_item['sent_id'] = item['data_id']
        tokens = item['context']['tokens']
        tmp_item['sentence'] = ' '.join(tokens[1:-1]).replace(' ##', '')
        tmp_item['events'] = {}
        # print('*'*10)
        # prinassertt(item)
        #  1==0
        predicted_results = []
        for event_id, predicted_trigger in enumerate(item['predicted_triggers']):
            predicted_types = sorted(item['tc_types'][str(event_id)], key=lambda x:x[0], reverse=True)
            node_to_prob = {node:prob for prob, node in predicted_types}
            if predicted_types[0][0] > threshold:
                neighbour_nodes = {'parents':{}, 'children':{}}
                for node in node_relation[predicted_types[0][1]]['parents']:
                    if node in node_to_prob:
                        neighbour_nodes['parents'][node] = node_to_prob[node]
                for node in node_relation[predicted_types[0][1]]['child']:
                    if node in node_to_prob:
                        neighbour_nodes['children'][node] = node_to_prob[node]
                top_1_node = predicted_types[0][1]
                for node in node_relation[top_1_node]['parents']:
                    if node in node_to_prob and predicted_types[0][0] - node_to_prob[node] < c_m_p_threshold:
                        # predicted_results.append((predicted_trigger, top_1_node, predicted_types[0][0], neighbour_nodes, node, node_to_prob[node]))
                        predicted_results.append((predicted_trigger, node))
                        break
                else:
                    # predicted_results.append((predicted_trigger, top_1_node, predicted_types[0][0], neighbour_nodes, top_1_node, predicted_types[0][0]))    
                    predicted_results.append((predicted_trigger, top_1_node))    
                # predicted_results.append((predicted_trigger, predicted_types[0][1], predicted_types[0][0], neighbour_nodes))    
        gold_results = list(zip(item['context']['mention_idxs'], item['true_label']))

        def add_to_results(results, key_name = 'pred', complex = False):
            if not complex:
                for trigger_id, e_type in results:
                    trigger_word = tokens[trigger_id[0]: trigger_id[1] + 1]
                    trigger_word = ' '.join(trigger_word).replace(' ##', '')

                    trigger_id = f'{trigger_word}({trigger_id[0]}-{trigger_id[1] + 1})'
                    if trigger_id not in tmp_item['events']:
                        tmp_item['events'][trigger_id] = {}
                    name, des, node_id = id2node[e_type]
                    tmp_item['events'][trigger_id][key_name] = f"{node_id[4:]}: {name} - {des}"
            else:
                for trigger_id, top1_type, top1_score, neighbour_nodes, pred_type, pred_score in results:
                    trigger_word = tokens[trigger_id[0]: trigger_id[1] + 1]
                    trigger_word = ' '.join(trigger_word).replace(' ##', '')

                    trigger_id = f'{trigger_word}({trigger_id[0]}-{trigger_id[1] + 1})'
                    if trigger_id not in tmp_item['events']:
                        tmp_item['events'][trigger_id] = {}
                    top1_name, top1_des, top1_node_id = id2node[top1_type]
                    pred_name, pred_des, pred_node_id = id2node[pred_type]
                    tmp_item['events'][trigger_id][key_name] = {
                        'pred': f"{pred_node_id[4:]}: {pred_name} - {pred_des} [{pred_score}]",
                        'top1': f"{top1_node_id[4:]}: {top1_name} - {top1_des} [{top1_score}]",
                        'parents_of_top1': [f"{id2node[node_id][2][4:]}: {id2node[node_id][0]}" if node_id not in neighbour_nodes['parents'] else f"{id2node[node_id][2][4:]}: {id2node[node_id][0]} [{neighbour_nodes['parents'][node_id]}]" for node_id in node_relation[top1_type]['parents']],
                        'children_of_top1': [f"{id2node[node_id][2][4:]}: {id2node[node_id][0]}" if node_id not in neighbour_nodes['children'] else f"{id2node[node_id][2][4:]}: {id2node[node_id][0]} [{neighbour_nodes['children'][node_id]}]" for node_id in node_relation[top1_type]['child']],
                        'pc_in_top_10': any(node_id in neighbour_nodes['parents'] for node_id in node_relation[top1_type]['parents']) or any(node_id in neighbour_nodes['children'] for node_id in node_relation[top1_type]['child'])
                    }

        add_to_results(gold_results, 'gold')
        add_to_results(predicted_results, 'pred')

        # for trigger_id in tmp_item['events']:
        #     if 'pred' in tmp_item['events'][trigger_id] and 'gold' in tmp_item['events'][trigger_id]:
        #         if tmp_item['events'][trigger_id]['gold'] in tmp_item['events'][trigger_id]['pred']['pred']:
        #             tmp_item['events'][trigger_id]['correctness'] = 'trigger TP; event correct'
        #         else:
        #             tmp_item['events'][trigger_id]['correctness'] = 'trigger TP; event wrong'
        
        test_results.append(tmp_item)

    with open(f'./cache/test_final_results_step_2_{threshold}.json', 'w') as f:
        f.write(json.dumps(test_results, indent=True))

# send_to_zoey()
# assert 1==0
def hit_k(c_m_p_threshold, eval_on_gold = False):
    matched_trigger = 0
    hit_at_k_cnt = {x:0 for x in [1,2,5,10]}
    hit_at_k_cnt_in_top_10 = {x:0 for x in [1,2,5,10]}
    matched_trigger_in_top_10 = 0

    test_results = []
    for item in tqdm(predict_samples):
        tmp_item = {}
        tmp_item['sent_id'] = item['data_id']
        tokens = item['context']['tokens']
        tmp_item['sentence'] = ' '.join(tokens[1:-1]).replace(' ##', '')
        tmp_item['events'] = {}
        predicted_results = []

        gold_results = list(zip(item['context']['mention_idxs'], item['true_label']))
        if eval_on_gold:
            trigger_iter = enumerate(item['context']['mention_idxs'])
        else:
            trigger_iter = enumerate(item['predicted_triggers'])
        for event_id, predicted_trigger in trigger_iter:
            predicted_types = sorted(item['tc_types'][str(event_id)], key=lambda x:x[0], reverse=True)
            for gold_trigger, gold_type in gold_results:
                if gold_trigger == predicted_trigger:
                    matched_trigger += 1
                    in_top_10 = gold_type in [x[1] for x in predicted_types]
                    if in_top_10:
                        matched_trigger_in_top_10 += 1
                    for k in hit_at_k_cnt:
                        k_types = [x[1] for x in predicted_types[:k]]
                        if gold_type in k_types:
                            hit_at_k_cnt[k] += 1
                            if in_top_10:
                                hit_at_k_cnt_in_top_10[k] += 1
                    break
            


    scores = {}
    scores['matched_trigger'] = matched_trigger
    scores['matched_trigger_in_top_10'] = matched_trigger_in_top_10
    for k in hit_at_k_cnt:
        scores[f"Hit@{k}"] = hit_at_k_cnt[k]/matched_trigger
    for k in hit_at_k_cnt_in_top_10:
        scores[f"Hit@{k}_in_top_10"] = hit_at_k_cnt_in_top_10[k]/matched_trigger_in_top_10
    print(json.dumps(scores, indent=True))



def get_score(c_m_p_threshold, eval_on_gold = False):
    num_TD_correct = 0
    num_TD_gold = 0
    num_TD_predict = 0

    num_TC_correct = 0
    num_TC_gold = 0
    num_TC_predict = 0

    test_results = []
    for item in tqdm(predict_samples):
        tmp_item = {}
        tmp_item['sent_id'] = item['data_id']
        tokens = item['context']['tokens']
        tmp_item['sentence'] = ' '.join(tokens[1:-1]).replace(' ##', '')
        tmp_item['events'] = {}
        predicted_results = []
        if eval_on_gold:
            trigger_iter = enumerate(item['context']['mention_idxs'])
        else:
            trigger_iter = enumerate(item['predicted_triggers'])
        for event_id, predicted_trigger in trigger_iter:
            predicted_types = sorted(item['tc_types'][str(event_id)], key=lambda x:x[0], reverse=True)
            node_to_prob = {node:prob for prob, node in predicted_types}
            if predicted_types[0][0] > threshold:
                top_1_node = predicted_types[0][1]
                for node in node_relation[top_1_node]['parents']:
                    if node in node_to_prob and predicted_types[0][0] - node_to_prob[node] < c_m_p_threshold:
                        predicted_results.append((predicted_trigger, node))
                        break
                else:
                    predicted_results.append((predicted_trigger, top_1_node))    
        gold_results = list(zip(item['context']['mention_idxs'], item['true_label']))

        def add_to_results(results, key_name = 'pred'):
            for trigger_id, e_type in results:
                trigger_word = tokens[trigger_id[0]: trigger_id[1] + 1]
                trigger_word = ' '.join(trigger_word).replace(' ##', '')

                trigger_id = f'{trigger_word}({trigger_id[0]}-{trigger_id[1] + 1})'
                if trigger_id not in tmp_item['events']:
                    tmp_item['events'][trigger_id] = {}
                name, des, _ = id2node[e_type]
                tmp_item['events'][trigger_id][key_name] = f'{name}: {des}'

        add_to_results(predicted_results, 'pred')
        add_to_results(gold_results, 'gold')

        gold_trigger = set(tuple(x[0]) for x in gold_results)
        assert len(gold_trigger) == len(gold_results)
        
        for p_trigger, p_etype in predicted_results:
            for g_trigger, g_etype in gold_results:
                if g_trigger == p_trigger:
                    num_TD_correct += 1
                    num_TC_gold += 1
                    num_TC_predict += 1
                    if p_etype == g_etype:
                        num_TC_correct += 1
                    break
        num_TD_predict += len(predicted_results)
        num_TD_gold += len(gold_results)
        
        test_results.append(tmp_item)

    # with open(output_file, 'w') as f:
    #     f.write(json.dumps(test_results, indent=True))

    scores = {}
    print(num_TD_correct, num_TD_gold, num_TD_predict)
    print(num_TC_correct, num_TC_gold, num_TC_predict)
    scores['num_TI_correct'] = num_TD_correct
    scores['num_TI_gold'] = num_TD_gold
    scores['num_TI_predict'] = num_TD_predict
    scores['num_TC_correct'] = num_TC_correct

    scores["TI_prec"] = num_TD_correct/num_TD_predict
    scores["TI_recall"] = num_TD_correct/num_TD_gold
    scores["TI_F1"] = 2 * scores["TI_prec"] * scores["TI_recall"] / (scores["TI_prec"] + scores["TI_recall"])
    scores["TC_accuracy"] = num_TC_correct/num_TD_correct
    scores["TC_prec"] = num_TC_correct/num_TD_predict
    scores["TC_recall"] = num_TC_correct/num_TD_gold
    scores["TC_F1"] = 2 * scores["TC_prec"] * scores["TC_recall"] / (scores["TC_prec"] + scores["TC_recall"])
    print(json.dumps(scores, indent=True))



def group_by_roleset(c_m_p_threshold, eval_on_gold = True):
    covered_roleset = []
    with open('./covered_rolesets_in_added_train_data_step_1.txt', 'r') as f:
        for line in f.readlines():
            covered_roleset.append(int(line.replace('\n', '')))

    clean_roleset = []
    for roleset in mapping_dict:
        if len(mapping_dict[roleset]) == 1:
            clean_roleset.append(roleset2id[roleset])
            
    covered_roleset_event_cnt = 0
    clean_roleset_event_cnt = 0
    other_roleset_event_cnt = 0

    k_list = [1,2,5,10]

    covered_hit_at_k_cnt = {x:0 for x in k_list}
    clean_hit_at_k_cnt = {x:0 for x in k_list}
    other_hit_at_k_cnt = {x:0 for x in k_list}

    for item in tqdm(predict_samples):

        for event_id, trigger in enumerate(item['context']['mention_idxs']):
            gold_type = item['true_label'][event_id]
            roleset_id = item['roleset_ids'][event_id]
            predicted_types = sorted(item['tc_types'][str(event_id)], key=lambda x:x[0], reverse=True)
        
            if roleset_id in clean_roleset:
                clean_roleset_event_cnt += 1
            elif roleset_id in covered_roleset:
                covered_roleset_event_cnt += 1
            else:
                other_roleset_event_cnt += 1

            for k in k_list:
                k_types = [x[1] for x in predicted_types[:k]]
                if gold_type in k_types:
                    if roleset_id in clean_roleset:
                        clean_hit_at_k_cnt[k] += 1
                    elif roleset_id in covered_roleset:
                        covered_hit_at_k_cnt[k] += 1
                    else:
                        other_hit_at_k_cnt[k] += 1


            


    scores = {}
    scores['clean_roleset_event_cnt'] = clean_roleset_event_cnt
    scores['covered_roleset_event_cnt'] = covered_roleset_event_cnt
    scores['other_roleset_event_cnt'] = other_roleset_event_cnt
    scores['total_cnt'] = clean_roleset_event_cnt + covered_roleset_event_cnt + other_roleset_event_cnt
    for k in k_list:
        scores[f"Hit@{k}_clean"] = clean_hit_at_k_cnt[k]/clean_roleset_event_cnt
        scores[f"Hit@{k}_covered"] = covered_hit_at_k_cnt[k]/covered_roleset_event_cnt
        scores[f"Hit@{k}_other"] = other_hit_at_k_cnt[k]/other_roleset_event_cnt
        scores[f"Hit@{k}_total"] = (clean_hit_at_k_cnt[k] + covered_hit_at_k_cnt[k] + other_hit_at_k_cnt[k]) / scores['total_cnt']
    print(json.dumps(scores, indent=True))

# for c_m_p_threshold in tqdm(torch.linspace(0,0.2,100)):
#     print('*'*10)
#     print(f"c_m_p_threshold: {c_m_p_threshold}")
#     get_score(c_m_p_threshold)
# group_by_roleset(0.065, eval_on_gold = eval_on_gold)
hit_k(0.065, eval_on_gold = eval_on_gold)
get_score(0.065, eval_on_gold = eval_on_gold)



