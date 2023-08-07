import json
from tqdm import tqdm
threshold=0.9
print(f'only selecting those cases whose top1 - top2 >= {threshold} ')
# model_dir = './exp/experiments_type_ranking/all_data_new_ontology_step_2/bert_base/epoch_1'
# original_data_file = './cache/type_ranking_results_with_top_20_events_train_set_new_ontology_step_2.json'
# new_data_file = './cache/type_ranking_results_with_top_20_events_train_set_new_ontology_step_3.json'

model_dir = 'exp/experiments_type_ranking/all_data_kairos/bert_base/epoch_0'
original_data_file = './cache/type_ranking_results_with_top_20_events_train_set_kairos.json'
new_data_file = './cache/type_ranking_results_with_top_20_events_train_set_kairos_predicted.json'

predicted_results = []
with open(f'{model_dir}/bts_prediction_results_on_trainset.jsonl', 'r') as f:
    for line in tqdm(f.readlines()):
        line = json.loads(line)
        predicted_results.append(line)
print(f'total events with results: {len(predicted_results)}')
cur_p = 0

covered_rolesets = set()
cnt_selected_events = 0
cnt_clean_events = 0
with open(original_data_file, 'r') as f:
    data = json.load(f)
for data_idx, item in tqdm(enumerate(data)):
    data[data_idx]['predicted_labels_by_bts'] = {}
    for eid, candidate_set in enumerate(item['label_idx']):
        if len(candidate_set) <= 1:
            cnt_clean_events += 1
            continue
        cur_result = predicted_results[cur_p]
        cur_p += 1
        assert cur_result['sent_id'] == item['data_id'] and cur_result['event_id'] == eid and len(candidate_set) == len(cur_result['scores'])

        sorted_cands = sorted(list(zip(candidate_set, cur_result['scores'])), key=lambda x:x[1], reverse=True)
        if sorted_cands[0][1] - sorted_cands[1][1] < threshold:
            continue
        data[data_idx]['predicted_labels_by_bts'][eid] = sorted_cands[0][0]
        cnt_selected_events += 1
        covered_rolesets.add(data[data_idx]['roleset_ids'][eid])
    if cur_p == len(predicted_results):
        break

print(f'clean events: {cnt_clean_events}')
print(f'selected {cnt_selected_events} events')
print(f'num of covered roleset in selected events: {len(covered_rolesets)}')
# with open('./covered_rolesets_in_added_train_data_step_1.txt', 'w') as f:
#     for roleset in covered_rolesets:
#         f.write(f'{roleset}\n')
# assert 1==0
with open(new_data_file, 'w') as f:
    f.write(json.dumps(data))




