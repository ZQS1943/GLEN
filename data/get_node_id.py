import json
from utils import xpo, node_to_id
from collections import defaultdict

roleset_to_xpo_nodes = defaultdict(set)
for node in xpo:
    for roleset in xpo[node]['pb_roleset']:
        roleset_to_xpo_nodes[roleset].add(node)


new_mapping_dict = {}
for roleset in roleset_to_xpo_nodes:
    nodes_list = []
    for node in roleset_to_xpo_nodes[roleset]:
        nodes_list.append({
            'node_code': node,
            'node_name': xpo[node]['name'],
            'node_id': node_to_id[node],
            'node_description': xpo[node]['wd_description']
        })
    new_mapping_dict[roleset] = nodes_list

with open('./data/roleset_to_nodes_detail.json', 'w') as f:
    f.write(json.dumps(new_mapping_dict, indent=True))
