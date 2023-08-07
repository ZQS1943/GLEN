import json
from collections import defaultdict
# with open('./data/roleset_to_nodes_final.json', 'r') as f:
#     mapping_dict = json.load(f)

with open('./data/roleset_to_nodes_detail.json', 'r') as f:
    mapping_dict_detail = json.load(f)



with open('./data/xpo_final_version.json', 'r') as f:
    xpo = json.load(f)


mapping_dict = defaultdict(list)
for node in xpo:
    for roleset in xpo[node]['pb_roleset']:
        mapping_dict[roleset].append(node)

node_to_id = {x:i + 1 for i, x in enumerate(xpo)}
id_to_node = {i + 1:x for i, x in enumerate(xpo)}

# for i,x in enumerate(xpo):
#     print(f"{i + 1} {x}: {xpo[x]['name']} - {xpo[x]['wd_description']} - {xpo[x]['pb_roleset']}")


