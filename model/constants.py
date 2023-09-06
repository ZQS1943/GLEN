from data.data_preprocessing import read_xpo
from collections import defaultdict
SENT_TAG = "[unused0]"
EVENT_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"
SEN_EVENT_DIV_TAG = '[unused3]'
TRG_TAG = '[unused4]'
relation2idx = {'parent':0, 'child':1, 'similar':2}
xpo, xpo_used, node2id, id2node, roleset2id, id2roleset, id2node_detail = read_xpo()

def get_node_relation():
    wd2name = {}
    wd2node = {}
    wdrelations = defaultdict(lambda: defaultdict(set))
    node_relation = defaultdict(lambda: defaultdict(list))

    for node in xpo:
        wd_id = xpo[node]['wd_node']
        wd2name[wd_id] = xpo[node]['name']
        wd2node[wd_id] = node

    for node in xpo:
        wd_id = xpo[node]['wd_node']
        if 'overlay_parents' in xpo[node]:
            for parent in xpo[node]['overlay_parents']:
                if parent['wd_node'] in wd2name:
                    wdrelations[wd_id]['parents'].add(parent['wd_node'])
                    wdrelations[parent['wd_node']]['child'].add(wd_id)
        if 'similar_nodes' in xpo[node]:
            for s_node in xpo[node]['similar_nodes']:
                if s_node['wd_node'] in wd2name:
                    wdrelations[wd_id]['similar_nodes'].add(s_node['wd_node'])

    for wd_id in wdrelations:
        id = node2id[wd2node[wd_id]]
        for key in wdrelations[wd_id]:
            node_relation[id][key] = [node2id[wd2node[w_id]] for w_id in wdrelations[wd_id][key]]
    

    return node_relation


node_relation = get_node_relation()