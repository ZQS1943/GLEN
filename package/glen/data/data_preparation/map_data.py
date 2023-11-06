import json
import os
from .utils import read_amr, read_ontonotes
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl
import codecs

domain_list = []

def get_amr_data(dir_name = 'data/source_data/LDC2019E81_Abstract_Meaning_Representation_AMR_Annotation_Release_3.0/data/alignments/unsplit/'):

    def clean_text(text):
        del_ch = ['@-@', '@–@', '-@']
        for ch in del_ch:
            text = text.replace(ch,'-')
        text = text.replace('@/@', '/').replace('/?', '?').replace('/-', '-').replace('/.', '.')
        return text
    sentence_dict = defaultdict(lambda:defaultdict(dict))
    
    for root, dirs, files in os.walk(dir_name):
        for file_name in files:
            file_name = root + '/' + file_name
            raw_amrs = open(file_name, 'r').read()
            
            sents = read_amr(raw_amrs)
            
            for sent in tqdm(sents):
                tmp = sent.sentid.split('.')
                document_name = '.'.join(tmp[:-1])
                s_id = int(tmp[-1]) - 1
                sentence = sent.sent
                if '</a>' in sentence or '</b>' in sentence or '</i>' in sentence:
                    continue
                sentence = clean_text(sentence)
                sentence_dict[file_name][document_name][s_id] = sentence
    # for key in sentence_dict:
    #     sentence_dict[key] = dict(sentence_dict[key])
    return dict(sentence_dict)

def get_ontonotes_data(dir_name = "data/source_data/ontonotes-release-5.0/data/files/data/english/annotations"):
    def clean_text(text):
        del_ch = ['@-@', '@–@', '-@']
        for ch in del_ch:
            text = text.replace(ch,'-')
        text = text.replace('@/@', '/').replace('/?', '?').replace('/-', '-').replace('/.', '.')
        return text
    sentence_dict = defaultdict(dict)
    for root, dirs, files in os.walk(dir_name):
        for file_name in tqdm(files):
            if not file_name.endswith('.parse'):
                continue
            file_path = root + '/' + file_name
            raw_text = open(file_path, 'r').read()
            sents = read_ontonotes(raw_text)
            for sen_id, sent in enumerate(sents):
                sent = clean_text(sent)
                sentence_dict[file_path[:-6]][sen_id] = sent
    return dict(sentence_dict)
onto_cache_path = "onto_dict.pickle"
if not os.path.exists(onto_cache_path):
    onto_dict = get_ontonotes_data()
    with open(onto_cache_path, 'wb') as file:
        pkl.dump(onto_dict, file)
amr_cache_path = "amr_dict.pickle"
if not os.path.exists(amr_cache_path):
    amr_dict = get_amr_data()
    with open(amr_cache_path, 'wb') as file:
        pkl.dump(amr_dict, file)


with open(onto_cache_path, 'rb') as file:
    onto_dict = pkl.load(file)
with open(amr_cache_path, 'rb') as file:
    amr_dict = pkl.load(file)

data_dir = "./data/data_split"
for stage in ["train", "dev_annotated", "test_annotated"]:
    source_file = os.path.join(data_dir, f"{stage}_unprepared.json")
    target_file = os.path.join(data_dir, f"{stage}.json")
    with open(source_file, 'r') as f:
        data = json.load(f)

    for index, item in tqdm(enumerate(data)):
        if "ontonotes" in item["domain"]:
            path = 'data/source_data/ontonotes-release-5.0/data/files/data/english/annotations' + item['document'][len("./propbank-release/data/ontonotes"):-len(".gold_conll")]
            data[index]['sentence'] = onto_dict[path][item['s_id']]
        elif "LDC2019E81_Abstract_Meaning_Representation_AMR_Annotation_Release_3.0" in item["domain"]:
            path = 'data/source_data' + item['domain'][1:]            
            data[index]['sentence'] = amr_dict[path][item['document']][item['s_id']]
    with open(target_file, 'w') as f:
        f.write(json.dumps(data))  
