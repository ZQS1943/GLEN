import argparse
import os
import json
from collections import defaultdict

SENT_TAG = "[unused0]"
EVENT_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"
SEN_EVENT_DIV_TAG = '[unused3]'
TRG_TAG = '[unused4]'

relation2idx={'parent':0, 'child':1, 'similar':2}

roleset2id = {'other': 0}
with open('./data/roleset_list.txt', 'r') as f:
    for i, line in enumerate(f.read().split('\n')):
        roleset2id[line] = i + 1
id2roleset = {v:k for k, v in roleset2id.items()}

with open('./tmp_roleset_ids.txt', 'w') as f:
    for i in range(len(id2roleset)):
        f.write(f'{i}: {id2roleset[i]}\n')

with open('./data/xpo_glen.json', 'r') as f:
    xpo = json.load(f)
xpo_used = [0]
for node in xpo:
    if 'removing_reason' in xpo[node] and len(xpo[node]['removing_reason']):
        xpo_used.append(0)
        continue
    if len(xpo[node]['pb_roleset']):
        xpo_used.append(1)
    else:
        xpo_used.append(0)
id2node = {i + 1:(xpo[x]['name'], xpo[x]['wd_description'], x) for i, x in enumerate(xpo)}
id2node[0] = ('others', 'other event types', 'other')
node2id = {x:i + 1 for i, x in enumerate(xpo)}


def get_node_relation(xpo, id2node):
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

node_relation = get_node_relation(xpo, node2id)


class EDParser(argparse.ArgumentParser):
    def __init__(
        self, add_ed_args=True, add_model_args=False, 
        description='Event Detenction parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_ed_args,
        )
        self.ed_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['ed_HOME'] = self.ed_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_ed_args:
            self.add_ed_args()
        if add_model_args:
            self.add_model_args()

    def add_ed_args(self, args=None):
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--alpha", default=0.5, type=float, 
            help="alpha for lp"
        )
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", 
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=10, type=int) 
        parser.add_argument(
            "--seed", type=int, default=52313, help="random seed for initialization"
        )

    def add_model_args(self, args=None):
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=64,
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        ) 
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_false",
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.add_argument(
            "--out_dim", type=int, default=1, help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="Whether to add an additonal linear projection on top of BERT.",
        )
        parser.add_argument(
            "--data_path",
            default="./data/tokenized_final_no_other",
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )
        parser.add_argument(
            "--mention_aggregation_type",
            default=None,
            type=str,
            help="Type of mention aggregation (None to just use [CLS] token, "
            "'all_avg' to average across tokens in mention, 'fl_avg' to average across first/last tokens in mention, "
            "'{all/fl}_linear' for linear layer over mention, '{all/fl}_mlp' to MLP over mention)",
        )
        parser.add_argument(
            "--no_mention_bounds",
            dest="no_mention_bounds",
            action="store_true",
            default=True,
            help="Don't add tokens around target mention. MUST BE FALSE IF 'mention_aggregation_type' is NONE",
        )
        parser.add_argument(
            "--mention_scoring_method",
            dest="mention_scoring_method",
            default="qa_linear",
            type=str,
            help="Method for generating/scoring mentions boundaries (options: 'qa_mlp', 'qa_linear', 'BIO')",
        )
        parser.add_argument(
            "--max_mention_length",
            dest="max_mention_length",
            default=10,
            type=int,
            help="Maximum length of span to consider as candidate mention",
        )

    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )
        parser.add_argument(
            "--num_negatives",
            default=2,
            type=int,
            help="The number of negative samples when training event trigger matcher with new loss.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=50,
            type=int,
            help="The number of warm up steps.",
        )
        parser.add_argument(
            "--with_label_propagation", action="store_true", help="Whether to run propagation."
        )
        parser.add_argument(
            "--wb_name",
            default='default',
            type=str,
            help="The name of wb process",
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )
        parser.add_argument(
            "--hidden_size", default=768, type=int, 
            help="hidden size of bert model"
        )
        parser.add_argument(
            "--data_truncation", default=-1, type=int, 
            help="truncate data for debug"
        )
        parser.add_argument(
            "--linear_dim", default=128, type=int, 
            help="output size of linear size"
        )
        parser.add_argument(
            "--similarity_metric", default='cosine', type=str, 
            help="metric for computing the similarity score between sentence and event types"
        )
        parser.add_argument(
            "--train_batch_size", default=8, type=int, 
            help="Total batch size for training."
        )
        parser.add_argument(
            "--eval_batch_size", default=8, type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=1e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=5, 
            help="Interval of loss printing",
        )
        parser.add_argument(
           "--eval_interval",
            type=int,
            default=25000,
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=1, 
            help="Interval for model saving"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=4,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--loss_type",
            type=str,
            default="margin_loss",
            help="Which type of loss to use in event type ranking.",
        )
        parser.add_argument(
            "--shuffle", type=bool, default=False, 
            help="Whether to shuffle train data",
        )
        # TODO DELETE LATER!!!
        parser.add_argument(
            "--start_idx",
            default=None,
            type=int,
        )
        parser.add_argument(
            "--end_idx",
            default=None,
            type=int,
        )
        parser.add_argument(
            "--last_epoch",
            default=0,
            type=int,
            help="Epoch to restore from when pretraining",
        )
        parser.add_argument(
            "--train_samples_path",
            default='./cache/processed_mention_data_with_top_100_events_annotated_dev_set.json',
            type=str,
            required=False,
            help="The full path to the train_samples.",
        )
        parser.add_argument(
            "--path_to_trainer_state",
            default=None,
            type=str,
            required=False,
            help="The full path to the last checkpoint's training state to load.",
        )
        parser.add_argument(
            '--dont_distribute_train_samples',
            default=False,
            action="store_true",
            help="Don't distribute all training samples across the epochs (go through all samples every epoch)",
        )
        parser.add_argument(
            "--load_cand_enc_only",
            default=False,
            action="store_true",
            help="Only load the candidate encoder from saved model path",
        )
        parser.add_argument(
            "--cand_enc_path",
            default="models/all_entities_large.t7",
            type=str,
            required=False,
            help="Filepath to the saved entity encodings.",
        )
        parser.add_argument(
            "--cand_token_ids_path",
            default="data/node_tokenized_ids_64.pt",
            type=str,
            required=False,
            help="Filepath to the saved tokenized entity descriptions.",
        )
        parser.add_argument(
            "--get_losses",
            default=False,
            action="store_true",
            help="Get losses during evaluation",
        )

        parser.add_argument(
            "--prediction_data",
            default="test",
            type=str,
            help="name for data to be predicted",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="Train / validation / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results.",
        )
        parser.add_argument(
            "--encode_batch_size", 
            default=8, 
            type=int, 
            help="Batch size for encoding."
        )
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="Path for candidate pool",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for candidate encoding",
        )
        
