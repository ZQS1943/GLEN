import argparse

def define_arguments(parser):
    parser.add_argument(
        "--seed",
        default=25,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--max_context_length",
        default=128,
        type=int,
        help="The maximum total input sequence length.",
    )
    parser.add_argument(
        "--path_to_model",
        default=None,
        type=str,
        help="The full path to the model to load(only in prediction mode).",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model.",
    )
    parser.add_argument(
        "--data_path",
        default="./data/data_preprocessed/",
        type=str,
        help="The path to the data.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output directory where generated output file (model, etc.) is to be dumped.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=50,
        type=int,
        help="The number of warm up steps.",
    )
    parser.add_argument(
        "--wb_name",
        default='default',
        type=str,
        help="The name of wb process",
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
        "--train_batch_size", default=8, type=int, 
        help="Total batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int,
        help="Total batch size for evaluation.",
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--threshold",
        default=0.9,
        type=float,
        help="In self-labeling for type classifier, only selecting cases whose P(top1 event) - P(top2 event) >= threshold ",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--print_interval", 
        type=int, 
        default=5, 
        help="Interval of loss printing",
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
    # TODO: refine this later
    parser.add_argument(
        "--TC_train_data_path",
        default=None,
        type=str,
        required=False,
        help="The path to the train data of Type Classification model.",
    )  
    parser.add_argument(
        "--cand_token_ids_path",
        default="./data/data_preprocessed/node_tokenized_ids_64_with_event_tag.pt",
        type=str,
        required=False,
        help="Filepath to the saved tokenized event descriptions.",
    )
    parser.add_argument(
        "--predict_set",
        default="test_set",
        type=str,
        required=False,
        help="Choose from train_set, dev_set, test_set",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="use the top k event types for each sentence",
    )
        
def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    params = args.__dict__
    return params