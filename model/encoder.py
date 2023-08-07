import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pytorch_transformers.modeling_bert import (
    BertModel,
    BertForMaskedLM
)
import math
from pytorch_transformers.tokenization_bert import BertTokenizer
from model.optimizer import get_bert_optimizer
from model.allennlp_span_utils import batched_span_select

def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

def loss_for_type_ranking(scores, candidate_label_sets, negative_smaples, device, th=1.0):
    loss = 0
    bs = scores.size()[0]
    for score, cand_sets, nega_samples in zip(scores, candidate_label_sets, negative_smaples):
        max_nega_score = torch.max(score[nega_samples])
        for cand_set in cand_sets:
            max_cand_score = torch.max(score[cand_set])
            # print(max_cand_score, max_nega_score)
            loss += 1/len(cand_sets) * torch.max(torch.tensor(.0).to(device), th - max_cand_score + max_nega_score)
    return loss/bs

def loss_for_type_classification(yes_scores, cand_mask, device, th = 0.1):
    max_nega_score = torch.max(yes_scores[~cand_mask])
    max_cand_score = torch.max(yes_scores[cand_mask])
    print(max_nega_score, max_cand_score)
    loss = torch.max(torch.tensor(.0).to(device), th - max_cand_score + max_nega_score)
    return loss


def batch_reshape_mask_left(
    input_t, selected, pad_idx=0, left_align_mask=None
):
    """
    Left-aligns all ``selected" values in input_t, which is a batch of examples.
        - input_t: >=2D tensor (N, M, *)
        - selected: 2D torch.Bool tensor, 2 dims same size as first 2 dims of `input_t` (N, M)
        - pad_idx represents the padding to be used in the output
        - left_align_mask: if already precomputed, pass the alignment mask in
            (mask on the output, corresponding to `selected` on the input)
    Example:
        input_t  = [[1,2,3,4],[5,6,7,8]]
        selected = [[0,1,0,1],[1,1,0,1]]
        output   = [[2,4,0],[5,6,8]]
    """
    batch_num_selected = selected.sum(1)
    max_num_selected = batch_num_selected.max()

    # (bsz, 2)
    repeat_freqs = torch.stack([batch_num_selected, max_num_selected - batch_num_selected], dim=-1)
    # (bsz x 2,)
    repeat_freqs = repeat_freqs.view(-1)

    if left_align_mask is None:
        # (bsz, 2)
        left_align_mask = torch.zeros(input_t.size(0), 2).to(input_t.device).bool()
        left_align_mask[:,0] = 1
        # (bsz x 2,): [1,0,1,0,...]
        left_align_mask = left_align_mask.view(-1)
        # (bsz x max_num_selected,): [1 xrepeat_freqs[0],0 x(M-repeat_freqs[0]),1 xrepeat_freqs[1],0 x(M-repeat_freqs[1]),...]
        left_align_mask = left_align_mask.repeat_interleave(repeat_freqs)
        # (bsz, max_num_selected)
        left_align_mask = left_align_mask.view(-1, max_num_selected)

    # reshape to (bsz, max_num_selected, *)
    input_reshape = torch.Tensor(left_align_mask.size() + input_t.size()[2:]).to(input_t.device, input_t.dtype).fill_(pad_idx)
    input_reshape[left_align_mask] = input_t[selected]
    # (bsz, max_num_selected, *); (bsz, max_num_selected)
    return input_reshape, left_align_mask


def load_biencoder(params):
    # Init model
    biencoder = EncoderRanker(params)
    return biencoder


def get_submodel_from_state_dict(state_dict, prefix):
    # get only submodel specified with prefix 'prefix' from the state_dict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = key[len(prefix)+1:]  # +1 for '.'  
            new_state_dict[key] = value
    return new_state_dict


class mentionScoresHead(nn.Module):
    def __init__(
        self, bert_output_dim, scoring_method="qa_linear", max_mention_length=10, 
    ):
        super(mentionScoresHead, self).__init__()
        self.scoring_method = scoring_method
        self.max_mention_length = max_mention_length
        if self.scoring_method == "qa_linear":
            self.bound_classifier = nn.Linear(bert_output_dim, 3)
        elif self.scoring_method == "qa_mlp" or self.scoring_method == "qa":  # for back-compatibility
            self.bound_classifier = nn.Sequential(
                nn.Linear(bert_output_dim, bert_output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(bert_output_dim, 3),
            )
        else:
            raise NotImplementedError()

    def forward(self, bert_output, mask_ctxt):
        '''
        Retuns scores for *inclusive* mention boundaries
        '''
        # (bs, seqlen, 3)
        logits = self.bound_classifier(bert_output)
        if self.scoring_method[:2] == "qa":
            # (bs, seqlen, 1); (bs, seqlen, 1); (bs, seqlen, 1)
            start_logprobs, end_logprobs, mention_logprobs = logits.split(1, dim=-1)
            # (bs, seqlen)
            start_logprobs = start_logprobs.squeeze(-1)
            end_logprobs = end_logprobs.squeeze(-1)
            mention_logprobs = mention_logprobs.squeeze(-1)
            # impossible to choose masked tokens as starts/ends of spans
            start_logprobs = torch.where(~mask_ctxt, -float("Inf"), start_logprobs)
            end_logprobs = torch.where(~mask_ctxt, -float("Inf"), end_logprobs)
            mention_logprobs = torch.where(~mask_ctxt, -float("Inf"), mention_logprobs)

            # take sum of log softmaxes:
            # log p(mention) = log p(start_pos && end_pos) = log p(start_pos) + log p(end_pos)
            # DIM: (bs, starts, ends)
            mention_scores = start_logprobs.unsqueeze(2) + end_logprobs.unsqueeze(1)
            # (bs, starts, ends)
            mention_cum_scores = torch.zeros(mention_scores.size(), dtype=mention_scores.dtype).to(mention_scores.device)
            # add ends
            mention_logprobs_end_cumsum = torch.zeros(mask_ctxt.size(0), dtype=mention_scores.dtype).to(mention_scores.device)
            for i in range(mask_ctxt.size(1)):
                mention_logprobs_end_cumsum += mention_logprobs[:,i]
                mention_cum_scores[:,:,i] += mention_logprobs_end_cumsum.unsqueeze(-1)
            # subtract starts
            mention_logprobs_start_cumsum = torch.zeros(mask_ctxt.size(0), dtype=mention_scores.dtype).to(mention_scores.device)
            for i in range(mask_ctxt.size(1)-1):
                mention_logprobs_start_cumsum += mention_logprobs[:,i]
                mention_cum_scores[:,(i+1),:] -= mention_logprobs_start_cumsum.unsqueeze(-1)

            # DIM: (bs, starts, ends)
            mention_scores += mention_cum_scores

            # DIM: (starts, ends, 2) -- tuples of [start_idx, end_idx]
            mention_bounds = torch.stack([
                torch.arange(mention_scores.size(1)).unsqueeze(-1).expand(mention_scores.size(1), mention_scores.size(2)),  # start idxs
                torch.arange(mention_scores.size(1)).unsqueeze(0).expand(mention_scores.size(1), mention_scores.size(2)),  # end idxs
            ], dim=-1).to(mask_ctxt.device)
            # DIM: (starts, ends)
            mention_sizes = mention_bounds[:,:,1] - mention_bounds[:,:,0] + 1  # (+1 as ends are inclusive)

            # Remove invalids (startpos > endpos, endpos > seqlen) and renormalize
            # DIM: (bs, starts, ends)
            valid_mask = (mention_sizes.unsqueeze(0) > 0) & mask_ctxt.unsqueeze(1)
            # DIM: (bs, starts, ends)
            mention_scores[~valid_mask] = -float("inf")  # invalids have logprob=-inf (p=0)
            # DIM: (bs, starts * ends)
            mention_scores = mention_scores.view(mention_scores.size(0), -1)
            # DIM: (bs, starts * ends, 2)
            mention_bounds = mention_bounds.view(-1, 2)
            mention_bounds = mention_bounds.unsqueeze(0).expand(mention_scores.size(0), mention_scores.size(1), 2)
        
        if self.max_mention_length is not None:
            mention_scores, mention_bounds = self.filter_by_mention_size(
                mention_scores, mention_bounds, self.max_mention_length,
            )

        return mention_scores, mention_bounds
    
    def filter_by_mention_size(self, mention_scores, mention_bounds, max_mention_length):
        '''
        Filter all mentions > maximum mention length
        mention_scores: torch.FloatTensor (bsz, num_mentions)
        mention_bounds: torch.LongTensor (bsz, num_mentions, 2)
        '''
        # (bsz, num_mentions)
        mention_bounds_mask = (mention_bounds[:,:,1] - mention_bounds[:,:,0] <= max_mention_length)
        # (bsz, num_filtered_mentions)
        mention_scores = mention_scores[mention_bounds_mask]
        mention_scores = mention_scores.view(mention_bounds_mask.size(0),-1)
        # (bsz, num_filtered_mentions, 2)
        mention_bounds = mention_bounds[mention_bounds_mask]
        mention_bounds = mention_bounds.view(mention_bounds_mask.size(0),-1,2)
        return mention_scores, mention_bounds

class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss


class GetContextEmbedsHead(nn.Module):
    def __init__(self, ctxt_output_dim, cand_output_dim):
        super(GetContextEmbedsHead, self).__init__()
        assert ctxt_output_dim == cand_output_dim

    def forward(self, bert_output, mention_bounds):
        '''
        bert_output
            (bs, seqlen, embed_dim)
        mention_bounds: both bounds are inclusive [start, end]
            (bs, num_spans, 2)
        '''
        # get embedding of [CLS] token
        if mention_bounds.size(0) == 0:
            return mention_bounds
        
        (
            embedding_ctxt,  # (batch_size, num_spans, max_batch_span_width, embedding_size)
            mask,  # (batch_size, num_spans, max_batch_span_width)
        ) = batched_span_select(
            bert_output,  # (batch_size, sequence_length, embedding_size)
            mention_bounds,  # (batch_size, num_spans, 2)
        )
        embedding_ctxt[~mask] = 0  # 0 out masked elements
        # embedding_ctxt = (batch_size, num_spans, max_batch_span_width, embedding_size)
        embedding_ctxt = embedding_ctxt.sum(2) / mask.sum(2).float().unsqueeze(-1)        
        return embedding_ctxt

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None,
    ):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask, DEBUG=False):
        if DEBUG:
            import pdb
            pdb.set_trace()
        try:
            output_bert, output_pooler, _ = self.bert_model(
                token_ids, segment_ids, attention_mask
            )
        except RuntimeError as e:
            print(token_ids.size())
            print(segment_ids.size())
            print(attention_mask.size())
            print(e)
            import pdb
            pdb.set_trace()
            output_bert, output_pooler, _ = self.bert_model(
                token_ids, segment_ids, attention_mask
            )

        if self.additional_linear is not None:
            # embeddings = (batch_size, embedding_size)
            embeddings = output_pooler
        else:
            # embeddings = (batch_size, embedding_size)
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result

class EncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(EncoderModule, self).__init__()
        bert = BertModel.from_pretrained(params["bert_model"], output_hidden_states=True)

        self.encoder = BertEncoder(
            bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )

        self.config = bert.config

        bert_output_dim = bert.embeddings.word_embeddings.weight.size(1)

        self.mention_aggregation_type = params.get('mention_aggregation_type', None)
        self.classification_heads = nn.ModuleDict({})
        self.linear_compression = None
        if self.mention_aggregation_type is not None:
            classification_heads_dict = {'get_context_embeds': GetContextEmbedsHead(
                bert_output_dim,
                bert_output_dim,
            )}
            classification_heads_dict['mention_scores'] = mentionScoresHead(
                bert_output_dim,
                params["mention_scoring_method"],
                params.get("max_mention_length", 10),
            )
            self.classification_heads = nn.ModuleDict(classification_heads_dict)

    def get_raw_ctxt_encoding(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
    ):
        """
            Gets raw, shared context embeddings from BERT,
            to be used by both mention detector and event type classifier

        Returns:
            torch.FloatTensor (bsz, seqlen, embed_dim)
        """
        raw_ctxt_encoding, _, _ = self.encoder.bert_model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
        )
        return raw_ctxt_encoding

    def get_ctxt_mention_scores(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        raw_ctxt_encoding = None,
    ):
        """
            Gets mention scores using raw context encodings

        Inputs:
            raw_ctxt_encoding: torch.FloatTensor (bsz, seqlen, embed_dim)
        Returns:
            torch.FloatTensor (bsz, num_total_mentions): mention scores/logits
            torch.IntTensor (bsz, num_total_mentions): mention boundaries
        """
        # (bsz, seqlen, embed_dim)
        if raw_ctxt_encoding is None:
            raw_ctxt_encoding = self.get_raw_ctxt_encoding(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )

        # (num_total_mentions,); (num_total_mentions,)
        return self.classification_heads['mention_scores'](
            raw_ctxt_encoding, mask_ctxt,
        )

    def prune_ctxt_mentions(
        self,
        mention_logits,
        mention_bounds,
        num_cand_mentions,
        threshold,
    ):
        '''
            Prunes mentions based on mention scores/logits (by either
            `threshold` or `num_cand_mentions`, whichever yields less candidates)

        Inputs:
            mention_logits: torch.FloatTensor (bsz, num_total_mentions)
            mention_bounds: torch.IntTensor (bsz, num_total_mentions)
            num_cand_mentions: int
            threshold: float
        Returns:
            torch.FloatTensor(bsz, max_num_pred_mentions): top mention scores/logits
            torch.IntTensor(bsz, max_num_pred_mentions, 2): top mention boundaries
            torch.BoolTensor(bsz, max_num_pred_mentions): mask on top mentions
            torch.BoolTensor(bsz, total_possible_mentions): mask for reshaping from total possible mentions -> max # pred mentions
        '''
        # (bsz, num_cand_mentions); (bsz, num_cand_mentions)
        top_mention_logits, mention_pos = mention_logits.topk(num_cand_mentions, sorted=True)
        # (bsz, num_cand_mentions, 2)
        #   [:,:,0]: index of batch
        #   [:,:,1]: index into top mention in mention_bounds
        mention_pos = torch.stack([torch.arange(mention_pos.size(0)).to(mention_pos.device).unsqueeze(-1).expand_as(mention_pos), mention_pos], dim=-1)
        # (bsz, num_cand_mentions)
        top_mention_pos_mask = torch.sigmoid(top_mention_logits).log() > threshold
        # (total_possible_mentions, 2)
        #   tuples of [index of batch, index into mention_bounds] of what mentions to include
        mention_pos = mention_pos[top_mention_pos_mask | (
            # 2nd part of OR: if nothing is > threshold, use topK that are > -inf
            ((top_mention_pos_mask.sum(1) == 0).unsqueeze(-1)) & (top_mention_logits > -float("inf"))
        )]
        mention_pos = mention_pos.view(-1, 2)
        # (bsz, total_possible_mentions)
        #   mask of possible logits
        mention_pos_mask = torch.zeros(mention_logits.size(), dtype=torch.bool).to(mention_pos.device)
        mention_pos_mask[mention_pos[:,0], mention_pos[:,1]] = 1
	# (bsz, max_num_pred_mentions, 2)
        chosen_mention_bounds, chosen_mention_mask = batch_reshape_mask_left(mention_bounds, mention_pos_mask, pad_idx=0)
        # (bsz, max_num_pred_mentions)
        chosen_mention_logits, _ = batch_reshape_mask_left(mention_logits, mention_pos_mask, pad_idx=-float("inf"), left_align_mask=chosen_mention_mask)
        return chosen_mention_logits, chosen_mention_bounds, chosen_mention_mask, mention_pos_mask

    def get_ctxt_embeds(
        self,
        raw_ctxt_encoding,
        mention_bounds,
    ):
        """
            Get candidate scores + embeddings associated with passed-in mention_bounds

        Input
            raw_ctxt_encoding: torch.FloatTensor (bsz, seqlen, embed_dim)
                shared embeddings straight from BERT
            mention_bounds: torch.IntTensor (bsz, max_num_pred_mentions, 2)
                top mention boundaries

        Returns
            torch.FloatTensor (bsz, max_num_pred_mentions, embed_dim)
        """
        # (bs, max_num_pred_mentions, embed_dim)
        embedding_ctxt = self.classification_heads['get_context_embeds'](raw_ctxt_encoding, mention_bounds)
        if self.linear_compression is not None:
            embedding_ctxt = self.linear_compression(embedding_ctxt)
        return embedding_ctxt

    def forward_ctxt(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        gold_mention_bounds=None,
        gold_mention_bounds_mask=None,
        num_cand_mentions=50,
        topK_threshold=-4.5,
        get_mention_scores=True,
    ):
        """
        If jie is set, returns mention embeddings of passed-in mention bounds
        Otherwise, uses top-scoring mentions
        """

        if self.mention_aggregation_type is None:
            '''
            OLD system: don't do mention aggregation (use tokens around mention)???
            '''
            embedding_ctxt = self.encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )
            # linear mapping to correct context length (may not need)
            if self.linear_compression is not None:
                embedding_ctxt = self.linear_compression(embedding_ctxt)
            return embedding_ctxt, None, None, None

        else:
            '''
            NEW system: aggregate mention tokens
            '''
            # (bs, seqlen, embed_size)
            raw_ctxt_encoding = self.get_raw_ctxt_encoding(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            )

            top_mention_bounds = None
            top_mention_logits = None
            extra_rets = {}
            if get_mention_scores:
                mention_logits, mention_bounds = self.get_ctxt_mention_scores(
                    token_idx_ctxt, segment_idx_ctxt, mask_ctxt, raw_ctxt_encoding,
                )
                extra_rets['all_mention_logits'] = mention_logits
                extra_rets['all_mention_bounds'] = mention_bounds
                if gold_mention_bounds is None:
                    (
                        top_mention_logits, top_mention_bounds, top_mention_mask, all_mention_mask,
                    ) = self.prune_ctxt_mentions(
                        mention_logits, mention_bounds, num_cand_mentions, topK_threshold,
                    )
                    extra_rets['mention_logits'] = top_mention_logits.view(-1)
                    extra_rets['all_mention_mask'] = all_mention_mask

            if top_mention_bounds is None:
                # use gold mention
                top_mention_bounds = gold_mention_bounds
                top_mention_mask = gold_mention_bounds_mask

            assert top_mention_bounds is not None
            assert top_mention_mask is not None

            # (bs, num_pred_mentions OR num_gold_mentions, embed_size)
            embedding_ctxt = self.get_ctxt_embeds(
                raw_ctxt_encoding, top_mention_bounds,
            )
            # for merging dataparallel, only 1st dimension can differ...
            return {
                "mention_reps": embedding_ctxt.view(-1, embedding_ctxt.size(-1)),
                "mention_bounds": top_mention_bounds.view(-1, top_mention_bounds.size(-1)),
                "mention_masks": top_mention_mask.view(-1),
                "mention_dims": torch.tensor(top_mention_mask.size()).unsqueeze(0).to(embedding_ctxt.device),
                **extra_rets
            }

    def forward_candidate(
        self,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        try:
            return self.encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        except:
            print('cannot foward this event type')
            print(token_idx_cands.size())
            print(segment_idx_cands.size())
            print(mask_cands.size())
            return torch.rand(token_idx_cands.size()).to(token_idx_cands.device)

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
        gold_mention_bounds=None,
        gold_mention_bounds_mask=None,
        num_cand_mentions=50,
        topK_threshold=-4.5,
        get_mention_scores=True,
    ):
        """
        If gold_mention_bounds is set, returns mention embeddings of passed-in mention bounds
        Otherwise, uses top-scoring mentions
        """
        embedding_ctxt = embedding_cands = top_mention_mask = \
                top_mention_logits = top_mention_bounds = all_mention_mask = \
                all_mention_logits = all_mention_bounds = max_num_pred_mentions = None

        context_outs = None
        cand_outs = None
        if token_idx_ctxt is not None:
            context_outs = self.forward_ctxt(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
                gold_mention_bounds=gold_mention_bounds,
                gold_mention_bounds_mask=gold_mention_bounds_mask,
                num_cand_mentions=num_cand_mentions, topK_threshold=topK_threshold,
                get_mention_scores=get_mention_scores,
            )
        if token_idx_cands is not None:
            cand_outs = self.forward_candidate(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return context_outs, cand_outs

    def upgrade_state_dict_named(self, state_dict):
        print("why use this?")
        assert 1==0
        prefix = ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            if head_name not in current_head_names:
                print(
                    'WARNING: deleting classification head ({}) from checkpoint '
                    'not present in current model: {}'.format(head_name, k)
                )
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

class EventTriggerMatchingYN(torch.nn.Module):
    def __init__(self, params, model_path=None):
        super(EventTriggerMatchingYN, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        self.yes_id = self.tokenizer.convert_tokens_to_ids(['yes'])[0]
        self.no_id = self.tokenizer.convert_tokens_to_ids(['no'])[0]
        # init model
        self.model = BertForMaskedLM.from_pretrained(params["bert_model"])
        model_path = params.get("path_to_model", None)
        print(model_path)
        if model_path is not None:
            # assert 1==0
            self.load_model(model_path)
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.loss_fact = nn.BCELoss()
        # self.loss_fact = nn.BCEWithLogitsLoss()
    def load_model(self, fname, cpu=False):
        if cpu or not torch.cuda.is_available():
            state_dict = torch.load(fname, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(fname)
        
        # self.model.upgrade_state_dict_named(state_dict)
        self.load_state_dict(state_dict)
        
    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def forward_new_loss(
        self, input_ids, mask_token_mask, 
        cand_mask=None, 
        return_loss=True
    ):
        token_embed = self.model(input_ids)[0]
        mask_token_embed = token_embed[mask_token_mask]
        yes_no_scores = torch.index_select(mask_token_embed, dim=1, index=torch.tensor([self.yes_id, self.no_id]).to(self.device))
        yes_no_scores = F.softmax(yes_no_scores, dim=-1)
        yes_scores = yes_no_scores[:,0]

        if return_loss:
            loss = loss_for_type_classification(yes_scores, cand_mask, self.device)
            return yes_scores, loss
        return yes_scores, 0

    def forward(self, input_ids, mask_token_mask, labels = None,return_loss=True):
        token_embed = self.model(input_ids)[0]
        mask_token_embed = token_embed[mask_token_mask]
        yes_no_scores = torch.index_select(mask_token_embed, dim=1, index=torch.tensor([self.yes_id, self.no_id]).to(self.device))
        yes_no_scores = F.softmax(yes_no_scores, dim=-1)
        yes_scores = yes_no_scores[:,0]

        if return_loss:
            loss = self.loss_fact(yes_scores, labels)
            return yes_scores, loss
        return yes_scores, 0

class EventTriggerMatching(torch.nn.Module):
    def __init__(self, params, cross_entropy=False, model_path=None):
        super(EventTriggerMatching, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        # model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.matching_classifier = nn.Sequential(
            nn.Linear(params['hidden_size'], params['hidden_size']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(params['hidden_size'], 1),
        )
        self.matching_classifier = self.matching_classifier.to(self.device)

        if cross_entropy:
            self.loss_fact = nn.CrossEntropyLoss()
        else:
            self.loss_fact = nn.MultiLabelMarginLoss()
    
    def load_model(self, fname, cpu=False):
        if cpu or not torch.cuda.is_available():
            state_dict = torch.load(fname, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(fname)
        
        self.model.upgrade_state_dict_named(state_dict)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = EncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def event_type_encode(self, event_type_input):
        # TODO: check mask
        token_idx, segment_idx, mask = to_bert_input(
            event_type_input, self.NULL_IDX
        )
        et_embed = self.model.encoder.bert_model(token_idx, segment_idx, mask)[0]
        et_embed = self.linear(et_embed)
        et_embed = torch.nn.functional.normalize(et_embed, p=2, dim=2) 
        return et_embed 

    def encode_candidate(self, event_type_input):
        token_idx, segment_idx, mask = to_bert_input(
            event_type_input, self.NULL_IDX
        )
        et_embed = self.model.encoder.bert_model(token_idx, segment_idx, mask)[0]
        et_embed = self.linear(et_embed)
        et_embed = torch.nn.functional.normalize(et_embed, p=2, dim=2) 
        return et_embed 

    def sentence_encode(self, sentence_input):
        token_idx, segment_idx, mask = to_bert_input(
            sentence_input, self.NULL_IDX
        )
        s_embed = self.model.encoder.bert_model(token_idx, segment_idx, mask)[0]
        s_embed = self.linear(s_embed)
        s_embed = torch.nn.functional.normalize(s_embed, p=2, dim=2)

        return  s_embed

    def forward(
        self, concatenated_input,
        trigger_word=False,
        label=None,
        return_loss=True,
        only_sentence=False
    ):
        """
        """
        input_dim = len(concatenated_input.size())
        if input_dim == 3:
            bs, k, l  = concatenated_input.size()
            concatenated_input = concatenated_input.reshape(bs * k, -1)
        token_idx, segment_idx, mask = to_bert_input(
            concatenated_input, self.NULL_IDX
        )
        input_emb = self.model.encoder.bert_model(token_idx, segment_idx, mask)
        if only_sentence:
            if trigger_word:
                return input_emb[0]
            else:
                return input_emb[1]
                
        input_emb = input_emb[1]
        scores = self.matching_classifier(input_emb)
        scores = torch.squeeze(scores)
        if input_dim == 3:
            scores = scores.reshape(bs, -1)
        if return_loss:
            loss = self.loss_fact(scores, label)
            return loss, scores
        else:
            return 0, scores

class TriggerLocalizer(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TriggerLocalizer, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)
        self.model = self.model.to(self.device)
    
    def load_model(self, fname, cpu=False):
        if cpu or not torch.cuda.is_available():
            state_dict = torch.load(fname, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(fname)
        
        # self.upgrade_state_dict_named(state_dict)
        self.load_state_dict(state_dict)

    def build_model(self):
        self.model = EncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(
        self, cands, gold_mention_bounds=None, gold_mention_bounds_mask=None,
        num_cand_mentions=50, topK_threshold=-4.5,
        get_mention_scores=True,
    ):
        """
        if gold_mention_bounds specified, selects according to gold_mention_bounds,
        otherwise selects according to top-scoring mentions

        Returns: Dictionary
            mention_reps: torch.FloatTensor (bsz, max_num_pred_mentions, embed_dim): mention embeddings
            mention_masks: torch.BoolTensor (bsz, max_num_pred_mentions): mention padding mask
            mention_bounds: torch.LongTensor (bsz, max_num_pred_mentions, 2)
            (
            mention_logits: torch.FloatTensor (bsz, max_num_pred_mentions): mention scores/logits
            all_mention_mask: torch.BoolTensor ((bsz, all_cand_mentions)
            all_mention_logits: torch.FloatTensor (bsz, all_cand_mentions): all mention scores/logits
            all_mention_bounds: torch.LongTensor (bsz, all_cand_mentions, 2): all mention bounds
            )
        """
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        context_outs, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands,
            None, None, None,
            gold_mention_bounds=gold_mention_bounds,
            gold_mention_bounds_mask=gold_mention_bounds_mask,
            num_cand_mentions=num_cand_mentions,
            topK_threshold=topK_threshold,
            get_mention_scores=get_mention_scores
        )
        if context_outs['mention_dims'].size(0) <= 1:
            for key in context_outs:
                if 'all' in key or key == 'mention_dims':
                    continue
                context_outs[key] = context_outs[key].view([context_outs['mention_dims'][0,0], -1] + list(context_outs[key].size()[1:]))
            return context_outs

        '''
        Reshape to (bs, num_mentions, *), iterating across GPUs
        '''
        def init_tensor(shape, dtype, init_value):
            return init_value * torch.ones(
                shape
            ).to(dtype=dtype, device=context_outs['mention_dims'].device)

        bs = cands.size(0)
        n_pred_mentions = context_outs['mention_dims'][:,1].max()
        context_outs_reshape = {}
        for key in context_outs:
            if 'all' in key or key == 'mention_dims':
                context_outs_reshape[key] = context_outs[key]
                continue
            # (bsz, max_num_pred_mentions, *)
            context_outs_reshape[key] = init_tensor(
                [bs, n_pred_mentions] + list(context_outs[key].size()[1:]),
                context_outs[key].dtype,
                -float("inf") if 'logit' in key else 0,
            )

        for idx in range(len(context_outs['mention_dims'])):
            # reshape
            gpu_bs = context_outs['mention_dims'][idx, 0]
            b_width = context_outs['mention_dims'][idx, 1]

            start_idx = (context_outs['mention_dims'][:idx, 0] * context_outs['mention_dims'][:idx, 1]).sum()
            end_idx = start_idx + b_width * gpu_bs

            s_reshape = context_outs['mention_dims'][:idx, 0].sum()
            e_reshape = s_reshape + gpu_bs
            for key in context_outs_reshape:
                if 'all' in key or key == 'mention_dims':
                    continue
                if len(context_outs[key].size()) == 1:
                    target_tensor = context_outs[key][start_idx:end_idx].view(gpu_bs, b_width)
                else:
                    target_tensor = context_outs[key][start_idx:end_idx].view(gpu_bs, b_width, -1)
                context_outs_reshape[key][s_reshape:e_reshape, :b_width] = target_tensor

        return context_outs_reshape


    def forward(
        self, concatenated_input,
        gold_mention_bounds=None,
        gold_mention_bounds_mask=None,
        return_loss=True,
    ):

        context_outs = self.encode_context(
                concatenated_input, gold_mention_bounds=gold_mention_bounds,
                gold_mention_bounds_mask=gold_mention_bounds_mask,
            )

        mention_logits = context_outs['all_mention_logits']
        mention_bounds = context_outs['all_mention_bounds']
        # print(mention_logits, mention_logits.shape)
        # print(mention_bounds, mention_bounds.shape)
        

        if not return_loss:
            return None, mention_logits, mention_bounds

        '''
        COMPUTE mention LOSS (TRAINING MODE)
        '''
        span_loss = self.get_span_loss(
            gold_mention_bounds=gold_mention_bounds, 
            gold_mention_bounds_mask=gold_mention_bounds_mask,
            mention_logits=mention_logits, mention_bounds=mention_bounds,
        )
        return span_loss, mention_logits, mention_bounds 

    def get_span_loss(
        self, gold_mention_bounds, gold_mention_bounds_mask, mention_logits, mention_bounds,
    ):
        """
        gold_mention_bounds (bs, num_mentions, 2)
        gold_mention_bounds_mask (bs, num_mentions):
        mention_logits (bs, all_mentions)
        menion_bounds (bs, all_mentions, 2)
        """
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")

        gold_mention_bounds[~gold_mention_bounds_mask] = -1  # ensure don't select masked to score
        # triples of [ex in batch, mention_idx in gold_mention_bounds, idx in mention_bounds]
        # use 1st, 2nd to index into gold_mention_bounds, 1st, 3rd to index into mention_bounds
        # print(mention_bounds.unsqueeze(1), mention_bounds.unsqueeze(1).shape)
        # print(gold_mention_bounds.unsqueeze(2), gold_mention_bounds.unsqueeze(2).shape)
        # assert 1==0
        gold_mention_pos_idx = ((
            mention_bounds.unsqueeze(1) - gold_mention_bounds.unsqueeze(2)  # (bs, num_mentions, start_pos * end_pos, 2)
        ).abs().sum(-1) == 0).nonzero()
        # gold_mention_pos_idx should have 1 entry per masked element
        # (num_gold_mentions [~gold_mention_bounds_mask])
        gold_mention_pos = gold_mention_pos_idx[:,2]

        # (bs, total_possible_spans)
        gold_mention_binary = torch.zeros(mention_logits.size(), dtype=mention_logits.dtype).to(gold_mention_bounds.device)
        gold_mention_binary[gold_mention_pos_idx[:,0], gold_mention_pos_idx[:,2]] = 1

        # prune masked spans
        mask = mention_logits != -float("inf")
        masked_mention_logits = mention_logits[mask]
        masked_gold_mention_binary = gold_mention_binary[mask]

        # (bs, total_possible_spans)
        span_loss = loss_fct(masked_mention_logits, masked_gold_mention_binary)

        return span_loss

class TypeClassifier(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypeClassifier, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)
        self.model = self.model.to(self.device)

        self.linear = nn.Linear(params['hidden_size'], params['linear_dim'], bias=False).to(self.device)

        self.similarity_metric = params['similarity_metric']
    
    def load_model(self, fname, cpu=False):
        if cpu or not torch.cuda.is_available():
            state_dict = torch.load(fname, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(fname)
        # self.model.upgrade_state_dict_named(state_dict)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = EncoderModule(self.params)

    # def save_model(self, output_dir):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     model_to_save = get_model_obj(self.model) 
    #     output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    #     output_config_file = os.path.join(output_dir, CONFIG_NAME)
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     model_to_save.config.to_json_file(output_config_file)

    # def get_optimizer(self, optim_states=None, saved_optim_type=None):
    #     return get_bert_optimizer(
    #         [self.model],
    #         self.params["type_optimization"],
    #         self.params["learning_rate"],
    #         fp16=self.params.get("fp16"),
    #     )

    def event_type_encode(self, event_type_input):
        # TODO: check mask
        token_idx, segment_idx, mask = to_bert_input(
            event_type_input, self.NULL_IDX
        )
        et_embed = self.model.encoder.bert_model(token_idx, segment_idx, mask)[0]
        et_embed = self.linear(et_embed)
        et_embed = torch.nn.functional.normalize(et_embed, p=2, dim=2) 
        return et_embed 

    def encode_candidate(self, event_type_input, one_vec=False):
        token_idx, segment_idx, mask = to_bert_input(
            event_type_input, self.NULL_IDX
        )
        et_embed = self.model.encoder.bert_model(token_idx, segment_idx, mask)
        if one_vec:
            et_embed = et_embed[1]
            return et_embed
        else:
            et_embed = et_embed[0]
        et_embed = self.linear(et_embed)
        et_embed = torch.nn.functional.normalize(et_embed, p=2, dim=2) 
        return et_embed 

    def sentence_encode(self, sentence_input):
        token_idx, segment_idx, mask = to_bert_input(
            sentence_input, self.NULL_IDX
        )
        s_embed = self.model.encoder.bert_model(token_idx, segment_idx, mask)[0]
        s_embed = self.linear(s_embed)
        s_embed = torch.nn.functional.normalize(s_embed, p=2, dim=2)

        return  s_embed

    def score(self, et_embed, s_embed):
        if self.similarity_metric == 'cosine':
            et_embed = et_embed.unsqueeze(0)
            s_embed = s_embed.unsqueeze(1)
            # print(et_embed.size(), s_embed.size())
            return (s_embed @ et_embed.permute(0, 1, 3, 2)).max(2).values.sum(2)

        assert self.similarity_metric == 'l2'
        raise NotImplementedError
        return (-1.0 * ((et_embed.unsqueeze(2) - s_embed.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def forward(
        self, sentence_input, event_type_input,
        cand_encs=None,
        label=None,
        margin_label=None,
        return_loss=True,
    ):
        """
        """
        if cand_encs is None:
            scores = self.score(self.event_type_encode(event_type_input), self.sentence_encode(sentence_input))
            # loss_function = nn.MSELoss()
            # loss_function = nn.BCELoss()
            # loss = loss_function(scores, label)

            loss_function = nn.MultiLabelMarginLoss()
            loss = loss_function(scores, margin_label)
            # print(loss)
            # assert 1==0

            return loss, scores 
        else:
            scores = self.score(cand_encs, self.sentence_encode(sentence_input))
            # scores = scores.sigmoid()
            # print(scores, scores.size())
            return 0, scores

class TypeRanking(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypeRanking, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.model = BertModel.from_pretrained(params["bert_model"])
        model_path = params.get("path_to_model", None)
        self.linear = nn.Linear(params['hidden_size'], params['linear_dim'], bias=False)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.linear = self.linear.to(self.device)
        self.similarity_metric = params['similarity_metric']
    
    def load_model(self, fname, cpu=False):
        print(f'load model from {fname}')
        if cpu or not torch.cuda.is_available():
            state_dict = torch.load(fname, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)
        

    # def save_model(self, output_dir):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     model_to_save = get_model_obj(self.model) 
    #     output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    #     output_config_file = os.path.join(output_dir, CONFIG_NAME)
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     model_to_save.config.to_json_file(output_config_file)

    # def get_optimizer(self, optim_states=None, saved_optim_type=None):
    #     return get_bert_optimizer(
    #         [self.model],
    #         self.params["type_optimization"],
    #         self.params["learning_rate"],
    #         fp16=self.params.get("fp16"),
    #     )

    def event_type_encode(self, event_type_input):
        # TODO: check mask
        # token_idx, segment_idx, mask = to_bert_input(
        #     event_type_input, self.NULL_IDX
        # )
        et_embed = self.model(event_type_input)[0]
        et_embed = self.linear(et_embed)
        et_embed = torch.nn.functional.normalize(et_embed, p=2, dim=2) 
        return et_embed 

    def encode_candidate(self, event_type_input, one_vec=False):
        # token_idx, segment_idx, mask = to_bert_input(
        #     event_type_input, self.NULL_IDX
        # )
        et_embed = self.model(event_type_input)
        if one_vec:
            et_embed = et_embed[1]
            return et_embed
        else:
            et_embed = et_embed[0]
        et_embed = self.linear(et_embed)
        et_embed = torch.nn.functional.normalize(et_embed, p=2, dim=2) 
        return et_embed 

    def sentence_encode(self, sentence_input):
        s_embed = self.model(sentence_input)[0]
        s_embed = self.linear(s_embed)
        s_embed = torch.nn.functional.normalize(s_embed, p=2, dim=2)

        return  s_embed

    def score(self, et_embed, s_embed):
        if self.similarity_metric == 'cosine':
            et_embed = et_embed.unsqueeze(0)
            s_embed = s_embed.unsqueeze(1)
            # print(et_embed.size(), s_embed.size())
            return (s_embed @ et_embed.permute(0, 1, 3, 2)).max(2).values.sum(2)

        assert self.similarity_metric == 'l2'
        raise NotImplementedError
        return (-1.0 * ((et_embed.unsqueeze(2) - s_embed.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def forward_new_loss(
        self, sentence_input, event_type_input,
        candidate_label_sets=None,
        negative_smaples=None,
        cand_encs=None,
        return_loss=True,
    ):
        """
        """
        # print(sentence_input, sentence_input.shape)
        # print(event_type_input, event_type_input.shape)
        # assert 1==0
        if cand_encs is None:
            scores = self.score(self.event_type_encode(event_type_input), self.sentence_encode(sentence_input))
            loss = loss_for_type_ranking(scores, candidate_label_sets, negative_smaples, self.device)

            return loss, scores 
        else:
            scores = self.score(cand_encs, self.sentence_encode(sentence_input))
            return 0, scores

    def forward(
        self, sentence_input, event_type_input,
        cand_encs=None,
        label=None,
        margin_label=None,
        return_loss=True,
    ):
        """
        """
        if cand_encs is None:
            scores = self.score(self.event_type_encode(event_type_input), self.sentence_encode(sentence_input))
            # loss_function = nn.MSELoss()
            # loss_function = nn.BCELoss()
            # loss = loss_function(scores, label)

            loss_function = nn.MultiLabelMarginLoss()
            loss = loss_function(scores, margin_label)
            # print(loss)
            # assert 1==0

            return loss, scores 
        else:
            scores = self.score(cand_encs, self.sentence_encode(sentence_input))
            # scores = scores.sigmoid()
            # print(scores, scores.size())
            return 0, scores

class LabelPropagationMLP(nn.Module):
    def __init__(self, params, drop=0.1, negative_slope=0.2, edge_types=3,edge_features=20):
        super().__init__()
        self.hidden_size = params['hidden_size']
        self.mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size * 4, norm=True)
        self.attn_scale = math.sqrt(self.hidden_size)

        self.attn_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_e = nn.Linear(edge_features, self.hidden_size)

        self.drop = nn.Dropout(drop)
        self.feat_activation = nn.GELU()
        self.edge_type_emb = nn.Embedding(edge_types , edge_features)

    def forward(self, g, embedding): # return the graph with propagated scores
        with g.local_scope():
            graph = g.local_var() 
            # print(graph)
            # print(graph.edges())
            # print(graph.ndata)
            # print(graph.edata)
            # assert 1==0
            h = self.mlp(embedding)
            # h = embedding
            # print('*'*10)
            # print('h: ', h, h.shape)
            edge_type_feat = self.feat_activation(self.edge_type_emb(g.edata['type_int']))
            # print('edge_type_feat: ', edge_type_feat, edge_type_feat.shape)

            attn_query = self.attn_l(h) 
            attn_key_1  = self.attn_r(h)
            graph.ndata.update({'attn_q': attn_query, 'attn_k_1': attn_key_1})

            attn_key_2 = self.attn_e(edge_type_feat) 
            # print('attn_query: ', attn_query, attn_query.shape)
            # print('attn_key_1: ', attn_key_1, attn_key_1.shape)
            # print('attn_key_2: ', attn_key_2, attn_key_2.shape)

            graph.edata.update({'attn_k_2': attn_key_2})

            graph.apply_edges(fn.u_add_e('attn_k_1', 'attn_k_2', 'attn_k'))
            # print('attn_k: ', graph.edata['attn_k'], graph.edata['attn_k'].shape)
            graph.apply_edges(fn.v_dot_e('attn_q', 'attn_k','alpha'))
            
            scaled_alpha = graph.edata.pop('alpha') / self.attn_scale
            graph.edata['a'] = self.drop(edge_softmax(graph, scaled_alpha))
            # graph.edata['a'] = edge_softmax(graph, scaled_alpha)
            # print('a: ', graph.edata['a'], graph.edata['a'].shape)
            # update scores
            # print(graph.ndata['scores'])
            # print(graph.edata['a'])
            # graph.apply_edges(fn.v_mul_e('scores', 'a', 'a_scores'))
            # graph.update_all(fn.copy_e('a_scores', 'm'), fn.sum('m', 'updated_scores'))
            graph.update_all(fn.u_mul_e('scores', 'a', 'a_scores'), fn.sum('a_scores', 'updated_scores'))
            candidate_scores = graph.ndata['updated_scores'][graph.ndata['is_cand']]
            # print('a_scores: ', graph.edata['a_scores'], graph.edata['a_scores'].shape)
            # print('updated_scores: ', graph.ndata['updated_scores'], graph.ndata['updated_scores'].shape)
            # print('candidate_scores: ', candidate_scores, candidate_scores.shape)
            
            return graph

class LabelPropagation(nn.Module):
    def __init__(self, params, drop=0.1, negative_slope=0.2, edge_types=3,edge_features=20):
        super().__init__()
        self.hidden_size = params['hidden_size']
        self.attn_scale = math.sqrt(self.hidden_size)

        self.attn_l = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_e = nn.Linear(edge_features, self.hidden_size)

        self.drop = nn.Dropout(drop)
        self.feat_activation = nn.GELU()
        self.edge_type_emb = nn.Embedding(edge_types , edge_features)

    def forward(self, g, embedding): # return the graph with propagated scores
        with g.local_scope():
            graph = g.local_var() 
            # print(graph)
            # print(graph.edges())
            # print(graph.ndata)
            # print(graph.edata)
            # assert 1==0
            h = self.drop(embedding)
            # h = embedding
            # print('*'*10)
            # print('h: ', h, h.shape)
            edge_type_feat = self.feat_activation(self.edge_type_emb(g.edata['type_int']))
            # print('edge_type_feat: ', edge_type_feat, edge_type_feat.shape)

            attn_query = self.attn_l(h) 
            attn_key_1  = self.attn_r(h)
            graph.ndata.update({'attn_q': attn_query, 'attn_k_1': attn_key_1})

            attn_key_2 = self.attn_e(edge_type_feat) 
            # print('attn_query: ', attn_query, attn_query.shape)
            # print('attn_key_1: ', attn_key_1, attn_key_1.shape)
            # print('attn_key_2: ', attn_key_2, attn_key_2.shape)

            graph.edata.update({'attn_k_2': attn_key_2})

            graph.apply_edges(fn.u_add_e('attn_k_1', 'attn_k_2', 'attn_k'))
            # print('attn_k: ', graph.edata['attn_k'], graph.edata['attn_k'].shape)
            graph.apply_edges(fn.v_dot_e('attn_q', 'attn_k','alpha'))
            
            scaled_alpha = graph.edata.pop('alpha') / self.attn_scale
            graph.edata['a'] = self.drop(edge_softmax(graph, scaled_alpha))
            # graph.edata['a'] = edge_softmax(graph, scaled_alpha)
            # print('a: ', graph.edata['a'], graph.edata['a'].shape)
            # update scores
            # print(graph.ndata['scores'])
            # print(graph.edata['a'])
            # graph.apply_edges(fn.v_mul_e('scores', 'a', 'a_scores'))
            # graph.update_all(fn.copy_e('a_scores', 'm'), fn.sum('m', 'updated_scores'))
            graph.update_all(fn.u_mul_e('scores', 'a', 'a_scores'), fn.sum('a_scores', 'updated_scores'))
            candidate_scores = graph.ndata['updated_scores'][graph.ndata['is_cand']]
            # print('a_scores: ', graph.edata['a_scores'], graph.edata['a_scores'].shape)
            # print('updated_scores: ', graph.ndata['updated_scores'], graph.ndata['updated_scores'].shape)
            # print('candidate_scores: ', candidate_scores, candidate_scores.shape)
            
            return graph

class LabelPropagationReplace(nn.Module):
    def __init__(self, params, drop=0.1, negative_slope=0.2, edge_types=3,edge_features=20):
        super().__init__()
        self.hidden_size = params['hidden_size']
        self.mlp = MLP(self.hidden_size, 1, self.hidden_size * 4, norm=True)
        self.drop = nn.Dropout(drop)

    def forward(self, g, embedding): # return the graph with propagated scores
        with g.local_scope():
            print('*'*10)
            graph = g.local_var() 
            h=embedding
            # h = self.drop(embedding)
            print('h: ', h, h.shape)
            scores = self.mlp(h)
            print('scores: ', scores, scores.shape)

            graph.ndata['updated_scores'] = scores
            
            # candidate_scores = graph.ndata['updated_scores'][graph.ndata['is_topk']]
            return graph

class SimpleLabelPropagationNoBert(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )        
        # self.graph_encoder = LabelPropagationReplace(params)
        self.graph_encoder = LabelPropagation(params)
        # self.graph_encoder = LabelPropagationMLP(params)

        self.graph_encoder = self.graph_encoder.to(self.device)

        self.loss_fact = nn.CrossEntropyLoss()
        self.NULL_IDX = 0
    
    def forward(self, g, label):
        b_sz = len(label)
        # print(b_sz)
        embeds = g.ndata['embedding']
        # print(embeds, embeds.shape)
        # for 
        graph = self.graph_encoder(g, embeds)
        graphs = dgl.unbatch(graph)

        loss = 0
        scores = []
        for g, l in zip(graphs, label):
            cur_score = g.ndata['updated_scores'][g.ndata['is_cand']].reshape(-1)
            scores.append(cur_score)
            
            loss += self.loss_fact(cur_score, l)
            # print('loss: ', loss)
            # print(cur_score, l, loss)
        return loss, scores
        # print(scores, scores.shape)
        # scores = scores.reshape(b_sz,-1)
        # # print(scores, scores.shape)
        # # print(label, label.shape)

        # loss = self.loss_fact(scores, label)
        # # print(loss)
        # # assert 1==0
        # return loss, scores

class SimpleLabelPropagation(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        bert = BertModel.from_pretrained(params["bert_model"], output_hidden_states=True)

        self.encoder = BertEncoder(
            bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        
        self.graph_encoder = LabelPropagation(params)

        self.encoder = self.encoder.to(self.device)
        self.graph_encoder = self.graph_encoder.to(self.device)

        self.loss_fact = nn.CrossEntropyLoss()
        self.NULL_IDX = 0
    
    def forward(self, g, label, k=100):
        b_sz = len(label)
        # print(b_sz)
        node_input_ids = g.ndata['tokenized_ids']
        num_node, _  = node_input_ids.size()
        if num_node > k:
            embed_results = []
            for i in range(0, num_node, k):
                tmp_ids = node_input_ids[i:i + k, :]
                token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                    tmp_ids, self.NULL_IDX
                )
                tmp_embeds = self.encoder(token_idx_cands, segment_idx_cands, mask_cands)
                embed_results.append(tmp_embeds)
            embeds = torch.cat(embed_results, dim=0)
        else:
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                node_input_ids, self.NULL_IDX
            )
            embeds = self.encoder(token_idx_cands, segment_idx_cands, mask_cands)
        graph = self.graph_encoder(g, embeds)
        graphs = dgl.unbatch(graph)

        loss = 0
        scores = []
        for g, l in zip(graphs, label):
            cur_score = g.ndata['updated_scores'][g.ndata['is_cand']].reshape(-1)
            scores.append(cur_score)
            
            loss += self.loss_fact(cur_score, l)
            # print('loss: ', loss)
            # print(cur_score, l, loss)
        return loss, scores

class NodeSelectionModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
                
        heads = [self.params['n_heads'], ] * self.params['n_layers']
        self.graph_encoder = EdgeAwareGAT(
            num_layers=self.params['n_layers'], 
            in_dim=self.params['hidden_size'],
            num_hidden=self.params['hidden_size'],
            num_classes=self.params['hidden_size'],
            heads=heads,
            edge_types=2,
            edge_features=20,
            node_types=2,
            node_features=20,
            activation=F.gelu,
            feat_drop=.1,
            attn_drop=.1,
            residual=True
        )
        self.graph_encoder = self.graph_encoder.to(self.device)

        self.predictor = ClassificationHead(self.params['hidden_size'], num_labels=1, dropout_prob=.1)
        self.predictor = self.predictor.to(self.device)

        self.loss_fact = nn.CrossEntropyLoss()

    def freeze_model(self, module):
        for param in list(module.parameters()):
            param.requires_grad = False
        return 
    

    def predict(self, inputs):
        '''
        A forward only pass for response generation.
        '''
        pass 
    def forward(self, g, label, return_loss=True) -> torch.Tensor:
        b_sz = len(label)
        feature = g.ndata['embedding']

        # select one from the following two lines
        node_embeddings = self.graph_encoder(g, feature) #
        # node_embeddings = feature

        # graph_list = dgl.unbatch(g)
        sentence_mask = g.ndata['is_sent']
        sentence_embedding = node_embeddings[sentence_mask]
        scores = self.predictor(sentence_embedding)
        scores = scores.reshape(b_sz,-1)

        logits = self.loss_fact(scores, label)


        return logits, scores

class EncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(EncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)
        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu or not torch.cuda.is_available():
            state_dict = torch.load(fname, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(fname)
        
        self.model.upgrade_state_dict_named(state_dict)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = EncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(
        self, cands, gold_mention_bounds=None, gold_mention_bounds_mask=None,
        num_cand_mentions=50, topK_threshold=-4.5,
        get_mention_scores=True,
    ):
        """
        if gold_mention_bounds specified, selects according to gold_mention_bounds,
        otherwise selects according to top-scoring mentions

        Returns: Dictionary
            mention_reps: torch.FloatTensor (bsz, max_num_pred_mentions, embed_dim): mention embeddings
            mention_masks: torch.BoolTensor (bsz, max_num_pred_mentions): mention padding mask
            mention_bounds: torch.LongTensor (bsz, max_num_pred_mentions, 2)
            (
            mention_logits: torch.FloatTensor (bsz, max_num_pred_mentions): mention scores/logits
            all_mention_mask: torch.BoolTensor ((bsz, all_cand_mentions)
            all_mention_logits: torch.FloatTensor (bsz, all_cand_mentions): all mention scores/logits
            all_mention_bounds: torch.LongTensor (bsz, all_cand_mentions, 2): all mention bounds
            )
        """
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        context_outs, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands,
            None, None, None,
            gold_mention_bounds=gold_mention_bounds,
            gold_mention_bounds_mask=gold_mention_bounds_mask,
            num_cand_mentions=num_cand_mentions,
            topK_threshold=topK_threshold,
            get_mention_scores=get_mention_scores
        )
        if context_outs['mention_dims'].size(0) <= 1:
            for key in context_outs:
                if 'all' in key or key == 'mention_dims':
                    continue
                context_outs[key] = context_outs[key].view([context_outs['mention_dims'][0,0], -1] + list(context_outs[key].size()[1:]))
            return context_outs

        '''
        Reshape to (bs, num_mentions, *), iterating across GPUs
        '''
        def init_tensor(shape, dtype, init_value):
            return init_value * torch.ones(
                shape
            ).to(dtype=dtype, device=context_outs['mention_dims'].device)

        bs = cands.size(0)
        n_pred_mentions = context_outs['mention_dims'][:,1].max()
        context_outs_reshape = {}
        for key in context_outs:
            if 'all' in key or key == 'mention_dims':
                context_outs_reshape[key] = context_outs[key]
                continue
            # (bsz, max_num_pred_mentions, *)
            context_outs_reshape[key] = init_tensor(
                [bs, n_pred_mentions] + list(context_outs[key].size()[1:]),
                context_outs[key].dtype,
                -float("inf") if 'logit' in key else 0,
            )

        for idx in range(len(context_outs['mention_dims'])):
            # reshape
            gpu_bs = context_outs['mention_dims'][idx, 0]
            b_width = context_outs['mention_dims'][idx, 1]

            start_idx = (context_outs['mention_dims'][:idx, 0] * context_outs['mention_dims'][:idx, 1]).sum()
            end_idx = start_idx + b_width * gpu_bs

            s_reshape = context_outs['mention_dims'][:idx, 0].sum()
            e_reshape = s_reshape + gpu_bs
            for key in context_outs_reshape:
                if 'all' in key or key == 'mention_dims':
                    continue
                if len(context_outs[key].size()) == 1:
                    target_tensor = context_outs[key][start_idx:end_idx].view(gpu_bs, b_width)
                else:
                    target_tensor = context_outs[key][start_idx:end_idx].view(gpu_bs, b_width, -1)
                context_outs_reshape[key][s_reshape:e_reshape, :b_width] = target_tensor

        return context_outs_reshape

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None,
            token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands

    def label_propogation(self, embedding_cands, embedding_neighbor, neighbor_mask, neighbor_relations, alpha = 0.5):
        print(embedding_cands.shape) 
        # torch.Size([4, 768])
        print(embedding_neighbor.shape) 
        # torch.Size([4, 11, 768])
        print(neighbor_mask.shape) 
        # torch.Size([4, 11])
        print(neighbor_relations.shape) 
        # torch.Size([4, 11])

        embedding_neighbor[~neighbor_mask] = 0
        # print(neighbor_mask)
        # print(embedding_neighbor, embedding_neighbor.shape)
        embedding_neighbor = embedding_neighbor.sum(dim=1)
        # print(embedding_neighbor, embedding_neighbor.shape) torch.Size([4, 768])
        neighbor_mask = neighbor_mask.sum(dim=-1).view(neighbor_mask.shape[0], -1).to(torch.float)
        # print(neighbor_mask, neighbor_mask.shape)
        embedding_neighbor = embedding_neighbor/neighbor_mask
        print(embedding_neighbor, embedding_neighbor.shape)

        embedding_cands += embedding_neighbor*alpha
        # print(embedding_cands, embedding_cands.shape)
        # torch.Size([4, 768])
        

        return embedding_cands

    # Score candidates given context input and label input
    # If text_encs/cand_encs is provided (pre-computed), text_vecs/cand_vecs is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        text_encs=None,  # pre-computed mention encoding
        cand_encs=None,  # pre-computed candidate encoding.
        cand_mask=None,
        gold_mention_bounds=None,
        gold_mention_bounds_mask=None,
        num_cand_mentions=50,
        mention_threshold=-4.5,
        get_mention_scores=True,
        hard_negs=False,  # (if training) passed in a subset of hard negatives
        hard_negs_mask=None,  # (if hard negs training) mask for gold candidate mentions on all inputs (pos + negs)
        # neighbor_n ask=None,
    ):
        """
        text_vecs (bs, max_ctxt_size):
        cand_vecs (bs, max_num_gold_mentions, 1, max_cand_size):
        text_encs (batch_num_mentions, embed_size): Pre-encoded mention vectors, masked before input
        cand_encs (num_ents_to_match [batch_num_total_ents/all_ents], embed_size): Pre-encoded candidate vectors, masked before input
        """
        '''
        Compute context representations and/or get mention scores
        '''
        if text_encs is None or get_mention_scores:
            # embedding_ctxt: (bs, num_gold_mentions/num_pred_mentions, embed_size)
            context_outs = self.encode_context(
                text_vecs, gold_mention_bounds=gold_mention_bounds,
                gold_mention_bounds_mask=gold_mention_bounds_mask,
                num_cand_mentions=num_cand_mentions,
                topK_threshold=mention_threshold,
                get_mention_scores=get_mention_scores,
            )

        mention_logits = None
        mention_bounds = None
        if get_mention_scores:
            mention_logits = context_outs['all_mention_logits']
            mention_bounds = context_outs['all_mention_bounds']

        if text_encs is None:
            if gold_mention_bounds is None:
                # (all_batch_pred_mentions, embed_size)
                embedding_ctxt = context_outs['mention_reps'][context_outs['mention_masks']]
            else:
                # (all_batch_pred_mentions, embed_size)
                embedding_ctxt = context_outs['mention_reps'][gold_mention_bounds_mask]
        else:
            # Context encoding is given, do not need to re-compute
            embedding_ctxt = text_encs

        '''
        Compute candidate representations
        '''
        if cand_encs is None:
            # Train time: Compute candidates in batch and compare in-batch negatives
            # cand_vecs: (bs, num_gold_mentions, num_nodes, cand_width) -> (batch_num_gold_mentions, cand_width)
            cand_vecs = cand_vecs[cand_mask].squeeze(1)
            # (batch_num_gold_mentions, embed_dim)
            # print(cand_vecs, cand_vecs.shape) 
            # torch.Size([4, 128])
            embedding_cands = self.encode_candidate(cand_vecs)
            # print("embedding_cands: ",embedding_cands, embedding_cands.shape)

            # neighbor_mask_label_level = neighbor_mask.any(dim=-1)
            # print(neighbor_mask_label_level, neighbor_mask_label_level.shape) 
            # # torch.Size([2, 14])
            # neighbor_nodes = neighbor_nodes[neighbor_mask_label_level]
            # bs, max_label, _ = neighbor_nodes.shape
            # neighbor_nodes = torch.reshape(neighbor_nodes,(bs * max_label,-1))
            # print("neighbor_nodes: ",neighbor_nodes,neighbor_nodes.shape)
            # # torch.Size([4, 11, 128]) torch.Size([44, 128])
            
            # # (batch_num_gold_mentions, max_num_neighbors)
            # embedding_neighbor = self.encode_candidate(neighbor_nodes)
            # print("embedding_neighbor: ", embedding_neighbor,embedding_neighbor.shape)
            # # torch.Size([44, 768])
            # embedding_neighbor = torch.reshape(embedding_neighbor, (bs, max_label,-1))


            # neighbor_mask = neighbor_mask[neighbor_mask_label_level]
            # print(neighbor_mask, neighbor_mask.shape) 
            # # torch.Size([4, 11])
            # neighbor_relations = neighbor_relations[neighbor_mask_label_level]

            # embedding_cands = self.label_propogation(embedding_cands, embedding_neighbor, neighbor_mask, neighbor_relations)
        else:
            # (batch_num_gold_mentions, embed_dim)
            embedding_cands = cand_encs
        '''
        Do inner-product search, or obtain scores on hard-negative entities
        '''
        # matmul across all cand_encs (in-batch, if cand_encs is None, or across all cand_encs)
        # (all_batch_pred_mentions, num_cands)
        # similarity score between ctxt i and cand j
        # print(embedding_ctxt.shape) torch.Size([3, 768])
        # print(embedding_cands.shape) torch.Size([4, 768])
        all_scores = embedding_ctxt.mm(embedding_cands.t())
            
        return all_scores, mention_logits, mention_bounds


    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(
        self, context_input, cand_input,
        text_encs=None,  # pre-computed mention encoding.
        cand_encs=None,  # pre-computed candidate embeddings 
        mention_logits=None,  # pre-computed mention logits
        mention_bounds=None,  # pre-computed mention bounds
        label_input=None,  # labels for passed-in (if hard negatives training)
        gold_mention_bounds=None,
        gold_mention_bounds_mask=None,
        cand_mask=None,
        # neighbor_nodes=None,
        # neighbor_relations=None,
        # neighbor_mask=None,
        label_probability=None,
        return_loss=True,
    ):
        """
        text_encs/cand_encs/label_inputs masked before training
        In-batch negs training: cand_encs None, label_inputs None, return_loss True
        Hard negs training: cand_encs non-None, label_inputs non-None, return_loss True
            cand_encs = all entities in batch + additional hard negatives
        Inference: cand_encs non-None, label_inputs None, return_loss False
            cand_encs = all entities in DB

        cand_encs
           non-None: set of candidate encodings to search in
           None: compute in-batch candidate vectors (used as negatives if train mode)
        label_inputs
           non-None: labels to use for hard negatives training
           None: random negatives training and/or inference
        """
        hard_negs = label_input is not None

        '''
        GET CANDIDATE SCORES
        '''
        scores, out_mention_logits, out_mention_bounds = self.score_candidate(
            context_input, cand_input,
            hard_negs=hard_negs,
            cand_encs=cand_encs,
            cand_mask=cand_mask,
            text_encs=text_encs,
            # neighbor_nodes=neighbor_nodes,
            # neighbor_relations=neighbor_relations,
            # neighbor_mask=neighbor_mask,
            gold_mention_bounds=gold_mention_bounds,
            gold_mention_bounds_mask=gold_mention_bounds_mask,
            get_mention_scores=True,
            # get_mention_scores=(return_loss and (mention_logits is None or mention_bounds is None)),
        )

        if mention_logits is None:
            mention_logits = out_mention_logits
        if mention_bounds is None:
            mention_bounds = out_mention_bounds

        if not return_loss:
            return None, scores, mention_logits, mention_bounds

        '''
        COMPUTE mention LOSS (TRAINING MODE)
        '''
        span_loss = 0
        if mention_logits is not None and mention_bounds is not None:
            N = context_input.size(0)  # batch size
            M = gold_mention_bounds.size(1)  # num_mentions per instance (just 1, so far)
            # 1 value
            span_loss = self.get_span_loss(
                gold_mention_bounds=gold_mention_bounds, 
                gold_mention_bounds_mask=gold_mention_bounds_mask,
                mention_logits=mention_logits, mention_bounds=mention_bounds,
            )

        # print(scores, scores.shape)
        # tensor([[16.9521, 18.3104, 15.9726, 11.8880],
        # [18.7696, 21.7191, 37.6406, 36.7603],
        # [21.4338, 21.7517, 42.0159, 26.1037]], device='cuda:0') torch.Size([3, 4])

    #     print(cand_mask, cand_mask.shape)
    #     tensor([[[ True,  True, False, False, False, False, False, False],
    #      [False, False, False, False, False, False, False, False],
    #      [False, False, False, False, False, False, False, False],
    #      [False, False, False, False, False, False, False, False],
    #      [False, False, False, False, False, False, False, False]],

    #     [[ True, False, False, False, False, False, False, False],
    #      [ True, False, False, False, False, False, False, False],
    #      [False, False, False, False, False, False, False, False],
    #      [False, False, False, False, False, False, False, False],
    #      [False, False, False, False, False, False, False, False]]],
    #    device='cuda:0') torch.Size([2, 5, 8])

        # scores: (bs*num_mentions [filtered], bs*num_mentions [filtered])
        # print("*"*10)
        # print(label_probability,label_probability.shape)
        # print(cand_mask, cand_mask.shape)
        label_probability = label_probability[cand_mask]

        # print(label_probability,label_probability.shape)
        # print(scores,scores.shape)
        target = torch.zeros_like(scores)
        j = 0
        i = 0
        for mention in cand_mask:
            for label in mention:
                flag = 0
                for m in label:
                    if m:
                        target[i][j] = label_probability[j]
                        j += 1
                        flag = 1
                if flag:
                    i += 1
        # print(target, target.shape)
                
        # for j, p in enumerate(label_probability):
        #     print(i,j)
        #     target[i][j] = p
        #     sum_p += p
        #     if sum_p == 1:
        #         sum_p = 0
        #         i += 1

        loss_fact = softCrossEntropy()

        # log P(entity|mention) + log P(mention) = log [P(entity|mention)P(mention)]
        loss = loss_fact(scores, target) + span_loss

        return loss, scores, mention_logits, mention_bounds 

    def get_span_loss(
        self, gold_mention_bounds, gold_mention_bounds_mask, mention_logits, mention_bounds,
    ):
        """
        gold_mention_bounds (bs, num_mentions, 2)
        gold_mention_bounds_mask (bs, num_mentions):
        mention_logits (bs, all_mentions)
        menion_bounds (bs, all_mentions, 2)
        """
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")

        gold_mention_bounds[~gold_mention_bounds_mask] = -1  # ensure don't select masked to score
        # triples of [ex in batch, mention_idx in gold_mention_bounds, idx in mention_bounds]
        # use 1st, 2nd to index into gold_mention_bounds, 1st, 3rd to index into mention_bounds
        gold_mention_pos_idx = ((
            mention_bounds.unsqueeze(1) - gold_mention_bounds.unsqueeze(2)  # (bs, num_mentions, start_pos * end_pos, 2)
        ).abs().sum(-1) == 0).nonzero()
        # gold_mention_pos_idx should have 1 entry per masked element
        # (num_gold_mentions [~gold_mention_bounds_mask])
        gold_mention_pos = gold_mention_pos_idx[:,2]

        # (bs, total_possible_spans)
        gold_mention_binary = torch.zeros(mention_logits.size(), dtype=mention_logits.dtype).to(gold_mention_bounds.device)
        gold_mention_binary[gold_mention_pos_idx[:,0], gold_mention_pos_idx[:,2]] = 1

        # prune masked spans
        mask = mention_logits != -float("inf")
        masked_mention_logits = mention_logits[mask]
        masked_gold_mention_binary = gold_mention_binary[mask]

        # (bs, total_possible_spans)
        span_loss = loss_fct(masked_mention_logits, masked_gold_mention_binary)

        return span_loss

def to_bert_input(token_idx, null_idx):
    """
    token_idx is a 2D tensor int.
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
