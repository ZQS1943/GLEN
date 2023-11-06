import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import (
    BertModel,
    BertForMaskedLM
)
from pytorch_transformers.tokenization_bert import BertTokenizer
from .allennlp_span_utils import batched_span_select

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


class mentionScoresHead(nn.Module):
    def __init__(
        self, bert_output_dim, max_mention_length=10, 
    ):
        super(mentionScoresHead, self).__init__()
        self.max_mention_length = max_mention_length
        self.bound_classifier = nn.Linear(bert_output_dim, 3)

    def forward(self, bert_output, mask_ctxt):
        '''
        Retuns scores for *inclusive* mention boundaries
        '''
        # (bs, seqlen, 3)
        logits = self.bound_classifier(bert_output)

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
        self, bert_model
    ):
        super(BertEncoder, self).__init__()
        self.bert_model = bert_model

    def forward(self, token_ids, segment_ids, attention_mask):
        try:
            output_bert, _, _ = self.bert_model(
                token_ids, segment_ids, attention_mask
            )
        except RuntimeError as e:
            print(token_ids.size())
            print(segment_ids.size())
            print(attention_mask.size())
            print(e)
            import pdb
            pdb.set_trace()
            output_bert, _, _ = self.bert_model(
                token_ids, segment_ids, attention_mask
            )

        return output_bert[:, 0, :]

class EncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(EncoderModule, self).__init__()
        bert = BertModel.from_pretrained(params["bert_model"], output_hidden_states=True)

        self.encoder = BertEncoder(
            bert
        )

        self.config = bert.config

        bert_output_dim = bert.embeddings.word_embeddings.weight.size(1)

        self.classification_heads = nn.ModuleDict({})
        self.linear_compression = None

        classification_heads_dict = {'get_context_embeds': GetContextEmbedsHead(
            bert_output_dim,
            bert_output_dim,
        )}
        classification_heads_dict['mention_scores'] = mentionScoresHead(
            bert_output_dim,
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

class TypeClassifier(torch.nn.Module):
    def __init__(self, params, model_path=None):
        super(TypeClassifier, self).__init__()
        self.params = params
        self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"] 
        )
        self.yes_id = self.tokenizer.convert_tokens_to_ids(['yes'])[0]
        self.no_id = self.tokenizer.convert_tokens_to_ids(['no'])[0]
        # init model
        self.model = BertForMaskedLM.from_pretrained(params["bert_model"])
        model_path = params.get("path_to_model", None)
        ckpt_path = params.get("path_to_ckpt", None)
        if ckpt_path is not None:
            model_path = os.path.join(ckpt_path, "type_classifier.bin")
        if model_path is not None:
            print(f'load model from {model_path} ...')
            self.load_state_dict(torch.load(model_path))

        self.model = self.model.to(self.device)
        self.loss_fact = nn.BCELoss()

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

class TriggerIdentifier(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TriggerIdentifier, self).__init__()
        self.params = params
        self.device = torch.device("cuda")
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"] 
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        ckpt_path = params.get("path_to_ckpt", None)
        if ckpt_path is not None:
            model_path = os.path.join(ckpt_path, "trigger_identifier.bin")
        if model_path is not None:
            print(f'load model from {model_path} ...')
            self.load_state_dict(torch.load(model_path))

        self.model = self.model.to(self.device)
    
    def build_model(self):
        self.model = EncoderModule(self.params)

    def encode_context(
        self, cands, gold_mention_bounds=None, gold_mention_bounds_mask=None,
        num_cand_mentions=50, topK_threshold=-4.5,
        get_mention_scores=True,
    ):
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
        

        if not return_loss:
            return None, mention_logits, mention_bounds


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

class TypeRanking(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypeRanking, self).__init__()
        self.params = params
        self.device = torch.device("cuda")
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"] 
        )
        # init model
        self.model = BertModel.from_pretrained(params["bert_model"])
        self.linear = nn.Linear(params['hidden_size'], params['linear_dim'], bias=False)
        model_path = params.get("path_to_model", None)
        ckpt_path = params.get("path_to_ckpt", None)
        if ckpt_path is not None:
            model_path = os.path.join(ckpt_path, "type_ranking.bin")
        if model_path is not None:
            print(f'load model from {model_path} ...')
            self.load_state_dict(torch.load(model_path))

        

        self.model = self.model.to(self.device)
        self.linear = self.linear.to(self.device)


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
        et_embed = et_embed.unsqueeze(0)
        s_embed = s_embed.unsqueeze(1)
        # print(et_embed.size(), s_embed.size())
        return (s_embed @ et_embed.permute(0, 1, 3, 2)).max(2).values.sum(2)

    def forward(
        self, sentence_input, event_type_input,
        candidate_label_sets=None,
        negative_smaples=None,
        cand_encs=None,
    ):
        """
        """
        if cand_encs is None:
            scores = self.score(self.event_type_encode(event_type_input), self.sentence_encode(sentence_input))
            loss = loss_for_type_ranking(scores, candidate_label_sets, negative_smaples, self.device)

            return loss, scores 
        else:
            scores = self.score(cand_encs, self.sentence_encode(sentence_input))
            return 0, scores

def to_bert_input(token_idx, null_idx):
    """
    token_idx is a 2D tensor int.
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
