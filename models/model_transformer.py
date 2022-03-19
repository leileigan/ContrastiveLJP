#-*- coding:utf-8 _*-  
# @Author: Leilei Gan
# @Time: 2020/11/04
# @Contact: 11921071@zju.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import datetime
import utils.config
import random
from torch.autograd import Variable


BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

def positional_embedding(x, min_timescale=1.0, max_timescale=1.0e4):
    """ positional embedding """

    batch, length, channels = list(x.size())
    assert (channels % 2 == 0)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1.))

    position = torch.arange(0, length).float()
    inv_timescales = torch.arange(0, num_timescales).float()
    if x.is_cuda:
        position = position.cuda()
        inv_timescales = inv_timescales.cuda()

    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    scaled_time = position.unsqueeze(1).expand(
        length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
    # scaled time is now length x num_timescales
    # length x channels
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)

    return signal.unsqueeze(0).expand(batch, length, channels)


class FNNLayer(nn.Module):
    """ FNN layer """

    def __init__(self, hidden_size, inner_linear, relu_dropout=0.1):
        super(FNNLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(hidden_size, inner_linear),
                                   nn.ReLU(inplace=True), nn.Dropout(relu_dropout))
        self.conv2 = nn.Linear(inner_linear, hidden_size)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        return outputs


class ResidualConnection(nn.Module):
    """ residual connection """

    def __init__(self, residual_dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, x, y):
        return x + self.dropout(y)


class LayerNormalization(nn.Module):
    """ layer normalization """

    def __init__(self, num_features, eps=1e-6, affine=True):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.)
            self.bias.data.fill_(0.)

    def forward(self, inputs):
        b, t, _ = list(inputs.size())
        mean = inputs.mean(2).view(b, t, 1).expand_as(inputs)
        input_centered = inputs - mean
        std = input_centered.pow(2).mean(2).add(self.eps).sqrt()
        output = input_centered / std.view(b, t, 1).expand_as(inputs)

        if self.affine:
            w = self.weight.view(1, 1, -1).expand_as(output)
            b = self.bias.view(1, 1, -1).expand_as(output)
            output = output * w + b

        return output


class ScaledDotProductAttention(nn.Module):
    """ scaled dot-product attention """

    def __init__(self, hidden_size, num_heads, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = (hidden_size // num_heads) ** -0.5
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, attn_mask=None):
        '''

        :param q: (batch_size x head, seq_len, emb_size/head)
        :param k: (batch_size x head, seq_len, emb_size/head)
        :param v: (batch_size x head, seq_len, emb_size/head)
        :param attn_mask:
        :return: (batch_size x head, seq_len, emb_size/head)
        '''
        q = q * self.scale
        attn = torch.bmm(q, k.transpose(1, 2))  # result: (batch_size x head , seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            #  attn.data.masked_fill_(attn_mask, -float('inf'))
            attn.data.masked_fill_(attn_mask, -1e9)

        attn = F.softmax(attn.view(-1, k.size(1)), dim=-1).view(-1, q.size(1), k.size(1))
        # attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v)

        return outputs


class MultiHeadAttention(nn.Module):
    """ multi-head attention """

    def __init__(self, hidden_size, num_heads, attention_dropout=0.1, share=True):
        super(MultiHeadAttention, self).__init__()
        self.share = share
        self.num_heads = num_heads

        if share:
            self.w_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        else:
            self.w_q = nn.Linear(hidden_size, hidden_size, bias=True)
            self.w_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

        self.attention = ScaledDotProductAttention(hidden_size, num_heads, attention_dropout)
        self.prj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, q, k, v, attn_mask=None):

        if self.share:
            qs, ks, vs = torch.split(self.w_qkv(q), split_size_or_sections=q.size(-1), dim=-1)
            #  qs: (batch_size, seq_len, emb_size)
        else:
            qs = self.w_q(q)
            ks, vs = torch.split(self.w_kv(k), split_size_or_sections=k.size(-1), dim=-1)

        # split and concat
        q_ = torch.cat(torch.chunk(qs, self.num_heads, dim=-1), dim=0)  # (h*B, L, C/h)
        k_ = torch.cat(torch.chunk(ks, self.num_heads, dim=-1), dim=0)  # (h*B, L, C/h)
        v_ = torch.cat(torch.chunk(vs, self.num_heads, dim=-1), dim=0)  # (h*B, L, C/h)

        outputs = self.attention.forward(q_, k_, v_, attn_mask=attn_mask.repeat(self.num_heads, 1, 1))
        # (h*B, L, C/h)
        outputs = torch.cat(torch.split(outputs, qs.size(0), dim=0), dim=-1)
        # (h, L, C)
        outputs = self.prj(outputs)

        return outputs


class EncoderBlock(nn.Module):
    """ encoder block """

    def __init__(self, hidden_size, num_heads, inner_linear, attention_dropout,
                 residual_dropout, relu_dropout):
        super(EncoderBlock, self).__init__()

        self.layer_norm1 = LayerNormalization(hidden_size)
        self.layer_norm2 = LayerNormalization(hidden_size)

        self.residual_connection1 = ResidualConnection(residual_dropout)
        self.residual_connection2 = ResidualConnection(residual_dropout)

        self.self_attention = MultiHeadAttention(hidden_size, num_heads, attention_dropout, share=True)
        self.fnn_layer = FNNLayer(hidden_size, inner_linear, relu_dropout)

    def forward2(self, inputs, self_attn_mask):

        residual = inputs
        self_attn_outputs = self.self_attention.forward(inputs, inputs, inputs, self_attn_mask)
        sub_layer_outputs = self.layer_norm1.forward(self.residual_connection1.forward(residual, self_attn_outputs))

        residual = sub_layer_outputs
        fnn_layer_outputs = self.fnn_layer.forward(sub_layer_outputs)
        sub_layer_outputs = self.layer_norm2.forward(self.residual_connection2.forward(residual, fnn_layer_outputs))

        return sub_layer_outputs

    def forward(self, inputs, self_attn_mask, word_context=None):

        residual = inputs

        self_attn_outputs = self.self_attention.forward(inputs, inputs, inputs, self_attn_mask)
        sub_layer_outputs = self.layer_norm1.forward(self.residual_connection1.forward(residual, self_attn_outputs))

        residual = sub_layer_outputs
        # add matched word context
        if word_context is not None:
            sub_layer_outputs = torch.cat((sub_layer_outputs, word_context), dim=2)

        fnn_layer_outputs = self.fnn_layer.forward(sub_layer_outputs)
        sub_layer_outputs = self.layer_norm2.forward(self.residual_connection2.forward(residual, fnn_layer_outputs))

        return sub_layer_outputs


def get_attn_padding_mask(seq_q, seq_k, pad):
    """ indicate the padding-related part to mask """

    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(pad).unsqueeze(1).expand(mb_size, len_q, len_k)
    subsequent_padding_mask = seq_q.data.eq(pad).unsqueeze(2).expand(mb_size, len_q, len_k)

    self_attn_mask = pad_attn_mask | subsequent_padding_mask
    return self_attn_mask


class TransformerEncoder(nn.Module):
    """ transformer encoder """

    def __init__(self, context_size=250, hidden_size=256, num_layers=4, num_heads=4, inner_linear=1024,
                 embed_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1, relu_dropout=0.1, pad=0):
        super(TransformerEncoder, self).__init__()
        print("self attention network hyper parameters: ")
        print("hidden size: %d" % hidden_size)
        print("num layers: %d" % num_layers)
        print("num heads: %d" % num_heads)
        print("inner linear size: %d" % inner_linear)
        print("embed dropout rate: %.2f" % embed_dropout)
        print("residual dropout rate: %.2f" % residual_dropout)
        print("attention dropout rate: %.2f" % attention_dropout)
        print("relu dropout rate: %.2f" % relu_dropout)
        self.dropout = nn.Dropout(embed_dropout)
        self.pad = pad
        self.W = nn.Linear(context_size, hidden_size)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(hidden_size, num_heads, inner_linear, attention_dropout,
                                                          residual_dropout, relu_dropout) for _ in range(num_layers)])

    def forward(self, x, src_seq):
        """ input include attend words """
        self_attn_mask = get_attn_padding_mask(src_seq, src_seq, self.pad)  # result: (mb_size, len_q, len_k)

        position_embedding = Variable(positional_embedding(x), requires_grad=False)
        x = x + position_embedding
        encoder_inputs = self.W(self.dropout(x))

        enc_outputs = encoder_inputs
        for enc_layer in self.encoder_blocks:
            enc_outputs = enc_layer.forward(enc_outputs, self_attn_mask)

        return enc_outputs


class LawModel(nn.Module):

    def __init__(self, config: utils.config.Data):
        super(LawModel, self).__init__()

        self.word_dim = config.word_emb_dim
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_embedding_layer = torch.nn.Embedding(config.word_alphabet_size, config.word_emb_dim, padding_idx=0)
        self.use_logical = True

        if config.pretrain_word_embedding is not None:
            self.word_embedding_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embedding_layer.weight.requires_grad = False
        else:
            self.word_embedding_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.lstm_layer1 = TransformerEncoder(self.word_dim, hidden_size=self.hidden_dim * 2)
        self.lstm_layer2 = nn.LSTM(self.hidden_dim * 8, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.claim_classifier = torch.nn.Linear(self.hidden_dim * 10, 3)
        self.claim_dropout = torch.nn.Dropout(0.2)
        self.claim_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)

        self.attn_c = nn.Parameter(torch.Tensor(self.hidden_dim * 10, 1).cuda())

        nn.init.uniform_(self.attn_c, -0.1, 0.1)

        self.use_logical = False
        print('use logical:', self.use_logical)


    def att_flow_layer(self, doc_rep, claim_rep):
        """
        :param doc_rep: [batch_size, max_doc_seq_lens, hidden_dim * 2]
        :param claim_rep: [batch_size * max_claims_num, max_claim_seq_lens, hidden_dim * 2]
        :return:
        """
        batch_size, max_doc_seq_len, hidden_dim= doc_rep.size()

        max_claim_seq_len = claim_rep.size(1)
        claim_rep = claim_rep.contiguous().view(batch_size, -1, hidden_dim) # [batch_size, max_claims_num * max_claim_seq_lens, hidden_dim * 2]

        if max_doc_seq_len < max_claim_seq_len:
            doc_rep = torch.cat([doc_rep, torch.zeros(batch_size, max_claim_seq_len - max_doc_seq_len, hidden_dim).cuda()], dim=1)
            max_doc_seq_len = doc_rep.size(1)

        ### beging debug
        claim_doc_sims = torch.bmm(claim_rep, doc_rep.transpose(1, 2))  # [batch_size, max_claims_num * max_claim_seq_lens, max_doc_seq_lens]
        matrix = claim_doc_sims.view(batch_size, -1, max_claim_seq_len, max_doc_seq_len)
        # print('claim2 matrix:', matrix[1, :8, :148].cpu().tolist())
        claim_to_doc_attn = torch.bmm(F.softmax(claim_doc_sims, dim=2), doc_rep).view(-1, max_claim_seq_len, hidden_dim)  # [batch_size * max_claims_num, max_claim_seq_lens, hidden_dim * 2]


        # doc_to_claim signifies which claim words are most relevant to each doc words.
        top_context_words, _ = torch.sort(claim_doc_sims, dim=2, descending=True) # [batch_size, max_claims_num * max_claim_seq_lens, max_doc_seq_lens]
        top_context_words = top_context_words.view(-1, max_claim_seq_len, max_doc_seq_len) # [batch_size * max_claims_num, max_claims_seq_lens, max_doc_seq_lens]
        claim_rep = claim_rep.view(-1, max_claim_seq_len, hidden_dim) #[batch_size * max_claims_num, max_claim_seq_lens, hidden_dim * 2]
        doc_to_claim_attn = torch.bmm(F.softmax(top_context_words[:,:,:max_claim_seq_len].transpose(1, 2), dim=2), claim_rep)  # [batch_size * max_claims_num, max_claim_seq_lens, hidden_dim * 2]

        return claim_to_doc_attn, doc_to_claim_attn # [batch_size * max_claims_num, max_claims_seq_lens, hidden_dim * 2], [batch_size * max_claims_num, max_claims_seq_lens, hidden_dim * 2]


    def claims_logical_operator(self, claim_logits, claim_softmax_attns, claims_num, claims_type, rate_fact_labels, claim_predicts, facts_y):

        # claim_logits: [batch_size, max_claims_num, 3]

        mask = torch.zeros_like(claim_logits)

        for idx, num in enumerate(claims_num.long().tolist()):
            # Rule1: 其余诉请为2 -> 诉讼费为2

            if '诉讼费' in claims_type[idx] and len(claims_type[idx]) > 1:
                ss_index = claims_type[idx].index('诉讼费')
                if 0 not in claim_predicts[idx, :ss_index].cpu().tolist() and 1 not in claim_predicts[idx, :ss_index].cpu().tolist() \
                    and 0 not in claim_predicts[idx, ss_index+1:].cpu().tolist() and 1 not in claim_predicts[idx, ss_index+1:].cpu().tolist():
                    mask[idx, ss_index, 2] = max(0, claim_softmax_attns[idx, :ss_index, 2].sum().item() + \
                                                    claim_softmax_attns[idx, ss_index+1:, 2].sum().item() - (num - 1) + 1)

            if '本息' in claims_type[idx]:
                bx_index = claims_type[idx].index('本息')
                # Rule4: 未约定利率 and 逾期利息起始时间非法 -> 非本息成立
                if facts_y[idx, 12].item() == 2 and facts_y[idx, 3].item() == 2:
                    mask[idx, bx_index, 2] = -1e2


                # Rule5: 利息超过银行同期利率四倍 -> 非本息成立
                if rate_fact_labels[idx].item() == -1:
                    #and facts_predicts[idx, 11].item() == 1:
                    # pre_condition = facts_predicts_prob[idx, 11, 1].log()
                    mask[idx, bx_index, 2] = -1e2


            if '利息' in claims_type[idx]:
                lx_index = claims_type[idx].index('利息')
                # Rule4: 未约定利率 and 逾期利息起始时间非法 -> 非本息成立
                if facts_y[idx, 12].item() == 2 and facts_y[idx, 3].item() == 2:
                    mask[idx, lx_index, 2] = -1e2

                # Rule5: 利息超过银行同期利率四倍 -> 非本息成立
                if rate_fact_labels[idx].item() == -1:
                    #and facts_predicts[idx, 11].item() == 1:
                    # pre_condition = facts_predicts_prob[idx, 11, 1].log()
                    mask[idx, lx_index, 2] = -1e2


        constrained_claim_logits = claim_logits + mask

        return constrained_claim_logits

    def claim_classifier_layer(self, claim_out, claim_doc_hat, doc_claim_hat, input_claims_y, claims_num, claims_type,
                               rate_fact_labels, input_fact):
        batch_size, max_claim_num = input_claims_y.size()
        claim_doc_aware = torch.cat([claim_out, claim_doc_hat, torch.mul(claim_out, claim_doc_hat), torch.mul(claim_out, doc_claim_hat)], dim=2) # [batch_size, max_seq_len, 1024]
        hidden = None
        claim_doc_aware_hidden, _ = self.lstm_layer2.forward(claim_doc_aware, hidden) # [batch_size, max_seq_len, 256]
        # print('claim doc aware size:', claim_doc_aware.size())
        # print('claim doc aware hidden size:', claim_doc_aware_hidden.size())

        claim_rep = torch.cat([claim_doc_aware, claim_doc_aware_hidden], dim=2) # [batch_size, max_seq_len, 128 * 10]
        attn_c_weights = torch.matmul(claim_rep, self.attn_c)
        attn_c_out = F.softmax(attn_c_weights, dim=1)
        claim_rep = torch.sum(claim_rep * attn_c_out, dim=1) # [batch_size, 128 * 10]
        claim_logits = self.claim_classifier(claim_rep).view(batch_size, max_claim_num, 3)  # [batch_size, 3]

        claim_probs = F.softmax(claim_logits, dim=2)
        claim_log_softmax = F.log_softmax(claim_logits, dim=2).view(batch_size * max_claim_num, 3)
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)
        claim_predicts = claim_predicts.view(batch_size, max_claim_num)

        if self.use_logical:
            claim_logits = self.claims_logical_operator(claim_logits, claim_probs, claims_num, claims_type, rate_fact_labels, claim_predicts, input_fact)

        claim_probs = F.softmax(claim_logits, dim=2)
        claim_log_softmax = F.log_softmax(claim_logits, dim=2).view(batch_size * max_claim_num, 3)
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)
        loss_claim = self.claim_loss(claim_log_softmax, input_claims_y.view(batch_size*max_claim_num).long())
        claim_predicts = claim_predicts.view(batch_size, max_claim_num)

        return claim_predicts, loss_claim, claim_probs


    def forward(self, input_doc_ids,  input_sentences_lens, input_fact, input_claims_y, input_claims_type,
                                doc_texts, claims_texts, batch_claim_ids, claims_num, claims_sentence_lens, rate_fact_labels, wenshu_texts, doc_sentences_num):
        """
        :param input_x: [batch_size, max_sentences_num, max_doc_seq_lens]
        :param input_sentences_nums: [batch_size]
        :param input_sentences_lens: [batch_size, sentences_num]
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size, max_claims_num]
        :param batch_claim_ids: [batch_size, max_claims_num, max_claim_seq_lens]
        :return:
        """
        self.lstm_layer2.flatten_parameters()
        batch_size, max_claims_num, max_claims_seq_len = batch_claim_ids.size()
        doc_word_embeds = self.word_embedding_layer.forward(input_doc_ids)
        doc_out = self.lstm_layer1.forward(doc_word_embeds, input_doc_ids)  # [batch_size, max_doc_seq_lens, hidden_dim]

        claim_word_embeds = self.word_embedding_layer.forward(batch_claim_ids.view(batch_size, max_claims_num * max_claims_seq_len))
        claim_out = self.lstm_layer1.forward(claim_word_embeds, batch_claim_ids.view(batch_size, max_claims_num * max_claims_seq_len))  # [batch_size, max_claim_seq_lens, hidden_dim]
        claim_out = claim_out.view(batch_size * max_claims_num, max_claims_seq_len, -1)

        claim_to_doc_attn, doc_to_claim_attn = self.att_flow_layer(doc_out, claim_out)

        # claim_to_doc_attn: [batch_siz, max_claim_seq_len, hidden_dim]
        # doc_to_claim_attn: [batch_size, max_claim_seq_len, hidden_dim]
        claim_predicts, loss_claim, claim_predicts_prob = self.claim_classifier_layer(claim_out, claim_to_doc_attn,
                                                                                      doc_to_claim_attn, input_claims_y,
                                                                                      claims_num,
                                                                                      input_claims_type,
                                                                                      rate_fact_labels, input_fact)
        return loss_claim, claim_predicts


if __name__ == '__main__':
   print(datetime.datetime.now())
