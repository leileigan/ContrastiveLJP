# -*- coding: utf-8 -*-
# @Time    : 2020/4/26 3:33 下午
# @Author  : Leilei Gan
# @Contact : 11921071@zju.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import datetime
import utils.config
import random

BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)
from utils.functions import debug_log


class LawModel(nn.Module):

    def __init__(self, config: utils.config.Data):
        super(LawModel, self).__init__()
        self.word_dim = config.word_emb_dim
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_embedding_layer = torch.nn.Embedding(config.word_alphabet_size, self.word_dim)


        if config.pretrain_word_embedding is not None:
            self.word_embedding_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embedding_layer.weight.requires_grad = False
        else:
            self.word_embedding_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.fact_gru = nn.GRU(self.word_dim, hidden_size = self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.claim_gru = nn.GRU(self.word_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.final_gru = nn.GRUCell(self.hidden_dim * 4, hidden_size=self.hidden_dim * 2)

        self.mutual_w_f = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.mutual_w_p = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.mutual_u_p = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.mutual_v = nn.Linear(self.hidden_dim * 2, 1)

        self.convs = nn.ModuleList(nn.Conv2d(1, item[1], (item[0], self.hidden_dim * 2)) for item in
                                   list(zip(config.filters_size, config.num_filters)))
        self.cnn_dropout = nn.Dropout(0.5)

        self.claim_classifier = torch.nn.Linear(self.hidden_dim * 2, 3)
        self.claim_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)

        self.attn_c = nn.Parameter(torch.Tensor(self.hidden_dim * 10, 1).cuda())
        nn.init.uniform_(self.attn_c, -0.1, 0.1)


    def conv_and_pool(self, input_x, conv):
        x = F.relu(conv(input_x)).squeeze(3)  # [batch_size, out_channel, seq_len - kernel_size + 1, hidden_size]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def claim_classifier_layer(self, claim_rep, input_claims_y):
        # print('claim out size:', claim_out.size())
        # print('claim doc hat size:', claim_doc_hat.size())
        # print('doc claim hat size:', doc_claim_hat.size())

        claim_logits = self.claim_classifier(claim_rep) # [batch_size, 3]

        claim_probs = F.softmax(claim_logits, dim=1)
        claim_log_softmax = F.log_softmax(claim_logits, dim=1)
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)
        loss_claim = self.claim_loss(claim_log_softmax, input_claims_y.long())

        return claim_predicts, loss_claim, claim_probs

    def point_wise_mutual_attention(self, u_f, u_p, v_p):
        """

        :param u_f: [B, L, d]
        :param u_p: [B, d]
        :param v_p: [B, d]
        :return:
        """

        B, L, _ = u_f.size()
        T, _ = u_p.size()
        max_claims_num = int(T/B)
        expand_u_p = u_p.unsqueeze(1).repeat(1, L, 1)
        expand_v_p = v_p.unsqueeze(1).repeat(1, L, 1)
        expand_u_f = u_f.repeat(max_claims_num, 1, 1)
        activates = torch.tanh(self.mutual_w_f(expand_u_f) + self.mutual_w_p(expand_u_p) + self.mutual_u_p(expand_v_p)) # [B, L, d]
        scores = self.mutual_v(activates).squeeze(2) #[B, L]
        softmax_scores = torch.softmax(scores, dim=1).unsqueeze(2)
        summaries = torch.sum(expand_u_f * softmax_scores, dim=1)
        return softmax_scores, summaries # [B, L], [B, d]

    def forward(self, input_x,  claim_seq_tensor, input_sentences_lens, claim_sentence_lens, input_fact, input_claims_y ):
        """

        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """
        # print('input x type:', type(input_x))
        self.fact_gru.flatten_parameters()
        self.claim_gru.flatten_parameters()

        batch_size,  max_claims_sequence_len = claim_seq_tensor.size()

        doc_word_embeds = self.word_embedding_layer.forward(input_x)
        hidden = None
        doc_out, hidden = self.fact_gru.forward(doc_word_embeds,  hidden) # [batch_size, max_doc_seq_lens, word_dim]

        claim_word_embeds = self.word_embedding_layer.forward(claim_seq_tensor)
        #[batch_size * max_claims_num, max_claims_sequence_len, word_dim]
        hidden = None
        claim_out, hidden = self.claim_gru.forward(claim_word_embeds, hidden) # [batch_size * max_claims_num, max_claim_seq_lens, hidden_dim]

        hidden_temp = torch.zeros(batch_size, self.hidden_dim * 2).cuda()
        hidden_vector = None
        for i in range(max_claims_sequence_len):
            scores, summaries = self.point_wise_mutual_attention(doc_out, claim_out[:, i, :], hidden_temp)
            input = torch.cat((claim_out[:, i, :], summaries), dim=1)
            hidden_temp = self.final_gru.forward(input, hidden_temp)
            if hidden_vector is None:
                hidden_vector = hidden_temp.unsqueeze(1)
            else:
                hidden_vector = torch.cat((hidden_vector, hidden_temp.unsqueeze(1)), dim=1)

        hidden_vector = hidden_vector.unsqueeze(1)
        hidden_conv = torch.cat(tuple([self.conv_and_pool(hidden_vector, conv) for conv in self.convs]), 1) #[batch_size * max_claims_num, 64 * 4]
        hidden_conv = self.cnn_dropout(hidden_conv)

        # claim_to_doc_attn: [batch_siz, max_claim_seq_len, hidden_dim]
        # doc_to_claim_attn: [batch_size, max_claim_seq_len, hidden_dim]
        claim_predicts, loss_claim, claim_predicts_prob = self.claim_classifier_layer(hidden_conv, input_claims_y)

        return loss_claim, claim_predicts


if __name__ == '__main__':
   print(datetime.datetime.now())

