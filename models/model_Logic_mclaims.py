#-*- coding:utf-8 _*-  
# @Author: Leilei Gan
# @Time: 2020/05/10
# @Contact: 11921071@zju.edu.cn

# -*- coding: utf-8 -*-
# @Time    : 2020/4/27 5:17 下午
# @Author  : Leilei Gan
# @Contact : 11921071@zju.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import datetime
import utils.data

BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"

class ClaimEncoder(nn.Module):
    def __init__(self, config: utils.data.Data, word_embedding_layer):
        super(ClaimEncoder, self).__init__()
        self.word_embedding_layer = word_embedding_layer

        self.lstm = nn.LSTM(config.word_emb_dim, hidden_size=config.HP_hidden_dim // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attn_p = nn.Parameter(torch.Tensor(config.HP_hidden_dim, 1))
        nn.init.uniform_(self.attn_p, -0.1, 0.1)

    def forward(self, input_claims_x, input_claims_nums, input_claims_lens):
        """

        :param input_claims_x: [batch_size, max_claim_num, max_sentence_len]
        :param input_claims_nums
        :param input_claims_lens
        :return:
        """
        word_embs = self.word_embedding_layer(input_claims_x)
        batch_size, max_claim_num, max_sentence_len = input_claims_x.size()
        lstm_input = word_embs.view(batch_size*max_claim_num, max_sentence_len, -1)
        hidden = None
        lstm_out, hidden = self.lstm.forward(lstm_input, hidden) # [batch_size * max_claim_num, max_sentence_len, hidden_dim]
        attn_weights = F.softmax(torch.matmul(lstm_out, self.attn_p))
        attn_outs = lstm_out * attn_weights
        claim_outs = torch.sum(attn_outs, dim=1)
        # [batch_size * max_claim_num, max_sentence_len, hidden_dim] -> [batch_size * max_claim_num, hidden_dim]
        claim_outs = claim_outs.view(batch_size, max_claim_num, -1)
        # [batch_size * max_claim_num, hidden_dim] -> [batch_size, max_claim_num, hidden_dim]
        return claim_outs

class DocEncoder(nn.Module):
    def __init__(self, config: utils.data.Data, word_embedding_layer):
        super(DocEncoder, self).__init__()
        self.word_embedding_layer  = word_embedding_layer
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_dim = config.word_emb_dim

        self.lstm1 = nn.LSTM(self.word_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True,
                             bidirectional=True)
        self.attn_p = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        nn.init.uniform_(self.attn_p, -0.1, 0.1)

    def forward(self, input_x, input_sentences_len):

        word_embeds = self.word_embedding_layer.forward(input_x)
        hidden = None
        lstm1_out, hidden = self.lstm1.forward(word_embeds, hidden) # [batch_size, max_sequence_lens, hidden_dim]
        attn_p_weights = torch.matmul(lstm1_out, self.attn_p) # [batch_size, max_sequence_lens]
        attn_p_out = F.softmax(attn_p_weights, dim=1)
        doc_out = lstm1_out * attn_p_out
        doc_out = torch.sum(doc_out, dim=1) # [batch_size, max_sequence_lens, hidden_dim] > [batch_size, hidden_dim]

        return doc_out # batch_size, hidden_dim


class LawModel(nn.Module):

    def __init__(self, config: utils.data.Data):
        super(LawModel, self).__init__()
        self.word_embeddings_layer = torch.nn.Embedding(config.word_alphabet_size, config.word_emb_dim, padding_idx=0)
        self.doc_encoder = DocEncoder(config, self.word_embeddings_layer)
        self.doc_dropout = torch.nn.Dropout(config.HP_dropout)

        self.fact_classifier = torch.nn.Linear(config.HP_hidden_dim, config.fact_num)
        self.fact_sigmoid = torch.nn.Sigmoid()

        self.claim_classifier = torch.nn.Linear(config.HP_hidden_dim, 3)
        self.bce_loss = torch.nn.BCELoss()
        #self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, size_average=True)
        self.nll_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)

        #----------------------logic operator---------------------------------
        self.linear_con = nn.Linear(1, 1)
        self.linear_dis = nn.Linear(1, 1)

        if config.HP_gpu:
            self.word_embeddings_layer = self.word_embeddings_layer.cuda()
            self.doc_encoder = self.doc_encoder.cuda()
            self.doc_dropout = self.doc_dropout.cuda()

            self.fact_classifier = self.fact_classifier.cuda()
            self.fact_sigmoid = self.fact_sigmoid.cuda()

            self.claim_classifier = self.claim_classifier.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.nll_loss = self.nll_loss.cuda()
            #self.cross_entropy_loss = self.cross_entropy_loss.cuda()

            self.linear_con = self.linear_con.cuda()
            self.linear_dis = self.linear_dis.cuda()


    def neg_log_likelihood_loss(self, input_x,  input_sentences_lens, input_fact, input_claims_y, input_claim_type):
        """

        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """
        doc_rep = self.doc_encoder.forward(input_x,  input_sentences_lens) # [batch_size, max_sequence_lens, hidden_dim]
        doc_rep = self.doc_dropout(doc_rep) # [batch_size, hidden_size]
        claim_outputs = self.claim_classifier(doc_rep) # [batch_size, 3]
        claim_log_softmax = F.log_softmax(claim_outputs, dim=1) # [batch_size, 3]
        loss_claim = self.nll_loss(claim_log_softmax, input_claims_y.long())
        #loss_claim = self.cross_entropy_loss(claim_outputs, input_claims_y.long())
        _, claim_predicts = torch.max(claim_outputs, dim=1) # [batch_size]

        fact_logits = self.fact_classifier(doc_rep) # [batch_size, hidden_dim] -> [batch_size, fact_num]
        fact_predicts_prob = self.fact_sigmoid(fact_logits) # [batch_size, fact_num]
        loss_fact = self.bce_loss(fact_predicts_prob, input_fact)
        fact_predicts = torch.round(fact_predicts_prob) # [batch_size, fact_num]
        #----------------------discrepency loss----------------------------------
        # dependency between facts and claim
        # 是否夫妻共同债务,
        # 是否物权担保,
        # 是否存在还款行为,
        # 是否约定利率,
        # 是否约定借款期限,
        # 是否约定保证期间,
        # 是否保证人不承担担保责任,
        # 是否保证人担保,
        # 是否约定违约条款,
        # 是否约定还款期限,
        # 是否超过诉讼时效,
        # 是否借款成立
        # claim label: 0:驳回， 1：支持，2：部分支持
        for idx, fact in enumerate(fact_predicts):
            if input_claim_type[idx] == '本金':
                if fact[11] == 1:
                    print('claim type 本金')
                    print('claim predicts:', claim_predicts[idx])
                    print('fact predicts prob:', fact_predicts_prob[idx][11])
                    logic_output = F.sigmoid(self.linear_con((fact_predicts_prob[idx][11] - 1).view(-1, 1))) #
                    print('logic output:', logic_output)
                    loss_claim += (claim_predicts[idx] - logic_output.view(1).squeeze(0)) ** 2
                elif fact[11] == 0:
                    print('claim type 本金')
                    print('claim predicts:', claim_predicts[idx])
                    print('fact predicts prob:', fact_predicts_prob[idx][11])
                    logic_output = F.sigmoid(self.linear_con((fact_predicts_prob[idx][11] - 1).view(-1, 1)))  #
                    print('logic output:', logic_output)
                    loss_claim += (claim_predicts[idx] - logic_output.view(1).squeeze(0)) ** 2

            elif input_claim_type[idx] == '利息':

                loss_claim = 0

            elif input_claim_type[idx] == '本息':
                if fact[11] == 1 and fact[3] == 1:
                # 借款成立 and 约定利息
                    print('claim type 本息')
                    print('claim predict:', claim_predicts[idx])
                    print('fact predicts prob: %s, %s' % (fact_predicts_prob[idx][11], fact_predicts_prob[idx][3]))
                    logic_output = F.sigmoid(self.linear_con((fact_predicts_prob[idx][11] + fact_predicts_prob[idx][3] - 2).view(-1, 1)))
                    print('logic output:', logic_output)
                    loss_claim += (claim_predicts[idx] - logic_output.view(1).squeeze(0)) ** 2
                elif fact[11] == 0 or fact[3] ==0:
                    logic_output = F.sigmoid(self.linear_dis((fact_predicts_prob[11] + fact_predicts_prob[3]).view(-1, 1)))
                    loss_claim += (claim_predicts[idx] - logic_output.view(1).squeeze(0)) ** 2

            elif input_claim_type[idx] == '担保':
                return

            elif input_claim_type[idx] == '违约':
                return


        return loss_claim, loss_fact, fact_predicts, claim_predicts


    def forward(self, input_x,  input_sentences_lens, input_fact, input_claims_y):
        #, input_sample_mask, input_sentences_mask
        doc_rep = self.doc_encoder.forward(input_x, input_sentences_lens)  # [batch_size, hidden_dim]
        doc_rep = self.doc_dropout(doc_rep) # [batch_size, hidden_size]
        claim_outputs = self.claim_classifier(doc_rep)
        claim_log_softmax = torch.nn.functional.log_softmax(claim_outputs, dim=1)
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)

        fact_logits = self.fact_classifier(doc_rep)  # [batch_size, hidden_dim] -> [batch_size, fact_num]
        fact_predicts_prob = self.fact_sigmoid(fact_logits)
        fact_predicts = torch.round(fact_predicts_prob)  # [batch_size, fact_num]

        return fact_predicts, claim_predicts


if __name__ == '__main__':
   print(datetime.datetime.now())
   debat_encoder = DebatEncoder(200, 200)
   input = torch.randn(32, 60, 50, 200) # [batch_size, max_utterance_num, max_seq_len, word_embs_dim]
   output = debat_encoder.forward(input)
   print(output.size())
