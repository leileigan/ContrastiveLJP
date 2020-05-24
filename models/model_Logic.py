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

        self.lstm_dropout = nn.Dropout(config.HP_lstmdropout)
        self.attn_p = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        nn.init.uniform_(self.attn_p, -0.1, 0.1)

    def forward(self, input_x, input_sentences_len):

        word_embeds = self.word_embedding_layer.forward(input_x)
        hidden = None
        lstm1_out, hidden = self.lstm1.forward(word_embeds, hidden) # [batch_size, max_sequence_lens, hidden_dim]

        lstm1_out = self.lstm_dropout.forward(lstm1_out)

        attn_p_weights = torch.matmul(lstm1_out, self.attn_p) # [batch_size, max_sequence_lens]
        attn_p_out = F.softmax(attn_p_weights, dim=1)
        doc_out = lstm1_out * attn_p_out
        doc_out = torch.sum(doc_out, dim=1) # [batch_size, max_sequence_lens, hidden_dim] > [batch_size, hidden_dim]

        return doc_out # batch_size, hidden_dim


class LawModel(nn.Module):

    def __init__(self, config: utils.data.Data):
        super(LawModel, self).__init__()
        self.word_embeddings_layer = torch.nn.Embedding(config.word_alphabet_size, config.word_emb_dim, padding_idx=0)

        if config.pretrain_word_embedding is not None:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embeddings_layer.weight.requires_grad = False
        else:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.doc_encoder = DocEncoder(config, self.word_embeddings_layer) # [batch_size, hidden_dim]

        self.fact_dense = torch.nn.Linear(config.HP_hidden_dim, 512) # [batch_size, 512]
        self.fact_drop = torch.nn.Dropout(config.HP_lstmdropout)
        self.fact_activation = torch.nn.ReLU()
        self.fact_classifier = torch.nn.Linear(512, config.fact_num * 3) # [batch_size, fact_num * 3]
        self.fact_embedding = torch.nn.Embedding(config.fact_num, config.fact_edim)

        self.claim_classifier = torch.nn.Linear(config.HP_hidden_dim + config.fact_edim, 3)
        self.claim_dropout = torch.nn.Dropout(0.2)
        self.bce_loss = torch.nn.BCELoss()
        self.nll_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)

        #----------------------logic operator---------------------------------
        self.linear_con = nn.Linear(1, 1)
        self.linear_dis = nn.Linear(1, 1)

        if config.HP_gpu:
            self.word_embeddings_layer = self.word_embeddings_layer.cuda()
            self.doc_encoder = self.doc_encoder.cuda()

            self.fact_dense = self.fact_dense.cuda()
            self.fact_activation = self.fact_activation.cuda()
            self.fact_classifier = self.fact_classifier.cuda()
            self.fact_drop = self.fact_drop.cuda()
            self.fact_sigmoid = self.fact_sigmoid.cuda()
            self.fact_embedding = self.fact_embedding.cuda()

            self.claim_classifier = self.claim_classifier.cuda()
            self.claim_dropout = self.claim_dropout.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.nll_loss = self.nll_loss.cuda()

            self.linear_con = self.linear_con.cuda()
            self.linear_dis = self.linear_dis.cuda()


    def neg_log_likelihood_loss(self, input_x,  input_sentences_lens, input_fact, input_claims_y, input_claim_type):
        """

        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size * fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """
        doc_rep = self.doc_encoder.forward(input_x,  input_sentences_lens) # [batch_size, max_sequence_lens, hidden_dim]

        fact_representation = self.fact_dense(doc_rep)  # [batch_size, hidden_dim] -> [batch_size, 512]
        batch_size = fact_representation.size(0)
        fact_num = fact_representation.size(1)
        fact_representation = self.fact_drop(fact_representation)
        fact_logits = self.fact_classifier(self.fact_activation(fact_representation))  # [batch_size, fact_num * 3]
        fact_logits = fact_logits.view(batch_size * fact_num, -1) # [batch_size * fact_num, 3]
        fact_log_softmax = F.log_softmax(fact_logits, dim=1)
        loss_fact = self.nll_loss(fact_log_softmax, input_fact.view(batch_size * fact_num))
        _, fact_predicts = torch.max(fact_logits, dim=1).view(batch_size, -1)  # [batch_size, fact_num]

        fact_emb = self.fact_embedding.forward(fact_predicts.long())  # [batch_size, fact_num, fact_emb_size]
        fact_emb = fact_logits.unsqueeze(2) * fact_emb
        avg_fact_emb = torch.sum(fact_emb, 1) / fact_num  # [batch_size, fact_emb_size]

        doc_rep = torch.cat((doc_rep, avg_fact_emb), dim=1)
        doc_rep = self.claim_dropout.forward(doc_rep)
        claim_outputs = self.claim_classifier(doc_rep) # [batch_size, 3]

        '''
        claim_log_softmax = F.log_softmax(claim_outputs, dim=1) # [batch_size, 3]
        loss_claim = self.nll_loss(claim_log_softmax, input_claims_y.long())
        #loss_claim = self.cross_entropy_loss(claim_outputs, input_claims_y.long())
        _, claim_predicts = torch.max(claim_outputs, dim=1) # [batch_size]
        '''
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
        # claim label: 0:驳回， 1：部分支持，2：支持
        claim_softmax_outputs = F.softmax(claim_outputs, dim=1)
        for idx, fact in enumerate(fact_predicts):
            if input_claim_type[idx] == '本金':
                if fact[11] == 1 and fact[2] == -1: # 本金支持
                    logic_output_con = F.sigmoid(self.linear_con((fact_predicts_prob[idx][11] + fact_predicts_prob[idx][2] - 2).view(-1, 1))) #
                    logic_output_negative = 1 - logic_output_con
                    logic_output = F.sigmoid(self.linear_dis(logic_output_negative + claim_softmax_outputs[idx][2]))
                    claim_softmax_outputs[idx][2] = logic_output
                    '''
                    print('claim type 本金')
                    print('claim predicts:', claim_predicts[idx])
                    print('fact predicts prob:', fact_predicts_prob[idx][11])
                    print('logic output:', logic_output)
                    '''
                elif fact[11] == 0: # 本金驳回
                    logic_output_con = F.sigmoid(self.linear_con((fact_predicts_prob[idx][11] - 1).view(-1, 1)))  #
                    logic_output_negative = 1 - logic_output_con
                    logic_output = F.sigmoid(self.linear_dis(logic_output_negative + claim_softmax_outputs[idx][0]))
                    claim_softmax_outputs[idx][0] = logic_output
                    ''''
                    print('claim type 本金')
                    print('claim predicts:', claim_predicts[idx])
                    print('fact predicts prob:', fact_predicts_prob[idx][11])
                    print('logic output:', logic_output)
                    '''
                elif fact[11] == 1 and fact[2] == 1: # 本金部分支持
                    logic_output_con = F.sigmoid(self.linear_con((fact_predicts_prob[idx][11] + fact_predicts_prob[idx][2] - 2).view(-1, 1)))
                    logic_output_negative = 1 - logic_output_con
                    logic_output = F.sigmoid(self.linear_dis(logic_output_negative + claim_softmax_outputs[idx][1]))
                    claim_softmax_outputs[idx][1] = logic_output
            '''
            elif input_claim_type[idx] == '本息':
                if fact[11] == 1 and fact[3] == 1:
                    # 借款成立 and 约定利息
                    print('claim type 本息')
                    print('claim predict:', claim_predicts[idx])
                    print('fact predicts prob: %s, %s' % (fact_predicts_prob[idx][11], fact_predicts_prob[idx][3]))
                    logic_output = F.sigmoid(self.linear_con((fact_predicts_prob[idx][11] + fact_predicts_prob[idx][3] - 2).view(-1, 1)))
                    print('logic output:', logic_output)
                    loss_claim += (claim_softmax_outputs[idx][2] - logic_output.view(1).squeeze(0)) ** 2

                elif fact[11] == 0 or fact[3] ==0:
                    logic_output = F.sigmoid(self.linear_dis((fact_predicts_prob[idx][11] + fact_predicts_prob[idx][3]).view(-1, 1)))
                    print('claim softmax output size:', claim_softmax_outputs.size())
                    print('claim softmax output', claim_softmax_outputs[idx])
                    print('logic output:', logic_output)
                    loss_claim += (claim_softmax_outputs[idx][0] - logic_output.view(1).squeeze(0)) ** 2
            
            elif input_claim_type[idx] == '担保':
                return
            elif input_claim_type[idx] == '违约':
                return
            '''

        claim_softmax_outputs = F.log_softmax(claim_softmax_outputs, dim=1)
        loss_claim = self.nll_loss(claim_softmax_outputs, input_claims_y.long())
        _, claim_predicts = torch.max(claim_outputs, dim=1)  # [batch_size]

        return loss_claim, loss_fact, fact_predicts, claim_predicts


    def forward(self, input_x,  input_sentences_lens, input_fact, input_claims_y, input_claim_type):


        """

        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """
        doc_rep = self.doc_encoder.forward(input_x, input_sentences_lens)  # [batch_size, max_sequence_lens, hidden_dim]

        fact_representation = self.fact_dense(doc_rep)  # [batch_size, hidden_dim] -> [batch_size, fact_num]
        fact_num = fact_representation.size(1)
        fact_representation = self.fact_drop(fact_representation)
        fact_logits = self.fact_classifier(self.fact_activation(fact_representation))  # [batch_size, fact_num]
        fact_predicts_prob = self.fact_sigmoid(fact_logits)
        # loss_fact = self.bce_loss(fact_predicts_prob, input_fact)
        fact_predicts = torch.round(fact_predicts_prob)  # [batch_size, fact_num]

        fact_emb = self.fact_embedding.forward(fact_predicts.long())  # [batch_size, fact_num, fact_emb_size]
        fact_emb = fact_predicts_prob.unsqueeze(2) * fact_emb
        avg_fact_emb = torch.sum(fact_emb, 1) / fact_num  # [batch_size, fact_emb_size]

        doc_rep = torch.cat((doc_rep, avg_fact_emb), dim=1)
        doc_rep = self.claim_dropout.forward(doc_rep)
        claim_outputs = self.claim_classifier(doc_rep)  # [batch_size, 3]
        # claim_log_softmax = F.log_softmax(claim_outputs, dim=1)  # [batch_size, 3]
        # loss_claim = self.nll_loss(claim_log_softmax, input_claims_y.long())
        # loss_claim = self.cross_entropy_loss(claim_outputs, input_claims_y.long())
        _, claim_predicts = torch.max(claim_outputs, dim=1)  # [batch_size]

        for idx, claim_type in enumerate(input_claim_type):
            if claim_type == '本金' and input_fact[idx][11] == 1 and claim_predicts[idx] != input_claims_y[idx].long():
                print('本金 predict error:', claim_predicts[idx])
                print('本金 input claim y:', input_claims_y[idx])

            elif claim_type == '本息' and input_fact[idx][3] == 1 and input_fact[idx][11] == 1 and claim_predicts[idx] != input_claims_y[idx].long():
                print('本息 predict error:', claim_predicts[idx])
                print('本息 input claim y:', input_claims_y[idx])

            elif claim_type == '利息' and input_fact[idx][3] == 1 and claim_predicts[idx] != input_claims_y[idx].long():
                print('利息 predict error:', claim_predicts[idx])
                print('利息 input claim y:', input_claims_y[idx])

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
        # claim_softmax_outputs = F.softmax(claim_outputs, dim=1)
        '''
        for idx, fact in enumerate(fact_predicts):
            if input_claim_type[idx] == '本金':
                print('本金 claim predicts:', claim_predicts[idx])
                print('本金 ground facts:', input_fact[idx][11])
                print('本金 fact predicts prob:', fact_predicts_prob[idx][11])

            elif input_claim_type[idx] == '本息':
                print('本息 claim predicts:', claim_predicts[idx])
                print('本息 ground facts:', input_fact[idx][11])
                print('本息 fact predicts prob:', fact_predicts_prob[idx][11])
            elif input_claim_type[idx] == '担保':
                return

            elif input_claim_type[idx] == '违约':
                return
        '''
        return fact_predicts, claim_predicts


if __name__ == '__main__':
   print(datetime.datetime.now())
