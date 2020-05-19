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
import utils.data
import random

BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


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
    def __init__(self, config: utils.data.Data):
        super(DocEncoder, self).__init__()
        self.word_embeddings_layer = torch.nn.Embedding(config.word_alphabet_size, config.word_emb_dim, padding_idx=0)
        if config.pretrain_word_embedding is not None:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embeddings_layer.weight.requires_grad = False
        else:
            self.word_embeddings_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_dim = config.word_emb_dim
        self.lstm_dropout = nn.Dropout(config.HP_lstmdropout)

        self.lstm1 = nn.LSTM(self.word_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True,
                             bidirectional=True)
        self.attn_p = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        nn.init.uniform_(self.attn_p, -0.1, 0.1)

    def forward(self, input_x, input_sentences_len):

        word_embeds = self.word_embeddings_layer.forward(input_x)
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

        self.doc_encoder = DocEncoder(config)

        self.fact_dense = torch.nn.Linear(config.HP_hidden_dim, 512)
        self.fact_drop = torch.nn.Dropout(config.HP_lstmdropout)
        self.fact_activation = torch.nn.ReLU()
        self.fact_classifier = torch.nn.Linear(512, config.fact_num)
        self.fact_sigmoid = torch.nn.Sigmoid()
        self.fact_embedding = torch.nn.Embedding(config.fact_num, 100)

        self.claim_classifier = torch.nn.Linear(config.HP_hidden_dim + 100, 3)
        self.bce_loss = torch.nn.BCELoss()
        self.nll_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)

        if config.HP_gpu:
            self.doc_encoder = self.doc_encoder.cuda()

            self.fact_dense = self.fact_dense.cuda()
            self.fact_activation = self.fact_activation.cuda()
            self.fact_classifier = self.fact_classifier.cuda()
            self.fact_drop = self.fact_drop.cuda()
            self.fact_sigmoid = self.fact_sigmoid.cuda()
            self.fact_embedding = self.fact_embedding.cuda()

            self.claim_classifier = self.claim_classifier.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.nll_loss = self.nll_loss.cuda()


    def neg_log_likelihood_loss(self, input_x,  input_sentences_lens, input_fact, input_claims_y):
        """

        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """
        doc_rep = self.doc_encoder.forward(input_x,  input_sentences_lens) # [batch_size, max_sequence_lens, hidden_dim]

        fact_representation = self.fact_dense(doc_rep)  # [batch_size, hidden_dim] -> [batch_size, fact_num]
        fact_num = fact_representation.size(1)
        fact_representation = self.fact_drop(fact_representation)
        fact_logits = self.fact_classifier(self.fact_activation(fact_representation)) # [batch_size, fact_num]
        fact_predicts_prob = self.fact_sigmoid(fact_logits)
        loss_fact = self.bce_loss(fact_predicts_prob, input_fact)
        fact_predicts = torch.round(fact_predicts_prob)  # [batch_size, fact_num]

        fact_emb = self.fact_embedding.forward(fact_predicts.long()) # [batch_size, fact_num, fact_emb_size]
        fact_emb = fact_predicts_prob.unsqueeze(2) * fact_emb
        avg_fact_emb = torch.sum(fact_emb, 1) / fact_num # [batch_size, fact_emb_size]

        doc_rep = torch.cat((doc_rep, avg_fact_emb), dim=1)
        doc_rep = F.dropout(doc_rep, p=0.2)
        claim_outputs = self.claim_classifier(doc_rep)  # [batch_size, 3]
        claim_log_softmax = F.log_softmax(claim_outputs, dim=1)
        loss_claim = self.nll_loss(claim_log_softmax, input_claims_y.long())
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)

        return loss_claim, loss_fact, fact_predicts, claim_predicts


    def forward(self, input_x,  input_sentences_lens, input_fact, input_claims_y):
        #, input_sample_mask, input_sentences_mask

        doc_rep = self.doc_encoder.forward(input_x, input_sentences_lens)  # [batch_size, hidden_dim]
        claim_outputs = self.claim_classifier(doc_rep)  # [batch_size, 3]
        claim_log_softmax = torch.nn.functional.log_softmax(claim_outputs, dim=1)
        loss_claim = self.nll_loss(claim_log_softmax, input_claims_y.long())
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)

        fact_representation = self.fact_dense(doc_rep)  # [batch_size, hidden_dim] -> [batch_size, fact_num]
        fact_logits = self.fact_classifier(self.fact_activation(fact_representation))
        fact_predicts_prob = self.fact_sigmoid(fact_logits)
        loss_fact = self.bce_loss(fact_predicts_prob, input_fact)
        fact_predicts = torch.round(fact_predicts_prob)  # [batch_size, fact_num]

        return loss_claim, loss_fact, fact_predicts, claim_predicts


if __name__ == '__main__':
   print(datetime.datetime.now())
   debat_encoder = DebatEncoder(200, 200)
   input = torch.randn(32, 60, 50, 200) # [batch_size, max_utterance_num, max_seq_len, word_embs_dim]
   output = debat_encoder.forward(input)
   print(output.size())
