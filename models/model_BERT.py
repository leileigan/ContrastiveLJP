#-*- coding:utf-8 _*-
# @Author: Leilei Gan
# @Time: 2020/05/11
# @Contact: 11921071@zju.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
from transformers import *
import datetime
import utils.data
import random

BERT_MODEL_PATH = "/home/ganleilei/data/data/BertPreModels/chinese_L-12_H-768_A-12/"
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
        print("Begin loading BERT model from path:", BERT_MODEL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.bert_model = BertModel.from_pretrained(BERT_MODEL_PATH)
        bert_params = list(self.bert_model.named_parameters())
        for p in bert_params:
            p[1].requires_grad = True
        print("Finish loading BERT model...")
        self.linear1 = nn.Linear(768, 256)

        '''
        self.attn_p = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        nn.init.uniform_(self.attn_p, -0.1, 0.1)
        '''

    def forward(self, word_text, input_x, input_sentences_len):

        pool_outs = self.load_bert_char_embedding(word_text, input_sentences_len)
        outs = self.linear1.forward(pool_outs)
        return outs # batch_size, hidden_dim

    def load_bert_char_embedding(self, word_text, word_seq_lens):
        max_len = max(word_seq_lens)
        max_len = min(max_len, 500)
        text = [' '.join(['[CLS]'] + item[:max_len] + ['[SEP]'] + (max_len - len(item[:max_len])) * ['[PAD]']) for item in word_text]
        tokenized_text = [self.tokenizer.tokenize(item) for item in text]
        indexed_token_ids = [self.tokenizer.convert_tokens_to_ids(item) for item in tokenized_text]
        tokens_tensor = torch.tensor(indexed_token_ids).cuda()

        # with torch.no_grad():
        _, pool_out = self.bert_model(tokens_tensor)

        return pool_out


class LawModel(nn.Module):

    def __init__(self, config: utils.data.Data):
        super(LawModel, self).__init__()
        self.doc_encoder = DocEncoder(config)
        self.doc_dropout = nn.Dropout(config.HP_lstmdropout)

        self.fact_classifier = torch.nn.Linear(config.HP_hidden_dim, config.fact_num)
        self.fact_sigmoid = torch.nn.Sigmoid()

        self.claim_classifier = torch.nn.Linear(config.HP_hidden_dim, 3)
        self.bce_loss = torch.nn.BCELoss()
        self.nll_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)

        if config.HP_gpu:
            self.doc_encoder = self.doc_encoder.cuda()
            self.doc_dropout = self.doc_dropout.cuda()

            self.fact_classifier = self.fact_classifier.cuda()
            self.fact_sigmoid = self.fact_sigmoid.cuda()

            self.claim_classifier = self.claim_classifier.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.nll_loss = self.nll_loss.cuda()

    def neg_log_likelihood_loss(self, word_text, input_x,  input_sentences_lens, input_fact, input_claims_y):
        """

        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """
        doc_rep = self.doc_encoder.forward(word_text, input_x, input_sentences_lens) # [batch_size, max_sequence_lens, hidden_dim]
        doc_rep = self.doc_dropout(doc_rep)

        claim_outputs = self.claim_classifier(doc_rep) # [batch_size, 3]
        claim_log_softmax = torch.nn.functional.log_softmax(claim_outputs, dim=1)
        loss_claim = self.nll_loss(claim_log_softmax, input_claims_y.long())
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)

        '''
        fact_logits = self.fact_classifier(doc_rep) # [batch_size, hidden_dim] -> [batch_size, fact_num]
        fact_predicts_prob = self.fact_sigmoid(fact_logits)
        loss_fact = self.bce_loss(fact_predicts_prob, input_fact)
        fact_predicts = torch.round(fact_predicts_prob) # [batch_size, fact_num]
        '''

        return loss_claim, claim_predicts


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
