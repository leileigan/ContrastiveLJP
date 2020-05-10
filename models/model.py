# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
from transformers import *
import datetime

BERT_MODEL_PATH = "/home/ganleilei/data/chinese_L-12_H-768_A-12/"

class Sequence(nn.Module):
    def __init__(self, data):
        super(Sequence, self).__init__()

        print("Begin loading BERT model from path:", BERT_MODEL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.bert_model = BertModel.from_pretrained(BERT_MODEL_PATH)
        bert_params = list(self.bert_model.named_parameters())
        for p in bert_params:
            p[1].requires_grad = True
        print("Finish loading BERT model...")

        self.linea1 = nn.Linear(data.bert_hidden_size, data.HP_hidden_dim)
        self.linea2 = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
        self.dropout = nn.Dropout(data.HP_dropout)
        self.gpu = data.HP_gpu

        if self.gpu:
            print("%s:begin copying data to gpu" % datetime.datetime.now())
            self.bert_model.cuda()
            self.linea1 = self.linea1.cuda()
            self.linea2 = self.linea2.cuda()
            self.dropout = self.dropout.cuda()
            print("%s: finish copying data to gpu" % datetime.datetime.now())

    def load_bert_char_embedding(self, word_text, word_seq_lens):
        max_len = max(word_seq_lens.data.tolist())
        max_len = min(max_len, 500)
        text = [' '.join(['[CLS]'] + item[:max_len] + ['[SEP]'] + (max_len - len(item[:max_len])) * ['[PAD]']) for item in word_text]
        tokenized_text = [self.tokenizer.tokenize(item) for item in text]
        indexed_token_ids = [self.tokenizer.convert_tokens_to_ids(item) for item in tokenized_text]
        tokens_tensor = torch.tensor(indexed_token_ids).cuda()

        # with torch.no_grad():
        _, pool_out = self.bert_model(tokens_tensor)

        return pool_out

    def forward(self, word_text, word_seq_lens):
        char_embs = self.load_bert_char_embedding(word_text, word_seq_lens)
        return self.linea2(self.dropout(self.linea1(char_embs)))
