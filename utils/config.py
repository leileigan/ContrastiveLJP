# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils.alphabet import Alphabet
from utils.functions import *
# import cPickle as pickle
import pickle
import gensim
import jieba, json
import random

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Config:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.word_emb_dim = 200

        self.pretrain_word_embedding = None
        self.accu_label_size = 119
        self.law_label_size = 103
        self.term_label_size = 12

        self.sent_len = 100
        self.doc_len = 15
        #  hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 128
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.2
        self.HP_lstmdropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.bert_hidden_size = 768

        # optimizer
        self.use_adam = True
        self.use_bert = False
        self.use_sgd = False
        self.use_adadelta = False
        self.use_warmup_adam = False
        self.mode = 'train'

        self.save_model_dir = ""
        self.save_dset_dir = ""

        self.hops = 3
        self.heads = 4

        self.filters_size = [1, 3, 4, 5]
        self.num_filters = [50, 50, 50, 50]

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     Word embedding size: %s" % (self.word_emb_dim))

        print("     Accu label     size: %s" % (self.accu_label_size))
        print("     Law label     size: %s" % (self.law_label_size))
        print("     Term label     size: %s" % (self.term_label_size))

        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Filter size:        :  %s" % (self.filters_size))
        print("     Number filters      :  %s" % (self.num_filters))
        print("DATA SUMMARY END.")
        sys.stdout.flush()


    def load_word_pretrain_emb(self, emb_path):
        self.pretrain_word_embedding = np.cast[np.float32](np.load(emb_path))
        self.word_emb_dim = self.pretrain_word_embedding.shape[1]
        print("word embedding size:", self.pretrain_word_embedding.shape)