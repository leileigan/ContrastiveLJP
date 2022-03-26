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
from transformers import BertTokenizer

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Config:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.word_emb_dim = 200
        self.pretrain_word_embedding = None
        self.word2id_dict = None
        self.id2word_dict = None
        self.bert_path = None

        self.accu_label_size = 119
        self.law_label_size = 103
        self.term_label_size = 12
        self.law_relation_threshold = 0.3

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
        self.HP_freeze_word_emb = True

        # optimizer
        self.use_adam = True
        self.use_bert = False
        self.use_sgd = False
        self.use_adadelta = False
        self.use_warmup_adam = False
        self.mode = 'train'

        self.save_model_dir = ""
        self.save_dset_dir = ""

        self.filters_size = [1, 3, 4, 5]
        self.num_filters = [50, 50, 50, 50]

        #contrastive learning
        self.moco_temperature = 0.07
        self.moco_queue_size = 65536
        self.moco_momentum = 0.999
        self.alpha = 0.1
        self.warm_epoch = 0
        self.confused_matrix = None


    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Bert Path:           %s" % (self.bert_path))
        print("     Accu label     size: %s" % (self.accu_label_size))
        print("     Law label     size:  %s" % (self.law_label_size))
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

        print("     Confused Matrix path:  %s" % (self.confused_matrix))
        print("     Temperature         :  %s" % (self.moco_temperature))
        print("     Momentum            :  %s" % (self.moco_momentum))
        print("     Queue size          :  %s" % (self.moco_queue_size))
        print("     Alpha               :  %s" % (self.alpha))

        print("DATA SUMMARY END.")
        sys.stdout.flush()


    def load_word_pretrain_emb(self, emb_path):
        self.pretrain_word_embedding = np.cast[np.float32](np.load(emb_path))
        self.word_emb_dim = self.pretrain_word_embedding.shape[1]
        print("word embedding size:", self.pretrain_word_embedding.shape)


class BertMocoConfig(object):
    
    """配置参数"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.bert_path = '/data/home/ganleilei/bert/bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

        self.MAX_SENTENCE_LENGTH = 250
        self.word_emb_dim = 200
        self.pretrain_word_embedding = None
        self.word2id_dict = None

        self.accu_label_size = 119
        self.law_label_size = 103
        self.term_label_size = 12
        self.law_relation_threshold = 0.3

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

        #contrastive loss
        self.HP_temperature = 0.7
        self.HP_alpha = 0.1
    

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


def seed_rand(SEED_NUM):
    torch.manual_seed(SEED_NUM)
    random.seed(SEED_NUM)
    np.random.seed(SEED_NUM)
    torch.cuda.manual_seed_all(SEED_NUM)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False