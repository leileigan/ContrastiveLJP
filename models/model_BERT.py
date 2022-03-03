# -*- coding: utf-8 -*-
# @Time    : 2020/4/26 3:33 下午
# @Author  : Leilei Gan
# @Contact : 11921071@zju.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import random
from transformers import BertConfig, BertModel, BertTokenizer
import sys

BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

class Config(object):

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

class LawModel(nn.Module):

    def __init__(self, config):
        super(LawModel, self).__init__()
        self.config = config
        self.bert_config = BertConfig.from_pretrained(config.bert_path, output_hidden_states=False)
        self.bert = BertModel.from_pretrained(config.bert_path, config=self.bert_config)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.accu_classifier = torch.nn.Linear(config.hidden_size, config.accu_label_size)
        self.accu_loss = torch.nn.NLLLoss()
        
        self.law_classifier = torch.nn.Linear(config.hidden_size, config.law_label_size)
        self.law_loss = torch.nn.NLLLoss()
        
        self.term_classifier = torch.nn.Linear(config.hidden_size, config.term_label_size)
        self.term_loss = torch.nn.NLLLoss()


    def classifier_layer(self, doc_out, accu_labels, law_labels, term_labels): #
        """
        :param doc_out: [batch_size, 2 * hidden_dim]
        :param accu_labels: [batch_size]
        :param law_labels: [batch_size]
        :param term_labels: [batch_size]
        """
        accu_logits = self.accu_classifier(doc_out)  # [batch_size, accu_label_size]
        accu_probs = F.softmax(accu_logits, dim=-1)
        accu_log_softmax = F.log_softmax(accu_logits, dim=-1)
        accu_loss = self.accu_loss(accu_log_softmax, accu_labels)
        _, accu_predicts = torch.max(accu_log_softmax, dim=-1) # [batch_size, accu_label_size]
        
        law_logits = self.law_classifier(doc_out)  # [batch_size, law_label_size]
        law_probs = F.softmax(law_logits, dim=-1)
        law_log_softmax = F.log_softmax(law_logits, dim=-1)
        law_loss = self.law_loss(law_log_softmax, law_labels)
        _, law_predicts = torch.max(law_log_softmax, dim=1) # [batch_size * max_claims_num]
        
        term_logits = self.term_classifier(doc_out)  # [batch_size, term_label_size]
        term_probs = F.softmax(term_logits, dim=-1)
        term_log_softmax = F.log_softmax(term_logits, dim=-1)
        term_loss = self.term_loss(term_log_softmax, term_labels)
        _, term_predicts = torch.max(term_log_softmax, dim=1) # [batch_size * max_claims_num]

        return accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss

        #return accu_predicts, accu_log_softmax, law_predicts, law_log_softmax, term_predicts, term_log_softmax

    def forward(self, fact_list, accu_labels, law_labels, term_labels):  #, input_sentences_lens, input_doc_len
        """ 
        Args:
            input_facts: [batch_size, max_sent_num, max_sent_seq_len]
            input_laws: [law_num, max_law_seq_len]
            law_labels: [batch_size]
            accu_labels : [batch_size]
            term_labels : [batch_size]
            input_sentences_lens : int 
            input_doc_len : int 

        Returns:
            [type]: [description]
        """
        #x = self.x_given_bert(fact_list)
        x = fact_list
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs.pooler_output
        accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss = self.classifier_layer(pooled, accu_labels, law_labels, term_labels)  # [batch_size, 3] 
        return accu_loss, law_loss, term_loss, accu_predicts, law_predicts, term_predicts

    def x_given_bert(self, fact_list):
        PAD, CLS = '[PAD]', '[CLS]'
        contents = []
        for line in fact_list:
                content = line.strip()
                if not content:
                    continue
                token = self.config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = self.config.tokenizer.convert_tokens_to_ids(token)

                if self.config.pad_size:
                    if len(token) < self.config.pad_size:
                        mask = [1] * len(token_ids) + [0] * (self.config.pad_size - len(token))
                        token_ids += ([0] * (self.config.pad_size - len(token)))
                    else:
                        mask = [1] * self.config.pad_size
                        token_ids = token_ids[:self.config.pad_size]
                        seq_len = self.config.pad_size
               
                contents.append([token_ids, seq_len, mask])
        x = torch.LongTensor([_[0] for _ in contents]).to(self.config.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in contents]).to(self.config.device)
        mask = torch.LongTensor([_[2] for _ in contents]).to(self.config.device)
        return x, seq_len, mask

    

if __name__ == '__main__':
   print(datetime.datetime.now())
