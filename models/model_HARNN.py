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
from utils.config import Config
import random

BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


class LawModel(nn.Module):

    def __init__(self, config: Config):
        super(LawModel, self).__init__()

        self.word_dim = config.word_emb_dim
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_embeddings_layer = torch.nn.Embedding(config.pretrain_word_embedding.shape[0], config.pretrain_word_embedding.shape[1], padding_idx=0)

        if config.pretrain_word_embedding is not None:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embeddings_layer.weight.requires_grad = not config.HP_freeze_word_emb
        else:
            self.word_embeddings_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.lstm_layer1 = nn.LSTM(self.word_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(self.hidden_dim * 2, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.attn_p = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        self.attn_q = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))

        nn.init.uniform_(self.attn_p, -0.1, 0.1)
        nn.init.uniform_(self.attn_q, -0.1, 0.1)

        self.accu_classifier = torch.nn.Linear(self.hidden_dim * 2, config.accu_label_size)
        self.accu_loss = torch.nn.NLLLoss()
        
        self.law_classifier = torch.nn.Linear(self.hidden_dim * 2, config.law_label_size)
        self.law_loss = torch.nn.NLLLoss()
        
        self.term_classifier = torch.nn.Linear(self.hidden_dim * 2, config.term_label_size)
        self.term_loss = torch.nn.NLLLoss()


    def classifier_layer(self, doc_out, accu_labels, law_labels, term_labels):
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
        """
        confuse_label_index = accu_labels.eq(1).nonzero().squeeze(-1)
        if confuse_label_index.size(0) > 0: 
            print("confuse label index size:", confuse_label_index.size())
            print("confusing accu probs size:", accu_probs[confuse_label_index].size())
            print("confusing logits 1:", accu_probs[confuse_label_index][:, 1])
            print("confusing logits 111:", accu_probs[confuse_label_index][:, 111])
        """
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


    def neg_log_likelihood_loss(self, input_facts, accu_labels, law_labels, term_labels, input_sentences_lens, input_doc_len):
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
        batch_size, max_sent_num, max_sent_seq_len = input_facts.size()
        doc_word_embeds = self.word_embeddings_layer.forward(input_facts.view(batch_size*max_sent_num, -1)) #[batch_size*max_sent_num, max_doc_seq_len, word_emb_size]

        hidden = None
        lstm1_out, hidden = self.lstm_layer1.forward(doc_word_embeds, hidden)
        # [batch_size*max_sent_num, max_sent_seq_len, hidden_dim * 2]
        attn_p_weights = torch.matmul(lstm1_out, self.attn_p)
        attn_p_outs = F.softmax(attn_p_weights, dim=1)
        doc_sent_rep = torch.sum(lstm1_out * attn_p_outs, dim=1).view(batch_size, max_sent_num, -1) #[batch_size, max_sent_num, hidden_dim*2]
        
        hidden = None
        lstm2_out, hidden = self.lstm_layer2.forward(doc_sent_rep, hidden)
        attn_q_weights = torch.matmul(lstm2_out, self.attn_q)
        attn_q_out = F.softmax(attn_q_weights, dim=1)
        doc_rep = torch.sum(lstm2_out*attn_q_out, dim=1) #[batch_size, hidden_dim * 2]

        accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss = self.classifier_layer(doc_rep, accu_labels, law_labels, term_labels)  # [batch_size, 3]
        return accu_loss, law_loss, term_loss, accu_predicts, law_predicts, term_predicts


    def get_hidden_state(self, input_facts, accu_labels, law_labels, term_labels, input_sentences_lens, input_doc_len):
        batch_size, max_sent_num, max_sent_seq_len = input_facts.size()
        doc_word_embeds = self.word_embeddings_layer.forward(input_facts.view(batch_size*max_sent_num, -1)) #[batch_size*max_sent_num, max_doc_seq_len, word_emb_size]

        hidden = None
        lstm1_out, hidden = self.lstm_layer1.forward(doc_word_embeds, hidden)
        # [batch_size*max_sent_num, max_sent_seq_len, hidden_dim * 2]
        attn_p_weights = torch.matmul(lstm1_out, self.attn_p)
        attn_p_outs = F.softmax(attn_p_weights, dim=1)
        doc_sent_rep = torch.sum(lstm1_out * attn_p_outs, dim=1).view(batch_size, max_sent_num, -1) #[batch_size, max_sent_num, hidden_dim*2]
        
        hidden = None
        lstm2_out, hidden = self.lstm_layer2.forward(doc_sent_rep, hidden)
        attn_q_weights = torch.matmul(lstm2_out, self.attn_q)
        attn_q_out = F.softmax(attn_q_weights, dim=1)
        doc_rep = torch.sum(lstm2_out*attn_q_out, dim=1) #[batch_size, hidden_dim * 2]

        return doc_rep


if __name__ == '__main__':
   print(datetime.datetime.now())
