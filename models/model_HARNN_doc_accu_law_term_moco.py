# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 15:33
# @Author  : Leilei Gan
# @Contact : 11921071@zju.edu.cn

from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import torch.optim as optim
import datetime
import math
from typing import List

class LawModel(nn.Module):

    def __init__(self, config):
        super(LawModel, self).__init__()

        self.config = config

        self.word_dim = config.word_emb_dim
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_embeddings_layer = torch.nn.Embedding(config.pretrain_word_embedding.shape[0], config.pretrain_word_embedding.shape[1], padding_idx=0)

        if config.pretrain_word_embedding is not None:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embeddings_layer.weight.requires_grad = False
        else:
            self.word_embeddings_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.lstm_layer1 = nn.LSTM(self.word_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True) # word lstm
        self.lstm_layer2 = nn.LSTM(self.hidden_dim * 2, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True) # sentence lstm

        self.attn_p = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        self.attn_q = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))

        nn.init.uniform_(self.attn_p, -0.1, 0.1)
        nn.init.uniform_(self.attn_q, -0.1, 0.1)

        self.accu_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, config.accu_label_size)
        )
        self.accu_loss = torch.nn.NLLLoss()
        
        self.law_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, config.law_label_size)
        )
        self.law_loss = torch.nn.NLLLoss()
        
        self.term_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, config.term_label_size)
        )
        self.term_loss = torch.nn.NLLLoss()

               
    def classifier_layer(self, doc_out, accu_labels, law_labels, term_labels):
        """
        :param doc_out: [batch_size, 4 * hidden_dim]
        :param law_out: [103, 4 * hidden_dim]
        :param accu_labels: [batch_size]
        :param law_labels: [batch_size]
        :param term_labels: [batch_size]
        """
        # print("doc out size:", doc_out.size())
        # print("law out size:", law_out.size())
        accu_logits = self.accu_classifier(doc_out)  # [batch_size, accu_label_size]
        accu_log_softmax = F.log_softmax(accu_logits, dim=-1)
        accu_loss = self.accu_loss(accu_log_softmax, accu_labels)
        _, accu_predicts = torch.max(accu_log_softmax, dim=-1) # [batch_size, accu_label_size]
        
        law_logits = self.law_classifier(doc_out)  # [batch_size, law_label_size]
        law_log_softmax = F.log_softmax(law_logits, dim=-1)
        law_loss = self.law_loss(law_log_softmax, law_labels)
        _, law_predicts = torch.max(law_log_softmax, dim=1) # [batch_size * max_claims_num]
        
        term_logits = self.term_classifier(doc_out)  # [batch_size, term_label_size]
        term_log_softmax = F.log_softmax(term_logits, dim=-1)
        term_loss = self.term_loss(term_log_softmax, term_labels)
        _, term_predicts = torch.max(term_log_softmax, dim=1) # [batch_size * max_claims_num]

        accu_rep = self.accu_classifier[1](self.accu_classifier[0](doc_out))
        law_rep = self.law_classifier[1](self.law_classifier[0](doc_out))
        term_rep = self.term_classifier[1](self.term_classifier[0](doc_out))

        return accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss, accu_rep, law_rep, term_rep


    def forward(self, input_facts, accu_labels, law_labels, term_labels, input_sentences_lens, input_doc_len):
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

        accu_preds, accu_loss, law_preds, law_loss, term_preds, term_loss, accu_rep, law_rep, term_rep = self.classifier_layer(doc_rep, accu_labels, law_labels, term_labels)  # [batch_size, 3]

        return accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds, doc_rep, accu_rep, law_rep, term_rep 


class MoCo(nn.Module):
    def __init__(self, config):
        super(MoCo, self).__init__()

        self.K = config.moco_queue_size
        self.m =config.moco_momentum
        self.T = config.moco_temperature
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = LawModel(config)
        self.encoder_k = LawModel(config)
        self.config = config
        self.confused_matrix = config.confused_matrix #[119, 119]
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("doc_feature_queue", torch.randn(self.K, config.HP_hidden_dim))
        self.register_buffer("accu_feature_queue", torch.randn(self.K, config.HP_hidden_dim))
        self.register_buffer("law_feature_queue", torch.randn(self.K, config.HP_hidden_dim))
        self.register_buffer("term_feature_queue", torch.randn(self.K, config.HP_hidden_dim))

        self.doc_feature_queue = nn.functional.normalize(self.doc_feature_queue.cuda(), dim=1)
        self.accu_feature_queue = nn.functional.normalize(self.accu_feature_queue.cuda(), dim=1)
        self.law_feature_queue = nn.functional.normalize(self.law_feature_queue.cuda(), dim=1)
        self.term_feature_queue = nn.functional.normalize(self.term_feature_queue.cuda(), dim=1)

        self.register_buffer("accu_label_queue", torch.randint(-1, 0, (self.K, 1)))
        self.accu_label_queue = self.accu_label_queue.cuda()
        self.register_buffer("law_label_queue", torch.randint(-1, 0, (self.K, 1)))
        self.law_label_queue = self.law_label_queue.cuda()
        self.register_buffer("term_label_queue", torch.randint(-1, 0, (self.K, 1)))
        self.term_label_queue = self.term_label_queue.cuda()

        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, doc_keys, accu_keys, law_keys, term_keys, accu_label_lists, law_label_lists, term_label_lists):
        batch_size = accu_keys.shape[0]
        accu_label_keys = accu_label_lists.unsqueeze(1)
        law_label_keys = law_label_lists.unsqueeze(1)
        term_label_keys = term_label_lists.unsqueeze(1)

        ptr = int(self.ptr)
        if ptr+batch_size > self.K:
            head_size = self.K - ptr
            head_doc_keys = doc_keys[: head_size]
            head_accu_keys = accu_keys[: head_size]
            head_law_keys = law_keys[: head_size]
            head_term_keys = term_keys[: head_size]

            head_accu_labels = accu_label_keys[: head_size]
            head_law_labels = law_label_keys[: head_size]
            head_term_labels = term_label_keys[: head_size]

            end_size = ptr + batch_size - self.K
            end_doc_keys = doc_keys[head_size:]
            end_accu_keys = accu_keys[head_size:]
            end_law_keys = law_keys[head_size:]
            end_term_keys = term_keys[head_size:]

            end_accu_labels = accu_label_keys[head_size:]
            end_law_labels = law_label_keys[head_size:]
            end_term_labels = term_label_keys[head_size:]

            # set head keys
            self.doc_feature_queue[ptr:, :] = head_doc_keys
            self.accu_feature_queue[ptr:, :] = head_accu_keys
            self.law_feature_queue[ptr:, :] = head_law_keys
            self.term_feature_queue[ptr:, :] = head_term_keys

            self.accu_label_queue[ptr:, :] = head_accu_labels
            self.law_label_queue[ptr:, :] = head_law_labels
            self.term_label_queue[ptr:, :] = head_term_labels

            # set tail keys
            self.doc_feature_queue[:end_size, :] = end_doc_keys
            self.accu_feature_queue[:end_size, :] = end_accu_keys
            self.law_feature_queue[:end_size, :] = end_law_keys
            self.term_feature_queue[:end_size, :] = end_term_keys

            self.accu_label_queue[:end_size, :] = end_accu_labels
            self.law_label_queue[:end_size, :] = end_law_labels
            self.term_label_queue[:end_size, :] = end_term_labels

        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.doc_feature_queue[ptr:ptr + batch_size, :] = doc_keys
            self.accu_feature_queue[ptr:ptr + batch_size, :] = accu_keys
            self.law_feature_queue[ptr:ptr + batch_size, :] = law_keys
            self.term_feature_queue[ptr:ptr + batch_size, :] = term_keys

            self.accu_label_queue[ptr:ptr+batch_size, :] = accu_label_keys
            self.law_label_queue[ptr:ptr+batch_size, :] = law_label_keys
            self.term_label_queue[ptr:ptr+batch_size, :] = term_label_keys

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.ptr[0] = ptr

    def _get_contra_loss(self, doc_query, accu_query, law_query, term_query, accu_label_lists, law_label_lists, term_label_lists):
        label_1_list = accu_label_lists.eq(1).nonzero(as_tuple=True)
        if label_1_list[0].size(0) > 0:
            label_1_index = label_1_list[0][0]
        else:
            label_1_index = -1
        accu_label_lists = accu_label_lists.unsqueeze(1)
        law_label_lists = law_label_lists.unsqueeze(1)
        term_label_lists = term_label_lists.unsqueeze(1)
        # [bsz, queue_size]
        accu_mask = torch.eq(accu_label_lists, self.accu_label_queue.T).float()
        law_mask = torch.eq(law_label_lists, self.law_label_queue.T).float()
        term_mask = torch.eq(term_label_lists, self.term_label_queue.T).float()

        # compute doc contra loss
        cos_sim_with_t = torch.div(torch.matmul(doc_query, self.doc_feature_queue.clone().detach().T), self.T)
         # for numerical stability
        logits_max, _ = torch.max(cos_sim_with_t, dim=1, keepdim=True)
        logits = cos_sim_with_t - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits + 1e-12)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        positive_accu_mask = accu_mask * term_mask
        mean_log_prob_pos = (positive_accu_mask * log_prob).sum(1) / \
            (positive_accu_mask.sum(1) + 1e-12)  # [bsz]
        doc_loss = -mean_log_prob_pos.mean()

        # compute accu contra loss
        cos_sim_with_t = torch.div(torch.matmul(accu_query, self.accu_feature_queue.clone().detach().T), self.T)
         # for numerical stability
        logits_max, _ = torch.max(cos_sim_with_t, dim=1, keepdim=True)
        logits = cos_sim_with_t - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits + 1e-12)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        positive_accu_mask = accu_mask
        mean_log_prob_pos = (positive_accu_mask * log_prob).sum(1) / \
            (positive_accu_mask.sum(1) + 1e-12)  # [bsz]
        accu_loss = -mean_log_prob_pos.mean()
        
        # compute law contra loss
        cos_sim_with_t = torch.div(torch.matmul(law_query, self.law_feature_queue.clone().detach().T), self.T)
         # for numerical stability
        logits_max, _ = torch.max(cos_sim_with_t, dim=1, keepdim=True)
        logits = cos_sim_with_t - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits + 1e-12)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        positive_law_mask = law_mask
        # positive_law_mask = law_mask
        mean_log_prob_pos = (positive_law_mask * log_prob).sum(1) / \
            (positive_law_mask.sum(1) + 1e-12)  # [bsz]
        law_loss = -mean_log_prob_pos.mean()

        # compute term contra loss
        cos_sim_with_t = torch.div(torch.matmul(term_query, self.term_feature_queue.clone().detach().T), self.T)
         # for numerical stability
        logits_max, _ = torch.max(cos_sim_with_t, dim=1, keepdim=True)
        logits = cos_sim_with_t - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits + 1e-12)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        # positive_term_mask = accu_mask * law_mask * term_mask
        positive_term_mask = term_mask
        mean_log_prob_pos = (positive_term_mask * log_prob).sum(1) / \
            (positive_term_mask.sum(1) + 1e-12)  # [bsz]
        term_loss = -mean_log_prob_pos.mean()

        return doc_loss, accu_loss, law_loss, term_loss, label_1_index

    def forward(self, legals, accu_label_lists, law_label_lists, term_lists, sent_lent, legals_len):
        # compute query features
        accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds, q_doc_feature, q_accu_feature, q_law_feature, q_term_feature = self.encoder_q(
            legals, accu_label_lists, law_label_lists, term_lists, sent_lent, legals_len)
        
        q_doc_feature = nn.functional.normalize(q_doc_feature, dim=1)
        q_accu_feature = nn.functional.normalize(q_accu_feature, dim=1)
        q_law_feature = nn.functional.normalize(q_law_feature, dim=1)
        q_term_feature = nn.functional.normalize(q_term_feature, dim=1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, _, _, _, _, _, k_doc_feature, k_accu_feature, k_law_feature, k_term_feature = self.encoder_k(legals, accu_label_lists, law_label_lists, term_lists, sent_lent, legals_len)
            
            k_doc_feature = nn.functional.normalize(k_doc_feature, dim=1)
            k_accu_feature = nn.functional.normalize(k_accu_feature, dim=1)
            k_law_feature = nn.functional.normalize(k_law_feature, dim=1)
            k_term_feature = nn.functional.normalize(k_term_feature, dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_doc_feature, k_accu_feature, k_law_feature, k_term_feature, accu_label_lists, law_label_lists, term_lists)
        contra_doc_loss, contra_accu_loss, contra_law_loss, contra_term_loss, _ = self._get_contra_loss(q_doc_feature, q_accu_feature, q_law_feature, q_term_feature, accu_label_lists, law_label_lists, term_lists)

        return contra_doc_loss, contra_accu_loss, contra_law_loss, contra_term_loss, accu_loss, law_loss, term_loss,accu_preds, law_preds, term_preds
    
    
    def predict(self, legals, legals_len, sent_lent,accu_label_lists, law_label_lists, term_lists):
        # compute query features
        accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds, q_doc_feature, q_accu_feature, q_law_feature, q_term_feature = self.encoder_q(legals, accu_label_lists, law_label_lists, term_lists, sent_lent, legals_len)
        
        #q = nn.functional.normalize(q, dim=1)
        #contra_loss, label_1_index = self._get_contra_loss(q, accu_label_lists)
        # if label_1_index != -1:
        #     print(
        #         f"Epoch: {epoch_idx}, Name: {name}, contra loss: {contra_loss.item()}, accu preds: {accu_preds[label_1_index].item()}, ground truth label: {accu_label_lists[label_1_index].item()}")
        #     print(''.join(raw_fact_list[label_1_index]))
        
        return accu_preds, law_preds, term_preds

if __name__ == '__main__':
   print(datetime.datetime.now())
