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
from transformers  import BertModel, BertTokenizer
import sys

BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

class LawModel(nn.Module):

    def __init__(self, config):
        super(LawModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.accu_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(config.hidden_size, config.accu_label_size)
        )
        self.accu_loss = torch.nn.NLLLoss()
        
        self.law_classifier = torch.nn.Linear(config.hidden_size, config.law_label_size)
        self.law_loss = torch.nn.NLLLoss()
        
        self.term_classifier = torch.nn.Linear(config.hidden_size, config.term_label_size)
        self.term_loss = torch.nn.NLLLoss()
        self.temperature = 0.7

    def classifier_layer(self, doc_out, accu_labels, law_labels, term_labels): #
        """
        :param doc_out: [batch_size, 2 * hidden_dim]
        :param accu_labels: [batch_size]
        :param law_labels: [batch_size]
        :param term_labels: [batch_size]
        """
        accu_logits = self.accu_classifier(doc_out)  # [batch_size, accu_label_size]
        # doc_out_dropout = doc_out.clone().detach()
        #accu_logits_dropout = self.accu_classifier(doc_out_dropout)
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

        #if accu_labels is not None:
        #    cl_loss = SupConLoss(temperature=self.temperature,
        #                         features=torch.stack([accu_logits, accu_logits_dropout], dim=1),
        #                         labels=accu_labels)

        return accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss, accu_logits

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
        x = fact_list
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs.pooler_output
        accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss, accu_logits = self.classifier_layer(pooled, accu_labels, law_labels, term_labels)  # [batch_size, 3] 
        return accu_logits, accu_loss, law_loss, term_loss, accu_predicts, law_predicts, term_predicts

class MoCo(nn.Module):
    def __init__(self, config, dim=119, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = LawModel(config)
        self.encoder_k = LawModel(config)
        self.config = config
       
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("feature_queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.feature_queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the label queue
        self.register_buffer("label_queue", torch.randint(-1, 0, (K, 1)))
        self.register_buffer("label_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, accu_label_lists):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # if ptr+batch_size > self.K:
        #     head_size = self.K - ptr
        #     head_keys = keys[: head_size]
        #     head_labels = labels[: head_size]
        #     end_size = ptr + batch_size - self.K
        #     end_keys = keys[head_size:]
        #     end_labels = labels[head_size:]
        #     self.queue[ptr:, :] = head_keys
        #     self.label_queue[ptr:, :] = head_labels
        #     self.queue[:end_size, :] = end_keys
        #     self.label_queue[:end_size, :] = end_labels
        # else:
        #     # replace the keys at ptr (dequeue and enqueue)
        #     self.queue[ptr:ptr + batch_size, :] = keys
        #     self.label_queue[ptr:ptr+batch_size, :] = keys
        
        assert self.K % batch_size == 0
        self.queue[ptr:ptr+batch_size, :] = keys
        self.label_queue[ptr:ptr+batch_size, :] = accu_label_lists.unsqueeze(1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        self.label_ptr[0] = ptr


    def _get_contra_loss(self, query, accu_label_lists):
        accu_label_lists = accu_label_lists.unsqueeze(1)
        mask = torch.eq(accu_label_lists, self.label_queue.T).float().cuda()
        query_queue_product = torch.einsum('nc,kc->nk', [query, self.queue.clone().detach().cuda()])
        query_queue_product =  query_queue_product / self.T
        exp_logits = torch.exp(query_queue_product)
        log_prob = query_queue_product - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

    def forward(self, fact_list, accu_label_lists, law_label_lists, term_lists):
        # compute query features
        q, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds = self.encoder_q(fact_list, accu_label_lists, law_label_lists, term_lists)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k, _, _, _, _, _, _ = self.encoder_k(fact_list, accu_label_lists, law_label_lists, term_lists)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, accu_label_lists)
        contra_loss = self._get_contra_loss(q, accu_label_lists)

        return contra_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds

if __name__ == '__main__':
   print(datetime.datetime.now())
