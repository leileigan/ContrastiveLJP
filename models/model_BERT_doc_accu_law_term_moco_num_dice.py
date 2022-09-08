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
from transformers import BertConfig, BertModel
import pickle as pk
from .model_DICE import DICE

num_target_classes = [83, 11, 55, 16, 37, 102, 52, 107, 61, 12, 58, 75, 78, 38, 69, 60, 54, 94, 110, 88, 19, 30, 59, 26, 51, 118, 86, 49, 7] # number sensitive classes
class LawModel(nn.Module):

    def __init__(self, config):
        super(LawModel, self).__init__()

        self.config = config
        self.bert_config = BertConfig.from_pretrained(config.bert_path, output_hidden_states=False)
        self.bert = BertModel.from_pretrained(config.bert_path, config=self.bert_config)
        for param in self.bert.parameters():
            param.requires_grad = True        

        self.accu_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert_config.hidden_size, config.mlp_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.mlp_size, config.accu_label_size)
        )
        self.accu_loss = torch.nn.NLLLoss()
        
        self.law_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert_config.hidden_size, config.mlp_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.mlp_size, config.law_label_size)
        )
        self.law_loss = torch.nn.NLLLoss()
        
        self.term_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert_config.hidden_size, config.mlp_size),
            torch.nn.ReLU(),
            torch.nn.Linear(cofig.mlp_size + 512, config.mlp_size),
            torch.nn.ReLU(),
            torch.nn.Linear(cofnig.mlp_size, config.term_label_size)
        )
        self.term_loss = torch.nn.NLLLoss()

        #数字相关编码
        print("dice config path:", config.dice_config_path)
        print("dice model path:", config.dice_model_path)
        self.dice_config = pk.load(open(config.dice_config_path, "rb"))
        self.dice_model = DICE(self.dice_config)
        self.dice_model.load_state_dict(torch.load(config.dice_model_path))
        for p in self.dice_model.parameters():
            p.requires_grad = True

               
    def classifier_layer(self, doc_out, accu_labels, law_labels, term_labels, money_hidden):
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
        
        term_rep = self.term_classifier[1](self.term_classifier[0](doc_out))
        num_doc_out = torch.cat([term_rep, money_hidden], dim=-1)
        term_logits = self.term_classifier[4](self.term_classifier[3](self.term_classifier[2](num_doc_out)))  # [batch_size, term_label_size]
        term_log_softmax = F.log_softmax(term_logits, dim=-1)
        term_loss = self.term_loss(term_log_softmax, term_labels)
        _, term_predicts = torch.max(term_log_softmax, dim=1) # [batch_size * max_claims_num]

        accu_rep = self.accu_classifier[1](self.accu_classifier[0](doc_out))
        law_rep = self.law_classifier[1](self.law_classifier[0](doc_out))

        return accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss, accu_rep, law_rep, term_rep


    def forward(self, input_facts, type_ids_list, attention_mask_list, accu_labels, law_labels, term_labels, money_amount_lists, drug_weight_lists):
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
        # encoder num
        num_mask = torch.eq(accu_labels.unsqueeze(1), torch.tensor(num_target_classes).cuda().unsqueeze(1).T).float() #
        num_mask = num_mask.sum(1).unsqueeze(-1) #[bsz]
        money_amount_hidden, drug_weight_hidden = self.dice_model.encode_num(money_amount_lists, drug_weight_lists) 
        # num1_hidden, num2_hidden = money_amount_hidden[0], money_amount_hidden[1]
        # dot_product = torch.matmul(num1_hidden, num2_hidden.T)
        # norm = torch.norm(num1_hidden, p=2, dim=-1) * torch.norm(num2_hidden, p=2, dim=-1)
        # diff = 1 - dot_product / norm
        # print("money num1:", money_amount_lists[0].tolist())
        # print("money num2:", money_amount_lists[1].tolist())
        # print("diff:", diff.item())

        outputs = self.bert.forward(input_ids=input_facts, attention_mask=attention_mask_list, token_type_ids=type_ids_list)
        doc_rep = outputs.pooler_output

        accu_preds, accu_loss, law_preds, law_loss, term_preds, term_loss, accu_rep, law_rep, term_rep = self.classifier_layer(doc_rep, accu_labels, law_labels, term_labels, money_amount_hidden)  # [batch_size, 3]

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
        self.register_buffer("doc_feature_queue", torch.randn(self.K, self.encoder_q.bert_config.hidden_size))
        self.register_buffer("accu_feature_queue", torch.randn(self.K, config.mlp_size))
        self.register_buffer("law_feature_queue", torch.randn(self.K, config.mlp_size))
        self.register_buffer("term_feature_queue", torch.randn(self.K, config.mlp_size))

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

    def forward(self, legals, type_ids_list, attention_mask_list, accu_label_lists, law_label_lists, term_lists, money_amount_lists, drug_weight_lists):
        # compute query features
        accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds, q_doc_feature, q_accu_feature, q_law_feature, q_term_feature = self.encoder_q(
            legals, type_ids_list, attention_mask_list, accu_label_lists, law_label_lists, term_lists, money_amount_lists, drug_weight_lists)
        
        q_doc_feature = nn.functional.normalize(q_doc_feature, dim=1)
        q_accu_feature = nn.functional.normalize(q_accu_feature, dim=1)
        q_law_feature = nn.functional.normalize(q_law_feature, dim=1)
        q_term_feature = nn.functional.normalize(q_term_feature, dim=1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, _, _, _, _, _, k_doc_feature, k_accu_feature, k_law_feature, k_term_feature = self.encoder_q(legals, type_ids_list, attention_mask_list, accu_label_lists, law_label_lists, term_lists, money_amount_lists, drug_weight_lists)
            
            k_doc_feature = nn.functional.normalize(k_doc_feature, dim=1)
            k_accu_feature = nn.functional.normalize(k_accu_feature, dim=1)
            k_law_feature = nn.functional.normalize(k_law_feature, dim=1)
            k_term_feature = nn.functional.normalize(k_term_feature, dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_doc_feature, k_accu_feature, k_law_feature, k_term_feature, accu_label_lists, law_label_lists, term_lists)
        contra_doc_loss, contra_accu_loss, contra_law_loss, contra_term_loss, _ = self._get_contra_loss(q_doc_feature, q_accu_feature, q_law_feature, q_term_feature, accu_label_lists, law_label_lists, term_lists)

        return contra_doc_loss, contra_accu_loss, contra_law_loss, contra_term_loss, accu_loss, law_loss, term_loss,accu_preds, law_preds, term_preds
    
    
    def predict(self, legals, type_ids_list, attention_mask_list, accu_label_lists, law_label_lists, term_lists, money_amount_lists, drug_lists):
        # compute query features
        accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds, q_doc_feature, q_accu_feature, q_law_feature, q_term_feature = self.encoder_q(legals, type_ids_list, attention_mask_list, accu_label_lists, law_label_lists, term_lists, money_amount_lists, drug_lists)
        
        #q = nn.functional.normalize(q, dim=1)
        #contra_loss, label_1_index = self._get_contra_loss(q, accu_label_lists)
        # if label_1_index != -1:
        #     print(
        #         f"Epoch: {epoch_idx}, Name: {name}, contra loss: {contra_loss.item()}, accu preds: {accu_preds[label_1_index].item()}, ground truth label: {accu_label_lists[label_1_index].item()}")
        #     print(''.join(raw_fact_list[label_1_index]))
        
        return accu_preds, law_preds, term_preds

if __name__ == '__main__':
   print(datetime.datetime.now())
