# -*- coding: utf-8 -*-
# @Time    : 2022/01/16 3:33 下午
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
from transformers import BertForSequenceClassification
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)


"""
class WeightingModel(nn.Module):
    
    def __init__(self,batch_size,hidden_size,emotion_size,encoder_type="electra"):
        super(WeightingModel, self).__init__()

        if encoder_type == "electra":
            options_name = "google/electra-base-discriminator"
            self.encoder_supcon_2 = BertForSequenceClassification.from_pretrained(options_name,num_labels=emotion_size)

            ## to make it faster
            self.encoder_supcon_2.electra.encoder.config.gradient_checkpointing=True


    def forward(self, text,attn_mask):

        supcon_fea_2 = self.encoder_supcon_2(text,attn_mask,output_hidden_states=True,return_dict=True)

        return supcon_fea_2.logits
"""

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

        self.warm_epoch = config.warm_epoch
    

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

        return accu_logits, accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss


    def forward(self, input_facts, accu_labels, law_labels, term_labels):
        """
        Args:
            input_facts: [batch_size, max_sent_num, max_sent_seq_len]
            input_laws: [law_num, max_law_seq_len]
            law_labels: [batch_size]
            accu_labels : [batch_size]
            term_labels : [batch_size]
        Returns:
            [type]: [description]
        """
        self.lstm_layer1.flatten_parameters()
        self.lstm_layer2.flatten_parameters()
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

        accu_logits, accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss = self.classifier_layer(doc_rep, accu_labels, law_labels, term_labels)
        return accu_logits, accu_loss, law_loss, term_loss, accu_predicts, law_predicts, term_predicts


class MoCo(nn.Module):
    def __init__(self, config: Config, dim=119):
        super(MoCo, self).__init__()

        self.K = config.moco_queue_size
        self.m =config.moco_momentum
        self.T = config.moco_temperature
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = LawModel(config)
        self.encoder_k = LawModel(config)
        self.config = config
       
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("feature_queue", torch.randn(self.K, dim))
        self.feature_queue = nn.functional.normalize(self.feature_queue.cuda(), dim=1)

        self.register_buffer("label_queue", torch.randint(-1, 0, (self.K, 1)))
        self.label_queue = self.label_queue.cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, accu_label_lists):
        """
        keys: [bsz, 119]
        accu_label_lists: [bsz]
        """
        batch_size = keys.shape[0]
        label_keys = accu_label_lists.unsqueeze(1)
        ptr = int(self.queue_ptr)

        if ptr+batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_labels = label_keys[: head_size]

            end_size = ptr + batch_size - self.K
            end_keys = keys[head_size:]
            end_labels = label_keys[head_size: ]

            self.feature_queue[ptr:, :] = head_keys
            self.label_queue[ptr:, :] = head_labels

            self.feature_queue[:end_size, :] = end_keys
            self.label_queue[:end_size, :] = end_labels
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.feature_queue[ptr:ptr + batch_size, :] = keys
            self.label_queue[ptr:ptr+batch_size, :] = label_keys

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
        self.label_ptr[0] = ptr


    def _get_contra_loss(self, query, accu_label_lists):
        label_1_list = accu_label_lists.eq(1).nonzero(as_tuple=True)
        if label_1_list[0].size(0) > 0: 
            label_1_index = label_1_list[0][0]
        else:
            label_1_index = -1
        accu_label_lists = accu_label_lists.unsqueeze(1)
        mask = torch.eq(accu_label_lists, self.label_queue.T).float() #[bsz, queue_size]
        query_queue_product = torch.einsum('nc,kc->nk', [query, self.feature_queue.clone().detach()])
        cos_sim = query_queue_product / torch.einsum('nc,kc->nk', [torch.norm(query, dim=1).unsqueeze(1), torch.norm(self.feature_queue.clone().detach(), dim=1).unsqueeze(1)])
        cos_sim_with_t = cos_sim/ self.T #[bsz, queue_size]
        neg_mask = torch.ne(accu_label_lists, self.label_queue.T).float() #[bsz, queue_size]
        # negative_sim: [bsz, 1] #所有负样本的和
        negative_sim_sum = (torch.exp(cos_sim_with_t) * neg_mask).sum(1, keepdim=True) #[bsz, 1]
        # positive_sim: [bsz, queue_size] #样本和队列里每个正样本的乘积
        positive_sim = torch.exp(cos_sim_with_t) #[bsz, queue_size]
        # sim_sum:  [bsz, queue_size]
        exp_logits = negative_sim_sum + positive_sim
        log_exp_logits = torch.log(exp_logits + 1e-12)
        # exp_logits = torch.exp(cos_sim_with_t)
        # log_exp_logits = torch.log((exp_logits * sum_mask).sum(1, keepdim=True) + 1e-12)
        log_prob = cos_sim_with_t - log_exp_logits #[bsz, queue_size]
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12) #[bsz]
        loss = -mean_log_prob_pos.mean()
        # if label_1_index != -1:
        #     # print("label queue:", self.label_queue.squeeze())
        #     # print("positive mask:", mask[label_1_index])
        #     print("positive cos sim select value:", torch.masked_select(
        #         cos_sim[label_1_index], mask[label_1_index].bool()).sort(dim=-1, descending=False)[0][:300])
        #     print("positive cos sim average value:", torch.masked_select(cos_sim[label_1_index], mask[label_1_index].bool()).mean())
        #     # print("hard neg mask:", hard_neg_mask[label_1_index])
        #     # print("hard neg mask select value:", torch.masked_select(query_queue_product[label_1_index], hard_neg_mask[label_1_index].bool()))
        #     print("neg cos sim select value:", torch.masked_select(
        #         cos_sim[label_1_index], neg_mask[label_1_index].bool()).sort(dim=-1, descending=True)[0][:300])
        #     print("neg cos sim average value:", torch.masked_select(
        #         cos_sim[label_1_index], neg_mask[label_1_index].bool()).mean())
        #     # print("log exp logits select value:", log_exp_logits[label_1_index])
        #     # print("log prob select value:", torch.masked_select(log_prob[label_1_index], mask[label_1_index].bool())[0][:300])
        #     # print("mean log prob select value:", mean_log_prob_pos)
        #     # print("query queue product:", query_queue_product[label_1_index])
        return loss, label_1_index


    def forward(self, fact_list, accu_label_lists, law_label_lists, term_lists):
        # compute query features
        q, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds = self.encoder_q(fact_list, accu_label_lists, law_label_lists, term_lists)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # print("accu loss:", q.size())
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k, _, _, _, _, _, _ = self.encoder_k(fact_list, accu_label_lists, law_label_lists, term_lists)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, accu_label_lists)
        contra_loss, _ = self._get_contra_loss(q, accu_label_lists)

        return contra_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds

    def predict(self, fact_list, raw_fact_list, accu_label_lists, law_label_lists, term_lists, epoch_idx, name):
        # compute query features
        q, _, _, _, accu_preds, law_preds, term_preds = self.encoder_q(fact_list, accu_label_lists, law_label_lists, term_lists)  # queries: NxC
        contra_loss, label_1_index = self._get_contra_loss(q, accu_label_lists)
        # if label_1_index != -1:
        #     print(
        #         f"Epoch: {epoch_idx}, Name: {name}, contra loss: {contra_loss.item()}, accu preds: {accu_preds[label_1_index].item()}, ground truth label: {accu_label_lists[label_1_index].item()}")
        #     print(''.join(raw_fact_list[label_1_index]))
        
        return accu_preds, law_preds, term_preds


if __name__ == '__main__':
   print(datetime.datetime.now())
