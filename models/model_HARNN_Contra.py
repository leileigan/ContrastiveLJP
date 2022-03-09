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

BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


def SupConLoss(temperature=0.07, contrast_mode='all', features=None, labels=None, mask=None):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
    """
    features: [bsz, n_views, dim]

    """
    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 1 indicates two items belong to same class
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # num of views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (bsz * views, dim)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature  # (bsz * views, dim)
        anchor_count = contrast_count  # num of views
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)  # (bsz, bsz)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bsz, 1)
    logits = anchor_dot_contrast - logits_max.detach()  # (bsz, bsz) set max_value in logits to zero
    # logits = anchor_dot_contrast

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                0)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    mask = mask * logits_mask  # 1 indicates two items belong to same class and mask-out itself

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # (anchor_cnt * bsz, contrast_cnt * bsz)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # compute mean of log-likelihood over positive
    if 0 in mask.sum(1):
        raise ValueError('Make sure there are at least two instances with the same class')
    # temp = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # loss
    # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss


class LawModel(nn.Module):

    def __init__(self, config: Config):
        super(LawModel, self).__init__()

        self.word_dim = config.word_emb_dim
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_embeddings_layer = torch.nn.Embedding(config.pretrain_word_embedding.shape[0], config.pretrain_word_embedding.shape[1], padding_idx=0)

        if config.pretrain_word_embedding is not None:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embeddings_layer.weight.requires_grad = False
        else:
            self.word_embeddings_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.lstm_layer1 = nn.LSTM(self.word_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(self.hidden_dim * 2, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.attn_p = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))
        self.attn_q = nn.Parameter(torch.Tensor(self.hidden_dim * 2, 1))

        nn.init.uniform_(self.attn_p, -0.1, 0.1)
        nn.init.uniform_(self.attn_q, -0.1, 0.1)

        self.accu_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.hidden_dim * 2, config.accu_label_size)
        )
        self.accu_loss = torch.nn.NLLLoss()
        
        self.law_classifier = torch.nn.Linear(self.hidden_dim * 2, config.law_label_size)
        self.law_loss = torch.nn.NLLLoss()
        
        self.term_classifier = torch.nn.Linear(self.hidden_dim * 2, config.term_label_size)
        self.term_loss = torch.nn.NLLLoss()

        self.temperature = config.temperature
    

    def classifier_layer(self, doc_out, accu_labels, law_labels, term_labels):
        """
        :param doc_out: [batch_size, 2 * hidden_dim]
        :param accu_labels: [batch_size]
        :param law_labels: [batch_size]
        :param term_labels: [batch_size]
        """
        accu_logits = self.accu_classifier(doc_out)  # [batch_size, accu_label_size]
        doc_out_dropout = doc_out.clone().detach()
        accu_logits_dropout = self.accu_classifier(doc_out_dropout)
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

        if accu_labels is not None:
            cl_loss = SupConLoss(temperature=self.temperature,
                                 features=torch.stack([accu_logits, accu_logits_dropout], dim=1),
                                 labels=accu_labels)

        return accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss, cl_loss




    def forward(self, input_facts, accu_labels, law_labels, term_labels):
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

        accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss, cl_loss = self.classifier_layer(doc_rep, accu_labels, law_labels, term_labels)  # [batch_size, 3]
        return accu_loss, law_loss, term_loss, cl_loss, accu_predicts, law_predicts, term_predicts



if __name__ == '__main__':
   print(datetime.datetime.now())
