#-*- coding:utf-8 _*-
# @Author: Leilei Gan
# @Time: 2020/05/13
# @Contact: 11921071@zju.edu.cn


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import datetime
import utils.config
import random


BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


class LawModel(nn.Module):

    def __init__(self, config: utils.config.Data):
        super(LawModel, self).__init__()
        self.word_dim = config.word_emb_dim
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_embedding_layer = torch.nn.Embedding(config.word_alphabet_size, config.word_emb_dim, padding_idx=0)
        self.rule1_lambda = config.rule1_lambda
        self.use_logical = True

        if config.pretrain_word_embedding is not None:
            self.word_embedding_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embedding_layer.weight.requires_grad = False
        else:
            self.word_embedding_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.convs = nn.ModuleList(nn.Conv2d(1, item[1], (item[0], self.word_dim)) for item in
                                   list(zip(config.filters_size, config.num_filters)))
        self.cnn_dropout = nn.Dropout(config.HP_lstmdropout)

        self.claim_classifier = torch.nn.Linear(config.HP_hidden_dim, 3)
        self.claim_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)
        print('use logical:', self.use_logical)

    def conv_and_pool(self, input_x, conv):
        x = F.relu(conv(input_x)).squeeze(3)  # [batch_size, out_channel, seq_len - kernel_size + 1, hidden_size]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def claims_logical_operator(self, claim_logits, claim_softmax_attns, claims_num, claims_type, rate_fact_labels, claim_predicts, facts_y):

        # claim_logits: [batch_size, max_claims_num, 3]

        mask = torch.zeros_like(claim_logits)

        for idx, num in enumerate(claims_num.long().tolist()):
            # Rule1: 其余诉请为2 -> 诉讼费为2

            if '诉讼费' in claims_type[idx] and len(claims_type[idx]) > 1:
                ss_index = claims_type[idx].index('诉讼费')
                if 0 not in claim_predicts[idx, :ss_index].cpu().tolist() and 1 not in claim_predicts[idx, :ss_index].cpu().tolist() \
                    and 0 not in claim_predicts[idx, ss_index+1:].cpu().tolist() and 1 not in claim_predicts[idx, ss_index+1:].cpu().tolist():
                    mask[idx, ss_index, 2] = max(0, claim_softmax_attns[idx, :ss_index, 2].sum().item() + \
                                                    claim_softmax_attns[idx, ss_index+1:, 2].sum().item() - (num - 1) + 1)

            if '本息' in claims_type[idx]:
                bx_index = claims_type[idx].index('本息')
                # Rule4: 未约定利率 and 逾期利息起始时间非法 -> 非本息成立
                if facts_y[idx, 12].item() == 2 and facts_y[idx, 3].item() == 2:
                    mask[idx, bx_index, 2] = -1e2


                # Rule5: 利息超过银行同期利率四倍 -> 非本息成立
                if rate_fact_labels[idx].item() == -1:
                    #and facts_predicts[idx, 11].item() == 1:
                    # pre_condition = facts_predicts_prob[idx, 11, 1].log()
                    mask[idx, bx_index, 2] = -1e2


            if '利息' in claims_type[idx]:
                lx_index = claims_type[idx].index('利息')
                # Rule4: 未约定利率 and 逾期利息起始时间非法 -> 非本息成立
                if facts_y[idx, 12].item() == 2 and facts_y[idx, 3].item() == 2:
                    mask[idx, lx_index, 2] = -1e2

                # Rule5: 利息超过银行同期利率四倍 -> 非本息成立
                if rate_fact_labels[idx].item() == -1:
                    #and facts_predicts[idx, 11].item() == 1:
                    # pre_condition = facts_predicts_prob[idx, 11, 1].log()
                    mask[idx, lx_index, 2] = -1e2


        constrained_claim_logits = claim_logits + mask

        return constrained_claim_logits


    def claim_classifier_layer(self, claim_out, doc_out, fact_aware_doc_rep, input_claims_y, claims_num, claims_type, rate_fact_labels, input_fact):
        """

        :param claim_out: [batch_size * max_claims_num, max_claims_sequence_len, 2 * hidden_dim]
        :param doc_out: [batch_size, max_doc_sequence_len, 2 * hidden_dim]
        :param fact_aware_doc_rep:
        :param input_claims_y:
        :return:
        """
        batch_size, max_claims_num = input_claims_y.size()
        fact_aware_doc_rep = torch.mean(fact_aware_doc_rep, dim=1).squeeze(1).repeat(max_claims_num, 1) #[batch_size * max_claims_num, 2 * hidden_dim]
        doc_out = doc_out.repeat(max_claims_num, 1, 1)

        claims_doc_out = torch.cat((doc_out, claim_out), dim=1) #[batch_size * max_claims_num, max_sequence_len, 2 * hidden_dim]
        attn_weights = torch.matmul(claims_doc_out, self.attn_p) #[batch_size * max_claims_num, max_seuqnce_len, 1]
        attn_logits = F.softmax(attn_weights, dim=1)
        attn_outs = torch.sum(claims_doc_out * attn_logits, dim=1)

        doc_claim_fact_rep = torch.cat((attn_outs, fact_aware_doc_rep), dim=1)
        claim_logits = self.claim_classifier(doc_claim_fact_rep).view(batch_size, max_claims_num, 3)  # [batch_size * max_claims_num, 3]

        claim_probs = F.softmax(claim_logits, dim=2)
        claim_log_softmax = F.log_softmax(claim_logits, dim=2).view(batch_size * max_claims_num, 3)
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)  # [batch_size * max_claims_num]
        claim_predicts = claim_predicts.view(batch_size, max_claims_num)

        if self.use_logical:
            claim_logits = self.claims_logical_operator(claim_logits, claim_probs, claims_num, claims_type, rate_fact_labels, claim_predicts, input_fact)

        claim_probs = F.softmax(claim_logits, dim=2)
        claim_log_softmax = F.log_softmax(claim_logits, dim=2).view(batch_size * max_claims_num, 3)
        loss_claim = self.claim_loss(claim_log_softmax, input_claims_y.view(batch_size*max_claims_num).long())
        _, claim_predicts = torch.max(claim_log_softmax, dim=1) # [batch_size * max_claims_num]
        claim_predicts = claim_predicts.view(batch_size, max_claims_num)

        return claim_predicts, loss_claim, claim_probs


    def neg_log_likelihood_loss(self, input_x,  input_sentences_lens, input_fact, input_claims_y, input_claims_type,
                                doc_texts, claims_texts, batch_claim_ids, input_claims_num, input_claims_len,
                                rate_fact_labels, wenshu_texts):
        """
        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """

        batch_size, max_claims_num, max_claims_seq_len = batch_claim_ids.size()
        # process fact description
        input_x = input_x.repeat(max_claims_num, 1).view(batch_size, max_claims_num, -1) #[batch_size, max_doc_seq_len] -> [batch_size, max_claims_num, max_doc_seq_len]
        input_x = torch.cat((input_x, batch_claim_ids), dim=2)
        doc_claim_word_embeds = self.word_embedding_layer.forward(input_x.view(batch_size*max_claims_num, -1)) #[batch_size * max_claims_num, max_len, word_embed_size]

        word_embeds = doc_claim_word_embeds.unsqueeze(1)  # [batch_size*max_claims_num, 1, max_len, word_embed_size]
        doc_claim_rep = torch.cat(tuple([self.conv_and_pool(word_embeds, conv) for conv in self.convs]), 1) #[batch_size * max_claims_num, 64 * 4]

        out = self.cnn_dropout(doc_claim_rep)

        claim_outputs = self.claim_classifier(out) # [batch_size*max_claims_num, 3]
        claim_log_softmax = torch.nn.functional.log_softmax(claim_outputs, dim=1)
        loss_claim = self.claim_loss(claim_log_softmax, input_claims_y.view(batch_size*max_claims_num).long())
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)
        claim_predicts = claim_predicts.view(batch_size, max_claims_num)

        return loss_claim,  claim_predicts


    def forward(self, input_x,  input_sentences_lens, input_fact, input_claims_y):
        #, input_sample_mask, input_sentences_mask
        doc_rep = self.doc_encoder.forward(input_x, input_sentences_lens)  # [batch_size, hidden_dim]
        claim_outputs = self.claim_classifier(doc_rep)
        claim_log_softmax = torch.nn.functional.log_softmax(claim_outputs, dim=1)
        _, claim_predicts = torch.max(claim_log_softmax, dim=1)

        fact_logits = self.fact_classifier(doc_rep)  # [batch_size, hidden_dim] -> [batch_size, fact_num]
        fact_predicts_prob = self.fact_sigmoid(fact_logits)
        fact_predicts = torch.round(fact_predicts_prob)  # [batch_size, fact_num]

        return fact_predicts, claim_predicts


if __name__ == '__main__':
   print(datetime.datetime.now())
