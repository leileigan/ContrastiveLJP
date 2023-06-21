#-*- coding:utf-8 _*-  
# @Author: Leilei Gan
# @Time: 2020/05/13
# @Contact: 11921071@zju.edu.cn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import *
import datetime
import utils.config
import random


BERT_MODEL_PATH = "/mnt/data/ganleilei/chinese_L-12_H-768_A-12/"
SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


class ClaimEncoder(nn.Module):
    def __init__(self, config: utils.config.Data, word_embedding_layer):
        super(ClaimEncoder, self).__init__()
        self.word_embedding_layer = word_embedding_layer

        self.lstm = nn.LSTM(config.word_emb_dim, hidden_size=config.HP_hidden_dim // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attn_p = nn.Parameter(torch.Tensor(config.HP_hidden_dim, 1))
        nn.init.uniform_(self.attn_p, -0.1, 0.1)

    def forward(self, input_claims_x, input_claims_nums, input_claims_lens):
        """

        :param input_claims_x: [batch_size, max_claim_num, max_sentence_len]
        :param input_claims_nums
        :param input_claims_lens
        :return:
        """
        word_embs = self.word_embedding_layer(input_claims_x)
        batch_size, max_claim_num, max_sentence_len = input_claims_x.size()
        lstm_input = word_embs.view(batch_size*max_claim_num, max_sentence_len, -1)
        hidden = None
        lstm_out, hidden = self.lstm.forward(lstm_input, hidden) # [batch_size * max_claim_num, max_sentence_len, hidden_dim]
        attn_weights = F.softmax(torch.matmul(lstm_out, self.attn_p))
        attn_outs = lstm_out * attn_weights
        claim_outs = torch.sum(attn_outs, dim=1)
        # [batch_size * max_claim_num, max_sentence_len, hidden_dim] -> [batch_size * max_claim_num, hidden_dim]
        claim_outs = claim_outs.view(batch_size, max_claim_num, -1)
        # [batch_size * max_claim_num, hidden_dim] -> [batch_size, max_claim_num, hidden_dim]
        return claim_outs

class DocEncoder(nn.Module):
    def __init__(self, config: utils.config.Data):

        super(DocEncoder, self).__init__()

        self.word_embeddings_layer = torch.nn.Embedding(config.word_alphabet_size, config.word_emb_dim, padding_idx=0)

        if config.pretrain_word_embedding is not None:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embeddings_layer.weight.requires_grad = False
        else:
            self.word_embeddings_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_dim = config.word_emb_dim

        print('filter sizes:', config.filters_size)
        print('num filters:', config.num_filters)

        self.convs = nn.ModuleList(nn.Conv2d(1, item[1], (item[0], self.word_dim)) for item in list(zip(config.filters_size, config.num_filters)))
        self.cnn_dropout = nn.Dropout(config.HP_lstmdropout)

    def conv_and_pool(self, input_x, conv):
        x = F.relu(conv(input_x)).squeeze(3) #[batch_size, out_channel, seq_len - kernel_size + 1, hidden_size]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_x, input_sentences_len):

        word_embeds = self.word_embedding_layer.forward(input_x) # [batch_size, seq_len, word_embed_size]
        # word_embeds = self.word_emb_dropout(word_embeds)
        word_embeds = word_embeds.unsqueeze(1) # [batch_size, 1, seq_len, word_embed_size]
        out = torch.cat(tuple([self.conv_and_pool(word_embeds, conv) for conv in self.convs]), 1)
        out = self.cnn_dropout(out)
        return out # batch_size, hidden_dim


class LawModel(nn.Module):

    def __init__(self, config: utils.config.Data):
        super(LawModel, self).__init__()

        self.doc_encoder = DocEncoder(config)

        self.fact_dense = torch.nn.Linear(config.HP_hidden_dim, 512)
        self.fact_activation = torch.nn.ReLU()
        self.fact_classifier = torch.nn.Linear(512, config.fact_num)
        self.fact_drop = torch.nn.Dropout(config.HP_lstmdropout)
        self.fact_sigmoid = torch.nn.Sigmoid()

        self.accu_classifier = torch.nn.Linear(config.HP_hidden_dim, 3)
        self.law_classifier = torch.nn.Linear()
        self.term_classifier = torch.nn.Linear()

        self.bce_loss = torch.nn.BCELoss()
        self.nll_loss = torch.nn.NLLLoss(ignore_index=-1, size_average=True)

        if config.HP_gpu:
            self.doc_encoder = self.doc_encoder.cuda()

            self.fact_classifier = self.fact_classifier.cuda()
            self.fact_activation = self.fact_activation.cuda()
            self.fact_drop = self.fact_drop.cuda()
            self.fact_sigmoid = self.fact_sigmoid.cuda()

            self.accu_classifier = self.claim_classifier.cuda()
            self.law_classifier = self.law_classifier.cuda()
            self.term_classifier = self.term_classifier.cuda()

            self.bce_loss = self.bce_loss.cuda()
            self.nll_loss = self.nll_loss.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def neg_log_likelihood_loss(self, input_facts, law_labels, accu_labels, term_labels, input_sentences_lens, input_doc_len):
        """

        :param input_x:
        :param input_sentences_nums
        :param input_sentences_lens:
        :param input_fact: [batch_size, fact_num]
        :param input_claims_y: [batch_size]
        :return:
        """
        fact_rep = self.doc_encoder.forward(input_facts,  input_sentences_lens) # [batch_size, hidden_dim]
        accu_outputs = self.claim_classifier(fact_rep) # [batch_size, 3]
        accu_log_softmax = F.log_softmax(accu_outputs, dim=1)
        accu_loss = self.nll_loss(accu_log_softmax, accu_labels.long())
        _, accu_preds = torch.max(accu_log_softmax, dim=1)

        law_rep = self.fact_dense(fact_rep)  # [batch_size, hidden_dim] -> [batch_size, fact_num]
        law_logits = self.law_classifier(self.fact_activation(law_rep))
        law_loss = self.bce_loss(law_logits, law_labels)
        law_preds = torch.round(law_logits)  # [batch_size, fact_num]

        term_rep = self.fact_dense(fact_rep)  # [batch_size, hidden_dim] -> [batch_size, fact_num]
        term_logits = self.fact_classifier(self.fact_activation(term_rep))
        term_predicts_prob = self.fact_sigmoid(term_logits)
        term_loss = self.bce_loss(term_predicts_prob, term_labels)
        term_preds = torch.round(term_predicts_prob)  # [batch_size, fact_num]

        return accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds 


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
   debat_encoder = DebatEncoder(200, 200)
   input = torch.randn(32, 60, 50, 200) # [batch_size, max_utterance_num, max_seq_len, word_embs_dim]
   output = debat_encoder.forward(input)
   print(output.size())
