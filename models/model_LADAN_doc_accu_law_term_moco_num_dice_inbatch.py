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
import random, math
from typing import List
import pickle as pk
from .model_DICE import DICE

from law_processed.law_processed import get_law_graph

def dynamic_partition(data, partitions, num_partitions):
    """
    Params:
        data: [N, Dim]
        partitions: [N, 1]
        num_partitions: int
    Return:
        A list of tensors
    """
    # print("data:", data)
    # print("partitions:", partitions)
    # print("number partitions:", num_partitions)
    res = []
    for i in range(num_partitions):
        # print("degree:", i)
        # print(f"cluster of degree {i},", data[torch.LongTensor(partitions).eq(i).nonzero().squeeze().tolist()].size())
        if len(data[torch.LongTensor(partitions).eq(i).nonzero().squeeze().tolist()].size()) < 2:
            res.append(data[torch.LongTensor(partitions).eq(i).nonzero().squeeze().tolist()].unsqueeze(0))
        else: 
            res.append(data[torch.LongTensor(partitions).eq(i).nonzero().squeeze().tolist()])
    return res


def dynamic_stitch(indices, data):
    # print("indices:", indices)
    n = sum(idx.numel() for idx in indices)
    # print("n:", n)
    res  = [None] * n
    for i, data_ in enumerate(data):
        idx = indices[i].view(-1)
        if len(idx) == 0:
            continue
        # print("idx:", idx)
        d = data_.view(idx.numel(), -1)
        # print("d size:", d.size())
        k = 0
        for idx_ in idx: 
            res[idx_] = d[k]
            k += 1
    res = torch.stack(res)
    return res


def attn_encoder_mask(q, k, fc_layer=None, mask=None, weights_regular=None, k_ori=False, div_norm=True):
    """[summary]

    Args:
        q ([torch.FloatTensor]): [..., seq_len, hidden_dim]
        k ([torch.FloatTensor]): [..., seq_len, hidden_dim]
        fc_layer ([type], optional): [description]. Defaults to None.
        mask ([torch.LongTensor], optional): [..., seq_len, 1]. Defaults to None.
        weights_regular ([type], optional): [description]. Defaults to None.
        k_ori (bool, optional): [description]. Defaults to False.
        div_norm (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # print("q size:", q.size())
    # print("k size:", k.size())
    # print("mask size:", mask.size())
    v = k
    k_shape = k.size()

    if fc_layer is not None:
        k = fc_layer(k)
    if not k_ori:
        v = k
    
    scores = torch.sum(k * q, dim=-1) #[..., seq_len, 1]
    if div_norm:
        scores = scores/math.sqrt(k_shape[-1])
    if mask is not None:
        scores = scores.squeeze()*mask
        scores = F.softmax(scores, dim=-1).unsqueeze(-1)
    else:
        scores = torch.softmax(scores, dim=-1)
    
    # print("v size:", v.size())
    # print("scores size:", scores.size())
    res =  torch.sum(v * scores, dim=1)
    # print("return size:", res.size())
    return res, scores

num_target_classes = [83, 11, 55, 16, 37, 102, 52, 107, 61, 12, 58, 75, 78, 38, 69, 60, 54, 94, 110, 88, 19, 30, 59, 26, 51, 118, 86, 49, 7] # number sensitive classes
class LawModel(nn.Module):

    def __init__(self, config):
        super(LawModel, self).__init__()

        self.config = config

        self.word_dim = config.word_emb_dim
        self.hidden_dim = config.HP_hidden_dim // 2
        self.word_embeddings_layer = torch.nn.Embedding(config.pretrain_word_embedding.shape[0], config.pretrain_word_embedding.shape[1], padding_idx=0)
        self.law_relation_threshold = config.law_relation_threshold

        if config.pretrain_word_embedding is not None:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
            self.word_embeddings_layer.weight.requires_grad = False
        else:
            self.word_embeddings_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(config.word_alphabet_size, config.word_emb_dim)))

        self.lstm_layer1 = nn.LSTM(self.word_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True) # word lstm
        self.lstm_layer2 = nn.LSTM(self.hidden_dim * 2, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True) # sentence lstm

        self.attn_p = nn.Parameter(torch.Tensor(1, self.hidden_dim * 2))
        self.attn_q = nn.Parameter(torch.Tensor(1, self.hidden_dim * 2))

        nn.init.uniform_(self.attn_p, -0.1, 0.1)
        nn.init.uniform_(self.attn_q, -0.1, 0.1)

        self.fully_attn_sent_1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        self.fully_attn_doc_1 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim*2)

        self.accu_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, config.accu_label_size)
        )
        self.accu_loss = torch.nn.NLLLoss()
        
        self.law_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, config.law_label_size)
        )
        self.law_loss = torch.nn.NLLLoss()
        
        self.term_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256 + 512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, config.term_label_size)
        )
        self.term_loss = torch.nn.NLLLoss()
        
        #数字相关编码
        self.dice_config = pk.load(open(config.dice_config_path, "rb"))
        self.dice_model = DICE(self.dice_config)
        self.dice_model.load_state_dict(torch.load(config.dice_model_path))
        for p in self.dice_model.parameters():
            p.requires_grad = True

        ### init graph
        # self.law_input, graph_list_1, graph_membership, neigh_index = get_law_graph(self.law_relation_threshold, config.word2id_dict, 15, 100)
        self.law_input, graph_list_1, graph_membership, neigh_index = get_law_graph(self.law_relation_threshold, config.word2id_dict, 15, 100, 'law_processed/law_label2index.pkl', self.config.law_label_size)
        self.law_input = torch.from_numpy(self.law_input).cuda()
        self.max_graph = len(graph_list_1)
        self.deg_list = [len(neigh_index[i]) for i in range(self.config.law_label_size)]
        self.graph_list = list(zip(*graph_membership))[1]
        # print("graph list 1:", graph_list_1)
        # print("neigh index:", neigh_index)
        # print("graph list:", self.graph_list)
        neigh_index = sorted(neigh_index.items(), key=lambda x: len(x[1]))
        self.max_deg = len(neigh_index[-1][1])
        t = 0
        self.adj_list = [[]]
        for i in range(self.config.law_label_size):
            each = neigh_index[i]
            if len(each[1]) != t:
                for j in range(t, len(each[1])):
                    self.adj_list.append([])
                t = len(each[1])
            self.adj_list[-1].append(each[1])
        # print("adj_list:", self.adj_list)

        ### graph distillation operator
        self.gdo_linear = torch.nn.Linear(4*self.hidden_dim, 2*self.hidden_dim)
        self.gdo_mlp = Sequential(
            torch.nn.Linear(2*self.hidden_dim, 2*self.hidden_dim),
            torch.nn.Tanh()
        )

        ### re-encoder
        self.re_encoder_linear_1 = torch.nn.Linear(4*self.hidden_dim, 2*self.hidden_dim)
        self.re_encoder_linear_2 = torch.nn.Linear(4*self.hidden_dim, 2*self.hidden_dim)
        self.lstm_layer3 = nn.LSTM(self.hidden_dim * 2, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True) # sentence lstm

        self.graph_classifier = torch.nn.Linear(self.hidden_dim * 2, self.max_graph)
        self.law_article_classifer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.config.law_label_size)
        )
        self.law_article_loss = torch.nn.NLLLoss()
        self.law_graph_community_loss = torch.nn.NLLLoss()
        self.fully_attn_sent_2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        self.fully_attn_doc_2 = torch.nn.Linear(self.hidden_dim*2, self.hidden_dim*2)

       
    def classifier_layer(self, doc_out, law_out, accu_labels, law_labels, term_labels, law_article_labels, money_hidden):
        """
        :param doc_out: [batch_size, 4 * hidden_dim]
        :param law_out: [103, 4 * hidden_dim]
        :param accu_labels: [batch_size]
        :param law_labels: [batch_size]
        :param term_labels: [batch_size]
        """
        # print("doc out size:", doc_out.size())
        # print("law out size:", law_out.size())
        law_article_logits = self.law_article_classifer(law_out)
        law_article_log_softmax = F.log_softmax(law_article_logits, dim=-1)
        law_article_loss = self.law_article_loss(law_article_log_softmax, law_article_labels)
        _, law_article_preds = torch.max(law_article_log_softmax, dim=-1)

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

        return accu_predicts, accu_loss, law_predicts, law_loss, term_predicts, term_loss, law_article_preds, law_article_loss, accu_rep, law_rep, term_rep

    def word_attn_lstm(self, input: torch.FloatTensor, mask: torch.BoolTensor):
        """
        Args:
            lstm_layer (torch.nn.Module)
            input (torch.FloatTensor): [batch_size*max_sent_num, max_sent_len, word_dim]
            mask: (torch.BoolTensor): [batch_size, max_sent_num, max_sent_len]
            batch_size (int)
        Returns:
            torch.FloatTensor: [bs, max_sent_num, sent_dim]
        """
        batch_size, max_sent_num, max_sent_len = mask.size()
        mask = mask.long().view(-1, max_sent_len)

        hidden = None
        lstm1_out, hidden = self.lstm_layer1.forward(input, hidden) # [batch_size*max_sent_num, max_sent_len, hidden_dim * 2]
        attn_p_weights = torch.matmul(lstm1_out, self.attn_p) # [batch_size*max_sent_num, max_sent_len, 1]
        attn_p_outs = F.softmax(attn_p_weights.squeeze() * mask, dim=-1).unsqueeze(-1) #[batch_size * max_sent_num, max_sent_len]
        sent_rep = torch.sum(lstm1_out * attn_p_outs, dim=1).view(batch_size, max_sent_num, -1) #[batch_size, max_sent_num, hidden_dim*2]
        return sent_rep
    

    def sent_attn_lstm(self, input: torch.FloatTensor, mask: torch.BoolTensor):
        """
        Args:
            lstm_layer (torch.nn.Module)
            input (torch.FloatTensor): [batch_size, max_sent_num, hidden_dim * 2]
            batch_size (int)
            mask (torch.BoolTensor): [batch_size, max_sent_num]
        Returns:
            torch.FloatTensor: [batch_size, hidden_dim*2]
        """
        mask = mask.long()

        hidden = None
        lstm2_out, hidden = self.lstm_layer2.forward(input, hidden) # [batch_size, max_sent_num, hidden_dim * 2]
        attn_q_weights = torch.matmul(lstm2_out, self.attn_q) #[batch_size, max_sent_num, 1]
        attn_q_outs = F.softmax(attn_q_weights.squeeze() * mask, dim=-1).unsqueeze(-1)
        doc_rep = torch.sum(lstm2_out * attn_q_outs, dim=1) #[batch_size, hidden_dim*2]
        return doc_rep


    def GDO(self, law_rep_partition: List[torch.FloatTensor], law_rep: torch.FloatTensor, indices):
        """[summary]

        Args:
            law_rep_partition (List[torch.FloatTensor]): A list of torch.FloatTensor, the size of each torch.FloatTensor is [number_of_degrees, hidden_size]
            law_rep (torch.FloatTensor): [law_label_size, hidden_size]
            indices (List): A list of lists

        Returns:
            (torch.FloatTensor): [law_label_size, hidden_size]
        """
        # print("law rep:", law_rep.size())
        article_new_list = [torch.tanh(law_rep_partition[0])]
        hidden_size = law_rep.size(-1)
        for i in range(1, self.max_deg+1):
            if i not in self.deg_list:
                article_new_list.append(torch.tanh(law_rep_partition[i]))
                continue
            # print(f"adj list of degree {i}:", self.adj_list[i])
            neigh_articles = torch.index_select(input=law_rep, dim=0, index=torch.LongTensor(list(chain(*self.adj_list[i]))).cuda()) #[number_of_degree_i, degree, hidden_size]
            neigh_articles = neigh_articles.view(-1, i, hidden_size)
            # print("neigh articles size:", neigh_articles.size())

            article = law_rep_partition[i] #[number_of_degree_i, hidden_size]
            if len(article.size()) == 1:
                article = article.unsqueeze(0)
            # print("article size:", article.size())
            article_1 = article.unsqueeze(1).expand(-1, i, -1)
            # print("article 1 size:", article_1.size())
            interaction_vec = torch.cat((article_1, neigh_articles), dim=-1) #[number_of_degree_i, degree, 2*hidden_size]
            neigh_articles = torch.mean(self.gdo_linear(interaction_vec), dim=1) # [number_of_degree_i, 2*hidden_size]
            new_article = self.gdo_mlp(article-neigh_articles) # [number_of_degree_i, 2*hidden_size]
            # print("new article size:", new_article.size())
            article_new_list.append(new_article)
        
        # print("article new list len:", len(article_new_list))
        law_conv = dynamic_stitch(indices, article_new_list)
        return law_conv


    def law_re_encoder(self, atten_list, law_sent_rep, law_mask, law_doc_mask):
        """[summary]

        Args:
            atten_list (List[torch.FloatTensor]): representations of graph community. 
            law_sent_rep (torch.FloatTensor): [num_of_law_articles*max_sent_num, max_sent_len, hidden_dim]
            law_mask (torch.LongTensor): [num_of_law_articles, max_sent_num, max_sent_len]
            law_doc_mask (torch.LongTensor): [num_of_law_articles, max_sent_num]
        Returns:
            torch.FloatTensor: [batch_size, max_sent_num, hidden_dim]
        """
        _, max_law_sent_num, max_law_sent_len = law_mask.size()
        law_u = torch.index_select(input=torch.stack(atten_list), dim=0, index=torch.LongTensor(self.graph_list).cuda()) #[num_of_law_articles, hidden_dim]
        # print("law sent rep:", law_sent_rep.size())
        # print("law mask size:", law_mask.size())

        u_law_w = self.re_encoder_linear_1(law_u).unsqueeze(1).repeat(max_law_sent_num, max_law_sent_len, 1) #[num_of_law_articles*max_law_sent_num, max_law_sent_len, hidden_dim]
        u_law_s = self.re_encoder_linear_2(law_u).unsqueeze(1).expand(-1, max_law_sent_num, -1) #[num_of_law_articles, max_law_sent_num, hidden_dim]
        # print("u law w:", u_law_w.size())
        # print("u law s:", u_law_s.size())
        ### attend original law_sent_rep to law graph community representations
        law_mask = law_mask.view(-1, max_law_sent_len)
        rep_law, _ = attn_encoder_mask(u_law_w, law_sent_rep, self.fully_attn_sent_2, law_mask, k_ori=True)
        # print("rep law size:", rep_law.size())
        rep_law = rep_law.view(-1, max_law_sent_num, 2*self.hidden_dim)

        hidden = None
        rep_law_2, _ = self.lstm_layer3(rep_law, hidden) #[num_of_law_articles, max_law_sent_num, hidden_dim]
        ### attend original law_doc_rep to law graph community representations
        rep_law_2, _ = attn_encoder_mask(u_law_s, rep_law_2, self.fully_attn_doc_2, law_doc_mask, k_ori=True)
        # print("rep law 2 size:", rep_law_2.size())
        return rep_law_2


    def fact_re_encoder(self, 
                        atten_list, 
                        fact_basic_doc_rep, 
                        fact_lstm_sent_rep, 
                        graph_community_labels,
                        fact_mask,
                        fact_doc_mask):
        """[summary]

        Args:
            atten_list: [graph_num, 4*hidden_size]
            fact_basic_doc_rep ([torch.FloatTensor]): [batch_size, hidden_size]
            fact_lstm_sent_rep ([torch.FloatTensor]): [batch_size * max_sent_num, max_sent_len ,hidden_size]
            fact_lstm_doc_rep ([torch.FloatTensor]): [batch_size, max_sent_num, hidden_size]
            law_basic_rep ([torch.FloatTensor]): [103, hidden_size]
            graph_community_labels ([]): [description]
        Returns:
            fact_re_rep [torch.FloatTensor]: [batch_size, 2*hidden_size]
            law_re_rep [torch.FloatTensor]: [183, 2*hidden_size]
        """
        ### graph community preds
        batch_size, max_fact_sent_num, max_fact_sent_len = fact_mask.size()
        # print("graph community labels:", graph_community_labels.size())
        fact_graph_choose_1 = self.graph_classifier(fact_basic_doc_rep) # [batch_size, 68]
        fact_graph_choose = F.softmax(fact_graph_choose_1, dim=-1) # [batch_size, 68]
        graph_choose_log_softmax = F.log_softmax(fact_graph_choose_1, dim=-1)
        # print("graph choose log softmax size:", graph_choose_log_softmax.size())
        graph_choose_loss = self.law_graph_community_loss(graph_choose_log_softmax, graph_community_labels)
        # print("graph choose loss:", graph_choose_loss)
        graph_choose_loss = graph_choose_loss / 128.0
        _, graph_preds = torch.max(graph_choose_log_softmax, dim=-1) #[batch_size]
        correct_graph = torch.eq(graph_community_labels, graph_preds)
        # print("correct graph:", correct_graph) 
        ### fact re-encoding
        # fact_graph_choose = torch.where()
        atten_tensor = torch.stack(atten_list, dim=0) # [68, 4*hidden_size]
        u_fact = torch.matmul(fact_graph_choose, atten_tensor) #[batch_size, 4*hidden_size]
        u_fact_w = self.re_encoder_linear_1(u_fact).unsqueeze(1).repeat(max_fact_sent_num, max_fact_sent_len, 1)
        u_fact_s = self.re_encoder_linear_2(u_fact).unsqueeze(1).expand(-1, max_fact_sent_num, -1)
        fact_mask = fact_mask.view(-1, max_fact_sent_len)
        rep_fact, _ = attn_encoder_mask(u_fact_w, fact_lstm_sent_rep, self.fully_attn_sent_2, fact_mask, k_ori=True)
        # print("rep fact size:", rep_fact.size())
        rep_fact = rep_fact.view(batch_size, max_fact_sent_num, -1)
        hidden = None
        rep_fact_2, _ = self.lstm_layer3(rep_fact, hidden)
        rep_fact_2, _ = attn_encoder_mask(u_fact_s, rep_fact_2, self.fully_attn_doc_2, fact_doc_mask, k_ori=True)
        return rep_fact_2, graph_choose_loss, graph_preds


    def forward(self, input_facts, accu_labels, law_labels, term_labels, input_sentences_lens, input_doc_len,
    money_amount_lists, drug_weight_lists):
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
        law_article_labels = torch.LongTensor(range(self.config.law_label_size)).cuda()
        gold_matrix_law = F.one_hot(law_labels, self.config.law_label_size).float() #[batch_size, 103]
        gold_matrix_accu = F.one_hot(accu_labels, self.config.accu_label_size).float() #[batch_size, 119]
        gold_matrix_time = F.one_hot(term_labels, self.config.term_label_size).float() #[batch_size, 12]
        
        # print("max graph:", self.max_graph)
        # print("gold matrix law size:", gold_matrix_law.size())
        # graph_labels = dynamic_partition(gold_matrix_law.transpose(1, 0), self.graph_list, self.max_graph)
        label = []
        for l in law_labels:
            label.append(list(self.graph_list)[l])
        graph_labels = torch.LongTensor(label).cuda()
        # print("graph labels:", label)
        # print("graph lables size:", graph_labels.size())

        batch_size, max_fact_sent_num, max_fact_sent_seq_len = input_facts.size()
        _, max_law_sent_num, max_law_sent_seq_len = self.law_input.size()

        fact_mask = ~input_facts.eq(self.config.word2id_dict["BLANK"]) 
        fact_sent_len = torch.sum(fact_mask, dim=-1) # [batch_size, max_sent_num, max_sent_seq_len] -> [batch_size, max_sent_num]
        fact_doc_mask = fact_sent_len.bool()
        fact_doc_len = torch.sum(fact_doc_mask, dim=-1) # [batch_size, max_sent_num] -> [batch_size]

        law_mask = ~self.law_input.eq(self.config.word2id_dict["BLANK"])
        law_sent_len = torch.sum(law_mask, dim=-1)
        law_doc_mask = law_sent_len.bool()
        law_doc_len = torch.sum(law_doc_mask, dim=-1)

        fact_word_embeds = self.word_embeddings_layer(input_facts.view(batch_size*max_fact_sent_num, -1)) #[batch_size*max_sent_num, max_doc_seq_len, word_emb_size]
        law_word_embeds = self.word_embeddings_layer(self.law_input.view(self.config.law_label_size*max_law_sent_num, -1))

        ### basic hrnn encoder
        hidden = None
        fact_lstm_sent_out, hidden = self.lstm_layer1.forward(fact_word_embeds, hidden) # [batch_size*max_sent_num, max_sent_len, hidden_dim * 2]
        fact_sent_rep, _ = attn_encoder_mask(self.attn_p, fact_lstm_sent_out, self.fully_attn_sent_1, fact_mask.view(-1, max_fact_sent_seq_len), k_ori=True)
        hidden = None
        law_lstm_sent_out, hidden = self.lstm_layer1.forward(law_word_embeds, hidden) # [batch_size*max_sent_num, max_sent_len, hidden_dim * 2]
        law_sent_rep, _ = attn_encoder_mask(self.attn_p, law_lstm_sent_out,self.fully_attn_sent_1, law_mask.view(-1, max_law_sent_seq_len), k_ori=True)

        hidden = None
        fact_lstm_doc_out, hidden = self.lstm_layer2.forward(fact_sent_rep.view(batch_size, max_fact_sent_num, -1), hidden) # [batch_size, max_sent_num, hidden_dim * 2]
        fact_doc_rep, _ = attn_encoder_mask(self.attn_q, fact_lstm_doc_out, self.fully_attn_doc_1, fact_doc_mask, k_ori=True) #[batch_size, hidden_size]
        hidden = None
        law_lstm_doc_out, hidden = self.lstm_layer2.forward(law_sent_rep.view(-1, max_law_sent_num, 2*self.hidden_dim), hidden) # [num_of_law_articles, max_sent_num, hidden_dim * 2]
        law_doc_rep, _= attn_encoder_mask(self.attn_q, law_lstm_doc_out, self.fully_attn_doc_1, law_doc_mask, k_ori=True) # [num_of_law_articles, hidden_size]
        
        ###graph interaction
        indices = dynamic_partition(torch.LongTensor(range(self.config.law_label_size)), self.deg_list, self.max_deg + 1) # clustering laws w.r.t their degrees
        law_rep_of_degrees = dynamic_partition(law_doc_rep, self.deg_list, self.max_deg + 1) # a list of [number_of_degrees, hidden_size]
        
        # first graph neural network layer
        law_conv = self.GDO(law_rep_of_degrees, law_doc_rep, indices)
        # second graph neural network layer
        sec_law_conv = self.GDO(law_rep_of_degrees, law_conv, indices)

        law_rep_of_degrees = dynamic_partition(sec_law_conv, self.graph_list, self.max_graph)
        atten_list = []
        for i in range(self.max_graph):
            u_max, _ = torch.max(law_rep_of_degrees[i], 0) #[2*hidden_size]
            u_min, _ = torch.min(law_rep_of_degrees[i], 0) #[2*hidden_size]
            atten_list.append(torch.cat([u_max, u_min], -1))  # size: [graph_num, 4*hidden_size] whether this u can use attention to get
        
        law_re_rep = self.law_re_encoder(atten_list, law_lstm_sent_out, law_mask, law_doc_mask)
        law_final_rep = torch.cat((law_doc_rep, law_re_rep), dim=-1)
        fact_re_rep, graph_choose_loss, graph_preds = self.fact_re_encoder(atten_list, fact_doc_rep, fact_lstm_sent_out, graph_labels, fact_mask, fact_doc_mask)
        fact_final_rep = torch.cat((fact_doc_rep, fact_re_rep), dim=-1)
        accu_preds, accu_loss, law_preds, law_loss, term_preds, term_loss, law_article_preds, law_article_loss, accu_rep, law_rep, term_rep = self.classifier_layer(fact_final_rep, law_final_rep, accu_labels, law_labels, term_labels, law_article_labels, money_amount_hidden)  # [batch_size, 3]
        return accu_loss, law_loss, term_loss, law_article_loss, graph_choose_loss, accu_preds, law_preds, term_preds, law_article_preds, graph_preds, fact_final_rep, accu_rep, law_rep, term_rep 


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
        self.register_buffer("doc_feature_queue", torch.randn(self.K, 2*config.HP_hidden_dim))
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

        accu_mask_inbatch = torch.eq(accu_label_lists, accu_label_lists.T).float()
        law_mask_inbatch = torch.eq(law_label_lists, law_label_lists.T).float()
        term_mask_inbatch = torch.eq(term_label_lists, term_label_lists.T).float()

        # compute inbatch doc contra loss
        cos_sim_with_t = torch.div(torch.matmul(doc_query, doc_query.clone().detach().T), self.T)
         # for numerical stability
        logits_max, _ = torch.max(cos_sim_with_t, dim=1, keepdim=True)
        logits = cos_sim_with_t - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits + 1e-12)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        positive_accu_mask = accu_mask_inbatch * term_mask_inbatch
        mean_log_prob_pos = (positive_accu_mask * log_prob).sum(1) / \
            (positive_accu_mask.sum(1) + 1e-12)  # [bsz]
        inbatch_doc_loss = -mean_log_prob_pos.mean()


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

        return inbatch_doc_loss, doc_loss, accu_loss, law_loss, term_loss, label_1_index

    def forward(self, legals, accu_label_lists, law_label_lists, term_lists, sent_lent, legals_len, money_amount_lists, drug_weight_lists):
        # compute query features
        accu_loss, law_loss, term_loss, law_article_loss, graph_choose_loss, accu_preds, law_preds, term_preds, law_article_preds, graph_preds, q_doc_feature, q_accu_feature, q_law_feature, q_term_feature = self.encoder_q(
            legals, accu_label_lists, law_label_lists, term_lists, sent_lent, legals_len, money_amount_lists, drug_weight_lists)

        q_doc_feature = nn.functional.normalize(q_doc_feature, dim=1)
        q_accu_feature = nn.functional.normalize(q_accu_feature, dim=1)
        q_law_feature = nn.functional.normalize(q_law_feature, dim=1)
        q_term_feature = nn.functional.normalize(q_term_feature, dim=1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, _, _, _, _, _, _, _, _, _, k_doc_feature, k_accu_feature, k_law_feature, k_term_feature = self.encoder_k(legals, accu_label_lists, law_label_lists, term_lists, sent_lent, legals_len, money_amount_lists, drug_weight_lists)
            
            k_doc_feature = nn.functional.normalize(k_doc_feature, dim=1)
            k_accu_feature = nn.functional.normalize(k_accu_feature, dim=1)
            k_law_feature = nn.functional.normalize(k_law_feature, dim=1)
            k_term_feature = nn.functional.normalize(k_term_feature, dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_doc_feature, k_accu_feature, k_law_feature, k_term_feature, accu_label_lists, law_label_lists, term_lists)
        inbatch_doc_loss, contra_doc_loss, contra_accu_loss, contra_law_loss, contra_term_loss, _ = self._get_contra_loss(q_doc_feature, q_accu_feature, q_law_feature, q_term_feature, accu_label_lists, law_label_lists, term_lists)

        #return contra_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds
        return inbatch_doc_loss, contra_doc_loss, contra_accu_loss, contra_law_loss, contra_term_loss, accu_loss, law_loss, term_loss, law_article_loss, graph_choose_loss, accu_preds, law_preds, term_preds, law_article_preds, graph_preds
    
    
    def predict(self, legals, accu_label_lists, law_label_lists, term_lists, sent_len, legals_len, money_amount_lists, drug_lists):
        # compute query features
        _, _, _, _, _, accu_preds, law_preds, term_preds, _, _, _, _, _, _ = self.encoder_q(legals, accu_label_lists, law_label_lists, term_lists, sent_len, legals_len, money_amount_lists, drug_lists)
        
        #q = nn.functional.normalize(q, dim=1)
        #contra_loss, label_1_index = self._get_contra_loss(q, accu_label_lists)
        # if label_1_index != -1:
        #     print(
        #         f"Epoch: {epoch_idx}, Name: {name}, contra loss: {contra_loss.item()}, accu preds: {accu_preds[label_1_index].item()}, ground truth label: {accu_label_lists[label_1_index].item()}")
        #     print(''.join(raw_fact_list[label_1_index]))
        
        return accu_preds, law_preds, term_preds

if __name__ == '__main__':
   print(datetime.datetime.now())
