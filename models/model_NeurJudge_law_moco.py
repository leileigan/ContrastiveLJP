import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import torch.optim as optim
import datetime
import random, math
from typing import List
import json
from torch.autograd import Variable


class Mask_Attention(nn.Module):
    def __init__(self):
        super(Mask_Attention, self).__init__()
    def forward(self, query, context):
        attention = torch.bmm(context, query.transpose(1, 2))
        mask = attention.new(attention.size()).zero_()
        mask[:,:,:] = -np.inf
        attention_mask = torch.where(attention==0, mask, attention)
        attention_mask = torch.nn.functional.softmax(attention_mask, dim=-1)
        mask_zero = attention.new(attention.size()).zero_()
        final_attention = torch.where(attention_mask!=attention_mask, mask_zero, attention_mask)
        context_vec = torch.bmm(final_attention, query)
        return context_vec

class Code_Wise_Attention(nn.Module):
    def __init__(self):
        super(Code_Wise_Attention, self).__init__()
    def forward(self,query,context):
        S = torch.bmm(context, query.transpose(1, 2))
        attention = torch.nn.functional.softmax(torch.max(S, 2)[0], dim=-1)
        context_vec = torch.bmm(attention.unsqueeze(1), context)
        return context_vec

class MaskGRU(nn.Module): 
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP=0.2):
        super(MaskGRU, self).__init__()
        self.gru_module = nn.GRU(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional)
        #self.drop_module = nn.Dropout(dropoutP)
    def forward(self, inputs ):
        self.gru_module.flatten_parameters()
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i, :seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)
        input_drop = input * mask_in
        H, _ = self.gru_module(input_drop)
        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i, :seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)
        output = H * mask
        return output

BASE="/data/ganleilei/law/ContrastiveLJP"
# Version of NeurJudge with nn.GRU    
class NeurJudge(nn.Module):
    def __init__(self,embedding):
        super(NeurJudge, self).__init__()
        self.charge_tong = json.load(open(BASE+'/NeurJudge_config_data/charge_tong.json'))
        self.art_tong = json.load(open(BASE+'/NeurJudge_config_data/art_tong.json'))
        self.id2charge = json.load(open(BASE+'/NeurJudge_config_data/id2charge.json'))
        self.data_size = 200
        self.hidden_dim = 150

        # self.embs = nn.Embedding(339503, 200)
        self.embs = nn.Embedding(embedding.shape[0], 200)
        self.embs.weight.data.copy_(torch.from_numpy(embedding))
        self.embs.weight.requires_grad = True

        self.encoder = nn.GRU(self.data_size,self.hidden_dim, batch_first=True, bidirectional=True)
        self.code_wise = Code_Wise_Attention()
        self.mask_attention = Mask_Attention()
        
        self.encoder_term = nn.GRU(self.hidden_dim * 6, self.hidden_dim*3, batch_first=True, bidirectional=True)
        self.encoder_article = nn.GRU(self.hidden_dim * 4, self.hidden_dim*2, batch_first=True, bidirectional=True)

        self.id2article = json.load(open(BASE+'/NeurJudge_config_data/id2article.json'))
        self.mask_attention_article = Mask_Attention()

        self.encoder_charge = nn.GRU(self.data_size,self.hidden_dim, batch_first=True, bidirectional=True)
        self.charge_pred = nn.Linear(self.hidden_dim*2,119)
        self.article_pred = nn.Linear(self.hidden_dim*4,103)
        self.time_pred = nn.Linear(self.hidden_dim*6,11)

        self.accu_loss = torch.nn.NLLLoss()
        self.law_loss = torch.nn.NLLLoss()
        self.term_loss = torch.nn.NLLLoss()
    

    def graph_decomposition_operation(self,_label,label_sim,id2label,label2id,num_label,layers=2):
        for i in range(layers):
            new_label_tong = []
            for index in range(num_label):
                Li = _label[index]
                Lj_list = []
                if len(label_sim[id2label[str(index)]]) == 0:
                    new_label_tong.append(Li.unsqueeze(0))
                else:
                    for sim_label in label_sim[id2label[str(index)]]:
                        Lj = _label[int(label2id[str(sim_label)])]
                        x1 = Li*Lj
                        x1 = torch.sum(x1,-1)
                        x2 = Lj*Lj
                        x2 = torch.sum(x2,-1)
                        x2 = x2+1e-10
                        xx = x1/x2
                        Lj = xx.unsqueeze(-1)*Lj
                        Lj_list.append(Lj)
                    Lj_list = torch.stack(Lj_list,0).squeeze(1)
                    Lj_list = torch.mean(Lj_list,0).unsqueeze(0)
                    new_label_tong.append(Li-Lj_list)  
            new_label_tong = torch.stack(new_label_tong,0).squeeze(1)
            _label = new_label_tong
        return _label

    def fact_separation(self,process,verdict_names,device,embs,encoder,circumstance,mask_attention,types):
        verdict, verdict_len = process.process_law(verdict_names,types)
        verdict = verdict.to(device)
        verdict_len = verdict_len.to(device)
        verdict = embs(verdict)
        verdict_hidden,_ = encoder(verdict)
        # Fact Separation
        scenario = mask_attention(verdict_hidden,circumstance)
        # vector rejection
        x3 = circumstance*scenario
        x3 = torch.sum(x3,2)
        x4 = scenario*scenario
        x4 = torch.sum(x4,2)
        x4 = x4+1e-10
        xx = x3/x4
        # similar vectors
        similar = xx.unsqueeze(-1)*scenario 
        # dissimilar vectors
        dissimilar = circumstance - similar
        return similar,dissimilar
    
    def forward(self, charge, charge_sent_len, article,
                article_sent_len, charge_tong2id, id2charge_tong, art2id, id2art,
                documents, sent_lent, process, device,
                accu_labels, law_labels, term_labels):
        # deal the case fact
        doc = self.embs(documents)
        d_hidden,_ = self.encoder(doc) 
        
        # the charge prediction
        df = d_hidden.mean(1)
        charge_out = self.charge_pred(df)

        accu_log_softmax = F.log_softmax(charge_out, dim=-1)
        accu_loss = self.accu_loss(accu_log_softmax, accu_labels)
        _, accu_predicts = torch.max(accu_log_softmax, dim=-1) # [batch_size, accu_label_size]
        charge_pred = charge_out.cpu().argmax(dim=1).numpy()
        charge_names = [self.id2charge[str(i)] for i in charge_pred]
        # Fact Separation for verdicts
        adc_vector, sec_vector = self.fact_separation(process = process,verdict_names = charge_names,device = device ,embs = self.embs,encoder = self.encoder, circumstance = d_hidden, mask_attention = self.mask_attention,types = 'charge')

        # the article prediction
        fact_article = torch.cat([d_hidden, adc_vector],-1) #[bsz, seq_len, 4*hidden_dim] -> [bsz, 4*hidden_dim]
        fact_legal_article_hidden,_ = self.encoder_article(fact_article)
        fact_article_hidden = fact_legal_article_hidden.mean(1)

        article_out = self.article_pred(fact_article_hidden)
        law_log_softmax = F.log_softmax(article_out, dim=-1)
        law_loss = self.law_loss(law_log_softmax, law_labels)
        _, law_predicts = torch.max(law_log_softmax, dim=1) # [batch_size * max_claims_num]
        article_pred = article_out.cpu().argmax(dim=1).numpy()
        article_names = [self.id2article[str(i)] for i in article_pred]
        # Fact Separation for sentencing
        ssc_vector, dsc_vector = self.fact_separation(process = process,verdict_names = article_names,device = device ,embs = self.embs,encoder = self.encoder, circumstance = sec_vector, mask_attention = self.mask_attention,types = 'article')


        # the term of penalty prediction change here
        # term_message = torch.cat([ssc_vector,dsc_vector],-1)
        term_message = torch.cat([d_hidden, ssc_vector,dsc_vector],-1)
        term_message,_ = self.encoder_term(term_message)

        fact_legal_time_hidden = term_message.mean(1)
        time_out = self.time_pred(fact_legal_time_hidden)
        term_log_softmax = F.log_softmax(time_out, dim=-1)
        term_loss = self.term_loss(term_log_softmax, term_labels)
        _, term_predicts = torch.max(term_log_softmax, dim=1) # [batch_size * max_claims_num]

        return accu_predicts, law_predicts, term_predicts, accu_loss, law_loss, term_loss, df
 
class NeurJudge_plus(nn.Module):
    def __init__(self,embedding):
        super(NeurJudge_plus, self).__init__()
        self.charge_tong = json.load(open(BASE+'/NeurJudge_config_data/charge_tong.json'))
        self.art_tong = json.load(open(BASE+'/NeurJudge_config_data/art_tong.json'))
        self.id2charge = json.load(open(BASE+'/NeurJudge_config_data/id2charge.json'))
        self.data_size = 200
        self.hidden_dim = 150

        # self.embs = nn.Embedding(339503, 200)
        self.embs = nn.Embedding(embedding.shape[0], 200)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad = False

        self.encoder = nn.GRU(self.data_size,self.hidden_dim, batch_first=True, bidirectional=True)
        self.code_wise = Code_Wise_Attention()
        self.mask_attention = Mask_Attention()
        
        self.encoder_term = nn.GRU(self.hidden_dim * 6, self.hidden_dim*3, batch_first=True, bidirectional=True)
        self.encoder_article = nn.GRU(self.hidden_dim * 8, self.hidden_dim*4, batch_first=True, bidirectional=True)

        self.id2article = json.load(open(BASE+'/NeurJudge_config_data/id2article.json'))
        self.mask_attention_article = Mask_Attention()

        self.encoder_charge = nn.GRU(self.data_size,self.hidden_dim, batch_first=True, bidirectional=True)
        self.charge_pred = nn.Linear(self.hidden_dim*6,119)
        self.article_pred = nn.Linear(self.hidden_dim*8,103)
        self.time_pred = nn.Linear(self.hidden_dim*6,11)

    def graph_decomposition_operation(self,_label,label_sim,id2label,label2id,num_label,layers=2):
        for i in range(layers):
            new_label_tong = []
            for index in range(num_label):
                Li = _label[index]
                Lj_list = []
                if len(label_sim[id2label[str(index)]]) == 0:
                    new_label_tong.append(Li.unsqueeze(0))
                else:
                    for sim_label in label_sim[id2label[str(index)]]:
                        Lj = _label[int(label2id[str(sim_label)])]
                        x1 = Li*Lj
                        x1 = torch.sum(x1,-1)
                        x2 = Lj*Lj
                        x2 = torch.sum(x2,-1)
                        x2 = x2+1e-10
                        xx = x1/x2
                        Lj = xx.unsqueeze(-1)*Lj
                        Lj_list.append(Lj)
                    Lj_list = torch.stack(Lj_list,0).squeeze(1)
                    Lj_list = torch.mean(Lj_list,0).unsqueeze(0)
                    new_label_tong.append(Li-Lj_list)  
            new_label_tong = torch.stack(new_label_tong,0).squeeze(1)
            _label = new_label_tong
        return _label

    def fact_separation(self,process,verdict_names,device,embs,encoder,circumstance,mask_attention,types):
        verdict, verdict_len = process.process_law(verdict_names,types)
        verdict = verdict.to(device)
        verdict_len = verdict_len.to(device)
        verdict = embs(verdict)
        verdict_hidden,_ = encoder(verdict)
        # Fact Separation
        scenario = mask_attention(verdict_hidden,circumstance)
        # vector rejection
        x3 = circumstance*scenario
        x3 = torch.sum(x3,2)
        x4 = scenario*scenario
        x4 = torch.sum(x4,2)
        x4 = x4+1e-10
        xx = x3/x4
        # similar vectors
        similar = xx.unsqueeze(-1)*scenario 
        # dissimilar vectors
        dissimilar = circumstance - similar
        return similar,dissimilar
    
    def forward(self,charge,charge_sent_len,article,\
    article_sent_len,charge_tong2id,id2charge_tong,art2id,id2art,\
    documents,sent_lent,process,device):
        # deal the semantics of labels (i.e., charges and articles) 
        charge = self.embs(charge)
        article = self.embs(article)
        charge,_ = self.encoder_charge(charge)
        article,_ = self.encoder_charge(article)
        _charge = charge.mean(1)
        _article = article.mean(1)
        # the original charge and article features
        ori_a = charge.mean(1)
        ori_b = article.mean(1)
        # the number of spreading layers is set as 2
        layers = 2
        # GDO for the charge graph
        new_charge = self.graph_decomposition_operation(_label = _charge, label_sim = self.charge_tong, id2label = id2charge_tong, label2id = charge_tong2id, num_label = 119, layers = 2)

        # GDO for the article graph
        new_article = self.graph_decomposition_operation(_label = _article, label_sim = self.art_tong, id2label = id2art, label2id = art2id, num_label = 103, layers = 2)

        # deal the case fact
        doc = self.embs(documents)
        batch_size = doc.size(0)
        d_hidden,_ = self.encoder(doc) 
        
        # L2F attention for charges
        new_charge_repeat = new_charge.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_hidden_charge = self.code_wise(new_charge_repeat,d_hidden)
        d_hidden_charge = d_hidden_charge.repeat(1,doc.size(1),1)
        
        a_repeat = ori_a.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_a = self.code_wise(a_repeat,d_hidden)
        d_a = d_a.repeat(1,doc.size(1),1)

        # the charge prediction
        fact_charge = torch.cat([d_hidden,d_hidden_charge,d_a],-1)
        #fact_charge_hidden = self.l_encoder([fact_charge,sent_lent.view(-1)])
        fact_charge_hidden = fact_charge
        df = fact_charge_hidden.mean(1)
        charge_out = self.charge_pred(df)
        
        charge_pred = charge_out.cpu().argmax(dim=1).numpy()
        charge_names = [self.id2charge[str(i)] for i in charge_pred]
        # Fact Separation for verdicts
        adc_vector, sec_vector = self.fact_separation(process = process,verdict_names = charge_names,device = device ,embs = self.embs,encoder = self.encoder, circumstance = d_hidden, mask_attention = self.mask_attention,types = 'charge')

        # L2F attention for articles
        new_article_repeat = new_article.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_hidden_article = self.code_wise(new_article_repeat,d_hidden)
        d_hidden_article = d_hidden_article.repeat(1,doc.size(1),1)

        b_repeat = ori_b.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_b = self.code_wise(b_repeat,d_hidden)
        d_b = d_b.repeat(1,doc.size(1),1)
        # the article prediction
        fact_article = torch.cat([d_hidden,d_hidden_article,adc_vector,d_b],-1)
        # fact_article_hidden = self.j_encoder([fact_article,sent_lent.view(-1)])
        # fact_article_hidden = fact_article_hidden.mean(1)
        fact_legal_article_hidden,_ = self.encoder_article(fact_article)

        fact_article_hidden = fact_legal_article_hidden.mean(1)
        article_out = self.article_pred(fact_article_hidden)

        article_pred = article_out.cpu().argmax(dim=1).numpy()
        article_names = [self.id2article[str(i)] for i in article_pred]

        # Fact Separation for sentencing
        ssc_vector, dsc_vector = self.fact_separation(process = process,verdict_names = article_names,device = device ,embs = self.embs,encoder = self.encoder, circumstance = sec_vector, mask_attention = self.mask_attention,types = 'article')

        # the term of penalty prediction change here
        # term_message = torch.cat([ssc_vector,dsc_vector],-1)
        term_message = torch.cat([d_hidden,ssc_vector,dsc_vector],-1)

        term_message,_ = self.encoder_term(term_message)

        fact_legal_time_hidden = term_message.mean(1)
        time_out = self.time_pred(fact_legal_time_hidden)

        return charge_out,article_out,time_out


class MoCo(nn.Module):
    def __init__(self, config):
        super(MoCo, self).__init__()

        self.K = config.moco_queue_size
        self.m =config.moco_momentum
        self.T = config.moco_temperature
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = NeurJudge(config.pretrain_word_embedding)
        self.encoder_k = NeurJudge(config.pretrain_word_embedding)
        self.config = config
        self.confused_matrix = config.confused_matrix #[119, 119]
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("accu_feature_queue", torch.randn(self.K, 2*config.HP_hidden_dim))
        self.accu_feature_queue = nn.functional.normalize(self.accu_feature_queue.cuda(), dim=1)

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
    def _dequeue_and_enqueue(self, keys, accu_label_lists, law_label_lists, term_label_lists):
        batch_size = keys.shape[0]
        accu_label_keys = accu_label_lists.unsqueeze(1)
        law_label_keys = law_label_lists.unsqueeze(1)
        term_label_keys = term_label_lists.unsqueeze(1)

        ptr = int(self.ptr)
        if ptr+batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_accu_labels = accu_label_keys[: head_size]
            head_law_labels = law_label_keys[: head_size]
            head_term_labels = term_label_keys[: head_size]

            end_size = ptr + batch_size - self.K
            end_keys = keys[head_size:]
            end_accu_labels = accu_label_keys[head_size:]
            end_law_labels = law_label_keys[head_size:]
            end_term_labels = term_label_keys[head_size:]

            # set head keys
            self.accu_feature_queue[ptr:, :] = head_keys
            self.accu_label_queue[ptr:, :] = head_accu_labels
            self.law_label_queue[ptr:, :] = head_law_labels
            self.term_label_queue[ptr:, :] = head_term_labels

            # set tail keys
            self.accu_feature_queue[:end_size, :] = end_keys
            self.accu_label_queue[:end_size, :] = end_accu_labels
            self.law_label_queue[:end_size, :] = end_law_labels
            self.term_label_queue[:end_size, :] = end_term_labels

        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.accu_feature_queue[ptr:ptr + batch_size, :] = keys
            self.accu_label_queue[ptr:ptr+batch_size, :] = accu_label_keys
            self.law_label_queue[ptr:ptr+batch_size, :] = law_label_keys
            self.term_label_queue[ptr:ptr+batch_size, :] = term_label_keys

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.ptr[0] = ptr

    def _get_contra_loss(self, query, accu_label_lists, law_label_lists, term_label_lists):
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
        positive_mask = law_mask

        # query_queue_product = torch.einsum('nc,kc->nk', [query, self.accu_feature_queue.clone().detach()])
        # cos_sim = query_queue_product / torch.einsum('nc,kc->nk', [torch.norm(query, dim=1).unsqueeze(
            # 1), torch.norm(self.accu_feature_queue.clone().detach(), dim=1).unsqueeze(1)])
        # cos_sim_with_t = cos_sim / self.T
        cos_sim_with_t = torch.div(torch.matmul(query, self.accu_feature_queue.clone().detach().T), self.T)

         # for numerical stability
        logits_max, _ = torch.max(cos_sim_with_t, dim=1, keepdim=True)
        logits = cos_sim_with_t - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits + 1e-12)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / \
            (positive_mask.sum(1) + 1e-12)  # [bsz]

        loss = -mean_log_prob_pos.mean()
        # if label_1_index != -1:
        #     # print("label queue:", self.label_queue.squeeze())
        #     # print("positive mask:", mask[label_1_index])
        #     # print("positive mask select value:", torch.masked_select(query_queue_product[label_1_index], mask[label_1_index].bool()))
        #     print("positive cos sim select value:", torch.masked_select(cos_sim[label_1_index], mask[label_1_index].bool()))
        #     # print("hard neg mask:", hard_neg_mask[label_1_index])
        #     # print("hard neg mask select value:", torch.masked_select(query_queue_product[label_1_index], hard_neg_mask[label_1_index].bool()))
        #     print("hard neg cos sim select value:", torch.masked_select(cos_sim[label_1_index], hard_neg_mask[label_1_index].bool()))
        #     # print("query queue product:", query_queue_product[label_1_index])
        return loss, label_1_index

    def forward(self, legals, legals_len, arts, arts_sent_lent,
                charge_tong2id, id2charge_tong, art2id, id2art, documents,
                sent_lent, process, device, accu_label_lists, law_label_lists, term_lists):
        # compute query features
        accu_preds, law_preds, term_preds, accu_loss, law_loss, term_loss, q_accu_feature = self.encoder_q(
            legals, legals_len, arts,
            arts_sent_lent, charge_tong2id, id2charge_tong, art2id, id2art, documents, sent_lent, process, device,
            accu_label_lists, law_label_lists, term_lists)
        
        q_accu_feature = nn.functional.normalize(q_accu_feature, dim=1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, _, _, _, _, _, k_accu_feature = self.encoder_q(legals, legals_len,
                                                              arts, arts_sent_lent, charge_tong2id, id2charge_tong, art2id, id2art, documents,
                                                              sent_lent, process, device, accu_label_lists, law_label_lists, term_lists)
            
            k_accu_feature = nn.functional.normalize(k_accu_feature, dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_accu_feature, accu_label_lists, law_label_lists, term_lists)
        contra_hmce_loss, _ = self._get_contra_loss(q_accu_feature, accu_label_lists, law_label_lists, term_lists)

        #return contra_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds
        return contra_hmce_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds
    
    
    def predict(self, legals,legals_len,arts,arts_sent_lent, \
                charge_tong2id,id2charge_tong,art2id,id2art,documents, \
                sent_lent,process,device, accu_label_lists, law_label_lists, term_lists):
        # compute query features
        accu_preds, law_preds, term_preds, _, _, _, _ = self.encoder_q(legals,legals_len,arts, \
            arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art,documents,sent_lent,process,device, \
            accu_label_lists, law_label_lists, term_lists)
        
        #q = nn.functional.normalize(q, dim=1)
        #contra_loss, label_1_index = self._get_contra_loss(q, accu_label_lists)
        # if label_1_index != -1:
        #     print(
        #         f"Epoch: {epoch_idx}, Name: {name}, contra loss: {contra_loss.item()}, accu preds: {accu_preds[label_1_index].item()}, ground truth label: {accu_label_lists[label_1_index].item()}")
        #     print(''.join(raw_fact_list[label_1_index]))
        
        return accu_preds, law_preds, term_preds
