from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import torch.optim as optim
import random, math
from typing import List
import json
from torch.autograd import Variable

DEVICE = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    
    indices = inverse.new_empty(unique.size(dim))
    for i, ix in enumerate(inverse):
        indices[ix] = perm[i]
    return unique, indices


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce'):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type
        # self.loss_item = [1, 0.0001, 0.000001]
        self.loss_item = layer_penalty

    def pow_2(self, value):
        return torch.pow(2, value)

    def get_penal(self, index):
        return torch.tensor(self.loss_item[index - 1])

    def forward(self, features, labels, features_queue, labels_queue):
        """
           features: [bsz, hidden_dim]
           labels: [bsz, 4]
           featues_queue: [queue_size, hidden_dim]
           labels_queue: [queue_size, 4]
        """
        mask = torch.ones(labels.shape).to(DEVICE) #[bsz, 4]
        mask_queue = torch.ones(labels_queue.shape).to(DEVICE) #[queue_size, 4]

        cumulative_loss = torch.tensor(0.0).to(DEVICE)
        max_loss_lower_layer = torch.tensor(float('-inf'))

        for l in range(1, labels.shape[1]): # l=1,2,3
            mask[:, labels.shape[1]-l:] = 0
            # print("layer:", l)
            # print("mask:", mask)
            layer_labels = labels * mask #[bsz, 4]
            # print("laber labels size:", layer_labels.size()) 
            mask_queue[:, labels.shape[1]-l:] = 0 #[bsz, 65536]
            # print("mask queue size:", mask_queue.size())
            layer_labels_queue = labels_queue * mask_queue
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels_queue), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(DEVICE)
            # print("mask labels size:", mask_labels.size())

            layer_loss = self.sup_con_loss(features, features_queue, mask=mask_labels)

            if self.loss_type == 'hmc':
                # cumulative_loss += self.layer_penalty(torch.tensor(
                #   1/(l)).type(torch.float)) * layer_loss
                cumulative_loss += self.get_penal(l).cuda() * layer_loss
                # print("layer:", l)
                # print("layer loss:", layer_loss.item())
                # print("weighted layer loss:", self.get_penal(l).cuda() * layer_loss.item())
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1/l).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            
            _, unique_indices = unique(layer_labels, dim=0)

            max_loss_lower_layer = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)

            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
            
            tmp = mask_labels.sum(0) > 0
            labels_queue = labels_queue[~tmp]
            mask_queue = mask_queue[~tmp]
            features_queue = features_queue[~tmp]
        

        return cumulative_loss


class MLP(nn.Module):
    """One hidden layer perceptron, with normalization."""

    def __init__(self, input_size = 512, hidden_size=2048, output_size=128):
        super(MLP, self).__init__()
        self._hidden_size = hidden_size
        self._output_size = output_size
        self.mlp = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=False),
            nn.BatchNorm1d(output_size, affine=False)
        )
    
    def forward(self, input):
        ans = self.mlp(input)
        return ans 


class MLP_moco(nn.Module):
    """One hidden layer perceptron, with normalization."""

    def __init__(self, input_size = 512, hidden_size=2048, output_size=128):
        super(MLP_moco, self).__init__()
        self._hidden_size = hidden_size
        self._output_size = output_size
        self.mlp = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, input):
        ans = self.mlp(input)
        return ans    

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, features_queue, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]
        mask = mask.float().to(DEVICE)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features_queue.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # compute log_prob
        exp_logits = torch.exp(logits + 1e-12)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        num = (mask.sum(1) > 0).sum()
        loss = loss.sum() / (num + 1e-12)
        return loss


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
        
        #定义hmce loss 对象
        self.hmce_loss = HMLC(layer_penalty=config.penalty, loss_type="hmc")

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("accu_feature_queue", torch.randn(self.K, 2*config.HP_hidden_dim))
        self.accu_feature_queue = nn.functional.normalize(self.accu_feature_queue.cuda(), dim=1)
        
        self.register_buffer("label_queue", torch.randint(-1, 0, (self.K, 4)))
        self.label_queue = self.label_queue.cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, accu_feature, labels):
        batch_size = accu_feature.shape[0]
        label_keys = labels
        ptr = int(self.queue_ptr)

        if ptr+batch_size > self.K:
            head_size = self.K - ptr
            accu_head_keys = accu_feature[: head_size]
            head_labels = label_keys[: head_size]

            end_size = ptr + batch_size - self.K
            accu_end_keys = accu_feature[head_size:]
            end_labels = label_keys[head_size: ]

            self.accu_feature_queue[ptr:, :] = accu_head_keys
            self.label_queue[ptr:, :] = head_labels

            self.accu_feature_queue[:end_size, :] = accu_end_keys
            self.label_queue[:end_size, :] = end_labels
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.accu_feature_queue[ptr:ptr + batch_size, :] = accu_feature
            self.label_queue[ptr:ptr+batch_size, :] = label_keys

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
        self.label_ptr[0] = ptr

    
    def merge_accu_term(self, accu, law, term):
        law = law.unsqueeze(1)
        term = term.unsqueeze(1)
        accu = accu.unsqueeze(1)
        labels = torch.cat((accu, law, term, term), 1)
        return labels

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

        labels = self.merge_accu_term(accu_label_lists, law_label_lists, term_lists)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k_accu_feature, labels)

        contra_hmce_loss = self.hmce_loss(q_accu_feature, labels, self.accu_feature_queue, self.label_queue)

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
