from unicodedata import bidirectional
from numpy import dot
import torch
import torch.nn as nn
import torch.nn.functional as F

class DICE(nn.Module):
    def __init__(self, config):
        super(DICE, self).__init__()
        self.num = 11
        self.data_size = config.word_embd_dim
        self.hidden_size = config.hidden_dim
        self.embs = nn.Embedding(self.num, self.data_size)   #待定数字格式
        self.embs.weight.requires_grad = True

        self.encoder = nn.GRU(self.data_size, self.hidden_size, batch_first=True, bidirectional=True)

        self.dice_loss = nn.MSELoss(reduction='mean')

    def forward(self, num1, num2, num_label):
        num1_emb = self.embs(num1)
        num2_emb = self.embs(num2)

        num1_op, _ = self.encoder(num1_emb)   # num1_op [bs, n, d], _[2, bs, d] 
        num2_op, _ = self.encoder(num2_emb)
        num1_hidden = num1_op[:, -1, :] 
        num2_hidden = num2_op[:, -1, :]

        dot_product = torch.einsum('nk,nk->n', [num1_hidden, num2_hidden])
        norm = torch.norm(num1_hidden, p=2, dim=-1) * torch.norm(num2_hidden, p=2, dim=-1)
        dis = 1 - dot_product / norm
        dice_loss = self.dice_loss(dis, num_label)
        
        return dice_loss, dis
    
    def num_eval(self, num1, num2):
        num1_emb = self.embs(num1)
        num2_emb = self.embs(num2)

        num1_op, _ = self.encoder(num1_emb)   # num1_op [bs, n, d], _[2, bs, d] 
        num2_op, _ = self.encoder(num2_emb)
        num1_hidden = num1_op[:, -1, :] 
        num2_hidden = num2_op[:, -1, :]

        dot_product = torch.einsum('nk,nk->n', [num1_hidden, num2_hidden])
        norm = torch.norm(num1_hidden, p=2, dim=-1) * torch.norm(num2_hidden, p=2, dim=-1)
        dis = 1 - dot_product / norm
        return dis
        
    def encode_num(self, num1, num2):

        num1_emb = self.embs(num1)
        num2_emb = self.embs(num2)

        num1_op, _ = self.encoder(num1_emb)   # num1_op [bs, n, d], _[2, bs, d] 
        num2_op, _ = self.encoder(num2_emb)
        num1_hidden = num1_op[:, -1, :] 
        num2_hidden = num2_op[:, -1, :]
        return num1_hidden, num2_hidden
        