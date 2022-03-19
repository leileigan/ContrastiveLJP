# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

seed_num = 2020
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)


def positional_embedding(x, min_timescale=1.0, max_timescale=1.0e4):
    """ positional embedding """

    batch, length, channels = list(x.size())
    assert (channels % 2 == 0)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1.))

    position = torch.arange(0, length).float()
    inv_timescales = torch.arange(0, num_timescales).float()
    if x.is_cuda:
        position = position.cuda()
        inv_timescales = inv_timescales.cuda()

    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    scaled_time = position.unsqueeze(1).expand(
        length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
    # scaled time is now length x num_timescales
    # length x channels
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)

    return signal.unsqueeze(0).expand(batch, length, channels)


class FNNLayer(nn.Module):
    """ FNN layer """

    def __init__(self, hidden_size, inner_linear, relu_dropout=0.1):
        super(FNNLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(hidden_size, inner_linear),
                                   nn.ReLU(inplace=True), nn.Dropout(relu_dropout))
        self.conv2 = nn.Linear(inner_linear, hidden_size)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        return outputs


class ResidualConnection(nn.Module):
    """ residual connection """

    def __init__(self, residual_dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, x, y):
        return x + self.dropout(y)


class LayerNormalization(nn.Module):
    """ layer normalization """

    def __init__(self, num_features, eps=1e-6, affine=True):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.)
            self.bias.data.fill_(0.)

    def forward(self, inputs):
        b, t, _ = list(inputs.size())
        mean = inputs.mean(2).view(b, t, 1).expand_as(inputs)
        input_centered = inputs - mean
        std = input_centered.pow(2).mean(2).add(self.eps).sqrt()
        output = input_centered / std.view(b, t, 1).expand_as(inputs)

        if self.affine:
            w = self.weight.view(1, 1, -1).expand_as(output)
            b = self.bias.view(1, 1, -1).expand_as(output)
            output = output * w + b

        return output


class ScaledDotProductAttention(nn.Module):
    """ scaled dot-product attention """

    def __init__(self, hidden_size, num_heads, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = (hidden_size // num_heads) ** -0.5
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, attn_mask=None):
        '''

        :param q: (batch_size x head, seq_len, emb_size/head)
        :param k: (batch_size x head, seq_len, emb_size/head)
        :param v: (batch_size x head, seq_len, emb_size/head)
        :param attn_mask:
        :return: (batch_size x head, seq_len, emb_size/head)
        '''
        q = q * self.scale
        attn = torch.bmm(q, k.transpose(1, 2))  # result: (batch_size x head , seq_len, seq_len)


        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            #  attn.data.masked_fill_(attn_mask, -float('inf'))
            attn.data.masked_fill_(attn_mask, -1e9)

        attn = F.softmax(attn.view(-1, k.size(1)), dim=-1).view(-1, q.size(1), k.size(1))
        # attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v)

        return outputs


class MultiHeadAttention(nn.Module):
    """ multi-head attention """

    def __init__(self, hidden_size, num_heads, attention_dropout=0.1, share=True):
        super(MultiHeadAttention, self).__init__()
        self.share = share
        self.num_heads = num_heads

        if share:
            self.w_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        else:
            self.w_q = nn.Linear(hidden_size, hidden_size, bias=True)
            self.w_kv = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

        self.attention = ScaledDotProductAttention(hidden_size, num_heads, attention_dropout)
        self.prj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, q, k, v, attn_mask=None):

        if self.share:
            qs, ks, vs = torch.split(self.w_qkv(q), split_size_or_sections=q.size(-1), dim=-1)
            #  qs: (batch_size, seq_len, emb_size)
        else:
            qs = self.w_q(q)
            ks, vs = torch.split(self.w_kv(k), split_size_or_sections=k.size(-1), dim=-1)

        # split and concat
        q_ = torch.cat(torch.chunk(qs, self.num_heads, dim=-1), dim=0)  # (h*B, L, C/h)
        k_ = torch.cat(torch.chunk(ks, self.num_heads, dim=-1), dim=0)  # (h*B, L, C/h)
        v_ = torch.cat(torch.chunk(vs, self.num_heads, dim=-1), dim=0)  # (h*B, L, C/h)

        outputs = self.attention.forward(q_, k_, v_, attn_mask=attn_mask.repeat(self.num_heads, 1, 1))
        # (h*B, L, C/h)
        outputs = torch.cat(torch.split(outputs, qs.size(0), dim=0), dim=-1)
        # (h, L, C)
        outputs = self.prj(outputs)

        return outputs


class EncoderBlock(nn.Module):
    """ encoder block """

    def __init__(self, hidden_size, num_heads, inner_linear, attention_dropout,
                 residual_dropout, relu_dropout):
        super(EncoderBlock, self).__init__()
        '''
        self.layer_norm1 = LayerNormalization(hidden_size)
        self.layer_norm2 = LayerNormalization(hidden_size)

        self.residual_connection1 = ResidualConnection(residual_dropout)
        self.residual_connection2 = ResidualConnection(residual_dropout)
        '''
        self.self_attention = MultiHeadAttention(hidden_size, num_heads, attention_dropout, share=True)

        # self.fnn_layer = FNNLayer(hidden_size, inner_linear, relu_dropout)

    def forward(self, inputs, self_attn_mask):

        # residual = inputs
        self_attn_outputs = self.self_attention.forward(inputs, inputs, inputs, self_attn_mask)
        '''
        sub_layer_outputs = self.layer_norm1.forward(self.residual_connection1.forward(residual, self_attn_outputs))

        residual = sub_layer_outputs
        fnn_layer_outputs = self.fnn_layer.forward(sub_layer_outputs)
        sub_layer_outputs = self.layer_norm2.forward(self.residual_connection2.forward(residual, fnn_layer_outputs))
        '''
        return self_attn_outputs


def get_attn_padding_mask(seq_q, seq_k, pad):
    """ indicate the padding-related part to mask """
    '''
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(pad).unsqueeze(1).expand(mb_size, len_q, len_k)
    subsequent_padding_mask = seq_q.data.eq(pad).unsqueeze(2).expand(mb_size, len_q, len_k)

    self_attn_mask = pad_attn_mask | subsequent_padding_mask
    return self_attn_mask


class TransformerEncoder(nn.Module):
    """ transformer encoder """

    def __init__(self, context_size=250, hidden_size=512, num_layers=2, num_heads=4, inner_linear=512,
                 embed_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1, relu_dropout=0.1, pad=0):
        super(TransformerEncoder, self).__init__()
        print("self attention network hyper parameters: ")
        print("hidden size: %d" % hidden_size)
        print("num layers: %d" % num_layers)
        print("num heads: %d" % num_heads)
        print("inner linear size: %d" % inner_linear)
        print("embed dropout rate: %.2f" % embed_dropout)
        print("residual dropout rate: %.2f" % residual_dropout)
        print("attention dropout rate: %.2f" % attention_dropout)
        print("relu dropout rate: %.2f" % relu_dropout)
        self.dropout = nn.Dropout(embed_dropout)
        self.pad = pad
        self.W = nn.Linear(context_size, hidden_size)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(hidden_size, num_heads, inner_linear, attention_dropout,
                                                          residual_dropout, relu_dropout) for _ in range(num_layers)])


    def forward(self, x, src_seq):
        """ input include attend words """
        self_attn_mask = get_attn_padding_mask(src_seq, src_seq, self.pad)  # result: (mb_size, len_q, len_k)
        '''
        Do not consider position.
        position_embedding = Variable(positional_embedding(x), requires_grad=False)
        x = x + position_embedding
        '''
        encoder_inputs = self.W(self.dropout(x))
        # encoder_inputs = self.W(x)
        enc_outputs = encoder_inputs

        for enc_layer in self.encoder_blocks:
            enc_outputs = enc_layer.forward(enc_outputs, self_attn_mask)

        return enc_outputs
