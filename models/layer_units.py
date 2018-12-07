# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

class ConcatAttentionUnit(nn.Module):

    def __init__(self, q_size, p_size, v_size):

        super(ConcatAttentionUnit, self).__init__()

        self.W1 = nn.Linear(q_size, v_size, bias=False)
        self.W2 = nn.Linear(p_size, v_size, bias=False)
        self.v = nn.Linear(v_size, 1, bias=False)

    def forward(self, hq, hp):

        _sq = self.W1(hq) #[batch_size, sen_q_len, v_size]
        _sp = self.W2(hp) #[batch_size, sen_p_len, v_size]
        sp = _sp.unsqueeze(dim=2) #[batch_size, sen_p_len, 1, v_size]
        sq = _sq.unsqueeze(dim=1) #[batch_size, 1, sen_q_len, v_size]
        h_plus = sp + sq #[batch_size, sen_p_en, sen_q_len, v_size]
        s = self.v(torch.tanh(h_plus)) #[batch_size, sen_p_len, sen_q_len, 1]

        a = F.softmax(s.squeeze(), dim=2) #[batch_size, sen_p_len, sen_q_len]

        q = a.bmm(hq) #[batch_size, sen_p_len, q_size]

        return q

class BilinearAttentionUnit(nn.Module):

    def __init__(self, q_size, p_size):
        super(BilinearAttentionUnit, self).__init__()
        self.W = nn.Linear(q_size, p_size)

    def forward(self, hq, hp):
        sq = self.W(hq) #[batch_size, sen_q_len, p_size]
        s = sq.bmm(hp.transpose(dim0=1, dim1=2)) #[batch_size, sen_q_len, sen_p_len]

        a = F.softmax(input=s, dim=1)

        q_att = a.transpose(dim0=1, dim1=2).bmm(hq) #[batch_size, sen_p_pen, q_size]

        return q_att

class DotAttentionUnit(nn.Module):
    def __init__(self, h_size, v_size):
        super(DotAttentionUnit, self).__init__()
        self.W = nn.Linear(h_size, v_size, bias=False)
        self.v = nn.Linear(v_size, 1, bias=False)

    def forward(self, hq, hp):
        hq_u = hq.unsqueeze(dim=1) #[batch_size, 1, sen_q_len, pq_size]
        hp_u = hp.unsqueeze(dim=2) #[batch_size, sen_p_len, 1, pq_size]
        h_mul = hq_u * hp_u #[batch_size, sen_p_len, sen_q_len, pq_size]

        s_w = torch.tanh(self.W(h_mul)) #[batch_size, sen_p_len, sen_q_len, v_size]
        s = self.v(s_w).squeeze() #[batch_size, sen_p_len, sen_q_len]

        a = F.softmax(input=s, dim=2)

        q_att = a.bmm(hq) #[batch_size, sen_p_len, pq_size]
        return q_att
class MinusAttentionUnit(nn.Module):
    def __init__(self, pq_size, v_size):
        super(MinusAttentionUnit, self).__init__()
        self.W = nn.Linear(pq_size, v_size, bias=False)
        self.v = nn.Linear(v_size, 1, bias=False)

    def forward(self, hq, hp):
        hq_u = hq.unsqueeze(dim=1).contiguous()
        hp_u = hp.unsqueeze(dim=2).contiguous()

        h_minus = hq_u - hp_u
        s_w = torch.tanh(self.W(h_minus))
        s = self.v(s_w).squeeze()
        a = F.softmax(input=s, dim=2)

        q_att = a.bmm(hq) #[batch_size, sen_p_len, pq_size]
        return q_att

class ParaAttentionUnit(nn.Module):
    def __init__(self, v_size, h_size, va_size):
        super(ParaAttentionUnit, self).__init__()
        self.va = nn.Parameter(torch.randn(va_size))
        self.v = nn.Linear(v_size, 1)
        self.W1 = nn.Linear(h_size, v_size)
        self.W2 = nn.Linear(va_size, v_size)

    def forward(self, h):
        w1 = self.W1(h)
        w2 = self.W2(self.va.unsqueeze(dim=0))
        w_plus = torch.tanh(w1 + w2)
        #pdb.set_trace()
        # s = self.v(w_plus.transpose(dim0=1, dim1=2))
        s = self.v(w_plus)
        s = s.squeeze()

        a = F.softmax(input=s, dim=1)
        a = a.unsqueeze(dim=1)

        x = a.bmm(h).squeeze() #[batch_size, h_size]?

        return x

class RqAttentionUnit(nn.Module):
    def __init__(self, v_size, ho_size, q_size):
        super(RqAttentionUnit, self).__init__()
        self.v = nn.Linear(v_size, 1)
        self.W1 = nn.Linear(ho_size, v_size)
        self.W2 = nn.Linear(q_size, v_size)

    def forward(self, h, rq):
        w1 = self.W1(h)
        w2 = self.W2(rq).unsqueeze(dim=1)

        w_plus = torch.tanh(w1 + w2)
        s = self.v(w_plus)
        s = s.squeeze()

        a = F.softmax(input=s, dim=1)
        a = a.unsqueeze(dim=1)

        x = a.bmm(h).squeeze() #[batch_size, h_size]

        return x

class RNNUnit(nn.Module):
    def __init__(self, rnn_model, input_size, hidden_size, bidirectional, num_layers, dropout_p=0):
        super(RNNUnit, self).__init__()

        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                               num_layers=num_layers, batch_first=True, dropout=dropout_p)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                              num_layers=num_layers, batch_first=True,dropout=dropout_p)
        else:
            raise ValueError('Wrong rnn_mode select %s, change to LSTM or GRU' % rnn_model)

    def forward(self, inp, inp_mask):
        """
        rnn。
        注：这里输入的inp中的最长句子，必须将最长长度占满，不能含有padding部分。
        Args:
            inp: [batch_size, sen_len, embedding_size]
            inp_mask: [batch_size, sen_len]
        Returns:
            out_encoder: [batch_size, sen_len, num_directions*hidden_size]
        """
        lengths = inp_mask.sum(dim=1).squeeze() #[batch_size]
        lengths_sort, idx_sort = torch.sort(input=lengths, descending=True)

        inp_sort = inp.index_select(dim=0, index=idx_sort)
        #pdb.set_trace()
        inp_pack = nn.utils.rnn.pack_padded_sequence(inp_sort, lengths_sort, batch_first=True)

        out_rnn, h_rnn = self.rnn(inp_pack)

        out_rnn_pad, _ = nn.utils.rnn.pad_packed_sequence(out_rnn, True, padding_value=0.0)

        _, idx_unsort = torch.sort(input=idx_sort, descending=False)
        out_encoder = out_rnn_pad.index_select(dim=0, index=idx_unsort)#在数据进入rnn前，对数据进行了排序处理，所以这里回复原顺序

        return out_encoder

class GatedGRUUnit(nn.Module):
    def __init__(self, rnn_model, in_size, hidden_size, bidirectional, n_layers, dropout_p):
        super(GatedGRUUnit, self).__init__()
        self.g = nn.Linear(in_size, in_size)
        self.gru = RNNUnit(rnn_model=rnn_model, input_size=in_size, hidden_size=hidden_size,
                         bidirectional=bidirectional, num_layers=n_layers, dropout_p=dropout_p)
    def forward(self, inp, inp_mask):
        inp_s = F.sigmoid(self.g(inp))
        inp_g = inp_s * inp
        out = self.gru(inp_g, inp_mask)
        return out