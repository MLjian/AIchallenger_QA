# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch
import torch.nn as nn
import pickle
from models.layer_units import ConcatAttentionUnit, BilinearAttentionUnit,  DotAttentionUnit,\
                               MinusAttentionUnit, GatedGRUUnit, ParaAttentionUnit, RqAttentionUnit, RNNUnit
import pdb
import numpy as np
import torch.nn.functional as F

class EmbeddingPreLayer(nn.Module):
    def __init__(self, embedding_path, padding_idx):
        super(EmbeddingPreLayer, self).__init__()
        self.padding_idx = padding_idx
        with open(embedding_path, 'rb') as f_w:
            num_vocab, embedding_size, _, embeddings = pickle.load(f_w)
            weights = torch.from_numpy(embeddings.astype(np.float32))
        self.embedding = nn.Embedding(num_embeddings=num_vocab, embedding_dim=embedding_size, padding_idx=self.padding_idx, _weight=weights)
        # self.embedding.weight.requires_grad_(requires_grad=False)
    def forward(self, sen_idx):
        mask = self.compute_mask(v=sen_idx)
        sen_emb = self.embedding(sen_idx)
        return sen_emb, mask

    def compute_mask(self, v):
        mask = torch.ne(v, self.padding_idx)#.float()
        return mask

class MultiwayMatchingLayer(nn.Module):
    def __init__(self, p_size, q_size, pq_size, vc_size, vd_size, vm_size):
        super(MultiwayMatchingLayer, self).__init__()
        self.ca = ConcatAttentionUnit(q_size, p_size, vc_size)
        self.ba = BilinearAttentionUnit(q_size, p_size)
        self.da = DotAttentionUnit(pq_size, vd_size)
        self.ma = MinusAttentionUnit(pq_size, vm_size)

    def forward(self, hq, hp):
        out_ca = self.ca(hq, hp)
        out_ba = self.ba(hq, hp)
        out_da = self.da(hq, hp)
        out_ma = self.ma(hq, hp)
        return out_ca, out_ba, out_da, out_ma

class InsideAggregationLayer(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers, dropout_p):
        super(InsideAggregationLayer, self).__init__()
        self.rnn_c = GatedGRUUnit('GRU', in_size, hidden_size, True, n_layers, dropout_p)
        self.rnn_b = GatedGRUUnit('GRU', in_size, hidden_size, True, n_layers, dropout_p)
        self.rnn_d = GatedGRUUnit('GRU', in_size, hidden_size, True, n_layers, dropout_p)
        self.rnn_m = GatedGRUUnit('GRU', in_size, hidden_size, True, n_layers, dropout_p)

    def forward(self, hp, hp_mask, out_ca, out_ba, out_da, out_ma):
        c = torch.cat((out_ca, hp), dim=2)
        b = torch.cat((out_ba, hp), dim=2)
        d = torch.cat((out_da, hp), dim=2)
        m = torch.cat((out_ma, hp), dim=2)
        out_c = self.rnn_c(c, hp_mask)
        out_b = self.rnn_c(b, hp_mask)
        out_d = self.rnn_c(d, hp_mask)
        out_m = self.rnn_c(m, hp_mask)
        return out_c, out_b, out_d, out_m

class MixedAggregationLayer(nn.Module):
    def __init__(self, h_size, va_size, v_size, hidden_size, bidirectional, num_layers, drop_p):
        super(MixedAggregationLayer, self).__init__()
        self.att = ParaAttentionUnit(v_size=v_size, h_size=h_size, va_size=va_size)
        self.gru = RNNUnit(rnn_model='GRU', input_size=h_size, hidden_size=hidden_size, bidirectional=bidirectional,
                           num_layers=num_layers, dropout_p=drop_p)

    def forward(self, ca, ba, da, ma, hp_mask):
        ca_uns, ba_uns, da_uns, ma_uns = ca.unsqueeze(dim=2), ba.unsqueeze(dim=2), da.unsqueeze(dim=2), ma.unsqueeze(dim=2)
        com = torch.cat((ca_uns, ba_uns, da_uns, ma_uns), dim=2)
        sen = []
        for i in range(com.size(1)):
            com_elem = com[:, i, :, :].squeeze(dim=1)#[batch_size, n_atts, h_size]
            word = self.att(com_elem) #[batch_size, h_size]
            sen.append(word.unsqueeze(dim=1))
        sen_h = torch.cat(sen, dim=1) #[batch_size, sen_len, h_size]

        out_gru = self.gru(sen_h, hp_mask) #[batch_size, sen_len, hidden_size]
        return out_gru

class QuestionMixedAttentionLayer(nn.Module):
    def __init__(self, vq_size, q_size, v_size, ho_size, vo_size):
        super(QuestionMixedAttentionLayer, self).__init__()
        self.q_att = ParaAttentionUnit(v_size=v_size, h_size=q_size, va_size=vq_size)
        self.ho_att = RqAttentionUnit(v_size=vo_size, ho_size=ho_size, q_size=q_size)

    def forward(self, ho, hq):
        rq = self.q_att(hq)

        rp = self.ho_att(ho, rq)
        return rp

class MLPLayer(nn.Module):
    def __init__(self, in_size, l1_size, l2_size, l3_size, n_classes):
        super(MLPLayer, self).__init__()
        self.l1 = nn.Linear(in_size, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.out = nn.Linear(l2_size, n_classes)

    def forward(self, inp):
        l1_o = torch.relu(self.l1(inp))
        l2_o = torch.relu(self.l2(l1_o))
        score = self.out(l2_o)
        return score

class DecissionLayer(nn.Module):

    def __init__(self, a_size, v_size, embedding_size):
        super(DecissionLayer, self).__init__()
        # self.sa = DotAttentionUnit(h_size=a_size, v_size=v_size)
        # self.a_encoder = RNNUnit(rnn_model='GRU', input_size=embedding_size, hidden_size=embedding_size,
        #                        bidirectional=True, num_layers=1, dropout_p=0)
        self.linear = nn.Linear(embedding_size, 2*embedding_size)

    def forward(self, encoder_out, a_emb):
        o_linear = self.linear(a_emb)
        score = o_linear.bmm(encoder_out.unsqueeze(dim=2)).squeeze() / 3
        return score