# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch.nn as nn
from models.layers import EmbeddingPreLayer, MultiwayMatchingLayer, InsideAggregationLayer, MixedAggregationLayer,\
                        QuestionMixedAttentionLayer, DecissionLayer
from models.layer_units import RNNUnit
import pdb

class MAN(nn.Module):
    def __init__(self, embedding_path, padding_idx, embedding_size):
        super(MAN, self).__init__()
        self.embedding = EmbeddingPreLayer(embedding_path=embedding_path, padding_idx=padding_idx)
        self.p_coder = RNNUnit(rnn_model='GRU', input_size=embedding_size, hidden_size=embedding_size,
                               bidirectional=True, num_layers=1, dropout_p=0)
        self.q_coder = RNNUnit(rnn_model='GRU', input_size=embedding_size, hidden_size=embedding_size,
                               bidirectional=True, num_layers=1, dropout_p=0)

        self.mm = MultiwayMatchingLayer(p_size=2*embedding_size, q_size=2*embedding_size, pq_size=2*embedding_size,
                                        vc_size=embedding_size, vd_size=embedding_size, vm_size=embedding_size)
        self.ia = InsideAggregationLayer(in_size=2*2*embedding_size, hidden_size=embedding_size, n_layers=1, dropout_p=0.2)
        self.ma = MixedAggregationLayer(h_size=2*embedding_size, hidden_size=embedding_size, va_size=embedding_size, v_size=embedding_size, bidirectional=True, num_layers=1, drop_p=0.2)
        self.qm = QuestionMixedAttentionLayer(vq_size=embedding_size, v_size=embedding_size, q_size=2*embedding_size, ho_size=2*embedding_size, vo_size=embedding_size)
        self.dec = DecissionLayer(a_size=2*embedding_size, v_size=embedding_size, embedding_size= embedding_size)

    def forward(self, q_words, p_words, a_words):

        p_emb, p_mask = self.embedding(p_words)
        q_emb, q_mask = self.embedding(q_words)
        hp = self.p_coder(p_emb, p_mask)

        hq = self.q_coder(q_emb, q_mask)

        o_ca, o_ba, o_da, o_ma = self.mm(hq=hq, hp=hp)

        o_ca_ia, o_ba_ia, o_da_ia, o_ma_ia = self.ia(hp, p_mask, o_ca, o_ba, o_da, o_ma)

        o_m = self.ma(o_ca_ia, o_ba_ia, o_da_ia, o_ma_ia, p_mask)

        rp = self.qm(o_m, hq)

        a_emb, a_mask = self.embedding(a_words)
        prob = self.dec(encoder_out=rp, a_emb=a_emb)

        return prob

