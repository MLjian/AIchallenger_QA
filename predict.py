# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""

from models.man import MAN
import torch
from utils.functions import test
import pickle
import numpy as np
with open('./data_preprocess/processed_data/ai_challenger_oqmrc_validtest_prepro_idxs.pkl', 'rb') as f_test:
    id_test, p_word, q_word, a_word, p_idx, q_idx, a_idx, label= pickle.load(f_test)
#id_test, p_test, q_test, a_test, cands = id_test[:1000], p_test[:1000], q_test[:1000], a_test[:1000], cands[:1000]


model_path = './trained_models/1_0.698.pt'
emb_path = './word2vec/sgns.zhihu.bigram-char_embedding.pkl'
model = MAN(embedding_path=
            emb_path, padding_idx=0, embedding_size=300)
model.cuda()
model.load_state_dict(torch.load(model_path))

to_path = './results/submission_a' + '_' + model_path.split('/')[-1][:-3] + '.txt'
test_preds = test(model, p_idx, q_idx, a_idx, id_test,a_word, 64, to_path)

test_preds = np.array(test_preds)
labels = np.array(label)
test_acc = np.equal(test_preds, labels).astype(float).mean()
print(test_acc)

