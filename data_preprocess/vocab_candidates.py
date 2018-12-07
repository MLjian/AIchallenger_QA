# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""

import pickle

with open('./candidates_vocabs_counts.pkl', 'rb') as f:
    counts = pickle.load(f)
    
with open('../word2vec/sgns.zhihu.bigram-char_embedding.pkl', 'rb') as f:
    _, _, word2id, _= pickle.load(f)
    
ans_count_2 = []
ans_vocabs = []
for word, count in counts.items():
    ans_vocabs.append(word)
    if count > 1:
        ans_count_2.append(word)

ans_count_2 = set(ans_count_2)
ans_vocabs = set(ans_vocabs)

emb_vocabs = set(word2id.keys())

result_1 = emb_vocabs & ans_vocabs
result_2 = emb_vocabs & ans_count_2
print(len(result_1), len(result_2))

add_words_2 = ans_count_2 - result_2
add_words_1 = ans_vocabs - result_1