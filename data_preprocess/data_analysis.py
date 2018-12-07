import pickle
import numpy as np
import pandas as pd
from collections import Counter

train_path = './processed_data/ai_challenger_oqmrc_trainingset_prepro.pkl'
valid_path = './processed_data/ai_challenger_oqmrc_validationset_prepro.pkl'
test_path = './processed_data/ai_challenger_oqmrc_testa_prepro.pkl'
with open(train_path, 'rb') as f_train:
    df_train = pickle.load(f_train)
with open(valid_path, 'rb') as f_valid:
    df_valid = pickle.load(f_valid)
with open(test_path, 'rb') as f_test:
    df_test = pickle.load(f_test)
df_all = pd.concat((df_train, df_valid, df_test))

df_all['p_length'] = df_all['p_word'].apply(len)
df_all['q_length'] = df_all['q_word'].apply(len)

print("------------passage lengths-------------------")
print(df_all['p_length'].describe())
p_len_arr = df_all['p_length'].values
p_len_count = np.bincount(p_len_arr)
p_len_thr = p_len_count[200:]
print('passage中句子长度大于等于200的句子个数为：{}'.format(p_len_thr.sum()))

print("------------question lengths-------------------")
print(df_all['q_length'].describe())
q_len_arr = df_all['q_length'].values
q_len_count = np.bincount(q_len_arr)
q_len_thr = q_len_count[20:]
print('question中句子长度大于等于20的句子个数为：{}'.format(q_len_thr.sum()))

answers_list = sum(list(df_all['a_word']), [])
answers_set = set(answers_list)
"""
answers_set_enter = list(map(lambda x: x + '\n', answers_set))
with open('./candidates_vocabs.txt', 'w') as f_vocabs:
    f_vocabs.writelines(answers_set_enter)
"""
counts = Counter(answers_list)
with open('./candidates_vocabs_counts.pkl', 'wb') as f_counts:
    pickle.dump(counts, f_counts)
print(answers_set)
print(counts)
print(len(answers_set)==len(counts))