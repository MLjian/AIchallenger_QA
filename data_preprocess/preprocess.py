# -*- coding: utf-8 -*-
"""
@brief: 对训练集/验证集/测试集进行分词；对训练集/验证集进行class标注；
@author: Jian
"""
import pandas as pd
import json
import pickle
import jieba
import pdb
import random

random.seed(666)

def json_to_df(data_path, have_ans):
    df = pd.DataFrame()
    with open(data_path, 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()
        q_id = []
        ur = []
        pas = []
        que = []
        alt = []
        if have_ans:
            ans = []
        for i, sample in enumerate(lines):
            sample = json.loads(sample)
            q_id.append(sample['query_id'])
            ur.append(sample['url'])
            pas.append(sample['passage'])
            que.append(sample['query'])
            alt.append(sample['alternatives'])
            if have_ans:
                ans.append(sample['answer'])
        df['query_id'] = q_id
        df['url'] = ur
        df['passage'] = pas
        df['query'] = que
        df['alternatives'] = alt
        if have_ans:
            df['answer'] = ans
    return df

def sentence_seg(sentence):
    sentence_seg = list(jieba.cut(sentence))
    return sentence_seg

def annotate(row):
    if row['answer'] not in row['a_word']:
        raise ValueError("标注失败，答案不在候选选项中！")
    for i, ans in enumerate(row['a_word']):
        if ans == row['answer']:
            return i
        
def split_ans(ans):
    ans = ans.split('|')
    if len(ans) > 3:
        print(ans)
        ans = ans[:3]
    if len(ans) < 3:
        print(ans)
        ans = ans + ['无法确定']*(3-len(ans))
    #pdb.set_trace()
    # ans_1 = ans[:2]
    # ans_2 = [ans[2]]
    # random.shuffle(ans_1)
    # ans =  ans_1 + ans_2
    random.shuffle(ans)
    return ans

def preprocess_train(train_path):
    df = json_to_df(train_path, True)

    df['p_word'] = df['passage'].apply(sentence_seg)
    df['q_word'] = df['query'].apply(sentence_seg)

    df['a_word'] = df['alternatives'].apply(split_ans)

    df['label'] = df.apply(annotate, axis=1)

    df.drop(columns=['query', 'passage', 'url', 'alternatives'], inplace=True)

    to_path = './processed_data/' + train_path.split('/')[-1][:-5] + '_prepro.pkl'
    with open(to_path, 'wb') as f_df:
        pickle.dump(df, f_df)
    return df

def preprocess_test(test_path):
    df = json_to_df(test_path, False)

    df['p_word'] = df['passage'].apply(sentence_seg)
    df['q_word'] = df['query'].apply(sentence_seg)

    df['a_word'] = df['alternatives'].apply(split_ans)

    df.drop(columns=['query', 'passage', 'url',  'alternatives'], inplace=True)

    to_path = './processed_data/' + test_path.split('/')[-1][:-5] + '_prepro.pkl'
    with open(to_path, 'wb') as f_df:
        pickle.dump(df, f_df)
    return df

if __name__ == '__main__':
    # train_path = '../data/ai_challenger_oqmrc_trainingset.json'
    # valid_path = '../data/ai_challenger_oqmrc_validationset.json'
    valid_path = '../data/ai_challenger_oqmrc_validtest.json'
    # test_path = '../data/ai_challenger_oqmrc_testa.json'
    #
    # df_train = preprocess_train(train_path)
    df_valid = preprocess_train(valid_path)
    # df_test = preprocess_test(test_path)
