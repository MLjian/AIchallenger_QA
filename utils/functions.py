# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch
import numpy as np
import pandas as pd
import pdb

def pad_sens(batch_sens, max_limit, padding_idx):
    """
    按一个batch中最长句子的长度对sentences进行padding和truncate
    Args:
        batch_sens:
        max_limit: 最长句子上限，每个batch的最大句子长度不得超过
        padding_idx: padding值
    Returns:
        batch_sens_padded: padding后的一个batch的句子
    """
    lengths = [len(x) for x in batch_sens]
    max_len = max(lengths)
    if max_len > max_limit:
        max_len = max_limit
    batch_sens_padded = [sen[:max_len] if len(sen)>=max_len else sen + [padding_idx]* (max_len - len(sen)) for sen in batch_sens]

    return batch_sens_padded

def eval(model, p_valid, q_valid, a_valid, y_valid, batch_size):

    model.eval()
    start_idx_valid = 0
    acc_sum = 0
    i = 0
    while start_idx_valid + batch_size <= len(y_valid):
        """取一batch数据"""
        end_idx_valid = start_idx_valid + batch_size
        p_valid_batch = pad_sens(p_valid[start_idx_valid:end_idx_valid], 200, 0)
        q_valid_batch = pad_sens(q_valid[start_idx_valid:end_idx_valid], 30, 0)
        a_valid_batch = pad_sens(a_valid[start_idx_valid:end_idx_valid], 3, 0)
        y_valid_batch = y_valid[start_idx_valid:end_idx_valid]
        start_idx_valid = end_idx_valid

        """数据预处理：按句子长度降序排序;pad_sequence"""
        p_valid_batch = torch.LongTensor(p_valid_batch).cuda()
        q_valid_batch = torch.LongTensor(q_valid_batch).cuda()
        a_valid_batch = torch.LongTensor(a_valid_batch).cuda()
        y_valid_batch = torch.LongTensor(y_valid_batch).cuda()

        """前向传播"""
        with torch.no_grad():
            output = model(q_valid_batch, p_valid_batch, a_valid_batch)

        """计算一个batch的准确率"""
        preds = torch.argmax(output, dim=1)
        acc_batch = torch.mean(torch.eq(preds, y_valid_batch).float())

        """累加每个batch的准确率"""
        acc_sum += acc_batch.item()
        i += 1

    acc_mean = acc_sum / i
    model.train()
    return acc_mean

def test(model, p_test, q_test, a_test, id_test, cands, batch_size, to_path):
    model.eval()
    start_idx_test = 0
    preds_test = []
    while start_idx_test + batch_size <= len(p_test):
        """取一batch数据"""
        end_idx_test = start_idx_test + batch_size
        p_test_batch = pad_sens(p_test[start_idx_test:end_idx_test], 200, 0)
        q_test_batch = pad_sens(q_test[start_idx_test:end_idx_test], 30, 0)
        a_test_batch = pad_sens(a_test[start_idx_test:end_idx_test], 3, 0)
        start_idx_test = end_idx_test

        """数据预处理：按句子长度降序排序;pad_sequence"""
        p_test_batch = torch.LongTensor(p_test_batch).cuda()
        q_test_batch = torch.LongTensor(q_test_batch).cuda()
        a_test_batch = torch.LongTensor(a_test_batch).cuda()


        """前向传播"""
        with torch.no_grad():
            output = model(q_test_batch, p_test_batch, a_test_batch)

        """预测"""
        preds_batch = torch.argmax(output, dim=1).tolist()
        preds_test += preds_batch
    if start_idx_test < len(p_test):
        p_test_batch = pad_sens(p_test[start_idx_test:], 200, 0)
        q_test_batch = pad_sens(q_test[start_idx_test:], 30, 0)
        a_test_batch = pad_sens(a_test[start_idx_test:], 3, 0)
        p_test_batch = torch.LongTensor(p_test_batch).cuda()
        q_test_batch = torch.LongTensor(q_test_batch).cuda()
        a_test_batch = torch.LongTensor(a_test_batch).cuda()
        with torch.no_grad():
            output = model(q_test_batch, p_test_batch, a_test_batch)
        preds_batch = torch.argmax(output, dim=1).tolist()
        preds_test += preds_batch

    submissions = []
    for id, pred, candidates in zip(id_test, preds_test, cands):

        pred_ans = candidates[pred]
        sub_sample = str(id) + '\t' + pred_ans
        submissions.append(sub_sample)
    outputs = '\n'.join(submissions)
    with open(to_path, 'w', encoding='utf-8') as f:
        f.write(outputs)

    return preds_test
