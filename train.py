import pickle
import torch
import torch.nn as nn
from models.man import MAN
import time
from utils.functions import pad_sens, eval
import pdb
import visdom

"""加载数据"""
with open('./data_preprocess/processed_data/ai_challenger_oqmrc_trainingset_prepro_idxs.pkl', 'rb') as f_train:
    id_train, p_word_train, q_word_train, a_word_train, p_idx_train, q_idx_train, a_idx_train, label_train = pickle.load(f_train)
#id_train, p_word_train, q_word_train, a_word_train, p_idx_train, q_idx_train, a_idx_train, label_train = id_train[:1000], p_word_train[:1000], q_word_train[:1000], a_word_train[:1000], p_idx_train[:1000], q_idx_train[:1000], a_idx_train[:1000], label_train[:1000]
with open('./data_preprocess/processed_data/ai_challenger_oqmrc_validationset_prepro_idxs.pkl', 'rb') as f_valid:
    id_valid, p_word_valid, q_word_valid, a_word_valid, p_idx_valid, q_idx_valid, a_idx_valid, label_valid = pickle.load(f_valid)
#id_valid, p_word_valid, q_word_valid, a_word_valid, p_idx_valid, q_idx_valid, a_idx_valid, label_valid = id_valid[:1000], p_word_valid[:1000], q_word_valid[:1000], a_word_valid[:1000], p_idx_valid[:1000], q_idx_valid[:1000], a_idx_valid[:1000], label_valid[:1000]

"""定义模型"""
emb_path = './word2vec/sgns.zhihu.bigram-char_embedding.pkl'
model = MAN(embedding_path= emb_path, padding_idx=0, embedding_size=300)
model.cuda()

"""定义损失函数"""
criterion = nn.CrossEntropyLoss()

"""定义优化器"""
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

if __name__ == '__main__':
    print("开始训练......................")
    batch_size = 32
    n_epochs = 10
    print_fre = 400
    losses_sum = 0
    acc_sum = 0

    j = 0

    for epoch in range(n_epochs):
        # if epoch != 0 :
        #     model.embedding.embedding.weight.requires_grad_(requires_grad=True)

        time_start = time.time()
        start_idx = 0
        while start_idx + batch_size <= len(label_train):
            """取一batch数据"""
            end_idx = start_idx + batch_size
            p_train_batch = pad_sens(p_idx_train[start_idx:end_idx], 200, 0)
            q_train_batch = pad_sens(q_idx_train[start_idx:end_idx], 30,0)
            a_train_batch = pad_sens(a_idx_train[start_idx:end_idx], 3, 0)
            label_train_batch = label_train[start_idx:end_idx]
            start_idx = end_idx

            """数据预处理：按句子长度降序排序;pad_sequence"""
            p_train_batch = torch.LongTensor(p_train_batch).cuda()
            q_train_batch = torch.LongTensor(q_train_batch).cuda()
            a_train_batch = torch.LongTensor(a_train_batch).cuda()
            label_train_batch = torch.LongTensor(label_train_batch).cuda()

            """前向传播"""
            output = model(q_train_batch, p_train_batch, a_train_batch)

            """BP"""
            loss = criterion(output, label_train_batch)
            optimizer.zero_grad()
            loss.backward()

            """更新参数"""
            optimizer.step()

            """可视化训练集accuracy和Loss"""
            acc_batch = torch.mean(torch.eq(torch.argmax(output, dim=1), label_train_batch).float()).item()
            j += 1
            losses_sum += loss.item()
            acc_sum += acc_batch
            if j % print_fre == 0:
                loss_mean = round(losses_sum / print_fre, 4)
                losses_sum = 0
                acc_mean = round(acc_sum / print_fre, 3)
                acc_sum = 0
                print("第{}epoch, 每{}个batch, train_loss : {}, train_acc : {}".format(epoch, print_fre, loss_mean, acc_mean))
        """可视化验证集准确率"""
        acc_valid = round(eval(model, p_idx_valid, q_idx_valid, a_idx_valid, label_valid, 16), 3)
        time_end = time.time()
        tim_gap = round((time_end - time_start) / 60, 2)
        print("---------------------第{}epoch结束, valid_acc:{}, 耗时:{}min-----------------------".format(epoch, acc_valid, tim_gap))
        model_to_path = './trained_models/' + str(epoch) + '_' + str(acc_valid) + '.pt'
        torch.save(model.state_dict(), model_to_path)
        torch.cuda.empty_cache()