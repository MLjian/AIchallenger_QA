import pickle


class Data2Idx():
    def __init__(self, data_path, word2id_path, have_label):
        self.data_path = data_path
        self.word2id_path = word2id_path
        self.have_label = have_label
        self.to_path = data_path[:-4] + '_idxs.pkl'

    def transform(self):
        with open(self.word2id_path, 'rb') as f:
            _, _, self.word2id, _ = pickle.load(f)

        with open(self.data_path, 'rb') as f:
            df = pickle.load(f) # columns=['p_word', 'q_word', 'a_word', 'label']

        df['p_idx'] = df['p_word'].apply(self.__sentence2idx)
        df['q_idx'] = df['q_word'].apply(self.__sentence2idx)
        df['a_idx'] = df['a_word'].apply(self.__sentence2idx)
       
        p_word_li = list(df['p_word'])
        q_word_li = list(df['q_word'])
        a_word_li = list(df['a_word'])

        p_idx_li = list(df['p_idx'])
        q_idx_li = list(df['q_idx'])
        a_idx_li = list(df['a_idx'])
        ids = list(df['query_id'])

        if self.have_label:
            label = list(df['label'])
            data = (ids, p_word_li, q_word_li, a_word_li, p_idx_li, q_idx_li, a_idx_li, label)
        else:
            data = (ids, p_word_li, q_word_li, a_word_li, p_idx_li, q_idx_li, a_idx_li)
        with open(self.to_path, 'wb') as f:
            pickle.dump(data, f)
        return df

    def __sentence2idx(self, sentence):
        idxs = [self.word2id[word] if word in self.word2id else self.word2id['<unk>'] for word in sentence]
        return idxs

if __name__ == '__main__':
    data_train = './processed_data/ai_challenger_oqmrc_trainingset_prepro.pkl'
    # data_valid = './processed_data/ai_challenger_oqmrc_validationset_prepro.pkl'
    data_valid = './processed_data/ai_challenger_oqmrc_validtest_prepro.pkl'
    data_test = './processed_data/ai_challenger_oqmrc_testa_prepro.pkl'
    w2i_path = '../word2vec/sgns.zhihu.bigram-char_embedding.pkl'

    # d2idx_train = Data2Idx(data_train, w2i_path, True)
    # df_train = d2idx_train.transform()
    #
    d2idx_valid = Data2Idx(data_valid, w2i_path, True)
    df_valid = d2idx_valid.transform()
    #
    # d2idx_test = Data2Idx(data_test, w2i_path, False)
    # df_test = d2idx_test.transform()



