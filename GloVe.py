import logging
import os
import re
import sys
from itertools import chain

import gensim
import pandas as pd
import torch
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

import pickle

embed_size = 300
max_len = 512

# 读取文件
train = pd.read_csv('../../Bag of Words Meets Bags of Popcorn/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('../../Bag of Words Meets Bags of Popcorn/testData.tsv', header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv('../../Bag of Words Meets Bags of Popcorn/unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)


# 获得review
def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    return words


# 函数encode_samples获得一个特征列表
def encode_samples(tokenized_samples):
    # 定义一个空的列表来存储特征
    features = []
    # 遍历每一个单词列表
    for sample in tokenized_samples:
        # 定义一个空的列表存储该sample里单词的特征
        feature = []
        # 遍历该sample中的每个token
        for token in sample:
            # 判断token是否存在于word_to_idx字典中
            if token in word_to_idx:
                # 如果存在将token在word_to_idx的索引值返回到feature列表中
                feature.append(word_to_idx[token])
            else:
                # 否则添加到feature中
                feature.append(0)
        # 将feature加入到features中
        features.append(feature)
    # 返回features
    return features


def pad_samples(features, maxlen=max_len, PAD=0):
    # 定义一个空列表padded_features
    padded_features = []
    # 遍历每个feature
    for feature in features:
        # 如果feature长度大于maxlen
        if len(feature) >= maxlen:
            # 将前maxlen元素作为padded_feature得 的值
            padded_feature = feature[:maxlen]
        else:
            # 否则将feature直接存入padded_feature中
            padded_feature = feature
            # 进行循环，条件为padded_feature的长度小于maxlen
            while len(padded_feature) < maxlen:
                # 如果符合循环条件则将PAD的值加入到padded_feature中
                padded_feature.append(PAD)
        # 将padded_feature的值加入到padded_features中
        padded_features.append(padded_feature)
    # 返回padded_features
    return padded_features


if __name__ == '__main__':
    # 获取正在运行的程序的名称
    program = os.path.basename(sys.argv[0])
    # 创建一个日志记录器，名字与正在运行的程序相同
    logger = logging.getLogger(program)
    # 日志记录器的基本格式为：时间，日志级别，消息
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # 日志记录器的根级别为INFO
    logging.root.setLevel(level=logging.INFO)
    # 记录并输出一条消息，显示正在运行的程序和传递给该程序的参数。
    logger.info("running %s" % ''.join(sys.argv))
    # 创建两个空的列表
    clean_train_reviews, train_labels = [], []
    # 循环读取train数据集中的review的值和对应的序号
    for i, review in enumerate(train["review"]):  # enumerate是将可遍历的数据对象组合成一个索引序列
        # 调用review_to_wordlist函数读取review存入clean_train_reviews中
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=False))
        # 将训练集中sentiment的值存入train_labels中
        train_labels.append(train["sentiment"][i])
    # 创建一个空的列表
    clean_test_reviews = []
    # 读取测试集中的review
    for review in test["review"]:
        # 调用review_to_wordlist函数读取review的值存入clean_test_reviews中
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=False))
    # 创建一个词汇表存储训练集和测试集中的所有单词
    vocab = set(chain(*clean_train_reviews)) | set(chain(*clean_test_reviews))
    # 获得词汇表的长度
    vocab_size = len(vocab)

    train_reviews, val_reviews, train_labels, val_labels = train_test_split(clean_train_reviews, train_labels,
                                                                            test_size=0.2, random_state=0)
    # 载入文件‘glove_model.txt’
    wvmodel_file = os.path.join('glove_model.txt')
    # 使用gensim.models.KeyedVectors来加载模型load_word2vec_format
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(wvmodel_file, binary=False,
                                                              encoding='utf-8')  # 运行文件，是否为二进制文件
    # 为词汇表中每个单词分配一个索引值，并给未知词分配索引0
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'
    # 将处理过后的train_reviews，val_reviews，clean_test_reviews，train_labels，val_labels转为二位张量
    train_features = torch.tensor(pad_samples(encode_samples(train_reviews)))
    val_features = torch.tensor(pad_samples(encode_samples(val_reviews)))
    test_features = torch.tensor(pad_samples(encode_samples(clean_test_reviews)))
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    # 权重矩阵初始化设置为数值全为0，大小为为vocab_size + 1, embed_size的矩阵
    weight = torch.zeros(vocab_size + 1, embed_size)
    # 进行循环
    for i in range(len(wvmodel.index_to_key)):
        # 异常处理，跳过无法转换为索引的单词
        try:
            index = word_to_idx[wvmodel.index_to_key[i]]
            print(i)
        except:
            continue
        # 先利用索引值从word_to_idx读取单词，再读取单词对应的向量的索引值，再读取向量，最后将读取出来的向量赋值给weight张量
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx_to_word[word_to_idx[wvmodel.index_to_key[i]]]))
    # 创建一个存储结果的pickle文件名为imdb_glove.pickle3
    pickle_file = os.path.join('pickle', 'imdb_glove.pickle3')
    # 将数据train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,vocab
    # 存入pickle文件中
    pickle.dump(
        [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
         vocab],
        open(pickle_file, 'wb'))
    print('data dumped!')


