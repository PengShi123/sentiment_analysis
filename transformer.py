import logging
import os
import sys
import time
import math
import re

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

test = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/testData.tsv", header=0, delimiter="\t", quoting=3)

num_epochs = 10
embed_size = 300
num_hiddens = 120
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.05
device = torch.device('cuda:0')
use_gpu = True

# Read data from files
train = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/labeledTrainData.tsv", header=0, delimiter="\t",
                    quoting=3)
test = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/testData.tsv", header=0, delimiter="\t",
                   quoting=3)


# 将一段话分为句子再将句子切割为单词
def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    return ' '.join(words)


class Vocab:
    def __init__(self, tokens=None):
        # 初始化一个空的列表，用于存储从索引到 token 的映射
        self.idx_to_token = list()
        # 初始化一个空的字典，用于存储从 token 到索引的映射
        self.token_to_idx = dict()
        # 检查tokens是否存在
        if tokens is not None:
            # 如果<unk>不在tokens中
            if "<unk>" not in tokens:
                # 则在tokens列表最后加上<unk>
                tokens = tokens + ["<unk>"]
            # 如果<unk>在tokens中，进行循环读取tokens中的token
            for token in tokens:
                # 将 token 添加到 idx_to_token 列表的末尾
                self.idx_to_token.append(token)
                # 将 token 的索引（即该 token 在 idx_to_token 列表中的位置）存储到 token_to_idx 字典中
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            # 将 <unk> 的索引存储到 self.unk 属性中
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, train, test, min_freq=1, reserved_tokens=None):
        # 初始化一个字典存储每个token在训练数据和测试数据中出现的频率
        token_freqs = defaultdict(int)
        # 遍历每个句子
        for sentence in train:
            # 遍历每个句子中单词
            for token in sentence:
                # 每个token遍历一次频率加1
                token_freqs[token] += 1
        # 遍历测试集的句子
        for sentence in test:
            # 遍历每个句子的token
            for token in sentence:
                # 每个token遍历一次频率加1
                token_freqs[token] += 1
        # 初始化一个列表，用于存储最终的唯一 tokens
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        # 将那些频率大于等于 min_freq 且不等于 <unk> 的 token 添加到 uniq_tokens 列表中
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        # 使用 uniq_tokens 列表作为参数来创建一个新的对象，并返回它
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词汇表的大小，即 idx_to_token 列表的长度
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 返回 token 在 token_to_idx 字典中对应的索引，如果 token 不在字典中则返回 <unk> 的索引
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 将一个 tokens 列表转换为一个 indices 列表，其中每个 token 都被替换为其在 token_to_idx 字典中对应的索引
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        # 将一个 indices 列表转换为一个 tokens 列表，其中每个索引都被替换为其在 idx_to_token 列表中对应的 token
        return [self.idx_to_token[index] for index in indices]


def length_to_mask(lengths):
    # 计算 lengths 张量中的最大值
    max_length = torch.max(lengths)
    # 创建一个形状为 (lengths.shape[0], max_length) 的张量，其中每个元素都是一个从 0 到 max_length - 1 的整数
    mask = torch.arange(max_length).expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)
    # 返回布尔掩码张量
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        # 初始化全0张量
        pe = torch.zeros(max_len, d_model)
        # 创建包含每个位置索引的张量 position，形状为 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建包含指数级递减数值的张量 div_term，形状为 (d_model // 2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 计算正弦和余弦函数的位置编码，并存储在 pe 张量的偶数索引和奇数索引位置
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将 pe 张量的形状变为 (1, max_len, d_model)，并注册为模块的缓冲区（buffer）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 重构张量x
        x = x.unsqueeze(1).repeat(1, 1, 64, 1)
        # 在输入数据 x 上添加位置编码，得到添加了位置编码的输出数据
        x = x + self.pe[:x.size(0), :]
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=128, activation: str = "relu"):
        # 调用父类的方法
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim  # 嵌入向量的维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 创建一个词嵌入层
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)  # 位置编码层
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout,
                                                   activation)  # 创建一个Transformer编码器层
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers)  # 使用多个Transformer编码器层创建一个完整的Transformer编码器
        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)  # 创建一个线性层，将编码后的隐藏状态映射到输出类别

    def forward(self, inputs, lengths):
        # transpose函数用于将第一维向量和第二维向量交换位置
        inputs = torch.transpose(inputs, 0, 1)  # 将输入数据的维度转换一下，使得批次大小（batch size）在第二个维度上
        hidden_states = self.embeddings(inputs)  # 将输入数据转换成嵌入向量
        hidden_states = self.position_embedding(hidden_states)  # 将嵌入向量转换为位置嵌入向量，得到添加了位置状态的隐藏信息
        attention_mask = length_to_mask(lengths) == False  # 根据输入序列的长度生成注意力掩码
        # 将添加了位置信息的隐藏状态通过Transformer编码器，得到编码后的隐藏状态
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        # 取编码后的隐藏状态的第一个维度（批次大小）的第一个元素（一般为0），以及所有的时间步和特征维度，作为最终的隐藏状态输出
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# 将多个样本合并成一个batch
def collate_fn(examples):
    # 存储每个样本的长度
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    # 存储每个样本的输入序列
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # 存储每个样本的目标类别
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度，batch_first=True表示batch维度在前（即第一维度）
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


if __name__ == '__main__':
    # 日志记录
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    # 处理数据
    clean_train_reviews, train_labels = [], []
    for i, review in enumerate(train["review"]):
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=False))
        train_labels.append(train["sentiment"][i])

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=False))
    # 借助处理过的训练数据和测试数据构建一个词汇表
    vocab = Vocab.build(clean_train_reviews, clean_test_reviews)
    # 将序号和标签组成一个元组
    train_reviews = [(vocab.convert_tokens_to_ids(sentence), train_labels[i])
                     for i, sentence in enumerate(clean_train_reviews)]
    # 为每个句子设置一个序号
    test_reviews = [vocab.convert_tokens_to_ids(sentence)
                    for sentence in clean_test_reviews]
    # 划分数据集，训练集占80%，验证集占20%
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(train_reviews, train_labels,
                                                                            test_size=0.2, random_state=0)
    # 将词汇长度，嵌入层大小，隐藏层包含的神经元个数，类别的数量
    net = Transformer(vocab_size=len(vocab), embedding_dim=embed_size, hidden_dim=num_hiddens, num_class=labels)
    # 调用gpu0
    net.to(device)
    # 损失函数用交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # 对全部参数使用Adam
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 包装数据
    train_set = TransformerDataset(train_reviews)
    val_set = TransformerDataset(val_reviews)
    test_set = TransformerDataset(test_reviews)
    # 数据的迭代，拼接数据，每迭代一次就打乱数据
    train_iter = torch.utils.data.DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    # 开始迭代
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, lengths, label in train_iter:
                print(feature, lengths, label)
                n += 1
                net.zero_grad()
                feature = Variable(feature.cuda())
                lengths = Variable(lengths.cuda())
                label = Variable(label.cuda())
                score = net(feature, lengths)
                loss = loss_function(score, label)
                loss.backward()
                optimizer.step()
                train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                         dim=1), label.cpu())
                train_loss += loss

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)
            with torch.no_grad():
                for val_feature, val_length, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.cuda()
                    val_length = val_length.cuda()
                    val_label = val_label.cuda()
                    val_score = net(val_feature)
                    val_loss = loss_function(val_score, val_label)
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    val_losses += val_loss
            end = time.time()
            runtime = end - start
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)
                              })
    test_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            for test_feature, in test_iter:
                test_feature = test_feature.cuda()
                test_score = net(test_feature)
                # test_pred.extent
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())

                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("../result/transformer.csv", index=False, quoting=3)
    logging.info('result saved!')
