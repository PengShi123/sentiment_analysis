import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from sklearn.metrics import accuracy_score

# 读入数据
test = pd.read_csv("/sentiment/data/testData.tsv", header=0, delimiter="\t", quoting=3)

# 迭代次数
num_epochs = 10
# 最大长度
max_len = 512
# 嵌入层大小
embed_size = 300
# 滤波器数量
num_filter = 128
# 滤波器大小
filter_size = 3
# 池化大小
pooling_size = 2
# 每层神经元个数
num_hiddens = 64
# 层数
num_layers = 2
# 支持双向
bidirectional = True
# 每个批次的数量
batch_size = 64
# 最终结果分为两类
labels = 2
# 学习率为0.8
lr = 0.8
# 使用gpu
device = torch.device('cuda:0')
use_gpu = True


class SentimentNet(nn.Module):
    # 初始化
    def __init__(self, embed_size, num_filter, filter_size, num_hiddens, num_layers, bidirectional, weight, labels,
                 use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_filter = num_filter
        self.filter_size = filter_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.labels = labels

        self.use_gpu = use_gpu

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        # 一维卷积
        self.conv1d = nn.Conv1d(self.embed_size, self.num_filter, self.filter_size, padding=1)
        # 激活函数为relu
        self.activate = F.relu
        # LSTM初始化
        self.encoder = nn.LSTM(input_size=max_len // pooling_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        # 判断是否为双向LSTM
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        # 将输入的数据通过嵌入层转换为嵌入向量
        embeddings = self.embedding(inputs)
        # 进行维度调整，再进行一维卷积，再利用激活函数得到卷积向量
        convolution = self.activate(self.conv1d(embeddings.permute([0, 2, 1])))
        # 对卷积向量进行最大池化
        pooling = F.max_pool1d(convolution, kernel_size=pooling_size)
        # 将最大池化的结果输入到编码器中得到最初状态和隐藏层的最终状态
        states, hidden = self.encoder(pooling.permute([1, 0, 2]))
        # 在第二个维度上将初始状态和最终状态拼接起来
        encoding = torch.cat([states[0], states[-1]], dim=1)
        # 通过解码器得到结果
        outputs = self.decoder(encoding)
        return outputs


if __name__ == '__main__':
    # 日志记录程序运行情况
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('../pickle/imdb_glove.pickle3')
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')
    net = SentimentNet(embed_size=embed_size, num_filter=num_filter, filter_size=filter_size,
                       num_hiddens=num_hiddens, num_layers=num_layers, bidirectional=bidirectional,
                       weight=weight, labels=labels, use_gpu=use_gpu)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    test_set = torch.utils.data.TensorDataset(test_features, )
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            for feature, label in train_iter:
                n += 1
                net.zero_grad()
                feature = Variable(feature.cuda())
                label = Variable(label.cuda())
                score = net(feature)
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
                for val_feature, val_label in val_iter:
                    m += 1
                    val_feature = val_feature.cuda()
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
                              'time': '%.2f' % (runtime)})
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
    result_output.to_csv("../result/cnn_lstm.csv", index=False, quoting=3)
    logging.info('result saved!')
