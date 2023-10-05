import logging
import os
import sys
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from sklearn.metrics import accuracy_score

# 读取数据
test = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/testData.tsv", header=0, delimiter="\t", quoting=3)
# 迭代次数
num_epochs = 10
# 嵌入层的大小
embed_size = 300
# 每个隐藏层的神经元个数
num_hiddens = 120
# 总共有两层隐藏层
num_layers = 2
# 双向初始赋值为0
bidirectional = True
# 每批的大小
batch_size = 64
# 最后分为两类
labels = 2
# 学习率为0.05
lr = 0.05
# 调用的gpu是cuda'0’
device = torch.device('cuda:0')
# 使用gpu
use_gpu = True


class SentimentNet(nn.Module):
    # 初始化数据
    def __init__(self, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        # 初始化LSTM模型
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)  # dropout是进行正则化防止过拟合的，0代表不用dropout

        if self.bidirectional:
            # 如果是双向的LSTM则输入大小是每层神经元个数的4倍，输出大小为labels的大小
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            # 如果不是双向的LSTM则输入大小是每层神经元个数的2倍，输出大小为labels的大小
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    # 定义前向传播算法
    def forward(self, inputs):
        # 将输入数据转换为嵌入向量
        embeddings = self.embedding(inputs)
        # 通过编码器获得初始状态和隐藏层的最后状态
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        # 将第一个状态和最后一个状态在第二个方向上拼接起来
        encoding = torch.cat([states[0], states[-1]], dim=1)
        # 解码器工作得到结果
        outputs = self.decoder(encoding)
        return outputs


if __name__ == '__main__':
    # 日志记录文件的运行情况
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    logging.info('loading data...')
    # 读取文件
    pickle_file = os.path.join('../pickle/imdb_glove.pickle3')
    # 读取文件中的数据
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')
    # 神经网络初始化
    net = SentimentNet(embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight,
                       labels=labels, use_gpu=use_gpu)
    # 调用gpu
    net.to(device)
    # 损失函数使用交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # 对神经网络中的所有参数进行Adm优化，Adm能够在训练过程中自动调整学习率，适应不同的参数和梯度
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 利用torch.utils.data.TensorDataset将train_features, train_labels组合成一个训练集
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    # 利用torch.utils.data.TensorDataset将val_features, val_labels组合成一个验证集
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    # 利用torch.utils.data.TensorDataset将test_features组合成一个测试集
    test_set = torch.utils.data.TensorDataset(test_features, )
    # 利用torch.utils.data.DataLoader进行训练集数据的迭代，迭代的内容为train_set，batch_size设置为64个批次，每个epoch开始时将数据打乱
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # 利用torch.utils.data.DataLoader进行训练集数据的迭代，迭代的内容为val_set，batch_size设置为64个批次，每个epoch开始时将数据打乱
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # 利用torch.utils.data.DataLoader进行训练集数据的迭代，迭代的内容为test_set，batch_size设置为64个批次，每个epoch开始时将数据打乱
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 开始迭代
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

            # tqdm.write('{epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f}' %
            #       (epoch, train_loss.data / n, train_acc / n, val_losses.data / m, val_acc / m, runtime))

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
    result_output.to_csv("../result/lstm.csv", index=False, quoting=3)
    logging.info('result saved!')
