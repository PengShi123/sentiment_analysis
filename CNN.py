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

'''
1、先给定固定参数的值：迭代次数，每个批次处理数据的大小，滤波器的信息、学习率等
2、定义一个情感分析网络的类，其中包括神经网络的初始化，前向传播的初始化
3、设定日志来记录程序的运行情况
3、获得后续所需的数据
4、将训练数据和验证数据输入到神经网络中
5、将测试数据输入到神经网络中
6、将结果写入文件中
'''
# 读取数据
test = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/testData.tsv", header=0, delimiter="\t", quoting=3)
num_epochs = 10
# 嵌入向量的维度
embed_size = 300
# 卷积层中滤波器的数量
num_filter = 128
# 卷积层中滤波器的大小
filter_size = 3
bidirectional = True
batch_size = 64
# 分类任务的类别数量
labels = 2
lr = 0.8
device = torch.device('cuda:0')
# 判断是否使用gpu计算
use_gpu = True


# 定义一个SentimentNet类，继承了pytorch的nn.Module类，可近似把Sentiment看成是pytorch类
class SentimentNet(nn.Module):
    # 初始化模型参数
    def __init__(self, embed_size, num_filter, filter_size, weight, labels, use_gpu, **kwargs):
        # 调用父类nn.Module的构造函数，传递其他参数
        super(SentimentNet, self).__init__(**kwargs)
        # 将use_gpu参数保存为实例变量
        self.use_gpu = use_gpu
        # 创建一个预训练的词嵌入层，权重由传入的weight提供
        self.embedding = nn.Embedding.from_pretrained(weight)
        # 将词嵌入层的权重设置为不可训练
        self.embedding.weight.requires_grad = False
        # 创建一个一维卷积层
        self.conv1d = nn.Conv1d(embed_size, num_filter, filter_size, padding=1)
        # 创建一个relu激活函数
        self.activate = F.relu
        # 创建一个线性层，用于将卷积层的输出映射到分类任务的类别数量
        self.decoder = nn.Linear(num_filter, labels)

    # 定义forward前向传播函数
    def forward(self, inputs):
        # 将输入的文本通过嵌入层转换为向量
        embeddings = self.embedding(inputs)
        # 对词嵌入向量进行卷积操作，然后应用ReLU激活函数。permute操作是为了调整维度顺序以适应卷积操作
        convolution = self.activate(self.conv1d(embeddings.permute([0, 2, 1])))
        # 对卷积后的特征图进行最大池化操作，提取最重要的特征
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        # 将池化后的特征通过线性层映射到分类任务的类别数量，得到模型的输出。squeeze操作是为了移除大小为1的维度
        outputs = self.decoder(pooling.squeeze(dim=2))
        # 返回outputs
        return outputs


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
    # 显示正在读入数据
    logging.info('loading data...')
    # 读入imdb_glove.pickle3文件
    pickle_file = os.path.join('../pickle/imdb_glove.pickle3')
    # 从imdb_glove.pickle3中读取train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx，
    # idx_to_word,vocab
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word,
     vocab] = pickle.load(open(pickle_file, 'rb'))
    # 读取完数据后输出data loaded
    logging.info('data loaded!')
    # 构建神经网络，初始化神经网络的变量：嵌入向量的维度，滤波器的数量，每个滤波器的大小，权重，标签，gpu使用情况
    net = SentimentNet(embed_size=embed_size, num_filter=num_filter, filter_size=filter_size,
                       weight=weight, labels=labels, use_gpu=use_gpu)
    # 将SentimentNet模型（或任何其他PyTorch模型）移动到指定的设备上，设备可以是CPU或GPU
    net.to(device)
    # 定义损失函数，利用交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义了一个随机梯度下降（SGD）优化器，用于更新模型中的所有参数，lr为学习率
    optimizer = optim.SGD(net.parameters(), lr=lr)
    # 利用torch.utils.data.TensorDataset将train_features, train_labels打包成一个训练数据集
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    # 利用torch.utils.data.TensorDataset将val_features, val_labels打包成一个验证数据集
    val_set = torch.utils.data.TensorDataset(val_features, val_labels)
    # 利用torch.utils.data.TensorDataset将test_features打包成一个测试数据集,测试数据可以没有test_labels
    test_set = torch.utils.data.TensorDataset(test_features, )
    # 利用torch.utils.data.DataLoader进行训练集数据的迭代，迭代的内容为train_set，batch_size设置为64个批次，每个epoch开始时将数据打乱
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # 利用torch.utils.data.DataLoader进行验证集数据的迭代，迭代的内容为batch_size，batch_size设置为64个批次，不打乱数据
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # 利用torch.utils.data.DataLoader进行测试集的迭代，迭代内容为batch_size，batch_size设置为64个批次，不打乱数据
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 进行循环，读取每个epoch
    for epoch in range(num_epochs):
        # 开始时间
        start = time.time()
        # 设置变量训练损失值和验证损失值都为0
        train_loss, val_losses = 0, 0
        # 设置变量训练精确度和验证精确度为0
        train_acc, val_acc = 0, 0
        # 设置n和m分别对训练数据和验证数据进行计数
        n, m = 0, 0
        # 创建一个进度条，总长度为训练数据迭代器的长度，desc为当前对应的第几个迭代数
        with tqdm(total=len(train_iter), desc='Epoch %d' % epoch) as pbar:
            # 每次迭代得到一个batch的数据和对应的标签。
            for feature, label in train_iter:
                # 对n进行赋值
                n += 1
                # 将梯度清0
                net.zero_grad()
                # 将上面的特征移动到GPU上计算
                feature = Variable(feature.cuda())
                # 将上面的标签移动到GPU上计算
                label = Variable(label.cuda())
                # 将输入的特征进行前向传播，得到分数
                score = net(feature)
                # 调用损失函数利用分数和标签计算获得损失值
                loss = loss_function(score, label)
                # 根据损失值计算每个参数的梯度
                loss.backward()
                # 根据梯度更新参数的数值
                optimizer.step()
                # 训练的精确度=之前的训练精度+精确值分数。score和label都是在cpu上计算，torch.argmax用来获得每个样本的预测类别
                train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                         dim=1), label.cpu())
                # 训练损失 = 之前训练损失 + 损失函数的结果
                train_loss += loss
                # 设置进度条的后缀信息，迭代进度，训练损失，训练准确率
                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                # 每次更新一个单位
                pbar.update(1)
            # 结束梯度计算
            with torch.no_grad():
                # 在验证集的迭代中获得测试集的特征和标签
                for val_feature, val_label in val_iter:
                    # 对m进行赋值
                    m += 1
                    # 利用gpu计算验证集的特征
                    val_feature = val_feature.cuda()
                    # 利用gpu计算验证集的标签
                    val_label = val_label.cuda()
                    # 输入的特征进入神经网络进行前向传播来获得分数
                    val_score = net(val_feature)
                    # 利用损失函数来得到验证集的孙树
                    val_loss = loss_function(val_score, val_label)
                    # 获得验证集的准确率
                    val_acc += accuracy_score(torch.argmax(val_score.cpu().data, dim=1), val_label.cpu())
                    # 将每个批次的损失叠加
                    val_losses += val_loss
            # 获得结束时间
            end = time.time()
            # 一个迭代使用的时间
            runtime = end - start
            # 设置进度条的后缀信息，迭代进度，训练损失，训练准确率，测试损失，测试精度，时间
            pbar.set_postfix({'epoch': '%d' % (epoch),
                              'train loss': '%.4f' % (train_loss.data / n),
                              'train acc': '%.2f' % (train_acc / n),
                              'val loss': '%.4f' % (val_losses.data / m),
                              'val acc': '%.2f' % (val_acc / m),
                              'time': '%.2f' % (runtime)})
    # 创建一个测试集的预测列表
    test_pred = []
    # 结束梯度下降
    with torch.no_grad():
        # 设置一个进度条，长度为测试集的迭代次数，名字为Prediction
        with tqdm(total=len(test_iter), desc='Prediction') as pbar:
            # 从每个批次中读取test_feature
            for test_feature, in test_iter:
                # 利用gpu计算test_feature
                test_feature = test_feature.cuda()
                # 利用神经网络实现前向传播获得分数
                test_score = net(test_feature)
                # 将多个元素加入到预测列表中，tolist()将numpy转为python列表
                test_pred.extend(torch.argmax(test_score.cpu().data, dim=1).numpy().tolist())
                # 每次更新一条
                pbar.update(1)
    # 输出数据的格式为id，sentiment
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    # 将数据存入文件cnn.csv中
    result_output.to_csv('../result/cnn.csv', index=False, quoting=3)
    # 提示运行结束
    logging.info('result saved!')
