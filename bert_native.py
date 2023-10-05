import os
import sys
import logging
import time

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/labeledTrainData.tsv", header=0, delimiter="\t",
                    quoting=3)
test = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/testData.tsv", header=0, delimiter="\t",
                   quoting=3)
BERT_PATH = "D:/pycharm project/pythonProject1/sentiment/bert_native/bert-base-uncased"


class TrainDataset(torch.utils.data.Dataset):  # torch.utils.data.Dataset数据集的基类，提供了__getitem__和__len__方法
    def __init__(self, encodings, labels=None):
        # 存储数据集中的输入数据
        self.encodings = encodings
        # labels是训练集对应的标签，比如分类任务中的类别标签
        self.labels = labels

    # 获取数据集中指定索引位置的数据
    def __getitem__(self, idx):
        # # 创建一个字典，通过索引从encodings中获取数据，并将其转换为PyTorch的tensor格式
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 从labels中获取对应索引位置的标签，并将其转换为PyTorch的tensor格式
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    # 获取数据集的长度
    def __len__(self):
        return len(self.labels)


class TestDataset(torch.utils.data.Dataset):
    # 初始化函数，接收两个参数：encodings和num_samples
    def __init__(self, encodings, num_samples=0):
        # encodings是一个字典，存储了经过分词和编码后的文本数据
        self.encodings = encodings
        # num_samples表示数据集中的样本数量，默认为0
        self.num_samples = num_samples

    # 重写__getitem__方法，用于获取指定索引位置的数据样本
    def __getitem__(self, idx):
        # 遍历encodings字典中的每一个键值对，将val中对应索引位置的值转换为PyTorch张量，并存储到item字典中
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    # 重写__len__方法，返回数据集的样本数量
    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    # 读取数据集
    train_texts, train_labels, test_texts = [], [], []
    for i, review in enumerate(train["review"]):
        train_texts.append(review)
        train_labels.append(train['sentiment'][i])
    for review in test['review']:
        test_texts.append(review)
    # train_texts和train_labels按照80 % 和20 % 的比例分割成训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    # 预训练模型
    tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH, model_max_length=64)
    # 对训练集、验证集和测试集的文本数据进行分词和编码
    # 其中，truncation=True表示如果文本长度超过模型的最大输入长度，则进行截断
    # padding=True表示如果文本长度小于模型的最大输入长度，则进行填充
    # 得到编码
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings, num_samples=len(test_texts))
    # 选择使用gpu还是cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BertForSequenceClassification.from_pretrained(BERT_PATH)
    model.to(device)
    model.train()
    # 数据载入每8个一个批次，每个批次都将数据打乱
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # 使用AdmW进行参数优化，设定学习率
    optim = optim.AdamW(model.parameters(), lr=5e-5)
    # 开始迭代，迭代次数为3次
    for epoch in range(3):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        with tqdm(total=len(train_loader), desc="Epoch %d" % epoch) as pbar:
            for batch in train_loader:
                n += 1
                optim.zero_grad()
                # 从batch中获取输入数据input_ids、attention_mask和标签labels，并将它们转移到指定的计算设备上（CPU或GPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # 将输入数据传递给模型进行处理，得到输出结果outputs
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                # 使用loss.backward()方法计算模型参数的梯度
                loss.backward()
                # 使用optim.step()方法根据梯度更新模型参数
                optim.step()
                train_acc += accuracy_score(torch.argmax(outputs.logits.cpu().data, dim=1), labels.cpu())
                train_loss += loss.cpu()

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)
            # no_grad()上下文管理器，它可以在评估模型时禁用梯度计算
            with torch.no_grad():
                for batch in val_loader:
                    m += 1

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                    val_loss = outputs.loss
                    val_acc += accuracy_score(torch.argmax(outputs.logits.cpu().data, dim=1), labels.cpu())
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
        with tqdm(total=len(test_loader), desc='Predction') as Pbar:
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                # 将模型在测试集上的预测结果添加到test_pred列表中
                test_pred.extend(torch.argmax(outputs.logits.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("../result/bert_native.csv", index=False, quoting=3)
    logging.info('result saved!')
