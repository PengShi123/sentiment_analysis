import os
import sys
import logging
import time

import pandas as pd
import torch.optim as optim

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train = pd.read_csv("../data/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("../data/testData.tsv", header=0,
                   delimiter="\t", quoting=3)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        # 定义encoding层
        self.encodings = encodings
        # 如果有标签则赋值
        if labels:
            self.labels = labels

    def __getitem__(self, idx):
        # 从encodings的结果中读取key和value，用tensor的方式存储key和val
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            # 将标签转为tensor，存入item的标签中
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # 返回标签的长度
        return len(self.labels)


class TestDataset(torch.utils.data.Dataset):
    # 初始化encodings和样本数量
    def __init__(self, encodings, num_samples=0):
        self.encodings = encodings
        self.num_samples = num_samples

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    # 加载日子
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    # 创建三个列表
    train_texts, train_labels, test_texts = [], [], []
    for i, review in enumerate(train["review"]):
        # 将review存到train_text中
        train_texts.append(review)
        # 将标签存到train_labels中
        train_labels.append(train['sentiment'][i])

    for review in test['review']:
        # 存取测试集的review
        test_texts.append(review)
    # 8：2的比例分割总的数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # 进行分词和编码，长度大于最大长度时截断，统一编码长度
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    # 处理数据
    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset = TestDataset(test_encodings, num_samples=len(test_texts))
    # 选择使用gpu还是cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 调用模型
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()
    # 数据载入根据批次大小来，每个批次都打乱
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
    # AdamW优化参数
    optim = optim.AdamW(model.parameters(), lr=5e-5)
    # 开始迭代
    for epoch in range(3):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0

        with tqdm(total=len(train_loader), desc="Epoch %d" % epoch) as pbar:
            for batch in train_loader:
                n += 1
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optim.step()
                train_acc += accuracy_score(torch.argmax(outputs.logits.cpu().data, dim=1), labels.cpu())
                train_loss += loss.cpu()

                pbar.set_postfix({'epoch': '%d' % (epoch),
                                  'train loss': '%.4f' % (
                                              train_loss.data / n),
                                  'train acc': '%.2f' % (train_acc / n)
                                  })
                pbar.update(1)

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
        with tqdm(total=len(test_loader), desc='Predction') as pbar:
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                test_pred.extend(torch.argmax(outputs.logits.cpu().data, dim=1).numpy().tolist())
                pbar.update(1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("../result/distilbert_native.csv", index=False, quoting=3)
    logging.info('result saved!')
