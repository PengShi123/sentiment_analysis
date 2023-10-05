import os
import sys
import logging
import datasets

import pandas as pd
import numpy as np

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 读取数据
train = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/labeledTrainData.tsv", header=0, delimiter="\t",
                    quoting=3)
test = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    # 程序的运行日志
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    # 按8：2的比例把总数据集分为训练集和验证集
    train, val = train_test_split(train, test_size=.2)
    # 生成由标签和review组成的字典
    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}
    # 处理数据
    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)
    # 预训练
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


    def preprocess_function(examples):
        # 切割样本，长度不能超过最大长度
        return tokenizer(examples['text'], truncation=True)


    # 对数据集中的每个数据都使用函数，预处理数据
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    # 拼接数据
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # 调用模型
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    # 精度
    metric = datasets.load_metric("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # 获得预测值
        predictions = np.argmax(logits, axis=-1)
        # 得出精度
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./results',  # 输出文件路径
        num_train_epochs=3,  # 迭代次数
        per_device_train_batch_size=4,  # 每个gpu计算的训练集的数据的批次大小
        per_device_eval_batch_size=8,  # # 每个设备上的评估批次大小为 12
        learning_rate=5e-6,  # 学习率
        warmup_steps=500,  # 学习率预热步数为 500
        weight_decay=0.01,  # 权重衰减系数为 0.01
        logging_dir='./logs',  # 日志文件存储位置
        logging_steps=100,  # 每100步记录一个日志
        save_strategy="no",  # 不保存模型权重
        evaluation_strategy="epoch"  # 每个 epoch 结束时进行评估
    )

    trainer = Trainer(
        model=model,
        args=training_args,  # 训练参数
        train_dataset=tokenized_train,  # 训练集数据
        eval_dataset=tokenized_val,  # 评估集数据
        tokenizer=tokenizer,  # 用于编码和解码文本的 tokenizer
        data_collator=data_collator,  # 用于合并和填充数据的 data collator
        compute_metrics=compute_metrics  # 用于计算评估指标的函数
    )

    trainer.train()
    # 测试集进行训练
    prediction_outputs = trainer.predict(tokenized_test)
    # 获得预测结果
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    # 输出测试集的结果
    print(test_pred)
    # 存储结果
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("../result/roberta_trainer.csv", index=False, quoting=3)
    logging.info('result saved!')
