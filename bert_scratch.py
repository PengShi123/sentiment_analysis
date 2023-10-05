import os
import sys
import logging
import datasets
import evaluate
import torch.nn as nn

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split

train = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/labeledTrainData.tsv", header=0, delimiter="\t",
                    quoting=3)
test = pd.read_csv("D:/pycharm project/pythonProject1/sentiment/data/testData.tsv", header=0, delimiter="\t",
                   quoting=3)
BERT_PATH = "D:/pycharm project/pythonProject1/sentiment/bert_native/bert-base-uncased"


class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 从config中获得标签数量
        self.num_labels = config.num_labels
        # 将config存储在当前对象的config变量中，以便后续使用
        self.config = config
        # 创建一个BertModel对象，用于提取文本特征
        self.bert = BertModel(config)
        # 根据config中的参数确定分类器的dropout概率，并创建一个Dropout层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性分类器，输入维度为config.hidden_size，输出维度为类别数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 调用post_init方法，进行额外的初始化操作（在父类中定义）
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # 从outputs中获取池化后的输出pooled_output（即[CLS]标记对应的输出）
        pooled_output = outputs[1]
        # 对pooled_output进行dropout操作
        pooled_output = self.dropout(pooled_output)
        # 将经过dropout的pooled_output传递给分类器进行处理，得到输出logits（即每个类别的预测分数）
        logits = self.classifier(pooled_output)
        # 创建一个交叉熵损失函数对象loss_fct
        loss_fct = nn.CrossEntropyLoss()
        # 使用loss_fct计算预测分数logits和真实标签labels之间的损失值loss
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None
            # 创建一个SequenceClassifierOutput对象，将损失值、预测分数、隐藏状态和注意力权重等信息打包在一起返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    # 划分训练集和验证集
    train, val = train_test_split(train, test_size=.2)
    # 创建三个字典
    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}
    # 将准备好的数据字典转换为 Hugging Face 提供的 datasets.Dataset 对象，这样可以更方便地进行数据的处理和加载
    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH, model_max_length=64)


    def preprocess_function(examples):
        # 使用 tokenizer 对 examples 中的 'text' 字段进行编码，truncation=True 表示如果文本长度超过了模型的最大输入长度，则进行截断处理
        return tokenizer(examples['text'], truncation=True)


    # 使用 map 方法对 train_dataset、val_dataset 和 test_dataset 中的数据进行预处理，并将处理后的数据分别保存在 tokenized_train、
    # tokenized_val 和 tokenized_test 中
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    # 创建一个 DataCollatorWithPadding 对象，用于在训练时将多个数据样本合并成一个批次，并进行必要的填充处理
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = BertScratch.from_pretrained(BERT_PATH)
    # 使用 datasets 库中的 load_metric 方法加载 accuracy 指标，用于在训练和验证时评估模型的性能
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        # 从 eval_pred 中获取模型的预测分数 logits 和真实标签 labels
        logits, labels = eval_pred
        # 使用 np.argmax 方法获取 logits 中最大值的索引，作为模型的预测结果
        predictions = np.argmax(logits, axis=-1)
        # 使用 metric 对象的 compute 方法计算模型的评估指标，并返回结果
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # 模型和训练过程中的输出将保存在 ./checkpoint 目录下
        num_train_epochs=3,  # 总共训练 3 个 epoch
        per_device_train_batch_size=6,  # 每个设备上的训练批次大小为 6
        per_device_eval_batch_size=12,  # 每个设备上的评估批次大小为 12
        warmup_steps=500,  # 学习率预热步数为 500
        weight_decay=0.01,  # 权重衰减系数为 0.01
        logging_dir='./logs',  # 训练日志将保存在 ./logs 目录下
        logging_steps=100,  # 每训练 100 步就记录一次日志
        save_strategy="no",  # 不保存模型权重
        evaluation_strategy="epoch"  # 每个 epoch 结束时进行评估
    )

    trainer = Trainer(
        model=model,  # 要训练的模型
        args=training_args,  # 训练参数
        train_dataset=tokenized_train,  # 训练数据集
        eval_dataset=tokenized_val,  # 验证数据集
        tokenizer=tokenizer,  # 用于编码和解码文本的 tokenizer
        data_collator=data_collator,  # 用于合并和填充数据的 data collator
        compute_metrics=compute_metrics  # 用于计算评估指标的函数
    )

    trainer.train()
    # 使用训练好的模型对测试数据集进行预测
    prediction_outputs = trainer.predict(tokenized_test)
    # 获取预测结果中概率最大的类别作为预测结果
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()  # flatten降维函数
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("../result/bert_scratch.csv", index=False, quoting=3)
    logging.info('result saved!')
