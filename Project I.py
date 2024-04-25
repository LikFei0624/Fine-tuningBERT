import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('/Users/likfei_0624/Desktop/Project I/reviews data.csv', encoding='ISO-8859-1')

# 选择需要的列并预处理
df = df[['reviews.text', 'reviews.rating']].dropna()
df['reviews.rating'] = df['reviews.rating'].astype(int)

# 划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['reviews.text'], df['reviews.rating'], test_size=0.1, random_state=42
)

# 初始化DistilBERT分词器
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 编码训练和验证数据
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

# 数据集类定义
class HotelReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 创建数据集
train_dataset = HotelReviewsDataset(train_encodings, train_labels.tolist())
val_dataset = HotelReviewsDataset(val_encodings, val_labels.tolist())

# 加载预训练的DistilBERT模型，适用于序列分类任务
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=df['reviews.rating'].nunique())

# 训练参数设置
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=16,  # 每个设备的训练批量
    per_device_eval_batch_size=64,   # 每个设备的评估批量
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,                # 日志记录步数
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 开始训练
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(eval_results)