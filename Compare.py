import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据
df = pd.read_csv('/Users/likfei_0624/Desktop/Project I/reviews data.csv', encoding='ISO-8859-1')
df = df[['reviews.text', 'reviews.rating']].dropna()
df['reviews.rating'] = df['reviews.rating'].astype(int)
train_texts, val_texts, train_labels, val_labels = train_test_split(df['reviews.text'], df['reviews.rating'], test_size=0.1, random_state=42)

# 初始化分词器
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 编码数据
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

# 数据集类
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

# 创建验证集
val_dataset = HotelReviewsDataset(val_encodings, val_labels.tolist())

# 加载原始模型
original_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=df['reviews.rating'].nunique())

# 加载微调后的模型
finetuned_model = DistilBertForSequenceClassification.from_pretrained('/Users/likfei_0624/Desktop/Project I/results/checkpoint-5500', num_labels=df['reviews.rating'].nunique())

# 评估设置
eval_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=64,
)

# 评估原始模型
original_trainer = Trainer(
    model=original_model,
    args=eval_args,
    eval_dataset=val_dataset
)

original_eval_results = original_trainer.evaluate()
print("Original model evaluation results:", original_eval_results)

# 评估微调后的模型
finetuned_trainer = Trainer(
    model=finetuned_model,
    args=eval_args,
    eval_dataset=val_dataset
)

finetuned_eval_results = finetuned_trainer.evaluate()
print("Finetuned model evaluation results:", finetuned_eval_results)