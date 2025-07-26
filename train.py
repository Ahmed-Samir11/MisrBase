import os
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import joblib

# Custom Trainer to fix device placement for LoRA
class FixedTrainer(Trainer):
    def prepare_inputs(self, inputs):
        device = self.model.device
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

# Config
MODEL_NAME = 'jhu-clsp/ettin-encoder-1b'
EPOCHS = 5
MAX_LEN = 256
OUTPUT_DIR = 'baseline_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
train = pd.read_csv('train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ':' + train.Misconception

# Enhanced feature engineering
def add_features(df):
    df['explanation_len'] = df['StudentExplanation'].fillna('').apply(len)
    df['mc_frac_count'] = df['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'FRAC_\d+_\d+|\\frac', x))
    )
    df['number_count'] = df['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'\b\d+\b', x))
    )
    df['operator_count'] = df['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'[\+\-\*/=]', x))
    )
    df['mc_answer_len'] = df['MC_Answer'].fillna('').apply(len)
    df['question_len'] = df['QuestionText'].fillna('').apply(len)
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)
    return df

train = add_features(train)

# is_correct feature
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

# Label encoding
y = train['target']
le = LabelEncoder()
train['label'] = le.fit_transform(y)
n_classes = len(le.classes_)

# Prompt engineering
def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This answer is incorrect."
    extra = (
        f"Additional Info: The explanation has {row['explanation_len']} characters "
        f"and includes {row['mc_frac_count']} fraction(s)."
    )
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}\n"
        f"{extra}"
    )

train['text'] = train.apply(format_input, axis=1)

# Tokenizer and dataset
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=MAX_LEN)

# Split train/val
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)
COLS = ['text', 'label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

# Model loading and LoRA setup
print('Loading model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=n_classes,
    torch_dtype=torch_dtype,
).to(device)

# Use correct LoRA target modules for XLM-RoBERTa/Ettin
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# Training arguments
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    bf16=use_bf16,
    fp16=not use_bf16,
)

# MAP@3 metric
def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    match = (top3 == labels[:, None])
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}

# Trainer
trainer = FixedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_map3,
)

trainer.train()

# Save model and label encoder
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))
print('Training complete. Model and label encoder saved to', OUTPUT_DIR) 