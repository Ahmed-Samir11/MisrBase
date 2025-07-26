import os
import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
import joblib
from datasets import Dataset

# Config
MODEL_DIR = 'baseline_model'
MAX_LEN = 256

# Load model, tokenizer, label encoder
print('Loading model, tokenizer, and label encoder...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

# Load test data
test = pd.read_csv('test.csv')

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

test = add_features(test)

# is_correct feature (copied logic from train)
# You must have the correct.csv or logic from training for this to work robustly in production.
# For now, we set is_correct to 0 (unknown) for all, or you can load/match from train if available.
test['is_correct'] = 0

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

test['text'] = test.apply(format_input, axis=1)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=MAX_LEN)

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)
ds_test.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Predict
trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

# Top 3 predictions
top3 = np.argsort(-probs, axis=1)[:, :3]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
print('Saved submission.csv') 