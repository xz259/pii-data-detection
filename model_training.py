import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                          DataCollatorForTokenClassification)
import evaluate
from datasets import Dataset
from functools import partial
from seqeval.metrics import recall_score, precision_score, f1_score

class CFG:
    max_length = 1024

df = pd.read_json('/data/train.json')

label_list = ['O', 'B-NAME_STUDENT', 'I-NAME_STUDENT', 'B-EMAIL', 'I-EMAIL',
              'B-USERNAME', 'I-USERNAME', 'B-ID_NUM', 'I-ID_NUM', 'B-PHONE_NUM',
              'I-PHONE_NUM', 'B-URL_PERSONAL', 'I-URL_PERSONAL', 'B-STREET_ADDRESS',
              'I-STREET_ADDRESS']

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

df['mapped_labels'] = df['labels'].apply(lambda labels: [label2id[label] for label in labels])

train_df, valid_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=1278)

def create_dataset(df):
    return Dataset.from_dict({col: df[col].tolist() for col in df})

train_ds = create_dataset(train_df)
valid_ds = create_dataset(valid_df)

tokenizer = AutoTokenizer.from_pretrained('deberta-v3-base')

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    for word_id in word_ids:
        label = labels[word_id] if word_id is not None else -100
        if label % 2 == 1: label += 1  
        new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=CFG.max_length)
    tokenized_inputs["labels"] = [align_labels_with_tokens(labels, tokenized_inputs.word_ids(i)) 
                                  for i, labels in enumerate(examples["mapped_labels"])]
    return tokenized_inputs

tokenized_train = train_ds.map(tokenize_and_align_labels, batched=True)
tokenized_valid = valid_ds.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=16)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for p, l in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for p, l in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    
    return {'recall': recall, 'precision': precision, 'f1': f1}

model = AutoModelForTokenClassification.from_pretrained(
    'deberta-v3-base', num_labels=len(label_list), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

training_args = TrainingArguments(
    fp16=True, 
    learning_rate=2e-5, 
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    num_train_epochs=4, 
    weight_decay=0.01, 
    evaluation_strategy='epoch', 
    save_strategy='epoch',
    load_best_model_at_end=True, 
    report_to='none'
    )

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_train, 
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer, 
    data_collator=data_collator, 
    compute_metrics=partial(compute_metrics)
    )

trainer.train()
trainer.save_model("/model_checkpoints/deberta_v3_base_1024")
tokenizer.save_pretrained("/model_checkpoints/deberta_v3_base_1024")