import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class CFG:
    max_length = 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/microsoft/deberta-v3-base'  

test = pd.read_json('/data/test.json')

def create_dataset(df):
    return Dataset.from_dict({col: df[col].tolist() for col in df})

test_ds = create_dataset(test)

tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
model = AutoModelForTokenClassification.from_pretrained(CFG.model_path).to(CFG.device)

def process_document(document, tokens, tokenizer, model):
    tokenized = tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=CFG.max_length, return_tensors="pt")
    input_ids = tokenized['input_ids'].to(CFG.device)
    
    with torch.no_grad():
        logits = model(input_ids).logits
    predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    
    pred_labels = [model.config.id2label[pred] for pred in predictions.flatten()]
    original_tokens = [tokens[wi] for wi in tokenized.word_ids() if wi is not None]
    
    return pd.DataFrame({
        'document': [document] * len(original_tokens),
        'token_number': list(range(len(original_tokens))),
        'label': pred_labels,
        'original_tokens': original_tokens
    }).query("label != 'O'") 

predictions = pd.concat([process_document(test_ds[i]['document'], test_ds[i]['tokens'], tokenizer, model) for i in range(len(test_ds))], ignore_index=True)