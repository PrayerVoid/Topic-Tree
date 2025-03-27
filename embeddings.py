import torch
from transformers import AutoModel, AutoTokenizer
from torch.multiprocessing import Pool, set_start_method
import pandas as pd
import numpy as np
import ast
import re
import os
from tqdm import tqdm

# 使用GPU加速（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "Alibaba-NLP/gte-modernbert-base"
local_model_path = "./模型调试"  # 替换为本地模型的实际路径（如果有）

if os.path.exists(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModel.from_pretrained(local_model_path).to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

def embedding(model, tokenizer, input_texts):
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt', is_split_into_words=True)
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    return embeddings

def load_dataset():
    dataset = pd.read_csv('模型调试/processed_data.csv')
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], utc=True).dt.tz_convert(None)
    dataset['text_normalized'] = dataset['text_normalized'].fillna('').apply(lambda x: ast.literal_eval(x) if x else [])
    dataset.info()
    return dataset

def process_text(text):
            return embedding(model, tokenizer, text).cpu().detach().numpy()


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    dataset = load_dataset()
    embeddings = []
    #process为进程数，根据实际配置调整
    with Pool(processes=4) as pool:
        embeddings = list(tqdm(pool.imap(process_text, dataset['text_normalized']), total=len(dataset), desc="Embedding texts"))
    
    dataset['embeddings'] = embeddings
    dataset['embeddings'] = dataset['embeddings'].apply(lambda x: np.array(ast.literal_eval(re.sub(r'\s+', ',', str(x).replace('\n', '').strip('[[  ]]')))))
    dataset.to_csv('模型调试/processed_data_embeddings.csv', index=False)