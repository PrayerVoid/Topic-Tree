import pandas as pd
import ast
import numpy as np
import re
def load_dataset():
    dataset = pd.read_csv('模型调试/processed_data_embeddings.csv')
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], utc=True).dt.tz_convert(None)
    dataset['text_normalized'] = dataset['text_normalized'].fillna('').apply(lambda x: ast.literal_eval(x) if x else [])
    dataset['embeddings'] = dataset['embeddings'].apply(lambda x: np.array(ast.literal_eval(re.sub(r'\s+', ',', str(x).replace('\n', '').strip('[[  ]]')))))
    dataset.info()
    return dataset

dataset=load_dataset()
dataset.to_csv('模型调试/last_version_data.csv',index=False)