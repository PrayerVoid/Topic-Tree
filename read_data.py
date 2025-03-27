import pandas as pd
import ast
import numpy as np
def load_dataset():
    dataset = pd.read_csv('模型调试/last_version_data.csv')
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], utc=True).dt.tz_convert(None)
    dataset['text_normalized'] = dataset['text_normalized'].fillna('').apply(lambda x: ast.literal_eval(x) if x else [])
    dataset['embeddings'] = dataset['embeddings'].apply(lambda x: np.array(str(x).strip('[  ]').split(), dtype=float))
    dataset.info()
    return dataset
