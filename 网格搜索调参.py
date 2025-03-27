# 导入必要的库
import numpy as np
import math
from numba import cuda
from read_data import load_dataset
from sklearn.metrics import silhouette_score
from datetime import datetime as time
import pandas as pd
import os
from graphviz import Digraph
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def temporal_coherence_index(similarities, fathers, timestamps):
    tci = 0
    valid_timestamps = [timestamps[i] for i in range(len(fathers)) if fathers[i] != -2]
    t_total = max(valid_timestamps) - min(valid_timestamps)
    edge = 0
    for i in range(1, len(fathers)):
        if fathers[i] >= 0:
            time_diff = abs(timestamps[i] - timestamps[fathers[i]])
            decay = np.exp(-time_diff / t_total)
            tci += similarities[i][fathers[i]] * decay
            edge += 1
    return tci / edge

def tree_shape_index(fathers):
    count = 0
    node_sum={}
    
    for i in range(len(fathers)):
        node = i
        if fathers[node] == -2:
            continue
        if fathers[node] == -1:
            count += 1
            node_sum[node]=[1,1]#自身作为树根节点C和A都为1
            continue
        node_sum[node]=[1,1]#自身作为子树根节点C和A都为1
        plus=1
        while fathers[node] != -1:
            plus+=1
            node_sum[fathers[node]][0]+=1#父节点A加1
            node_sum[fathers[node]][1]+=plus#父节点C加i
            node = fathers[node]
    
    # 计算异速标度律指标
    scaling_exponents = []
    for i,ans in node_sum.items():
        if ans[0] == 1:
            continue#防止除0
        x = np.log(ans[1]) / np.log(ans[0])
        scaling_exponents.append(x)
    
    scaling_index = np.mean(scaling_exponents) if scaling_exponents else 0
    return count, 2-abs(1.5-scaling_index)#链状为2，铺展为1

def cluster_effect(embeddings, fathers):
    roots = np.zeros(len(fathers), dtype=int)
    emb = []
    for i in range(len(fathers)):
        if fathers[i] == -2:
            roots[i] = -2
            continue
        node = i
        emb.append(i)
        while fathers[node] != -1:
            node = fathers[node]
        roots[i] = node
    labels = roots[emb]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0
    score = silhouette_score(embeddings[emb, :], labels, metric='cosine')
    return score

@cuda.jit
def calculate_semantic_similarities_kernel(embeddings, similarities, limit):
    idx = cuda.grid(1)
    if idx < limit:
        for j in range(idx):
            semantic_sim = 0.0
            norm_1 = 0.0
            norm_2 = 0.0
            for k in range(embeddings.shape[1]):
                semantic_sim += embeddings[idx, k] * embeddings[j, k]
                norm_1 += embeddings[idx, k] * embeddings[idx, k]
                norm_2 += embeddings[j, k] * embeddings[j, k]
            if norm_1 > 0 and norm_2 > 0:
                similarities[idx, j] = semantic_sim / (math.sqrt(norm_1) * math.sqrt(norm_2))

@cuda.jit
def calculate_temporal_weights_kernel(temporal_weights, limit, alpha, beta):
    idx = cuda.grid(1)
    if idx < limit:
        for j in range(idx):
            time_diff = (idx - j) / limit
            temporal_weights[idx, j] = 1 - alpha * (beta * time_diff + (1 - beta) * (math.exp(time_diff) - 1) / math.e)

@cuda.jit
def find_fathers_kernel(semantic_similarities, temporal_weights, fathers, limit, p):
    idx = cuda.grid(1)
    if idx < limit:
        max_similarity = -1
        father = -1
        for j in range(idx):
            combined_sim_first = semantic_similarities[idx, j]
            combined_sim = semantic_similarities[idx, j] * temporal_weights[idx, j]
            if combined_sim_first > p and combined_sim > max_similarity:
                max_similarity = combined_sim
                father = j
        fathers[idx] = father

def find_roots_and_mark_deletion(fathers, node_num=3):
    n = len(fathers)
    roots = [-1] * n
    tree_sizes = [0] * n
    def find_root(node):
        if fathers[node] == -1:
            return node
        if roots[node] == -1:
            roots[node] = find_root(fathers[node])
        return roots[node]
    for i in range(n):
        roots[i] = find_root(i)
    for root in roots:
        tree_sizes[root] += 1
    for i in range(n):
        if tree_sizes[roots[i]] <= node_num:
            fathers[i] = -2
    return fathers



def generate_topic_tree(dataset, node_num, p, start, limit, alpha=0.5, beta=2):
    embeddings = np.array(dataset['embeddings'].tolist()[start:start+limit], dtype=np.float32)
    timestamps = np.array(dataset['timestamp'].astype(np.int64).tolist()[start:start+limit], dtype=np.float32) / 1e9
    semantic_similarities = np.zeros((limit, limit), dtype=np.float32)
    temporal_weights = np.zeros((limit, limit), dtype=np.float32)
    fathers = np.full(limit, -1, dtype=np.int32)
    t_max = timestamps.max()
    t_min = timestamps.min()
    threadsperblock = 256
    blockspergrid = (limit + threadsperblock - 1) // threadsperblock
    d_embeddings = cuda.to_device(embeddings)
    d_timestamps = cuda.to_device(timestamps)
    d_semantic_similarities = cuda.to_device(semantic_similarities)
    d_temporal_weights = cuda.to_device(temporal_weights)
    d_fathers = cuda.to_device(fathers)
    calculate_semantic_similarities_kernel[blockspergrid, threadsperblock](d_embeddings, d_semantic_similarities, limit)
    calculate_temporal_weights_kernel[blockspergrid, threadsperblock]( d_temporal_weights, limit, alpha, beta)
    find_fathers_kernel[blockspergrid, threadsperblock](d_semantic_similarities, d_temporal_weights, d_fathers, limit, p)
    fathers = d_fathers.copy_to_host()
    semantic_similarities = d_semantic_similarities.copy_to_host()
    temporal_weights = d_temporal_weights.copy_to_host()
    fathers = find_roots_and_mark_deletion(fathers, node_num)
    similarities = semantic_similarities
    tci = temporal_coherence_index(similarities, fathers, timestamps)
    cluster_effect_value = cluster_effect(embeddings, fathers)
    count, scaling_index = tree_shape_index(fathers)
    
    return tci, cluster_effect_value, scaling_index, count, fathers

# 主函数
if __name__ == '__main__':
    dataset = load_dataset()
    alpha_values = [0.3,0.5,0.7]
    beta_values = [0.3,0.5,0.7]
    p_values = [0.7,0.8,0.9,0.95]
    node_num_values = [30]
    Lambda=0.5 # 控制tci的权重
    best_overall_index = -float('inf')
    best_params = None
    t = time.now()
    results = []

    lambda_best_params = {}

    for alpha in alpha_values:
        for beta in beta_values:
            for p in p_values:
                for node_num in node_num_values:
                    tci, cluster_effect_value, scaling_index, count, fathers = generate_topic_tree(dataset, node_num, p, 60000, 20000, alpha, beta)
                    overall_index = Lambda * tci + (1 - Lambda) * scaling_index
                    num_deleted_nodes = sum(1 for father in fathers if father == -2)
                    print(f"a={alpha}, b={beta}, p={p}, node_num={node_num}, count={count}, OverallIndex={round(overall_index, 5)}, tci={round(tci, 5)}, scaling_index={round(scaling_index, 5)}, DeletedNodes={num_deleted_nodes}")
                    results.append((tci, cluster_effect_value, scaling_index, alpha, beta, p, node_num, count, fathers))

    best_lambda_index = -float('inf')
    best_lambda_params = None
    for result in results:
        tci, cluster_effect_value, scaling_index, alpha, beta, p, node_num, count, fathers = result
        overall_index = Lambda * tci + (1 - Lambda) * scaling_index
        num_deleted_nodes = sum(1 for father in fathers if father == -2)
        if overall_index > best_lambda_index:
            best_lambda_index = overall_index
            best_lambda_params = (Lambda, alpha, beta, p, node_num, count, fathers)
    lambda_best_params[Lambda] = best_lambda_params
    if best_lambda_index > best_overall_index:
        best_overall_index = best_lambda_index
        best_params = best_lambda_params

    print(f"耗时:{time.now() - t}")
    for Lambda, params in lambda_best_params.items():
        print(f"最佳参数组合 (Lambda={Lambda}): α={params[1]}, β={params[2]}, P={params[3]}, node_num={params[4]}, count={params[5]}")
    print(f"\n最佳综合评价指标: OverallIndex={round(best_overall_index, 5)}\n\n\n")