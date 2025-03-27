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
import seaborn as sns

def analyze_tree_topics_text(dataset, fathers, start, limit):
    # 自定义停用词列表（英文）
    custom_stop_words = [
        'belt', 'road', 'china', 'development','silk', 'new','chinese','initiative','xi'
    ]

    combined_stop_words = set(ENGLISH_STOP_WORDS).union(set(custom_stop_words))

    # 找到每个节点的根节点
    roots = np.zeros(limit, dtype=int)
    for i in range(limit):
        if fathers[i] == -2:
            roots[i] = -2
            continue
        node = i
        while fathers[node] != -1:
            node = fathers[node]
        roots[i] = node

    unique_roots = np.unique(roots)
    tree_topics = {}

    for root in unique_roots:
        if root == -2:
            continue
        # 找到属于该树的所有节点
        tree_nodes = [i for i in range(limit) if roots[i] == root]
        tree_texts = list(chain.from_iterable(dataset.iloc[start + i, 5] for i in tree_nodes))
        
        # 使用TfidfVectorizer将文本转换为TF-IDF矩阵
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(combined_stop_words))
        X = vectorizer.fit_transform(tree_texts)
        
        # 计算共现词频矩阵
        co_occurrence_matrix = (X.T * X)
        np.fill_diagonal(co_occurrence_matrix.toarray(), 0)
        
        # 获取主题词
        feature_names = vectorizer.get_feature_names_out()
        topic_words = [feature_names[i] for i in np.argsort(co_occurrence_matrix.sum(axis=0)).tolist()[0][-10:]]
        
        tree_topics[root] = topic_words

    return tree_topics

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
def calculate_temporal_weights_kernel(timestamps, temporal_weights, limit, alpha, beta, t_max, t_min):
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

def normalize(value, min_value, max_value):
    if max_value - min_value == 0:
        return 0
    return (value - min_value) / (max_value - min_value)

def calculate_tree_statistics(dataset, fathers, start, limit):
    tree_statistics = {}
    node_statistics = {i: [[], [], [], [], []] for i in range(limit)}

    for i in range(limit):
        if fathers[i] == -2:
            continue
        node = i
        sentiment_polarity = dataset.iloc[start + i]['sentiment_polarity']
        likes = dataset.iloc[start + i]['likes']
        retweets = dataset.iloc[start + i]['retweets']
        replies = dataset.iloc[start + i]['replies']
        credibility = dataset.iloc[start + i]['credibility']

        while node != -1:
            if sentiment_polarity != 0:  # sentiment_polarity=0为缺失值
                node_statistics[node][0].append(sentiment_polarity)
            node_statistics[node][1].append(likes)
            node_statistics[node][2].append(retweets)
            node_statistics[node][3].append(replies)
            node_statistics[node][4].append(credibility)
            node = fathers[node]

    for node, stats in node_statistics.items():
        if not stats:  # 如果该节点没有统计数据，跳过
            continue
        sentiment_polarity_mean = np.mean(stats[0])
        sentiment_polarity_std = np.std(stats[0])
        
        # 归一化
        likes_mean = np.mean(stats[1])
        retweets_mean = np.mean(stats[2])
        replies_mean = np.mean(stats[3])
        max_value = max(likes_mean, retweets_mean, replies_mean)
        min_value = min(likes_mean, retweets_mean, replies_mean)
        likes_normalized = [normalize(x, min_value, max_value) for x in stats[1]]
        retweets_normalized = [normalize(x, min_value, max_value) for x in stats[2]]
        replies_normalized = [normalize(x, min_value, max_value) for x in stats[3]]
        
        likes_retweets_replies_mean = (np.mean(likes_normalized) + np.mean(retweets_normalized) + np.mean(replies_normalized)) / 3
        likes_retweets_replies_std = np.std(likes_normalized + retweets_normalized + replies_normalized)
        credibility_mean = np.mean(stats[4])
        credibility_std = np.std(stats[4])

        tree_statistics[node] = {
            'sentiment_polarity_mean': sentiment_polarity_mean,
            'sentiment_polarity_std': sentiment_polarity_std,
            'likes_retweets_replies_mean': likes_retweets_replies_mean,
            'likes_retweets_replies_std': likes_retweets_replies_std,
            'credibility_mean': credibility_mean,
            'credibility_std': credibility_std
        }

    return tree_statistics

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
    calculate_temporal_weights_kernel[blockspergrid, threadsperblock](d_timestamps, d_temporal_weights, limit, alpha, beta, t_max, t_min)
    find_fathers_kernel[blockspergrid, threadsperblock](d_semantic_similarities, d_temporal_weights, d_fathers, limit, p)
    fathers = d_fathers.copy_to_host()
    semantic_similarities = d_semantic_similarities.copy_to_host()
    temporal_weights = d_temporal_weights.copy_to_host()
    fathers = find_roots_and_mark_deletion(fathers, node_num)
    similarities = semantic_similarities
    tci = temporal_coherence_index(similarities, fathers, timestamps)
    cluster_effect_value = cluster_effect(embeddings, fathers)
    count, scaling_index = tree_shape_index(fathers)
    
    # 计算树的统计数据
    tree_statistics = calculate_tree_statistics(dataset, fathers, start, limit)
    
    return tci, cluster_effect_value, scaling_index, count, fathers, tree_statistics

def plot_tree_and_histogram(fathers, root, tree_statistics, output_dir, tree_count, stat_key):
    sns.set_theme(style="darkgrid")
    dot = Digraph(comment=f'Tree {tree_count} {stat_key}')
    diffs = []
    tree = [root]
    dot.node(str(root), f'{root}\n0', style='filled', fillcolor='white')

    for i, father in enumerate(fathers):
        if father in tree:
            tree.append(i)
            if i in tree_statistics and father in tree_statistics:
                tmp = tree_statistics[i][stat_key] - tree_statistics[father][stat_key]
                if tmp != 0:
                    diffs.append(tmp)
                dot.node(str(i), f'{i}\n{tmp:.5f}', style='filled', fillcolor='white')
                dot.edge(str(father), str(i))

    dot.render(os.path.join(output_dir, f'tree_{tree_count}_{stat_key}.gv'), view=False)

    plt.figure()
    sns.histplot(diffs, bins=np.arange(-1, 1, 0.1), edgecolor='black')
    plt.title(f'{stat_key} Diff Distribution for Tree {tree_count}')
    plt.xlabel('Diff')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'tree_{tree_count}_{stat_key}_diff_histogram.png'))
    plt.close()

def save_trees_to_file(fathers, alpha, beta, p, Lambda, dataset, start, limit, tree_statistics, output_dir='tree_output'):
    # 创建以参数命名的文件夹
    param_dir = os.path.join(output_dir, f'alpha_{alpha}_beta_{beta}_p_{p}_Lambda_{Lambda}')
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    
    tree_count = 0
    tree_topics = analyze_tree_topics_text(dataset, fathers, start, limit)
    tree_filename = os.path.join(param_dir, 'trees.txt')
    topics_filename = os.path.join(param_dir, 'topics.txt')
    stats_filename = os.path.join(param_dir, 'statistics.csv')
    
    with open(tree_filename, 'w', encoding='utf-8') as tree_file, open(topics_filename, 'w', encoding='utf-8') as topics_file, open(stats_filename, 'w', encoding='utf-8') as stats_file:
        stats_file.write("Tree,Sentiment Polarity Mean,Sentiment Polarity Std,Likes/Retweets/Replies Mean,Likes/Retweets/Replies Std,Credibility Mean,Credibility Std\n")
        for i, father in enumerate(fathers):
            if father == -1:
                tree_count += 1
                tree_file.write(f"\n\nTree {tree_count}:\n")
                tree_file.write(f"Root: {i}\n")
                nodes_to_visit = [i]
                dot = Digraph(comment=f'Tree {tree_count}')
                dot.node(str(i), str(i))
                dot_text = Digraph(comment=f'Tree {tree_count} Text')
                node_text = ', '.join(dataset.iloc[start + i]['text_normalized']) + f" (Timestamp: {dataset.iloc[start + i]['timestamp']})"
                dot_text.node(str(i), node_text)
                timestamps = [dataset.iloc[start + i]['timestamp']]
                while nodes_to_visit:
                    current_node = nodes_to_visit.pop(0)
                    for j, fth in enumerate(fathers):
                        if fth == current_node:
                            tree_file.write(f"Parent {fth} -> Node {j}\n")
                            nodes_to_visit.append(j)
                            dot.node(str(j), str(j))
                            dot.edge(str(fth), str(j))
                            node_text = ', '.join(dataset.iloc[start + j]['text_normalized']) + f" (Timestamp: {dataset.iloc[start + j]['timestamp']})"
                            dot_text.node(str(j), node_text)
                            dot_text.edge(str(fth), str(j))
                            timestamps.append(dataset.iloc[start + j]['timestamp'])
                dot.render(os.path.join(param_dir, f'tree_{tree_count}.gv'), view=False)
                dot_text.render(os.path.join(param_dir, f'tree_{tree_count}_text.gv'), view=False)
                # 输出树的主题词和时间信息
                if i in tree_topics:
                    min_timestamp = min(timestamps)
                    max_timestamp = max(timestamps)
                    timestamps_in_seconds = [ts.timestamp() for ts in timestamps]
                    avg_timestamp_in_seconds = np.mean(timestamps_in_seconds)
                    avg_timestamp = pd.to_datetime(avg_timestamp_in_seconds, unit='s')
                    topics_file.write(f"Tree {tree_count} Topics: {', '.join(tree_topics[i])}\n")
                    topics_file.write(f"Tree {tree_count} : Time Range: {min_timestamp} - {max_timestamp}, Average Time: {avg_timestamp}\n")
                
                # 输出根节点的三个参数的均值和标准差
                if i in tree_statistics:
                    stats = tree_statistics[i]
                    topics_file.write(f"Tree {tree_count} Statistics:\n")
                    topics_file.write(f"  Sentiment Polarity Mean: {stats['sentiment_polarity_mean']}\n")
                    topics_file.write(f"  Sentiment Polarity Std: {stats['sentiment_polarity_std']}\n")
                    topics_file.write(f"  Likes/Retweets/Replies Mean: {stats['likes_retweets_replies_mean']}\n")
                    topics_file.write(f"  Likes/Retweets/Replies Std: {stats['likes_retweets_replies_std']}\n")
                    topics_file.write(f"  Credibility Mean: {stats['credibility_mean']}\n")
                    topics_file.write(f"  Credibility Std: {stats['credibility_std']}\n\n\n\n")
                    stats_file.write(f"{tree_count},{stats['sentiment_polarity_mean']},{stats['sentiment_polarity_std']},{stats['likes_retweets_replies_mean']},{stats['likes_retweets_replies_std']},{stats['credibility_mean']},{stats['credibility_std']}\n")

                # 绘制树状图和统计图
                plot_tree_and_histogram(fathers,i, tree_statistics, param_dir, tree_count, 'sentiment_polarity_mean')
                plot_tree_and_histogram(fathers,i, tree_statistics, param_dir, tree_count, 'credibility_mean')
                plot_tree_and_histogram(fathers,i, tree_statistics, param_dir, tree_count, 'likes_retweets_replies_mean')

def plot_statistics(output_dir='tree_output'):

    sns.set_theme(style="darkgrid")
    stats_files = [os.path.join(root, file) for root, _, files in os.walk(output_dir) for file in files if file == 'statistics.csv']
    all_stats = pd.concat([pd.read_csv(file) for file in stats_files])

    # 计算变异系数
    all_stats['Likes/Retweets/Replies CV'] = all_stats['Likes/Retweets/Replies Std'] / all_stats['Likes/Retweets/Replies Mean']

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Tree Statistics Comparison')

    sns.barplot(x=all_stats.groupby('Tree')['Sentiment Polarity Mean'].mean().index, y=all_stats.groupby('Tree')['Sentiment Polarity Mean'].mean().values, ax=axes[0, 0])
    axes[0, 0].set_title('Sentiment Polarity Mean')

    sns.barplot(x=all_stats.groupby('Tree')['Sentiment Polarity Std'].mean().index, y=all_stats.groupby('Tree')['Sentiment Polarity Std'].mean().values, ax=axes[0, 1])
    axes[0, 1].set_title('Sentiment Polarity Std')

    sns.barplot(x=all_stats.groupby('Tree')['Likes/Retweets/Replies Mean'].mean().index, y=all_stats.groupby('Tree')['Likes/Retweets/Replies Mean'].mean().values, ax=axes[1, 0])
    axes[1, 0].set_title('Likes/Retweets/Replies Mean')

    sns.barplot(x=all_stats.groupby('Tree')['Likes/Retweets/Replies Std'].mean().index, y=all_stats.groupby('Tree')['Likes/Retweets/Replies Std'].mean().values, ax=axes[1, 1])
    axes[1, 1].set_title('Likes/Retweets/Replies Std')

    sns.barplot(x=all_stats.groupby('Tree')['Credibility Mean'].mean().index, y=all_stats.groupby('Tree')['Credibility Mean'].mean().values, ax=axes[2, 0])
    axes[2, 0].set_title('Credibility Mean')

    sns.barplot(x=all_stats.groupby('Tree')['Credibility Std'].mean().index, y=all_stats.groupby('Tree')['Credibility Std'].mean().values, ax=axes[2, 1])
    axes[2, 1].set_title('Credibility Std')

    sns.barplot(x=all_stats.groupby('Tree')['Likes/Retweets/Replies CV'].mean().index, y=all_stats.groupby('Tree')['Likes/Retweets/Replies CV'].mean().values, ax=axes[3, 0])
    axes[3, 0].set_title('Likes/Retweets/Replies CV')

    for ax in axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=4.0, w_pad=4.0, h_pad=6.0)
    plt.savefig(os.path.join(output_dir, 'tree_statistics_comparison.png'))

# 主函数
if __name__ == '__main__':
    dataset = load_dataset()
    alpha_values = [0.7]
    beta_values = [0.7]
    p_values = [0.8]
    node_num_values = [30]
    Lambda_values = [0.5] # 控制tci的权重
    best_overall_index = -float('inf')
    best_params = None
    t = time.now()
    results = []

    # 清空 tree_output 文件夹
    output_dir = 'tree_output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    lambda_best_params = {}

    for alpha in alpha_values:
        for beta in beta_values:
            for p in p_values:
                for node_num in node_num_values:
                    tci, cluster_effect_value, scaling_index, count, fathers, tree_statistics = generate_topic_tree(dataset, node_num, p, 60000, 20000, alpha, beta)
                    results.append((tci, cluster_effect_value, scaling_index, alpha, beta, p, node_num, count, fathers))

    for Lambda in Lambda_values:
        best_lambda_index = -float('inf')
        best_lambda_params = None
        for result in results:
            tci, cluster_effect_value, scaling_index, alpha, beta, p, node_num, count, fathers = result
            overall_index = Lambda * tci + (1 - Lambda) * scaling_index
            num_deleted_nodes = sum(1 for father in fathers if father == -2)
            print(f"Lambda={Lambda}, a={alpha}, b={beta}, p={p}, node_num={node_num}, count={count}, OverallIndex={round(overall_index, 5)}, tci={round(tci, 5)}, scaling_index={round(scaling_index, 5)}, DeletedNodes={num_deleted_nodes}")
            save_trees_to_file(fathers, alpha, beta, p, Lambda, dataset, 60000, 20000, tree_statistics, output_dir)
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

    # 绘制统计图像
    plot_statistics(output_dir)