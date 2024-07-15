import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, hp, tpe
from scipy.sparse.linalg import svds
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

# 加载数据
train_ratings = pd.read_csv('train_ratings.csv')
validation_ratings = pd.read_csv('validation_ratings.csv')
test_ratings = pd.read_csv('test_ratings.csv')


# 获取某个用户的评分
def get_user_ratings(user_id):
    return user_item_matrix.loc[user_id].dropna()


# 基于物品的协同过滤推荐
def item_based_recommendation(user_id, top_k=3, n_neigh=3):
    if user_id not in user_item_matrix.index:
        return pd.Series()

    user_ratings = get_user_ratings(user_id)
    recommendations = pd.Series(dtype=np.float64)

    for item, rating in user_ratings.items():
        if item not in user_item_matrix.columns:
            continue

        item_sim_scores = pd.Series(
            item_similarity[user_item_matrix.columns.get_loc(item)], index=user_item_matrix.columns
        ).sort_values(ascending=False)
        similar_items = item_sim_scores.index[:n_neigh]

        for idx, similar_item in enumerate(similar_items):
            if similar_item in user_ratings.index:
                continue
            recommendations.at[similar_item] = (
                recommendations.get(similar_item, 0) + item_sim_scores[similar_item] * rating
            )

    recommendations = recommendations.groupby(recommendations.index).sum()
    return recommendations.sort_values(ascending=False).head(top_k)


# 合并训练集和验证集
combined_train_validation_ratings = pd.concat([train_ratings, validation_ratings])

# 创建新的用户-物品矩阵
user_item_matrix = combined_train_validation_ratings.pivot_table(
    index='user_code', columns='meal_code', values='Rating'
)

# 重新计算用户相似度和物品相似度矩阵
user_similarity = cosine_similarity(user_item_matrix.fillna(0))
item_similarity = cosine_similarity(user_item_matrix.fillna(0).T)


# 定义目标函数
def objective(params):
    top_k = int(params['top_k'])
    n_neigh = int(params['n_neigh'])

    hits = 0
    recall_total = 0
    precision_total = 0

    for user_id in test_ratings['user_code'].unique():
        user_test_ratings = test_ratings[test_ratings['user_code'] == user_id]
        recommended_items = item_based_recommendation(user_id, top_k, n_neigh).index.tolist()

        for item_id in user_test_ratings['meal_code']:
            if item_id in recommended_items:
                hits += 1
        recall_total += len(user_test_ratings['meal_code'])
        precision_total += len(recommended_items)

    precision_value = hits / precision_total if precision_total > 0 else 0
    recall_value = hits / recall_total if recall_total > 0 else 0

    if precision_value + recall_value > 0:
        f1_value = 2 * precision_value * recall_value / (precision_value + recall_value)
    else:
        f1_value = 0

    print(top_k, n_neigh, (precision_value, recall_value, f1_value))

    return -f1_value  # Hyperopt最小化目标函数，所以我们返回负的F1值


# 定义搜索空间
space = {'top_k': hp.quniform('top_k', 1, 5, 1), 'n_neigh': hp.quniform('n_neigh', 1, 160, 1)}

# 优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best hyperparameters:", best)
