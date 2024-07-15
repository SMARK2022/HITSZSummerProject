import json
import re
import sys
import threading

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout,
                             QLabel, QListWidget, QMainWindow, QPushButton,
                             QVBoxLayout, QWidget)
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("推荐系统")
        self.setGeometry(100, 100, 800, 600)

        # 加载数据
        self.load_data()

        # 初始化UI
        self.initUI()

        # 设置推荐变量
        self.recommendation_started = False

    def load_data(self):
        # 读取 SQL 文件
        with open('./data/meal_list.sql', 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # 提取插入语句中的数据
        insert_statements = re.findall(
            r"INSERT INTO `meal_list` VALUES \((.*?)\);", sql_content, re.S
        )

        # 解析数据
        meal_data = []
        for statement in insert_statements:
            rows = statement.split("),(")
            for row in rows:
                row_data = row.split(',')
                meal_data.append([int(row_data[0]), row_data[1].strip("'"), row_data[2].strip("'")])

        # 创建 DataFrame
        self.meal_df = pd.DataFrame(meal_data, columns=['meal_no', 'MealID', 'meal_name'])

        # 读取 JSON 文件
        with open('./data/MealRatings_201705_201706.json', 'r', encoding='utf-8') as f:
            ratings_data = json.load(f)

        # 创建 DataFrame
        self.ratings_df = pd.DataFrame(ratings_data)

        # 处理重复评分记录
        self.ratings_df.sort_values(
            by=['UserID', 'MealID', 'ReviewTime'], ascending=[True, True, False], inplace=True
        )
        self.ratings_df.drop_duplicates(subset=['UserID', 'MealID'], keep='first', inplace=True)

        # 编码用户和菜品
        self.user_id_map = {
            user_id: idx for idx, user_id in enumerate(self.ratings_df['UserID'].unique())
        }
        self.meal_id_map = {
            meal_id: idx for idx, meal_id in enumerate(self.ratings_df['MealID'].unique())
        }

        self.ratings_df['user_code'] = self.ratings_df['UserID'].map(self.user_id_map)
        self.ratings_df['meal_code'] = self.ratings_df['MealID'].map(self.meal_id_map)

    def initUI(self):
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 顶部用户选择部分
        user_selection_layout = QHBoxLayout()
        self.user_combo = QComboBox()
        self.user_combo.setEditable(True)
        self.user_combo.addItems(self.ratings_df['UserID'].unique().tolist())
        self.user_button = QPushButton("确定")
        self.user_button.clicked.connect(self.load_user_data)

        user_selection_layout.addWidget(QLabel("选择用户:"))
        user_selection_layout.addWidget(self.user_combo)
        user_selection_layout.addWidget(self.user_button)

        main_layout.addLayout(user_selection_layout, 1)

        # 中间部分布局
        middle_layout = QHBoxLayout()

        # 左侧已选菜品部分
        selected_meals_group = QGroupBox("已选菜品")
        selected_meals_layout = QVBoxLayout(selected_meals_group)

        self.meal_combo = QComboBox()
        self.meal_combo.addItems(self.meal_df['meal_name'].tolist())
        self.meal_button = QPushButton("添加")
        self.meal_button.clicked.connect(self.add_meal)
        self.selected_meals_list = QListWidget()
        self.selected_meals_list.itemDoubleClicked.connect(self.remove_meal)

        meal_selection_layout = QHBoxLayout()
        meal_selection_layout.addWidget(self.meal_combo)
        meal_selection_layout.addWidget(self.meal_button)

        selected_meals_layout.addLayout(meal_selection_layout)
        selected_meals_layout.addWidget(self.selected_meals_list)

        middle_layout.addWidget(selected_meals_group, 2)

        # 右侧推荐部分
        recommendation_layout = QVBoxLayout()

        self.recommend_button = QPushButton("开始推荐")
        self.recommend_button.clicked.connect(self.start_recommendation)
        recommendation_layout.addWidget(self.recommend_button)

        self.user_based_group = self.create_recommendation_group("基于用户的协同过滤")
        self.item_based_group = self.create_recommendation_group("基于物品的协同过滤")
        self.als_group = self.create_recommendation_group("ALS推荐算法")

        recommendation_layout.addWidget(self.user_based_group)
        recommendation_layout.addWidget(self.item_based_group)
        recommendation_layout.addWidget(self.als_group)

        middle_layout.addLayout(recommendation_layout, 6)

        main_layout.addLayout(middle_layout, 9)

        # 初始化已选菜品集合
        self.selected_meals_set = set()

    def create_recommendation_group(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)

        for i in range(4):
            btn = QPushButton(f"推荐菜品 {i+1}")
            btn.clicked.connect(lambda checked, b=btn: self.order_meal(b.text()))
            layout.addWidget(btn)

        return group

    def order_meal(self, meal):
        if meal not in self.selected_meals_set:
            self.selected_meals_set.add(meal)
            self.selected_meals_list.addItem(meal)
            print(f"添加菜品: {meal}")
            self.check_recommendation_status()

    def load_user_data(self):
        user_id = self.user_combo.currentText()
        if user_id in self.user_id_map:
            user_code = self.user_id_map[user_id]
            user_meals = self.ratings_df[self.ratings_df['user_code'] == user_code]['MealID']
            meal_names = self.meal_df.set_index('MealID').loc[user_meals]['meal_name'].tolist()

            self.selected_meals_list.clear()
            self.selected_meals_set = set(meal_names)
            self.selected_meals_list.addItems(self.selected_meals_set)

            print(f"加载用户数据: {user_id}")
        else:
            self.selected_meals_list.clear()
            self.selected_meals_set.clear()
            print(f"新用户: {user_id}")

    def add_meal(self):
        meal = self.meal_combo.currentText()
        if meal not in self.selected_meals_set:
            self.selected_meals_set.add(meal)
            self.selected_meals_list.addItem(meal)
            print(f"添加菜品: {meal}")
            self.check_recommendation_status()

    def remove_meal(self, item):
        meal = item.text()
        if meal in self.selected_meals_set:
            self.selected_meals_set.remove(meal)
            self.selected_meals_list.takeItem(self.selected_meals_list.row(item))
            print(f"删除菜品: {meal}")
            self.check_recommendation_status()

    def check_recommendation_status(self):
        print("CHANGE")
        if self.recommendation_started:
            # 启动推荐线程
            threading.Thread(target=self.run_recommendations).start()

    def start_recommendation(self):

        # 更新推荐状态
        self.recommendation_started = True
        self.recommend_button.setEnabled(False)

        # 启动推荐线程
        threading.Thread(target=self.run_recommendations).start()

    def run_recommendations(self):

        # 获取当前用户所选择的菜品记录
        selected_meals = list(self.selected_meals_set)
        new_user_code = len(self.user_id_map)  # 为新用户创建新的 user_code
        self.ratings_df_new = self.ratings_df.copy()

        # 为新的 user_code 添加评分记录，假设评分为 5
        for meal in selected_meals:
            meal_id = self.meal_df[self.meal_df['meal_name'] == meal]['MealID'].values[0]
            meal_code = self.meal_id_map[meal_id]
            self.ratings_df_new = self.ratings_df_new._append(
                {
                    'UserID': f'new_user_{new_user_code}',
                    'MealID': meal_id,
                    'Rating': 5,
                    'ReviewTime': pd.Timestamp.now(),
                    'user_code': new_user_code,
                    'meal_code': meal_code,
                },
                ignore_index=True,
            )
        # 创建用户-物品矩阵
        user_item_matrix = self.ratings_df_new.pivot_table(
            index='user_code', columns='meal_code', values='Rating'
        )

        # 计算用户相似度和物品相似度矩阵
        user_similarity = cosine_similarity(user_item_matrix.fillna(0))
        item_similarity = cosine_similarity(user_item_matrix.fillna(0).T)
        print(user_item_matrix.iloc[-1].max())

        def compute_rmse(R, U, Vt):
            R_pred = np.dot(U, Vt)
            mse = np.sum((R - R_pred) ** 2) / np.count_nonzero(R)
            rmse = np.sqrt(mse)
            return rmse

        def als_train(R, k=10, max_iter=10, tol=0.001):
            num_users, num_items = R.shape
            U = np.random.rand(num_users, k)
            Vt = np.random.rand(k, num_items)
            R_demeaned = R - np.mean(R, axis=1).reshape(-1, 1)

            for i in range(max_iter):
                # Fix Vt and solve for U
                for u in range(num_users):
                    U[u, :] = np.linalg.solve(np.dot(Vt, Vt.T), np.dot(Vt, R_demeaned[u, :].T)).T

                # Fix U and solve for Vt
                for v in range(num_items):
                    Vt[:, v] = np.linalg.solve(np.dot(U.T, U), np.dot(U.T, R_demeaned[:, v]))

                rmse = compute_rmse(R_demeaned, U, Vt)
                # print(f"Iteration {i+1}/{max_iter}, RMSE: {rmse}")

                if rmse < tol:
                    break

            return U, Vt

        # ALS矩阵分解
        R = user_item_matrix.fillna(0).values
        U, Vt = als_train(R, k=8, max_iter=4, tol=0.001)
        predicted_ratings = np.dot(U, Vt) + np.mean(R, axis=1).reshape(-1, 1)
        predicted_ratings_df = pd.DataFrame(
            predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index
        )

        # 获取某个用户的评分
        def get_user_ratings(user_id):
            return user_item_matrix.loc[user_id].dropna()

        # 基于用户的协同过滤推荐
        def user_based_recommendation(user_id, top_k=4):
            if user_id not in user_item_matrix.index:
                return pd.Series()

            user_sim_scores = pd.Series(
                user_similarity[user_item_matrix.index.get_loc(user_id)], index=user_item_matrix.index
            )
            user_sim_scores = user_sim_scores.drop(user_id, errors='ignore').sort_values(ascending=False)
            similar_users = user_sim_scores.index[:2*top_k]

            recommendations = pd.Series(dtype=np.float64)
            for similar_user in similar_users:
                similar_user_ratings = get_user_ratings(similar_user)
                recommendations = pd.concat([recommendations, similar_user_ratings])

            recommendations = recommendations.groupby(recommendations.index).sum()
            user_rated_items = get_user_ratings(user_id).index
            recommendations = recommendations.drop(user_rated_items, errors='ignore')

            return recommendations.sort_values(ascending=False).head(top_k)

        # 基于物品的协同过滤推荐
        def item_based_recommendation(user_id, top_k=3):
            if user_id not in user_item_matrix.index:
                return pd.Series()

            user_ratings = get_user_ratings(user_id)
            recommendations = pd.Series(dtype=np.float64)

            for item, rating in user_ratings.items():
                if item not in user_item_matrix.columns:
                    continue

                item_sim_scores = pd.Series(
                    item_similarity[user_item_matrix.columns.get_loc(item)], index=user_item_matrix.columns
                )
                similar_items = item_sim_scores.sort_values(ascending=False).index[: 2 * top_k]

                for similar_item in similar_items:
                    if similar_item in user_ratings.index:
                        continue
                    recommendations.at[similar_item] = (
                        recommendations.get(similar_item, 0) + item_sim_scores[similar_item] * rating
                    )

            recommendations = recommendations.groupby(recommendations.index).sum()
            return recommendations.sort_values(ascending=False).head(top_k)

        # ALS推荐算法
        def als_user_recommendation(user_id, top_k=4):
            if user_id not in predicted_ratings_df.index:
                return pd.Series()

            user_ratings = predicted_ratings_df.loc[user_id].sort_values(ascending=False)
            user_rated_items = get_user_ratings(user_id).index
            recommendations = user_ratings.drop(user_rated_items, errors='ignore')
            return recommendations.head(top_k)

        # 三个推荐算法并行运行
        user_based_result = user_based_recommendation(new_user_code, top_k=4)
        item_based_result = item_based_recommendation(new_user_code, top_k=4)
        als_result = als_user_recommendation(new_user_code, top_k=4)
        print(user_based_result, item_based_result, als_result)

        # 更新UI
        self.update_recommendation_group(self.user_based_group, user_based_result)
        self.update_recommendation_group(self.item_based_group, item_based_result)
        self.update_recommendation_group(self.als_group, als_result)

        # 定义每个推荐算法的函数
        def run_user_based_recommendation():
            user_based_result =  user_based_recommendation(new_user_code, top_k=4)
            self.update_recommendation_group(self.user_based_group, user_based_result)

        def run_item_based_recommendation():
            item_based_result = item_based_recommendation(new_user_code, top_k=4)
            self.update_recommendation_group(self.item_based_group, item_based_result)

        def run_als_recommendation():
            als_result= als_user_recommendation(new_user_code, top_k=4)
            self.update_recommendation_group(self.als_group, als_result)

        # 创建线程来并行运行推荐算法
        threads = []

        for func in [
            run_user_based_recommendation,
            run_item_based_recommendation,
            run_als_recommendation,
        ]:
            thread = threading.Thread(target=lambda f: threads.append(f()), args=(func,))
            thread.start()

        for thread in threads:
            thread.join()

    def update_recommendation_group(self, group, recommendations):
        for i, meal_code in enumerate(recommendations.index):
            meal_name = self.meal_df[
                self.meal_df['MealID'] == list(self.meal_id_map.keys())[meal_code]
            ]['meal_name'].values[0]
            group.layout().itemAt(i).widget().setText(meal_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecommendationApp()
    window.show()
    sys.exit(app.exec_())
