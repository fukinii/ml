import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Shanghai_HMT_2010_cut.csv')
data = data.dropna()

data = data.drop(['cbwd'], axis=1)
# data = data.drop(['No'], axis=1)
# print(data.head(10))
data.info()
pres_median = data['PRES'].median()

# print(pres_median)

data['PRES'] = (data['PRES'] > pres_median).astype('int64')

# print(data.head(10))

from src.decision_tree import DecisionTree

X = data.drop(['PRES'], axis=1)
y = data['PRES']

''' Разделяем выборку на тестовую и обучающую '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

''' Создаем объект решабщего дерева и обучаем его'''
decision_tree_pd = DecisionTree(max_depth=5, min_node_size=10)
root_train_pd = decision_tree_pd.build_tree_through_df(X_train, y_train)

''' Создаем данные для проверки решений'''

X_test_numpy = X_test.to_numpy()
y_test_numpy = y_test.to_numpy()

y_test_numpy = y_test_numpy.reshape((1, len(y_test_numpy))).transpose().astype(int)
dataset_test = np.concatenate((X_test_numpy, y_test_numpy), axis=1)

'''========================================================================'''

res_pd = 0

for row in dataset_test:

    prediction_pandas = decision_tree_pd.predict(root_train_pd, row)

    if row[-1] == prediction_pandas:
        res_pd += 1

print(len(dataset_test), res_pd)

print(res_pd / len(dataset_test))

# print(data.columns)
# print(data.columns[0])
# print(type(data.columns))
decision_tree_pd.draw(root_train_pd, data.columns, 0)
