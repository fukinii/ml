import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.decision_tree import DecisionTree
from src.utils.utils import get_key

data_pd = pd.read_csv('Shanghai_HMT_2010_cut.csv')
data_pd = data_pd.dropna()
data_pd.info()
data_pd = data_pd.drop(['No', 'year', 'month', 'day'], axis=1)
# data_pd = X
# X.info()

pres_median = data_pd['PRES'].median()
data_pd['PRES'] = (data_pd['PRES'] > pres_median).astype('int64')
X = data_pd.drop(['PRES'], axis=1)
y = data_pd['PRES']

X.to_csv('X.csv')

''' Разделяем выборку на тестовую и обучающую '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_numpy = X_test.to_numpy()
y_numpy = y_test.to_numpy()

my_ds = DecisionTree(max_depth=7, min_node_size=5)
# my_ds.set_categorical_data_from_dataframe(X)

y_test_numpy = y_numpy.reshape((1, len(y_numpy))).transpose().astype(int)
dataset_test = np.concatenate((X_numpy, y_test_numpy), axis=1)

root_train_pd = my_ds.build_tree_through_df(X_train, y_train)

index = np.array([6, 14])

res_pd = 0
for row in dataset_test:

    prediction_pandas = my_ds.predict(root_train_pd, row)

    if row[-1] == prediction_pandas:
        res_pd += 1

print(len(dataset_test), res_pd, res_pd / len(dataset_test))

my_ds.draw(root_train_pd, X.columns, 0)
