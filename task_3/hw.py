import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from src.decision_tree import DecisionTree

data_pd = pd.read_csv('data/data_cut.csv')
data_pd = data_pd.dropna()
data_pd.info()

a = set()
for index, row in data_pd.iterrows():
    for item in row:
        if item == " ?":
            a.add(index)

data_pd.drop(list(a), axis=0, inplace=True)
# data_pd.to_csv('my_data_pd_old.csv')
# list_to_drop = ['Education', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native-country', 'Age']
# data_pd.drop(list_to_drop, axis=1, inplace=True)
data_pd.info()
# #
# df = my_data_pd[~my_data_pd[list(cols)].eq('?').all(axis = 1)]
# df.to_csv('df.csv')
# df = my_data_pd.drop(my_data_pd[my_data_pd.score < 50].index)
# my_data_pd
# my_data_pd.info()

data_pd['Money'] = (data_pd['Money'] == " <=50K").astype('int64')

# data_pd.to_csv('my_data_pd_old.csv')

# y.to_csv('y.csv')

X = data_pd.drop(['Money'], axis=1)
y = data_pd['Money']

''' Разделяем выборку на тестовую и обучающую '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_numpy = X_test.to_numpy()
y_numpy = y_test.to_numpy()

y_test_numpy = y_numpy.reshape((1, len(y_numpy))).transpose().astype(int)
dataset_test = np.concatenate((X_numpy, y_test_numpy), axis=1)

# my_ds = DecisionTree(max_depth=5, min_node_size=20)
# root_train_pd = my_ds.build_tree_through_df(X_train, y_train)
#
# with open('data/tree_pickle_3131_all_5_5.pickle', 'wb') as f:
#     pickle.dump(root_train_pd, f)
#
# with open('data/my_ds_5_5.pickle', 'wb') as f:
#     pickle.dump(my_ds, f)

with open('data/tree_pickle_3131_all_5_5_1640.pickle', 'rb') as f:
    root_train_pd = pickle.load(f)

with open('data/my_ds_5_5_1640.pickle', 'rb') as f:
    my_ds = pickle.load(f)

res_pd = 0
for row in dataset_test:

    prediction_pandas = my_ds.predict(root_train_pd, row)

    if row[-1] == prediction_pandas:
        res_pd += 1
    print("my_pred: ", prediction_pandas, "; true_val: ", row[-1])

print(len(dataset_test), res_pd, res_pd / len(dataset_test))

my_ds.draw(root_train_pd, X.columns, 0)
