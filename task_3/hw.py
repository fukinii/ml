import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree

data_pd = pd.read_csv('data/data_cut_300.csv')
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_numpy = X_test.to_numpy()
y_numpy = y_test.to_numpy()

y_test_numpy = y_numpy.reshape((1, len(y_numpy))).transpose().astype(int)
dataset_test = np.concatenate((X_numpy, y_test_numpy), axis=1)

# my_ds = DecisionTree(max_depth=5, min_node_size=20)
# root_train_pd = my_ds.fit(X_train, y_train)
#
# with open('data/my_ds_300_all_5_5_.pickle', 'wb') as f:
#     pickle.dump(my_ds, f)

# with open('data/tree_pickle_300_all_5_5_1727.pickle', 'rb') as f:
#     root_train_pd = pickle.load(f)
#
with open('data/my_ds_300_all_5_5_.pickle', 'rb') as f:
    my_ds = pickle.load(f)

res_pd = 0

_, accuracy = my_ds.predict(X_test, y_test)

# for row in dataset_test:
#
#     prediction_pandas = my_ds.single_predict(root_train_pd, row)
#
#     if row[-1] == prediction_pandas:
#         res_pd += 1
#     print("my_pred: ", prediction_pandas, "; true_val: ", row[-1])

print("accuracy = ", accuracy)

my_ds.draw(my_ds.root, X.columns, 0)

tree = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)

exp1_data_train = X_train
exp1_data_labels = y_train
tree.fit(exp1_data_train, exp1_data_labels)


