import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Shanghai_HMT_2010_cut.csv')
data = data.dropna()

data = data.drop(['cbwd'], axis=1)
print(data.head(10))
data.info()
pres_median = data['PRES'].median()

# print(pres_median)

data['PRES'] = (data['PRES'] > pres_median).astype('int64')

# print(data.head(10))

from src.decision_tree import DecisionTree

X = data.drop(['PRES'], axis=1)
y = data['PRES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# X_train_numpy = X_train.to_numpy()
# y_train_numpy = y_train.to_numpy()
#
# y_train_numpy = y_train_numpy.reshape((1, len(y_train_numpy))).transpose().astype(int)
#
# dataset_train = np.concatenate((X_train_numpy, y_train_numpy), axis=1)
#
# decision_tree = DecisionTree(5, 1)
#
# root = decision_tree.build_tree(dataset_train)
#
# res: int = 0
#
# for row in dataset_train:
#     prediction = decision_tree.predict(root, row)
#     if row[-1] == prediction:
#         res += 1
#     print('Expected=%d, Got=%d' % (row[-1], prediction))
#
# print(len(dataset_train), res)

X_test_numpy = X_test.to_numpy()
y_test_numpy = y_test.to_numpy()

y_test_numpy = y_test_numpy.reshape((1, len(y_test_numpy))).transpose().astype(int)
dataset_test = np.concatenate((X_test_numpy, y_test_numpy), axis=1)

decision_tree_test = DecisionTree(5, 1)
# decision_tree = DecisionTree(5, 1)
root_test = decision_tree_test.build_tree(dataset_test)

res_test = 0

for row in dataset_test:
    prediction = decision_tree_test.predict(root_test, row)
    if row[-1] == prediction:
        res_test += 1
    print('Expected=%d, Got=%d' % (row[-1], prediction))

print(len(dataset_test), res_test)

print(res_test / len(dataset_test))