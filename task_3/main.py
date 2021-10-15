import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Shanghai_HMT_2010.csv')

# data.info()
data = data.dropna()
data = data.drop(['cbwd'], axis=1)
# data = (data - data.mean()) / data.std()

pres_median = data['PRES'].median()
# print(pres_median)

data['PRES'] = (data['PRES'] > pres_median).astype('int64')

# print(data.head(10))

from src.decision_tree import DecisionTree

X = data.drop(['PRES'], axis=1)
y = data['PRES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_numpy = X_train.to_numpy()
y_train_numpy = y_train.to_numpy()

print(X_train)
print(y_train)

print(X_train_numpy)
print(y_train_numpy)

y_train_numpy = y_train_numpy.reshape((1, len(y_train_numpy))).transpose().astype(int)

print(y_train_numpy)

dataset = np.concatenate((X_train_numpy, y_train_numpy), axis=1)

print(dataset)


decision_tree = DecisionTree(5, 1)


root = decision_tree.build_tree(dataset)

for row in dataset:
    prediction = decision_tree.predict(root, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))