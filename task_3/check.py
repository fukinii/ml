import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.decision_tree import DecisionTree
from src.utils.utils import get_key

# def get_key(d, value):
#     for k, v in d.items():
#         if v == value:
#             return k


data_pd = pd.read_csv('Shanghai_HMT_2010_cut.csv')
data_pd = data_pd.dropna()

pres_median = data_pd['PRES'].median()
data_pd['PRES'] = (data_pd['PRES'] > pres_median).astype('int64')
X = data_pd.drop(['PRES'], axis=1)
y = data_pd['PRES']

X.to_csv('X.csv')

# categorical_features = X.columns[X.dtypes == object].tolist()
# # print(type(X.columns))
# a = np.arange(len(X.columns))
# categorical_feature_index = []
# for feature in categorical_features:
#     categorical_feature_index.append(a[feature == X.columns][0])
#
# print("categorical_features: ", categorical_features)
# print("categorical_feature_index: ", categorical_feature_index)
# print()
#
# cat_feature_value_list = []
#
# for feature in categorical_features:
#     cat_feature_value_list.append(set(X[feature]))
#
# print("cat_feature_value_list: ", cat_feature_value_list)
# print()
# list_of_dicts = []
#
# for i, feature in enumerate(categorical_features):
#     dict = {}
#     for ind_val, val in enumerate(cat_feature_value_list[i]):
#         dict[val] = ind_val + 1
#
#     list_of_dicts.append(dict)
#
# print("list_of_dicts: ", list_of_dicts)

''' Разделяем выборку на тестовую и обучающую '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_numpy = X.to_numpy()
y_numpy = y.to_numpy()

y_test_numpy = y_numpy.reshape((1, len(y_numpy))).transpose().astype(int)
dataset_test = np.concatenate((X_numpy, y_test_numpy), axis=1)

X_m = dataset_test

# list_of_dict_of_norms = []
# sorted_norms = []
# list_of_dict_cat_to_int = []
# list_of_dict_int_to_cat = []
#
# for id_feature, cat_feature_dict in enumerate(cat_feature_value_list):
#
#     dict_of_c_sizes = {key: 0 for key in cat_feature_dict}
#     dict_of_c_first_class_sizes = {key: 0 for key in cat_feature_dict}
#     dict_of_norms = {}
#     dict_cat_to_int = {}
#     dict_int_to_cat = {}
#
#     for row in X_m:
#         row_c = row[categorical_feature_index[id_feature]]
#         dict_of_c_sizes[row_c] += 1
#
#         if row[-1] == 1:
#             dict_of_c_first_class_sizes[row_c] += 1
#
#     for key in dict_of_c_sizes:
#         dict_of_norms[key] = dict_of_c_first_class_sizes[key] / dict_of_c_sizes[key]
#
#     list_of_dict_of_norms.append(dict_of_norms)
#
#     a = []
#     for key in dict_of_norms:
#         a.append(dict_of_norms[key])
#     a.sort()
#     sorted_norms.append(a)
#     print("dict_of_norms: ", dict_of_norms)
#     i = 0
#     for norm in a:
#         key = get_key(dict_of_norms, norm)
#         del dict_of_norms[key]
#         print(key, i)
#         dict_cat_to_int[key] = i
#         dict_int_to_cat[i] = key
#         i += 1
#         print(dict_cat_to_int)
#         print(dict_int_to_cat)
#     list_of_dict_cat_to_int.append(dict_cat_to_int)
#     list_of_dict_int_to_cat.append(dict_int_to_cat)
#
#     print(dict_of_c_sizes)
#     print(dict_of_c_first_class_sizes)
#     print()
#
# print(list_of_dict_of_norms)
# print(sorted_norms)
# print(list_of_dict_cat_to_int)
# print(list_of_dict_int_to_cat)
# print()

my_ds = DecisionTree(max_depth=5, min_node_size=5)

my_ds.set_categorical_data_from_dataframe(X)
list_of_dict_cat_to_int, list_of_dict_int_to_cat = my_ds.create_dict_for_cat_features(X_m)
print(list_of_dict_cat_to_int)
print(list_of_dict_int_to_cat)