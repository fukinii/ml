from catboost import Pool, CatBoostClassifier
import numpy as np
from emnist import extract_training_samples, list_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support

list_of_datasets = list_datasets()
print(list_of_datasets)

images, labels = extract_training_samples('letters')

print(images.shape)

# images = images[:10000]
# labels = labels[:10000]

images = np.reshape(images, (images.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

# cat_features = list(range(10))
train_dataset = Pool(data=X_train,
                     label=y_train)

eval_dataset = Pool(data=X_test,
                    label=y_test)

ref_iter = 100
ref_learning_rate = 0.1
ref_depth = 6

model = CatBoostClassifier(iterations=2,
                           learning_rate=ref_learning_rate,
                           depth=ref_depth)

# Fit model
model.fit(train_dataset)
# Get predicted classes
preds_class = model.predict(eval_dataset)

# print(a)

acc = accuracy_score(preds_class, y_test)
f1_micro = f1_score(y_test, preds_class, average='micro')
f1_macro = f1_score(y_test, preds_class, average='macro')
y_test_pool = Pool(preds_class, y_test)

# target_names = ['number 0', 'number 1', 'number 2', 'number 3', 'number 4', 'number 5', 'number 6', 'number 7',
#                 'number 8', 'number 9']
print(classification_report(y_test, preds_class))
print("++++++++++++++++++++++++++++++++++++++++++++")


a = 1
