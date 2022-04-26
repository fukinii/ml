from catboost import Pool, CatBoostClassifier
import numpy as np
from emnist import extract_training_samples, list_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support

list_of_datasets = list_datasets()
print(list_of_datasets)

images, labels = extract_training_samples('letters')

print(images.shape)

# images = images[:100000]
# labels = labels[:100000]

images = np.reshape(images, (images.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

cat_features = list(range(10))
train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features=cat_features)

eval_dataset = Pool(data=X_test,
                    label=y_test,
                    cat_features=cat_features)

model = CatBoostClassifier(iterations=5000,
                           learning_rate=1e-1,
                           depth=3)
                           # loss_function='MultiClass')

# Fit model
model.fit(train_dataset)
# Get predicted classes
preds_class = model.predict(eval_dataset)

acc = accuracy_score(preds_class, y_test)
f1_micro = f1_score(y_test, preds_class, average='micro')
f1_macro = f1_score(y_test, preds_class, average='macro')
y_test_pool = Pool(preds_class, y_test)

# target_names = ['number 0', 'number 1', 'number 2', 'number 3', 'number 4', 'number 5', 'number 6', 'number 7',
#                 'number 8', 'number 9']
print(classification_report(y_test, preds_class))
print("++++++++++++++++++++++++++++++++++++++++++++")


a = 1
