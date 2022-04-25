from catboost import Pool, CatBoostClassifier
import numpy as np
from emnist import extract_training_samples
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

images, labels = extract_training_samples('mnist')

print(images.shape)

# images = images[:1000000]
# labels = labels[:1000000]

images = np.reshape(images, (images.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

cat_features = list(range(10))
train_dataset = Pool(data=X_train,
                     label=y_train,
                     cat_features=cat_features)

eval_dataset = Pool(data=X_test,
                    label=y_test,
                    cat_features=cat_features)

model = CatBoostClassifier(iterations=10,
                           learning_rate=1,
                           depth=2,
                           loss_function='MultiClass')

# Fit model
model.fit(train_dataset)
# Get predicted classes
preds_class = model.predict(eval_dataset)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_dataset)
# Get predicted RawFormulaVal
preds_raw = model.predict(eval_dataset,
                          prediction_type='RawFormulaVal')

acc = accuracy_score(preds_class, y_test)
print(acc)
print(model.get_best_score())

a = 1
