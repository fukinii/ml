import numpy as np
from emnist import extract_training_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

images, labels = extract_training_samples('mnist')

images = images[:10000]
labels = labels[:10000]

images = np.reshape(images, (images.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

print(images.shape)
print(labels.shape)

rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1e-1, max_depth=2, random_state=0)

# rfc.fit(X_train, y_train)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)

a = 1

# eq = np.count_nonzero(y_pred == y_test)
#
# acc = eq / X_test.shape[0]
acc = accuracy_score(y_pred, y_test)
print(acc)
