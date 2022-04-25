import numpy as np
from emnist import list_datasets
from emnist import extract_training_samples

images, labels = extract_training_samples('letters')

print(images.shape)
print(labels.shape)

array_1d = np.reshape(images, (images.shape[0], -1, 1))

print(array_1d.shape)
