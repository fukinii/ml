import numpy as np
from sklearn import preprocessing
from emnist import extract_training_samples, list_datasets
from my_utils import pad_and_normalize

import pickle

list_of_datasets = list_datasets()
print(list_of_datasets)

images, labels = extract_training_samples('letters')

HEIGHT = images.shape[1]
WIDTH = images.shape[2]

print(images[0].shape)
print(images[0])

images32x32 = pad_and_normalize(images_array=images)

with open('letters.pickle', 'wb') as f:
    pickle.dump([images32x32, labels], f)

# images32x32 = np.zeros((images.shape[0], images.shape[1] + 4, images.shape[2] + 4), dtype=np.float64)
# for i, imgage_i in enumerate(images):
#     print(f'i = {i} из {images.shape[0]}')
#     images32x32[i] = np.pad(preprocessing.normalize(images[i], norm='max'), pad_width=2, mode='constant',
#                             constant_values=0.)

a = 1
