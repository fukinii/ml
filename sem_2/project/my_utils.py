import numpy as np
from sklearn import preprocessing
from emnist import extract_training_samples, list_datasets


def pad_and_normalize(images_array):
    images32x32 = np.zeros((images_array.shape[0], images_array.shape[1] + 4, images_array.shape[2] + 4),
                           dtype=np.float64)
    for i, image_i in enumerate(images_array):
        print(f'i = {i} из {images_array.shape[0]}')
        images32x32[i] = np.pad(preprocessing.normalize(image_i, norm='max'), pad_width=2, mode='constant',
                                constant_values=0.)

    return images32x32