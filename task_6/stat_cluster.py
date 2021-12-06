import numpy as np
import matplotlib.pyplot as plt
import random

description = np.loadtxt("docword.enron_cut.txt", max_rows=3, dtype=int)

doc_number = description[0]
word_dict_number = description[1]
nonzero_counts = description[2]

data = np.loadtxt("docword.enron_cut.txt", skiprows=3, dtype=int)
vocab = np.loadtxt("vocab.enron.txt", dtype=str)

num_of_centroids = 3
color_list = {}

for i in range(num_of_centroids):
    color_list[i] = np.array([np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])

markers_list = ['x', 'v', '^', 'o', ',', 4, 5, 6, 7, 8, 9, 's', 'p', 'h']

a = np.zeros((data.shape[1], 2))

# print(a)

borders = np.zeros((data.shape[1], 2))
for i in range(data.shape[1]):
    borders[i, 0] = np.min(data[:, i])
    borders[i, 1] = np.max(data[:, i])


centroids_coords = np.zeros((num_of_centroids, data.shape[1]))

for i in range(num_of_centroids):
    curr_centroid_coord = np.zeros((data.shape[1]))
    for j in range(data.shape[1]):
        curr_centroid_coord[j] = random.uniform(borders[j, 0], borders[j, 1])
    centroids_coords[i] = curr_centroid_coord

# k1 = np.min(enron[:, 0])
# k2 = np.max(enron[:, 0])
#
# l1 = np.min(enron[:, 1])
# l2 = np.max(enron[:, 1])
#
# m1 = np.min(enron[:, 2])
# m2 = np.max(enron[:, 2])

print(centroids_coords)
