import numpy as np
import matplotlib.pyplot as plt
import random
from stat_cluster_utils import calc_centroids
import pickle

data_file = "docword.enron.txt"

description = np.loadtxt(data_file, max_rows=3, dtype=int)

doc_number = description[0]
word_dict_number = description[1]
nonzero_counts = description[2]

data = np.loadtxt(data_file, skiprows=3, dtype=int)
vocab = np.loadtxt("vocab.enron.txt", dtype=str)

print("Данные прочитаны")

num_of_centroids = 5
color_list = {}

for i in range(num_of_centroids):
    color_list[i] = np.array([np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])

centroids_coords, match_numbers = calc_centroids(data, num_of_centroids=num_of_centroids)
print("Построены центроиды")

out = [centroids_coords, match_numbers]

with open('Centroids.pickle', 'wb') as f:
    pickle.dump(out, f)

# with open('Centroids.pickle', 'rb') as f:
#     out = pickle.load(f)
# centroids_coords = out[0]
# match_numbers = out[1]

fig_1 = plt.figure(figsize=(16, 10))
ax = fig_1.add_subplot(111)
ax.set(facecolor='black')
fig_1.set(facecolor='black')
# ax.set_xlim(l1, l2)
# ax.set_ylim(m1, m2)

for i in range(num_of_centroids):
    ax.scatter(data[:, 0][match_numbers == i], data[:, 1][match_numbers == i],
               c=color_list[i])

    ax.scatter(centroids_coords[i, 0], centroids_coords[i, 1], c=color_list[i], marker='*', s=200)

plt.show()

print(centroids_coords)
