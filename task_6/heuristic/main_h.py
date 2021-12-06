import numpy as np
import matplotlib.pyplot as plt
import random
from utils import update, calc_matrix_dist, thin_out_matrix, find_clusters
import matplotlib._color_data as mcd
import matplotlib.animation

# data_file = "../docword.enron.txt"
# description = np.loadtxt(data_file, max_rows=3, dtype=int)
#
# doc_number = description[0]
# word_dict_number = description[1]
# nonzero_counts = description[2]
#
# data = np.loadtxt(data_file, skiprows=3, dtype=int)
# vocab = np.loadtxt("vocab.enron.txt", dtype=str)
#
# print("Данные прочитаны")
#
# num_of_centroids = 5
# color_list = {}
#
# for i in range(num_of_centroids):
#     color_list[i] = np.array([np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])


mu1_x = 0
mu1_y = 1

mu2_x = 1
mu2_y = 1

mu3_x = 1
mu3_y = 0

sigma = 0.1

l1 = -1
l2 = 2

m1 = -1
m2 = 2

N = 100
random_points_group_1 = np.array([np.array([random.gauss(mu1_x, sigma), random.gauss(mu1_y, sigma)]) for i in range(N)])
random_points_group_2 = np.array([np.array([random.gauss(mu2_x, sigma), random.gauss(mu2_y, sigma)]) for i in range(N)])
random_points_group_3 = np.array([np.array([random.gauss(mu3_x, sigma), random.gauss(mu3_y, sigma)]) for i in range(N)])

# random_points_group_1 = np.array([np.array([random.uniform(l1, l2), random.uniform(m1, m2)]) for i in range(N)])
# random_points_group_2 = np.array([np.array([random.uniform(l1, l2), random.uniform(m1, m2)]) for i in range(N)])
# random_points_group_3 = np.array([np.array([random.uniform(l1, l2), random.uniform(m1, m2)]) for i in range(N)])


data = np.zeros((N * 3, 2))

data[0:N, :] = random_points_group_1
data[N: 2 * N, :] = random_points_group_2
data[2 * N: 3 * N, :] = random_points_group_3

num_of_centroids = 3

centroids_coords = np.array(
    [np.array([random.uniform(l1, l2), random.uniform(m1, m2)]) for i in range(num_of_centroids)])

match_numbers = np.zeros(3 * N, dtype=int)
min_distance = 0

color_list = {}

for i in range(num_of_centroids):
    color_list[i] = np.array([np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])

markers_list = ['x', 'v', '^', 'o', ',', 4, 5, 6, 7, 8, 9, 's', 'p', 'h']

fig_1 = plt.figure(figsize=(16, 10))
ax = fig_1.add_subplot(111)
# ax.set(facecolor='black')
# fig_1.set(facecolor='black')
# ax.set_xlim(l1, l2)
# ax.set_ylim(m1, m2)

print(data.shape)

ax.scatter(data[:, 0], data[:, 1])

plt.show()

matrix_dist = calc_matrix_dist(data)

dist_list = np.reshape(matrix_dist, data.shape[0] * data.shape[0])

plt.hist(dist_list)
plt.show()

matrix = thin_out_matrix(matrix_dist, 0.4)

clusters = find_clusters(matrix)

fig_1 = plt.figure(figsize=(16, 10))
ax = fig_1.add_subplot(111)
ax.set(facecolor='black')
fig_1.set(facecolor='black')
ax.set_xlim(l1, l2)
ax.set_ylim(m1, m2)

calc_num_of_centroids = len(clusters)

for i in range(calc_num_of_centroids):
    ax.scatter(data[clusters[i], 0], data[clusters[i], 1],
               c=color_list[i])

    # ax.scatter(centroids_coords[i, 0], centroids_coords[i, 1], c=color_list[i], marker='*', s=200)

plt.show()
