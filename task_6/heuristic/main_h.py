import pickle

import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib import ticker

from utils import update, calc_matrix_dist, sparse_matrix, find_clusters, build_task, calc_matrix_dist_parall

# import matplotlib.animation
#
# data_file = "../docword.kos.txt"
# description = np.loadtxt(data_file, max_rows=3, dtype=int)
#
# doc_number = description[0]
# word_dict_number = description[1]
# nonzero_counts = description[2]
#
# data = np.loadtxt(data_file, skiprows=3, dtype=int)
# vocab = np.loadtxt("../vocab.kos.txt", dtype=str)
#
# print("Данные прочитаны")
# point = build_task(data, word_dict_number, doc_number)

mu1_x = 0
mu1_y = 1

mu2_x = 1
mu2_y = 1

mu3_x = 1
mu3_y = 0

sigma = 0.15

l1 = -1
l2 = 2

m1 = -1
m2 = 2

N = 100
random_points_group_1 = np.array([np.array([random.gauss(mu1_x, sigma), random.gauss(mu1_y, sigma)]) for i in range(N)])
random_points_group_2 = np.array([np.array([random.gauss(mu2_x, sigma), random.gauss(mu2_y, sigma)]) for i in range(N)])
random_points_group_3 = np.array([np.array([random.gauss(mu3_x, sigma), random.gauss(mu3_y, sigma)]) for i in range(N)])

point = np.zeros((N * 3, 2))

point[0:N, :] = random_points_group_1
point[N: 2 * N, :] = random_points_group_2
point[2 * N: 3 * N, :] = random_points_group_3
num_of_centroids = 3

markers_list = ['x', 'v', '^', 'o', ',', 4, 5, 6, 7, 8, 9, 's', 'p', 'h']

matrix_dist = calc_matrix_dist(point)
print("Посчитана матрица расстояний")

# with open('matrix_dist_kos.pickle', 'rb') as f:
#     matrix_dist = pickle.load(f)

# with open('matrix_dist_kos.pickle', 'wb') as f:
#     pickle.dump(matrix_dist, f)


dist_list = np.reshape(matrix_dist, point.shape[0] * point.shape[0])

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
ax.set_xlim((0., 2))
ax.set_ylim((0., 4e3))
plt.hist(dist_list, bins=300)
plt.grid()
plt.savefig("Гистограмма_custom.png")
plt.show()


#
matrix = sparse_matrix(matrix_dist, 0.5)
print("Матрица расстояний разрежена")


clusters = find_clusters(matrix)
print("Найдены кластеры")


fig_1 = plt.figure(figsize=(16, 10))
ax = fig_1.add_subplot(111)
ax.set(facecolor='black')
fig_1.set(facecolor='black')
ax.set_xlim(l1, l2)
ax.set_ylim(m1, m2)

calc_num_of_centroids = len(clusters)
print(calc_num_of_centroids)
color_list = {}
for i in range(calc_num_of_centroids):
    color_list[i] = np.array([np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])

for i in range(calc_num_of_centroids):
    ax.scatter(point[clusters[i], 0], point[clusters[i], 1], c=color_list[i])

plt.savefig("Кластеризация_custom.png")
plt.show()

