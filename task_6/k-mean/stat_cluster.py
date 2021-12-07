import numpy as np
import matplotlib.pyplot as plt
import random
from stat_cluster_utils import calc_centroids, build_task, calc_intra_cluster_distance, calc_extra_cluster_distance
import pickle

# data_file = "../docword.enron.txt"
data_file = "../docword.kos.txt"

description = np.loadtxt(data_file, max_rows=3, dtype=int)

doc_number = description[0]
word_dict_number = description[1]
nonzero_counts = description[2]

data = np.loadtxt(data_file, skiprows=3, dtype=int)
# vocab = np.loadtxt("../vocab.enron.txt", dtype=str)
vocab = np.loadtxt("../vocab.kos.txt", dtype=str)

with open('../heuristic/matrix_dist_kos.pickle', 'rb') as f:
    matrix_dist_kos = pickle.load(f)

point = build_task(data, word_dict_number, doc_number)

# res_array = np.zeros((20, 4))
# res_array = []
# for num_of_centroids in range(2, 20):
#     centroids_coords, match_numbers, is_broken = calc_centroids(point, num_of_centroids=num_of_centroids)
#
#     if not is_broken:
#         print("Построены центроиды для num_of_centroids = ", num_of_centroids)
#         out = [centroids_coords, match_numbers]
#         intra = calc_intra_cluster_distance(data=point, match_numbers=match_numbers, matrix_dist_kos=matrix_dist_kos)
#         extra = calc_extra_cluster_distance(data=point, match_numbers=match_numbers, matrix_dist_kos=matrix_dist_kos)
#         # res_array[num_of_centroids] = np.array([num_of_centroids, intra, extra, intra / extra])
#         res_array.append(np.array([num_of_centroids, intra, extra, intra / extra]))
#         # res_array[num_of_centroids][0] = intra
#         # res_array[num_of_centroids][0] = extra
#         # res_array[num_of_centroids][0] = intra / extra
#     # else:
#     #     print("Построение центроиды для num_of_centroids = ", num_of_centroids, " завершилось с ошибкой")
#     #     res_array[num_of_centroids] = np.array([-1, -1, -1, -1])
#
# with open('res_array_sparse.pickle', 'wb') as f:
#     pickle.dump(res_array, f)

with open('res_array_sparse.pickle', 'rb') as f:
    res_array = pickle.load(f)
#
# res_array_c = np.copy(res_array)
# indexes = []
# for i, res in enumerate(res_array):
#     if res[0] == -1:
#         indexes.append(i)

# res_array_c = np.delete(res_array_c, indexes)
fig_1 = plt.figure(figsize=(16, 10))
ax = fig_1.add_subplot(111)
ax.grid()
# plt.hist(res_array_c[:, 1], bins=20)
for i in range(len(res_array)):
    # ax.scatter(res_array[i][0], res_array[i][0], c='r', label=0)
    # ax.scatter(res_array[i][0], res_array[i][1], c='m', label=1)
    # ax.scatter(res_array[i][0], res_array[i][2], c='g', label=2)
    ax.scatter(res_array[i][0], res_array[i][3], c='b')
ax.legend()
plt.show()
# for i in range(res_array.shape[0]):
#     ax.scatter(data[:, 0][match_numbers == i], data[:, 1][match_numbers == i],
#                c=color_list[i])
#
#     ax.scatter(centroids_coords[i, 0], centroids_coords[i, 1], c=color_list[i], marker='*', s=200)

# plt.show()

# print("Данные прочитаны")
#
# num_of_centroids = 17
# color_list = {}
#
# for i in range(num_of_centroids):
#     color_list[i] = np.array([np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])
#

#

# print("Построены центроиды")
# centroids_coords, match_numbers = calc_centroids(point, num_of_centroids=num_of_centroids)
# out = [centroids_coords, match_numbers]

# with open('Centroids_kos.pickle', 'wb') as f:
#     pickle.dump(out, f)


# file_centroids_kos = 'Centroids5_kos.pickle'
# with open(file_centroids_kos, 'rb') as f:
#     out = pickle.load(f)
# centroids_coords = out[0]
# match_numbers = out[1]

# intra = calc_intra_cluster_distance(data=point, match_numbers=match_numbers, matrix_dist_kos=matrix_dist_kos)
# extra = calc_extra_cluster_distance(data=point, match_numbers=match_numbers, matrix_dist_kos=matrix_dist_kos)

# print(intra, extra)

# fig_1 = plt.figure(figsize=(16, 10))
# ax = fig_1.add_subplot(111)
# ax.set(facecolor='black')
# fig_1.set(facecolor='black')
# ax.set_xlim(l1, l2)
# ax.set_ylim(m1, m2)

# for i in range(num_of_centroids):
#     ax.scatter(data[:, 0][match_numbers == i], data[:, 1][match_numbers == i],
#                c=color_list[i])
#
#     ax.scatter(centroids_coords[i, 0], centroids_coords[i, 1], c=color_list[i], marker='*', s=200)
#
# plt.show()

# print(centroids_coords)
