# import random
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def calc_intra_cluster_distance(data, match_numbers):
#
#
# def draw(data, match_numbers, centroids_coords, color_list):
#     fig_1 = plt.figure(figsize=(16, 10))
#     ax = fig_1.add_subplot(111)
#     ax.set(facecolor='black')
#     fig_1.set(facecolor='black')
#
#     for i in range(centroids_coords.shape[0]):
#         ax.scatter(data[:, 0][match_numbers == i], data[:, 1][match_numbers == i],
#                    c=color_list[i])
#
#         ax.scatter(centroids_coords[i, 0], centroids_coords[i, 1], c=color_list[i], marker='*', s=200)
#
#     plt.show()
#
#
# def get_color_list(num_of_centroids):
#     color_list = {}
#
#     for i in range(num_of_centroids):
#         color_list[i] = np.array(
#             [np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])
#
#     return color_list