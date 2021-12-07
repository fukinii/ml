import multiprocessing
from functools import partial

import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


def build_task(data, word_dict_number, doc_number):
    points = np.zeros((doc_number, word_dict_number))
    for p in data:
        points[p[0] - 1, p[1] - 1] = p[2]
    return points


def calc_matrix_dist(data):
    num_of_points = data.shape[0]
    matrix = np.zeros((num_of_points, num_of_points))
    matrix.fill(-10.)
    for i in range(1, num_of_points):
        if i % (10 ** ((np.log10(num_of_points) - 1) // 1)) == 0:
            print(i, num_of_points)

        for j in range(i):
            matrix[i, j] = calc_distance(data[i], data[j])

    return matrix


def calc_matrix_dist_parall(data):
    num_of_points = data.shape[0]
    matrix = np.zeros((num_of_points, num_of_points))
    pool = multiprocessing.Pool(processes=8)
    for i in range(num_of_points):
        print(i)


        calc_distance_ = partial(calc_distance, point2=data[i])
        a = pool.map(calc_distance_, data)

        # a = np.array(pool.map(partial(calc_matrix_dist_parall_single, point=data_1[i]), data_2))
        matrix[i, :] = np.array(a)
        # for j in range(1, num_of_points):
        #     matrix[i, j] = calc_distance(data_1[i], data_2[j])

    return matrix


def calc_matrix_dist_parall_single(data, point):
    num_of_points = data.shape[0]
    dist_list = np.zeros(num_of_points)
    for i in range(1, num_of_points):
        dist_list[i] = calc_distance(point, data[i])

    return dist_list


def sparse_matrix(matrix, r):
    matrix_out = matrix * ((matrix < r).astype(int))

    return matrix_out


def find_clusters(matrix):
    out = []
    num_of_points = matrix.shape[0]
    for i, point in enumerate(matrix):

        if i % (10 ** (np.log10(num_of_points) // 1)) == 0:
            print(i, num_of_points)

        mask = np.where(point > 0)

        current_set = np.unique(mask)
        current_set = np.append(current_set, i)

        flag = False
        for set_id, set_of_pts in enumerate(out):
            inter = np.intersect1d(set_of_pts, current_set)
            if len(inter) > 0:
                flag = True
                set_of_pts = np.union1d(set_of_pts, current_set)
                out[set_id] = set_of_pts

        if not flag:
            out.append(current_set)

    return out

# a = 1
def calc_distance(point1, point2):
    # a += 1
    return np.linalg.norm(point1 - point2)


def calc_center_of_mass(points):
    return np.sum(points, axis=0) / np.shape(points)[0]


def update(num, sol, match_numbers_in_time, all_points, color_list, num_of_centroids, l1, l2, m1, m2, ax_anim,
           markers_list):
    ax_anim.clear()
    ax_anim.set_xlim([l1, l2])
    ax_anim.set_ylim([m1, m2])

    current_pos = sol[num]
    for i in range(num_of_centroids):
        ax_anim.scatter(all_points[:, 0][match_numbers_in_time[num] == i],
                        all_points[:, 1][match_numbers_in_time[num] == i],
                        c=color_list[i], marker=markers_list[i])
        ax_anim.scatter(current_pos[i, 0], current_pos[i, 1], c=color_list[i], marker='*', s=250)


def calc_intra_cluster_distance(data, match_numbers):
    num_of_points = data.shape[0]
    distance = 0.
    count = 0.

    for i in range(2, num_of_points):
        for j in range(i):
            if match_numbers[i] == match_numbers[j]:
                distance += calc_distance(data[i], data[j])
                count += 1

    return distance / count
