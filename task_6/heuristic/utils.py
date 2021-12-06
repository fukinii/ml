import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


def calc_matrix_dist(data):
    num_of_points = data.shape[0]
    matrix = np.zeros((num_of_points, num_of_points))
    for i in range(1, num_of_points):
        for j in range(i):
            matrix[i, j] = calc_distance(data[i], data[j])

    return matrix


def thin_out_matrix(matrix, r):
    num_of_points = matrix.shape[0]
    matrix_out = np.copy(matrix)
    for i in range(1, num_of_points):
        for j in range(i):
            if matrix_out[i, j] > r:
                matrix_out[i, j] = -1
    return matrix_out


def find_clusters(matrix):
    out = []

    for i, point in enumerate(matrix):
        mask = np.where(point > 0)
        # current_set = set().update(mask)
        # current_set = set()
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


def calc_distance(point1, point2):
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
