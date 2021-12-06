import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


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
