import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
import random


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


def calc_centroids(all_points, N, TOLERANCE=1e-5):
    l1 = -1
    l2 = 2

    m1 = -1
    m2 = 2

    num_of_centroids = 3
    solution = []

    centroids_coords = np.array(
        [np.array([random.uniform(l1, l2), random.uniform(m1, m2)]) for i in range(num_of_centroids)])

    match_numbers_in_time = []
    match_numbers = np.zeros(3 * N, dtype=int)

    while 1:
        previous_centroids_coords = np.copy(centroids_coords)
        solution.append(previous_centroids_coords)
        for point_idx, point in enumerate(all_points):
            min_distance = 1e9
            closest_centroid = 0
            for centroid_idx, centroid in enumerate(centroids_coords):
                distance = calc_distance(point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = centroid_idx

            match_numbers[point_idx] = closest_centroid

        match_numbers_in_time.append(np.copy(match_numbers))

        for centroid_idx, centroid in enumerate(centroids_coords):
            tmp = all_points[match_numbers == centroid_idx]
            if np.shape(tmp)[0] != 0:
                centroids_coords[centroid_idx] = calc_center_of_mass(all_points[match_numbers == centroid_idx])

        unique_match_numbers = np.unique(match_numbers)

        if np.amax(np.abs(previous_centroids_coords - centroids_coords)) < TOLERANCE:
            if np.shape(unique_match_numbers)[0] < num_of_centroids:
                arr = np.arange(num_of_centroids)
                if arr[-1] != unique_match_numbers[-1]:
                    centroids_coords[-1] = np.array([random.uniform(l1, l2), random.uniform(m1, m2)])
                else:
                    for i in arr:
                        if np.unique(match_numbers)[i] != arr[i]:
                            impostor = np.arange(num_of_centroids)[i]
                            centroids_coords[i] = np.array([random.uniform(l1, l2), random.uniform(m1, m2)])
                            break
            else:
                break

    return all_points, match_numbers  # FIXME: некорректная работа