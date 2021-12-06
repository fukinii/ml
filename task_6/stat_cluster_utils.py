import numpy as np
import matplotlib.pyplot as plt
import random


def calc_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def calc_center_of_mass(points):
    return np.sum(points, axis=0) / np.shape(points)[0]


def calc_centroids(data, num_of_centroids):

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
