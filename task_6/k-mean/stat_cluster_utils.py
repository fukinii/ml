import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import repeat
import multiprocessing
from functools import partial


def build_task(data, word_dict_number, doc_number):
    # task = np.zeros()

    points = np.zeros((doc_number, word_dict_number))
    for p in data:
        points[p[0] - 1, p[1] - 1] = p[2]
    return points


def calc_intra_cluster_distance(data, match_numbers):
    num_of_points = data.shape[0]
    distance_ = 0.
    count_ = 0.

    for i in range(2, num_of_points):
        print(i, num_of_points)
        # pool = multiprocessing.Pool(processes=8)
        # a = range(i)
        # out = pool.map(
        #     partial(
        #         calc_intra_cluster_distance_parall, match_numbers=match_numbers, data=data, i=i
        #     ),
        #              a)
        # distance = np.array(out[0])
        # count = np.array(out[1])
        #
        # distance_ += np.sum(distance)
        # count_ += np.sum(count)
        for j in range(i):
            if match_numbers[i] == match_numbers[j]:
                distance_ += calc_distance(data[i], data[j])
                count_ += 1

    return distance_ / count_


def calc_intra_cluster_distance_parall(j, match_numbers, data, i):
    distance = 0.
    count = 0.

    if match_numbers[i] == match_numbers[j]:
        distance += calc_distance(data[i], data[j])
        count += 1
    return [distance, count]


def calc_extra_cluster_distance(data, match_numbers):
    num_of_points = data.shape[0]
    distance = 0.
    count = 0.
    for i in range(num_of_points):

        for j in range(i):
            if match_numbers[i] != match_numbers[j]:
                distance += calc_distance(data[i], data[j])
                count += 1

    return distance / count


def calc_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def calc_center_of_mass(points):
    return np.sum(points, axis=0) / np.shape(points)[0]


def calc_centroids(data, num_of_centroids, tolerance=1e-2):
    num_of_points = data.shape[0]
    dim = data.shape[1]

    borders = np.zeros((dim, 2))
    for i in range(dim):
        a = data[:, i]
        borders[i, 0] = np.min(a)
        borders[i, 1] = np.max(a)

    centroids_coords = np.zeros((num_of_centroids, dim))

    for i in range(num_of_centroids):
        curr_centroid_coord = np.zeros(dim)
        for j in range(dim):
            curr_centroid_coord[j] = random.uniform(borders[j, 0], borders[j, 1])
        centroids_coords[i] = curr_centroid_coord

    solution = []
    match_numbers_in_time = []

    match_numbers = np.zeros(num_of_points, dtype=int)

    iteration = 0
    while 1:
        previous_centroids_coords = np.copy(centroids_coords)
        solution.append(previous_centroids_coords)

        # pool = multiprocessing.Pool(processes=8)
        # match_numbers = np.array(pool.map(partial(match_parall, centroids_coords=centroids_coords), data))
        match_numbers = match(data, centroids_coords)

        match_numbers_in_time.append(np.copy(match_numbers))

        for centroid_idx, centroid in enumerate(centroids_coords):
            tmp = data[match_numbers == centroid_idx]
            if np.shape(tmp)[0] != 0:
                centroids_coords[centroid_idx] = calc_center_of_mass(data[match_numbers == centroid_idx])

        err = np.amax(np.abs(previous_centroids_coords - centroids_coords))
        print("Кластеризация:", iteration, err)

        if err < tolerance:
            unique_match_numbers = np.unique(match_numbers)
            if np.shape(unique_match_numbers)[0] < num_of_centroids:
                arr = np.arange(num_of_centroids)
                # print(arr, unique_match_numbers)
                diff = np.setdiff1d(arr, unique_match_numbers)
                # print(diff)

                for id in diff:
                    for j in range(dim):
                        centroids_coords[id][j] = random.uniform(borders[j, 0], borders[j, 1])
                a = previous_centroids_coords - centroids_coords
                print(a[np.nonzero(a)])
                # if arr[-1] != unique_match_numbers[-1]:
                #     for j in range(dim):
                #         centroids_coords[-1][j] = random.uniform(borders[j, 0], borders[j, 1])
                #     # centroids_coords[-1] = np.array([random.uniform(l1, l2), random.uniform(m1, m2)])
                # else:
                #     for i in arr:
                #         if np.unique(match_numbers)[i] != arr[i]:
                #             # impostor = np.arange(num_of_centroids)[i]
                #             # centroids_coords[i] = np.array([random.uniform(l1, l2), random.uniform(m1, m2)])
                #             for j in range(dim):
                #                 centroids_coords[i][j] = random.uniform(borders[j, 0], borders[j, 1])
                #             break
                print("Не нашли себя центроидов:", num_of_centroids - np.shape(unique_match_numbers)[0])
                print("Данные скорректированы")
            else:
                print("Расчет окончен")
                break
        iteration += 1

    return centroids_coords, match_numbers


def match(data, centroids_coords):
    num_of_points = data.shape[0]
    match_numbers = np.zeros(num_of_points, dtype=int)

    for point_idx, point in enumerate(data):
        min_distance = 1e9
        closest_centroid = 0
        for centroid_idx, centroid in enumerate(centroids_coords):
            distance = calc_distance(point, centroid)
            if distance < min_distance:
                min_distance = distance
                closest_centroid = centroid_idx

        match_numbers[point_idx] = closest_centroid

    return match_numbers


def match_parall(data, centroids_coords):
    num_of_points = 1
    match_numbers = np.zeros(num_of_points, dtype=int)

    min_distance = 1e9
    closest_centroid = 0
    for centroid_idx, centroid in enumerate(centroids_coords):
        distance = calc_distance(data, centroid)
        if distance < min_distance:
            min_distance = distance
            closest_centroid = centroid_idx

    # match_numbers[point_idx] = closest_centroid
    return closest_centroid


def draw(data, match_numbers, centroids_coords, color_list):
    fig_1 = plt.figure(figsize=(16, 10))
    ax = fig_1.add_subplot(111)
    ax.set(facecolor='black')
    fig_1.set(facecolor='black')

    for i in range(centroids_coords.shape[0]):
        ax.scatter(data[:, 0][match_numbers == i], data[:, 1][match_numbers == i],
                   c=color_list[i])

        ax.scatter(centroids_coords[i, 0], centroids_coords[i, 1], c=color_list[i], marker='*', s=200)

    plt.show()


def get_color_list(num_of_centroids):
    color_list = {}

    for i in range(num_of_centroids):
        color_list[i] = np.array(
            [np.array([random.uniform(0.5, 1.), random.uniform(0.5, 1.), random.uniform(0.5, 1.)])])

    return color_list