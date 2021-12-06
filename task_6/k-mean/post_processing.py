import multiprocessing
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import random
from stat_cluster_utils import calc_centroids, get_color_list, draw, calc_intra_cluster_distance, \
    calc_extra_cluster_distance

import pickle

with open('Centroids5_kos.pickle', 'rb') as f:
    out = pickle.load(f)

print("Считаны результаты кластеризации")

centroids_coords = out[0]
match_numbers = out[1]

data_file = "../docword.kos.txt"
description = np.loadtxt(data_file, max_rows=3, dtype=int)
print("Считано описание данных")

doc_number = description[0]
word_dict_number = description[1]
nonzero_counts = description[2]
data = np.loadtxt(data_file, skiprows=3, dtype=int)
print("Считаны данные")

vocab = np.loadtxt("../vocab.kos.txt", dtype=str)
print("Считан словарь")

num_of_centroids = centroids_coords.shape[0]
color_list = get_color_list(num_of_centroids)

# pool = multiprocessing.Pool(processes=8)
# intra_cluster_distance = np.array(pool.map(partial(calc_intra_cluster_distance, match_numbers=match_numbers), data))

intra_cluster_distance = calc_intra_cluster_distance(data, match_numbers)
print("Вычислено среднее внутрикластерное расстояние")

extra_cluster_distance = calc_extra_cluster_distance(data, match_numbers)
print("Вычислено среднее внекластерное расстояние")

print(intra_cluster_distance)

draw(data, match_numbers, centroids_coords, color_list)
