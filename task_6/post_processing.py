import numpy as np
import matplotlib.pyplot as plt
import random
from stat_cluster_utils import calc_centroids, get_color_list, draw

import pickle

with open('Centroids.pickle', 'rb') as f:
    out = pickle.load(f)

centroids_coords = out[0]
match_numbers = out[1]

data_file = "docword.enron.txt"
description = np.loadtxt(data_file, max_rows=3, dtype=int)
doc_number = description[0]
word_dict_number = description[1]
nonzero_counts = description[2]
data = np.loadtxt(data_file, skiprows=3, dtype=int)
vocab = np.loadtxt("vocab.enron.txt", dtype=str)

num_of_centroids = centroids_coords.shape[0]
color_list = get_color_list(num_of_centroids)

draw(data, match_numbers, centroids_coords, color_list)
