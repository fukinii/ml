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

