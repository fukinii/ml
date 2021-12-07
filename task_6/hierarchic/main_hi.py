import pickle

import numpy as np
import matplotlib.pyplot as plt
import random

from utils import build_task

import matplotlib._color_data as mcd
import matplotlib.animation

data_file = "../docword.kos.txt"
description = np.loadtxt(data_file, max_rows=3, dtype=int)

doc_number = description[0]
word_dict_number = description[1]
nonzero_counts = description[2]

data = np.loadtxt(data_file, skiprows=3, dtype=int)
vocab = np.loadtxt("../vocab.kos.txt", dtype=str)

print("Данные прочитаны")
point = build_task(data, word_dict_number, doc_number)

