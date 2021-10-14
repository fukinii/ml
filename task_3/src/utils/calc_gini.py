import sys
import numpy as np
from typing import Dict, Type, Sequence, Optional, List, Any


# def calc_gini(
#         groups: List[List],
#         classes: List
# ) -> float:
#     # Количество элементов в узле (во всех группах)
#     num_of_elements: int = sum([len(group) for group in groups])
#     gini: float = 0.0
#
#     for group in groups:
#         size = len(group)
#
#         # Если группа пустая, скипаем ее
#         if size == 0:
#             continue
#
#         score: float = 0.0
#
#         # Пробегаемся по классам и сравниваем классы содержмого с ними
#         for class_i in classes:
#             p: float = 0.
#             for element in group:
#
#                 # Если классы совпали, инкрементируем вероятность ...
#                 if element[-1] == class_i:
#                     p += 1
#
#             # ... и нормируем ее
#             p = p / np.double(size)
#             score += p * (1 - p)
#
#         # Суммируем критерий информативности
#         gini += score * size / float(num_of_elements)
#
#     return gini


# def do_split(
#         index: int,
#         value: float,
#         data: List
# ) -> List[List]:
#     # Создаем данные для левого и правого ребенка
#     left: List = []
#     right: List = []
#
#     # Пробегаемся по всем элеменным таблицы и разбиваем на детей в зависимости от неравенства
#     for row in data:
#         if row[index] < value:
#             left.append(row)
#         else:
#             right.append(row)
#
#     return [left, right]
#
#
# def get_split(
#         data: List
# ) -> Dict[str, Any]:
#     classes: List = list(row[-1] for row in data)
#     split_index, split_value, best_gini, best_split = sys.maxsize, sys.maxsize, sys.maxsize, None
#
#     out: Dict = {}
#
#     for index in range(len(data[0]) - 1):
#         for row in data:
#             groups = do_split(index, row[index], data)
#             gini = calc_gini(groups, classes)
#             # print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], gini))
#             if gini < best_gini:
#                 split_index, split_value, best_gini, best_split = index, row[index], gini, groups
#
#     out['index'] = split_index
#     out['value'] = split_value
#     out['groups'] = best_split
#
#     return out
