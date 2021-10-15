import sys
import numpy as np
from typing import Dict, List, Any, Union


class DecisionTree:

    def __init__(
            self,
            max_depth: int,
            min_node_size: int
    ):
        self.max_depth: int = max_depth
        self.min_node_size: int = min_node_size
        self.tree = None

    class TreeBinaryNode:
        def __init__(self, left_node=None, right_node=None, data_dict=None):
            if data_dict is None:
                data_dict = {}

            self.left: Union[DecisionTree.TreeBinaryNode, float] = left_node
            self.right: Union[DecisionTree.TreeBinaryNode, float] = right_node
            self.data_dict: Dict = data_dict

        def set_left_as_node(self, left_node):
            self.left: DecisionTree.TreeBinaryNode = left_node

        def set_left_as_num(self, left_num: int):
            self.left: int = left_num

        def set_right_as_node(self, right_node):
            self.left: DecisionTree.TreeBinaryNode = right_node

        def set_right_as_num(self, right_num: int):
            self.left: int = right_num

    class TreeNonBinaryNode:
        def __init__(self, data):
            self.children = []
            self.data = data

    @staticmethod
    def calc_gini(
            groups: List[List],
            classes: List
    ) -> float:
        # Количество элементов в узле (во всех группах)
        num_of_elements: int = sum([len(group) for group in groups])
        gini: float = 0.0

        for group in groups:
            size = len(group)

            # Если группа пустая, скипаем ее
            if size == 0:
                continue

            score: float = 0.0

            # Пробегаемся по классам и сравниваем классы содержмого с ними
            for class_i in classes:
                p: float = 0.
                for element in group:

                    # Если классы совпали, инкрементируем вероятность ...
                    if element[-1] == class_i:
                        p += 1

                # ... и нормируем ее
                p = p / np.double(size)
                score += p * (1 - p)

            # Суммируем критерий информативности
            gini += score * size / float(num_of_elements)

        return gini

    @staticmethod
    def do_single_split(
            index: int,
            value: float,
            data: List
    ) -> List[List]:
        # Создаем данные для левого и правого ребенка
        left: List = []
        right: List = []

        # Пробегаемся по всем элеменным таблицы и разбиваем на детей в зависимости от неравенства
        for row in data:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)

        return [left, right]

    def do_full_one_node_split(
            self,
            data: List
    ) -> Dict[str, Any]:
        classes: List = list(row[-1] for row in data)
        split_index, split_value, best_gini, best_split = sys.maxsize, sys.maxsize, sys.maxsize, None

        out: Dict = {}

        for index in range(len(data[0]) - 1):
            for row in data:
                groups = self.do_single_split(index, row[index], data)
                gini = self.calc_gini(groups, classes)
                # print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], gini))
                if gini < best_gini:
                    split_index, split_value, best_gini, best_split = index, row[index], gini, groups

        out['index'] = split_index
        out['value'] = split_value
        out['groups'] = best_split

        return out

    @staticmethod
    def create_value_of_last_node(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def do_split(self, node: TreeBinaryNode, current_depth):
        left_list, right_list = node.data_dict['groups']
        del (node.data_dict['groups'])

        if not left_list or not right_list:
            node.left = node.right = self.create_value_of_last_node(left_list + right_list)
            return

        if current_depth >= self.max_depth:
            node.left = self.TreeBinaryNode(left_node=self.create_value_of_last_node(left_list))
            node.right = self.TreeBinaryNode(left_node=self.create_value_of_last_node(right_list))
            return

        # if len(left_list) <= self.min_node_size:
        #     node.left = self.TreeBinaryNode(left_node=self.create_value_of_last_node(left_list))
        #     return
        # else:
        #     node.left = self.TreeBinaryNode(left_node=None, right_node=None,
        #                                     data_dict=self.do_full_one_node_split(left_list))
        #     self.do_split(node=node.left, current_depth=current_depth + 1)
        #
        # if len(right_list) <= self.min_node_size:
        #     node.right = self.TreeBinaryNode(right_node=self.create_value_of_last_node(right_list))
        #     return
        # else:
        #     node.right = self.TreeBinaryNode(left_node=None, right_node=None,
        #                                      data_dict=self.do_full_one_node_split(right_list))
        #     self.do_split(node=node.right, current_depth=current_depth + 1)

        node.left = self.do_recurse(data_list=left_list, depth=current_depth, i=-1)
        node.right = self.do_recurse(data_list=right_list, depth=current_depth, i=1)

    def do_recurse(self, data_list, depth, i):
        node: DecisionTree.TreeBinaryNode = DecisionTree.TreeBinaryNode()

        if len(data_list) <= self.min_node_size:
            if i == -1:
                node = self.TreeBinaryNode(left_node=self.create_value_of_last_node(data_list))
            elif i == 1:
                node = self.TreeBinaryNode(right_node=self.create_value_of_last_node(data_list))
            else:
                assert "Некорректный ввод i"
        else:
            node = self.TreeBinaryNode(left_node=None, right_node=None,
                                       data_dict=self.do_full_one_node_split(data_list))
            self.do_split(node=node, current_depth=depth + 1)

        return node

    def build_tree(self, data):
        root = self.do_full_one_node_split(data)
        root_node = self.TreeBinaryNode(left_node=None, right_node=None, data_dict=root)
        self.do_split(root_node, 1)
        return root_node
