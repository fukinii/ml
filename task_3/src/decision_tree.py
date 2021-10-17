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

            self.left: Union[DecisionTree.TreeBinaryNode, int] = left_node
            self.right: Union[DecisionTree.TreeBinaryNode, int] = right_node
            self.data_dict: Dict = data_dict

        def set_left_as_node(self, left_node):
            self.left: DecisionTree.TreeBinaryNode = left_node

        def set_left_as_num(self, left_num: int):
            self.left: int = left_num

        def set_right_as_node(self, right_node):
            self.right: DecisionTree.TreeBinaryNode = right_node

        def set_right_as_num(self, right_num: int):
            self.right: int = right_num

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
            for class_i in set(classes):
                p: float = 0.
                for index, element in enumerate(group):
                    # print(index)
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
                if gini < best_gini:
                    split_index, split_value, best_gini, best_split = index, row[index], gini, groups

        out['index'] = split_index
        out['value'] = split_value
        out['groups'] = best_split

        return out

    @staticmethod
    def create_value_of_last_node(group) -> int:
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def do_split(self, node: TreeBinaryNode, current_depth):
        left_list, right_list = node.data_dict['groups']
        del (node.data_dict['groups'])

        if not left_list or not right_list:
            node.left = node.right = self.create_value_of_last_node(left_list + right_list)
            return

        if current_depth >= self.max_depth:
            node.set_left_as_num(self.create_value_of_last_node(left_list))
            node.set_right_as_num(self.create_value_of_last_node(right_list))
            return

        node.left = self.do_recurse(data_list=left_list, depth=current_depth)
        node.right = self.do_recurse(data_list=right_list, depth=current_depth)

    def do_recurse(self, data_list, depth):

        node: Union[DecisionTree.TreeBinaryNode, int]

        if len(data_list) <= self.min_node_size:
            node: int = self.create_value_of_last_node(data_list)
        else:
            node: DecisionTree.TreeBinaryNode = self.TreeBinaryNode(left_node=None, right_node=None,
                                                                    data_dict=self.do_full_one_node_split(data_list))
            self.do_split(node=node, current_depth=depth + 1)

        return node

    def build_tree(self, data):
        root = self.do_full_one_node_split(data)
        root_node = self.TreeBinaryNode(left_node=None, right_node=None, data_dict=root)
        self.do_split(root_node, 1)
        return root_node

    def predict(self, node, row):

        if row[node.data_dict['index']] < node.data_dict['value']:
            if type(node.left) == DecisionTree.TreeBinaryNode:
                return self.predict(node=node.left, row=row)
            else:
                return node.left
        else:
            if type(node.right) == DecisionTree.TreeBinaryNode:
                return self.predict(node=node.right, row=row)
            else:
                return node.right
