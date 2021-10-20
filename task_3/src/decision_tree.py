import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
from src.utils.utils import get_key
import copy


class DecisionTree:
    """
    Класс решающего дерева
    """

    def __init__(
            self,
            max_depth: int,
            min_node_size: int
    ) -> None:
        """
        Конструктор класса решающего дерева

        :param max_depth: максимальаня глубина дерева
        :param min_node_size: минимальный размер узла (количества объектов в нем)
        """
        self.cat_feature_value_list: List = []
        self.max_depth: int = max_depth
        self.min_node_size: int = min_node_size
        self.tree = None
        self.column_of_features: pd.core.indexes.base.Index = None
        self.categorical_features_list: List = []
        self.categorical_feature_index_list: List = []
        self.dict_index_to_num_cat_feature = {}

    class TreeBinaryNode:
        """
        Класс узла решающего дерева
        """

        def __init__(self, left_node=None, right_node=None, data_dict=None) -> None:
            """
            Конструктор класса узла

            :param left_node: левый ребенок узла, которым может быть как следующим узлом, так и просто числом,
            определяющим класс данных, если построение дерева окончено
            :param right_node: правый ребенок узла, которым может быть как следующим узлом, так и просто числом,
            определяющим класс данных
            :param data_dict: словарь с данными узла, содержащий индекс фичи, пороговое значение и разбиваемые группы
            """
            if data_dict is None:
                data_dict = {}

            self.left: Union[DecisionTree.TreeBinaryNode, int] = left_node
            self.right: Union[DecisionTree.TreeBinaryNode, int] = right_node
            self.data_dict: Dict = data_dict

        def set_left_as_node(self, left_node) -> None:
            """
            Метод, записывающий в левого ребенка типа узел подаваемый на вход узел
            """
            self.left: DecisionTree.TreeBinaryNode = left_node

        def set_left_as_num(self, left_num: int):
            """
            Метод, записывающий в левого ребенка типа int подаваемое на вход число
            """
            self.left: int = left_num

        def set_right_as_node(self, right_node):
            """
            Метод, записывающий в правого ребенка типа узел подаваемый на вход узел
            """
            self.right: DecisionTree.TreeBinaryNode = right_node

        def set_right_as_num(self, right_num: int):
            """
            Метод, записывающий в правого ребенка типа int подаваемое на вход число
            """
            self.right: int = right_num

    class TreeNonBinaryNode:
        def __init__(self):
            pass

    @staticmethod
    def calc_gini(
            groups: List[List],
            classes: set
    ) -> float:
        """
        Метод для вычисления критерия информативности Джини

        :param groups: входные группы с данными, среди которых могут быть представители всех классов.
        Последняя ячейка хранит его номер
        :param classes: существующие классы
        :return: значение критерия информативности
        """
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
            threshold: float,
            data: List
    ) -> List[List]:
        """
        Метод для построения одного разбиения на основе уже вычисленного порога некоторой фичи
        :param index: индекс фичи
        :param threshold: пороговое значение
        :param data: список данных (иначе говоря - некоторый массив строк row исходной таблицы с данными)
        :return: два массива-ребенка
        """
        # Создаем данные для левого и правого ребенка
        left: List = []
        right: List = []

        # Пробегаемся по всем элеменным таблицы и разбиваем на детей в зависимости от значения в ячейке и порога
        for row in data:
            if type(row[index]) == str:
                a = 5
            if row[index] < threshold:
                left.append(row)
            else:
                right.append(row)

        return [left, right]

    def do_full_one_node_split(
            self,
            data: List
    ) -> Dict[str, Any]:
        """
        Произвести полное разбиение для входного массива строк row таблицы: пробежаться по всем фичам и соответствующим
        ячейкам с вычислением Джини для каждого случая. Выбирается наилучший Джини
        :param data: массив строк row с данными. Последняя ячейка хранит номер класса
        :return: словарь с индексом фичи, порогом и разбиением
        """

        # Создаем множество со всеми имеющимися классами в data
        classes: set = set(row[-1] for row in data)

        # Инициализируем используемые данные
        split_index, split_threshold, best_gini, best_split = sys.maxsize, sys.maxsize, sys.maxsize, None
        out: Dict = {}

        list_of_dict_cat_to_int, list_of_dict_int_to_cat = self.create_dict_for_cat_features(data)

        # Пробегаемся по всем фичам и ячейкам для вычисления наилучшего ДЖини
        for row in data:
            for index in range(len(data[0]) - 1):
                if index in self.categorical_feature_index_list:
                    # Строим конкретное разбиение для текущего значения index и data
                    str = row[index]

                    cat_num = self.dict_index_to_num_cat_feature[index]
                    cat_dict = list_of_dict_cat_to_int[cat_num]
                    new_data = self.replace_categorical_str_to_int(data, cat_dict, index)
                    c = cat_dict[str]
                    groups = self.do_single_split(index, c, new_data)

                    # Вычисляем для него Джини
                    gini = self.calc_gini(groups, classes)
                else:
                    # Строим конкретное разбиение для текущего значения index и data
                    groups = self.do_single_split(index, row[index], data)

                    # Вычисляем для него Джини
                    gini = self.calc_gini(groups, classes)

                # Если Джини лучше предыдущих, сохраняем его и соответствующие ему индекс, порог и разбиение
                if gini < best_gini:
                    split_index, split_threshold, best_gini, best_split = index, row[index], gini, groups
        #
        # if type(best_split[0][0][8]) == int:
        #     debug = 1

        if split_index in self.categorical_feature_index_list:
            # for feature_index in self.categorical_feature_index_list:
            cat_num = self.dict_index_to_num_cat_feature[split_index]
            cat_dict = list_of_dict_int_to_cat[cat_num]
            best_split[0] = self.replace_int_to_categorical_str(best_split[0], cat_dict, split_index)
            best_split[1] = self.replace_int_to_categorical_str(best_split[1], cat_dict, split_index)

        # Заполнение данных для вывода
        out['index'] = split_index
        out['threshold'] = split_threshold
        out['groups'] = best_split

        return out

    @staticmethod
    def create_value_of_last_node(group: List) -> int:
        """
        Метод, создающий финальное значение в листе на основе наиболее часто встречающегося класса в группе
        :param group: список строк row исходной таблицы
        :return: наиболее часто встречающийся класс
        """
        classes = [row[-1] for row in group]
        return max(set(classes), key=classes.count)

    def do_split(self, node: TreeBinaryNode, current_depth: int) -> None:
        """
        Рекурсивный метод для построения дерева
        :param node: входной узел
        :param current_depth: текущая глубина узла node
        :return: None
        """

        # Из текущего узла вытаскиваем уже найденные данные с группами
        left_list, right_list = node.data_dict['groups']

        # Удаляем содержимое словаря по ключу групп, так как нам эти данные больше не нужны
        del (node.data_dict['groups'])

        # Остановка построения дерева, если один из детей пустой
        if not left_list or not right_list:
            node.left = node.right = self.create_value_of_last_node(left_list + right_list)
            return

        # Остановка построения дерева, если глубина достаточно большая
        if current_depth >= self.max_depth:
            node.set_left_as_num(self.create_value_of_last_node(left_list))
            node.set_right_as_num(self.create_value_of_last_node(right_list))
            return

        # Вызов рекурсивной части, если не было выполнено одно из условий остановки
        node.left = self.do_recurse(data_list=left_list, depth=current_depth)
        node.right = self.do_recurse(data_list=right_list, depth=current_depth)

    def do_recurse(self, data_list: List, depth: int):
        """
        Рекурсивная часть метода do_split. Остановка вычислений, если текущий размер узла меньше минимального, либо
        вызов do_split
        :param data_list: список данных группы
        :param depth: текущая глубина
        :return: либо номер класса, если рекурсию необходимо завершить, либо узел TreeBinaryNode
        """

        # Инициализация узла
        node: Union[DecisionTree.TreeBinaryNode, int]

        # Если текущий размер узла меньше минимального, записываем номер класса в узел. Иначе - вызов do_split
        if len(data_list) <= self.min_node_size:
            node: int = self.create_value_of_last_node(data_list)
        else:
            data_list_debug = copy.deepcopy(self.do_full_one_node_split(data_list))
            node: DecisionTree.TreeBinaryNode = self.TreeBinaryNode(left_node=None, right_node=None,
                                                                    data_dict=self.do_full_one_node_split(data_list))
            debug = copy.deepcopy(node)
            self.do_split(node=node, current_depth=depth + 1)

        return node

    def build_tree(self, data: List):
        """
        Метод для построения дерева на основе входной таблицы, содержащей выборку с X и Y. Необходимо предварительная
        обработка с помещением номера класса объекта в конец таблицы

        :param data: список входных данных. Последний столбец - номер класса
        :return: узел корня с содержащимся в нем деревом
        """
        root = self.do_full_one_node_split(data)
        root_node: DecisionTree.TreeBinaryNode = self.TreeBinaryNode(left_node=None, right_node=None, data_dict=root)
        self.do_split(root_node, 1)
        return root_node

    def build_tree_through_df(self,
                              x: pd.core.frame.DataFrame,
                              y: pd.core.series.Series):
        """
        Построение дерева на основе разбитой выборки DataFrame
        :param x: выборка x для построения дерева
        :param y: выборка y для построения дерева
        :return: корневой узел дерева
        """
        assert type(x) == pd.core.frame.DataFrame, "Некорректный тип данных x_train"
        assert type(y) == pd.core.series.Series, "Некорректный тип данных y_train"

        self.set_categorical_data_from_dataframe(x)

        # Обернем все в numpy
        x_numpy = x.to_numpy()
        y_numpy = y.to_numpy()
        y_numpy = y_numpy.reshape((1, len(y_numpy))).transpose().astype(int)
        dataset = np.concatenate((x_numpy, y_numpy), axis=1)

        # Вызовем простроение дерева
        root = self.do_full_one_node_split(list(dataset))
        root_node = self.TreeBinaryNode(left_node=None, right_node=None, data_dict=root)
        self.do_split(root_node, 1)
        return root_node

    def predict(self, node, row: List) -> object:
        """

        :rtype: DecisionTree.TreeBinaryNode
        :param node: узел
        :param row: строка таблицы для предсказания класса
        :return: предсказанное значение класса
        """
        if row[node.data_dict['index']] < node.data_dict['threshold']:
            if type(node.left) == DecisionTree.TreeBinaryNode:
                return self.predict(node=node.left, row=row)
            else:
                return node.left
        else:
            if type(node.right) == DecisionTree.TreeBinaryNode:
                return self.predict(node=node.right, row=row)
            else:
                return node.right

    def draw(self, node, columns: pd.core.indexes.base.Index, current_depth):
        """
        Метод для отрисовки дерева
        :param node: текущий узел. В первый раз на вход подается корень
        :param columns: список фич из dataFrame
        :param current_depth: текущая глубина
        """
        if type(node) == DecisionTree.TreeBinaryNode:
            print("—" * current_depth, columns[node.data_dict['index']], "<", node.data_dict['threshold'])
            self.draw(node.left, columns, current_depth + 1)
            self.draw(node.right, columns, current_depth + 1)
        else:
            print("—" * current_depth, node)

    def set_categorical_data_from_dataframe(self, data: pd.core.frame.DataFrame):
        self.categorical_features_list = data.columns[data.dtypes == object].tolist()

        a = np.arange(len(data.columns))
        categorical_feature_index = []
        for i, feature in enumerate(self.categorical_features_list):
            categorical_feature_index.append(a[feature == data.columns][0])
            self.dict_index_to_num_cat_feature[categorical_feature_index[-1]] = i

        self.categorical_feature_index_list = categorical_feature_index

        assert len(self.cat_feature_value_list) == 0

        for feature in self.categorical_features_list:
            self.cat_feature_value_list.append(set(data[feature]))

    def create_dict_for_cat_features(self, x_m):

        list_of_dict_of_norms = []
        sorted_norms = []
        list_of_dict_cat_to_int = []
        list_of_dict_int_to_cat = []

        for id_feature, cat_feature_dict in enumerate(self.cat_feature_value_list):

            dict_of_c_sizes = {key: 0 for key in cat_feature_dict}
            dict_of_c_first_class_sizes = {key: 0 for key in cat_feature_dict}
            dict_of_norms = {}
            dict_cat_to_int = {}
            dict_int_to_cat = {}

            for row in x_m:
                row_c = row[self.categorical_feature_index_list[id_feature]]
                if row_c == 2:
                    debug = 1
                dict_of_c_sizes[row_c] += 1

                if row[-1] == 1:
                    dict_of_c_first_class_sizes[row_c] += 1

            for key in dict_of_c_sizes.copy():
                if dict_of_c_sizes[key] == 0:
                    del dict_of_c_sizes[key]
                    del dict_of_c_first_class_sizes[key]
                    continue

                dict_of_norms[key] = dict_of_c_first_class_sizes[key] / dict_of_c_sizes[key]

            list_of_dict_of_norms.append(dict_of_norms)

            a = []
            for key in dict_of_norms:
                a.append(dict_of_norms[key])
            a.sort()
            sorted_norms.append(a)

            i = 0
            for norm in a:
                key = get_key(dict_of_norms, norm)
                del dict_of_norms[key]

                dict_cat_to_int[key] = i
                dict_int_to_cat[i] = key
                i += 1

            list_of_dict_cat_to_int.append(dict_cat_to_int)
            list_of_dict_int_to_cat.append(dict_int_to_cat)

        return list_of_dict_cat_to_int, list_of_dict_int_to_cat

    @staticmethod
    def replace_categorical_str_to_int(data, dict_cat_to_int, index):
        new_data = copy.deepcopy(data)
        for row_id, row in enumerate(new_data):
            a = row
            cat = row[index]
            new_data[row_id][index] = dict_cat_to_int[cat]

        return new_data

    @staticmethod
    def replace_int_to_categorical_str(data, dict_int_to_cat, index):
        new_data = copy.deepcopy(data)
        for row_id, row in enumerate(new_data):
            a = row
            i = row[index]
            new_data[row_id][index] = dict_int_to_cat[i]

        return new_data
