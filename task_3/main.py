
from src.decision_tree import DecisionTree


decision_tree = DecisionTree(3, 1)

dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]

root = decision_tree.build_tree(dataset)

print(root.data_dict)

# a = 1
#
for row in dataset:
    prediction = decision_tree.predict(root, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))















# groups = [[[1, 1], [1, 0]], [[1, 1], [1, 0]]]
#
# classes = [0, 1]
#
# res = calc_gini(groups=groups, classes=classes)
#
# print(res)
# print(calc_gini([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
#
# dataset = [[1, 2], [1, 0], [1, 2], [1, 0]]
#
# left, right = do_split(1, 1, dataset)
#
# print(left, right)
#
# dataset = [[2.771244718, 1.784783929, 0],
#            [1.728571309, 1.169761413, 0],
#            [3.678319846, 2.81281357, 0],
#            [3.961043357, 2.61995032, 0],
#            [2.999208922, 2.209014212, 0],
#            [7.497545867, 3.162953546, 1],
#            [9.00220326, 3.339047188, 1],
#            [7.444542326, 0.476683375, 1],
#            [10.12493903, 3.234550982, 1],
#            [6.642287351, 3.319983761, 1]]
#
# split = get_split(dataset)
# # print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
# print(split['groups'][0])
# print(split['groups'][1])

