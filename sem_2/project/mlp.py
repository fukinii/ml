import json
import matplotlib.pyplot as plt

# fig, (ax1, ax2) = plt.subplots(
#     nrows=1, ncols=1,
#     figsize=(12, 10)
# )

# estimators = [20, 50, 100, 300, 500, 1000, 2000, 3000, 4000, 5000]
# accuracy = [0.66, 0.76, 0.81, 0.85, 0.85, 0.87, 0.88, 0.89, 0.89, 0.89]



plt.grid(True)
plt.xlabel("Num of estimators")
plt.ylabel("Accuracy")
plt.title('Etsimators')
# plt.plot(estimators, accuracy, color='blue', label='train')

plt.savefig('est(accuracy).png')

plt.show()

# with open("my_catboost_json/output_5000_01_6.json", 'r') as f:
#   data = json.load(f)
#
# learn_accuracy_list = data['1']
# data_len = len(learn_accuracy_list)
# test_accuracy_list = data['2']
# iterations = [i + 1 for i in range(data_len)]
#
# print(len(learn_accuracy_list))
#
# # fig, (ax1, ax2) = plt.subplots(
# #     figsize=(12, 9)
# # )
#
# plt.rcParams['savefig.facecolor'] = 'white'
# plt.rcParams['figure.facecolor'] = 'white'
#
# plt.grid(True)
# plt.title('Classification accuracy')
# plt.plot(iterations, learn_accuracy_list, color='blue', label='train')
# plt.plot(iterations, test_accuracy_list, color='orange', label='test')
# plt.legend()
#
# plt.xlabel("Number of estimators")
# plt.ylabel("Accuracy")
#
# plt.savefig("5000_01_6.png")
#
# plt.show()
