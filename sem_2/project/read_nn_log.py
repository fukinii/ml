import pandas as pd
import matplotlib.pyplot as plt

path = "/home/fukin/ml/homework/ml/sem_2/project/log.csv"

df = pd.read_csv(path, sep=',')


def plotgraph(epochs, acc, val_acc, savefig1, savefig2):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model ' + savefig2)
    plt.ylabel(savefig2)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.savefig(savefig1 + " " + savefig2)
    plt.show()


values = df.values

epochs = values[:, 0]
acc = values[:, 1]
val_acc = values[:, 3]
loss = values[:, 2]
val_loss = values[:, 4]

plotgraph(epochs, acc, val_acc, "Validation", "accuracy")
plotgraph(epochs, loss, val_loss, "Validation", "loss")
