import matplotlib.pyplot as plt


def plot_f1_score(results, num_epoch):
    epoch = [i for i in range(1, num_epoch + 1)]
    plt.plot(epoch, results)
    plt.title("F1 Score per Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.show()


def plot_acc_loss(accuracies, losses, num_epoch):
    epoch = [i for i in range(1, num_epoch + 1)]
    plt.plot(epoch, accuracies, label = "Accuracy")
    plt.plot(epoch, losses, label = "Loss")
    plt.legend()
    plt.show()