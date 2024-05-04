from matplotlib import pyplot as plt


# Plot the loss and accuracy values
def plot_loss_accuracy(loss_values, acc_values):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_values, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the figure to the logs folder
    plt.savefig('logs/loss_acc_plot.png')
