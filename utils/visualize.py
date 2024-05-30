from matplotlib import pyplot as plt

model_type = 'ViT'

# Plot the loss and accuracy values
def plot_loss_accuracy(loss_values, acc_values, model_type):
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
    # if model_type == 'BCDNet':
    #     plt.savefig('logs/BCDNet.png')
    # elif model_type == 'resnet':
    #     plt.savefig('logs/resnet.png')
    # elif model_type == 'Vit':
    plt.savefig('logs/ViT.png')
