import os
import numpy as np
import torch
import time
from models import BCDNet, resnet
from tqdm import tqdm
from modules.optimizer import Adam, SGD
from modules.scheduler import StepLR, CosineAnnealingLR
from modules.data_loader import BCDDataset, TrainLoader, ValLoader
from modules import data_loader
from utils.visualize import plot_loss_accuracy

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Prepare the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the dataset
dataset = BCDDataset('./data', data_loader.transformation)
train_set, val_set, test_set = data_loader.split(dataset)

# Create the checkpoints path
os.makedirs('./checkpoints', exist_ok=True)

# Define the main function
def main(epochs, optimizer_type, lr, scheduler_type, model_type):
    # Create the TrainLoader and ValLoader
    train_loader = TrainLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_loader = ValLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

    # Create the model
    if model_type == 'BCDNet':
        net = BCDNet.model
    elif model_type == 'resnet':
        net = resnet.model

    net.to(device)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    params = net.parameters()

    # Create the optimizer
    if optimizer_type == 'Adam':
        optimizer = Adam(params, lr=lr)
    elif optimizer_type == 'SGD':
        optimizer = SGD(params, lr=lr)
    else:
        raise ValueError(f'Invalid optimizer {optimizer_type}')

    # Create the scheduler
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise ValueError(f'Invalid scheduler {scheduler_type}')

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    # Define the Training parameters
    num_epochs = epochs
    best_acc = 0.0
    loss_values = []
    acc_values = []

    # record the start time
    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        loop = tqdm(train_loader, total=len(train_loader), dynamic_ncols=True)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_description(f'Training [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        # Validate the model
        if (epoch + 1) % 5 == 0:
            net.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                loop = tqdm(val_loader, total=len(val_loader), dynamic_ncols=True)
                for images, labels in loop:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
                    loop.set_description(f'Val [{epoch + 1}/{num_epochs}]')
                    loop.set_postfix(loss=val_loss, accuracy=accuracy)

                # Save the loss and accuracy values
                val_loss /= len(val_loader)
                loss_values.append(val_loss)
                acc_values.append(accuracy)

            # Save the best model
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(net.state_dict(), os.path.join('./checkpoints', f'{model_type}.pth'))

    # Record the end time
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f}s')
    
    # Plot the loss and accuracy values
    plot_loss_accuracy(loss_values, acc_values, model_type)


# Call the main function
if __name__ == '__main__':
    main(100, 'Adam', 0.0005, 'StepLR', 'BCDNet')
