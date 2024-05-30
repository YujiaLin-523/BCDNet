import numpy as np
import torch
from tqdm import tqdm
from modules import data_loader
from modules.data_loader import TestLoader, BCDDataset, transformation
from models import BCDNet, resnet, ViT


# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Prepare the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(batch_size, model_type):
    # Create the model
    if model_type == 'BCDNet':
        net = BCDNet.model
    elif model_type == 'resnet':
        net = resnet.model
    elif model_type == 'ViT':
        net = ViT.model
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)

    # Load the model from the checkpoint
    net.load_state_dict(torch.load(f'checkpoints/{model_type}.pth'))

    # Create the loss function
    loss = torch.nn.CrossEntropyLoss()

    # Create the test_set
    dataset = BCDDataset('./data', transformation)
    train_set, val_set, test_set = data_loader.split(dataset)

    # Create the TestLoader
    test_loader = TestLoader(test_set, batch_size=batch_size, shuffle=True)

    # Test the model
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(test_loader, total=len(test_loader))
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            loop.set_description(f'Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    test(batch_size=256, model_type='BCDNet')
    test(batch_size=256, model_type='resnet')
    test(batch_size=256, model_type='ViT')