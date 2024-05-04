import numpy as np
import torch
from tqdm import tqdm
from modules import data_loader
from modules.data_loader import TestLoader, BCDDataset, transformation
from models.BCDNet import BCDNet

# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Prepare the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
net = BCDNet().to(device)

# Load the model from the checkpoint
net.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Create the loss function
loss = torch.nn.CrossEntropyLoss()

# Create the test_set
dataset = BCDDataset('./data', transformation)
train_set, val_set, test_set = data_loader.split(dataset)

# Create the TestLoader
test_loader = TestLoader(test_set, batch_size=64, shuffle=True)

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
