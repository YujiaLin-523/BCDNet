import torch
from torchvision.models import resnet

model = resnet.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
