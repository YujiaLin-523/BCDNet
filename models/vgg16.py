import torch
import torchvision

model = torchvision.models.vgg16()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
