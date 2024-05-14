import torch
from torchvision import models

model = models.vit_b_16(weights=models.ViT_B_16_Weights)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)
