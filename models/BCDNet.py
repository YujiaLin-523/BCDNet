import torch
from torch import nn


class BCDNet(nn.Module):
    def __init__(self):
        super(BCDNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 1, padding=0),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 1, padding=0),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1 * 256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        # This print function is used to check the size of the output of the feature extractor.
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = BCDNet()

# This code block is used to test if the model is working correctly or not.
# test_net = BCDNet()
# test_input = torch.randn(1, 3, 256, 256)
# test_output = test_net(test_input)
# print(test_output.size())
