import torch.nn as nn
import torch.nn.functional as F

# A simple CNN for binary classification (ship vs no-ship)
class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 80 -> 40 -> 20 -> 10 after 3 pools
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)  # ship vs no-ship (2 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)