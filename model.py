import torch
import torch.nn as nn
import torch.nn.functional as F


class SleepStageCNN(nn.Module):
    def __init__(self, input_length: int, n_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool2(F.relu(self.bn2(self.conv2(dummy))))
            dummy = self.pool3(F.relu(self.bn3(self.conv3(dummy))))
            n_features = dummy.numel()

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
