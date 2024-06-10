import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.conv1_1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.conv1_3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattened size for fixed input size of 256x256
        self.flattened_size = 64 * 64 * 16

    def forward(self, img1, img2):

        combined_input = torch.cat((img1, img2), dim=1)

        x = F.relu(self.bn1_1(self.conv1_1(combined_input)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten

        return x

class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network
        self.fc1 = nn.Linear(self.base_network.flattened_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Prediction layer for time interval

    def forward(self, img1_1, img1_2, img2_1, img2_2):
        feat1 = self.base_network(img1_1, img1_2)
        feat2 = self.base_network(img2_1, img2_2)

        x = torch.cat((feat1, feat2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta_t = self.fc3(x)  # Predict the time interval

        return delta_t


    def get_embedding(self, img1, img2):
        return self.base_network(img1, img2)
