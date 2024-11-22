import torch.nn as nn
import torch.nn.functional as F

class CustomNeuralNetwork(nn.Module):
    def __init__(self):
        super(CustomNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        
        # self.batchnorm1 = nn.BatchNorm2d(4)
        # self.batchnorm2 = nn.BatchNorm2d(8)
        
        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        
        self.dropout = nn.Dropout(0.005)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        # x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 8 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)