import torch.nn as nn
import torch.nn.functional as F

class CustomNeuralNetwork(nn.Module):
    def __init__(self):
        super(CustomNeuralNetwork, self).__init__()
        # Simple but effective architecture
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Using 5x5 kernel for better feature capture
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.batchnorm2 = nn.BatchNorm2d(20)
        
        # Reduced FC layer size
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # Classifier
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)