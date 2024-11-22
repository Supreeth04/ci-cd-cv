from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def trainDataAndLoad():
    train_data = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=get_transforms()
    )

    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    return train_loader

def testDataAndLoad():
    test_data = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=get_transforms()
    )

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    return test_loader

