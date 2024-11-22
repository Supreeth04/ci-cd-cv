import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import data_file
from model import CustomNeuralNetwork, count_parameters

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = CustomNeuralNetwork().to(device)
    model_params = count_parameters(model)
    print(f"Total trainable parameters: {model_params:,}")
    
    # Adjusted training settings for smaller model
    optimizer = Adam(model.parameters(), lr=0.0001)  # Increased base learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.005,  # Increased max learning rate
        steps_per_epoch=60000,
        epochs=1,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=10.0
    )
    
    criterion = nn.CrossEntropyLoss()
    train_loader = data_file.trainDataAndLoad()
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
        # Print batch statistics
        if total % 2000 == 0:
            print(f'\nLoss: {running_loss/2000:.4f}, '
                  f'Running Accuracy: {100 * correct/total:.2f}%')
            running_loss = 0.0
    
    accuracy = correct / total
    print(f"\nFinal Training Accuracy: {accuracy:.4f}")
    
    return accuracy, model_params

if __name__ == "__main__":
    train_model()