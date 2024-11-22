import torch
from model import CustomNeuralNetwork, count_parameters
from model_train import train_model

def test_model_parameters():
    model = CustomNeuralNetwork()
    param_count = count_parameters(model)
    print(f"\nTesting parameter count...")
    print(f"Model has {param_count:,} parameters")
    assert param_count < 25000, f"Model has {param_count:,} parameters, exceeding limit of 25,000"
    print("✓ Parameter count test passed!")
    return True

def test_model_accuracy():
    print("\nTesting model accuracy...")
    accuracy, _ = train_model()
    assert accuracy > 0.95, f"Model accuracy {accuracy:.4f} is below required 0.95"
    print(f"✓ Accuracy test passed! ({accuracy:.4f})")
    return True

if __name__ == "__main__":
    test_model_parameters()
    test_model_accuracy()