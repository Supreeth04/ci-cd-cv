from model_train import train_model
from model import CustomNeuralNetwork, count_parameters

def main():
    print("Starting MNIST Model Training and Evaluation")
    print("-" * 50)
    
    accuracy, param_count = train_model()
    
    print("\nFinal Results:")
    print(f"Parameter Count: {param_count:,}")
    print(f"Training Accuracy: {accuracy:.4f}")
    print("\nRequirements Check:")
    print(f"Parameters < 25000: {'✓' if param_count < 25000 else '✗'}")
    print(f"Accuracy > 95%: {'✓' if accuracy > 0.95 else '✗'}")

if __name__ == "__main__":
    main()