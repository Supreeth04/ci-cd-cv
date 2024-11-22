# MNIST Classification with PyTorch

[![Model Tests](https://github.com/yourusername/mnist-classification/actions/workflows/build-pipeline.yml/badge.svg)](https://github.com/yourusername/mnist-classification/actions/workflows/build-pipeline.yml)

A lightweight CNN model for MNIST digit classification that achieves >95% accuracy in one epoch while maintaining parameter count under 25,000.

## Project Structure

```
├── model.py                    # Neural network architecture
├── model_train.py             # Training loop and optimization
├── data_file.py               # Data loading and preprocessing
├── test_model.py              # Testing functions
├── main.py                    # Main execution script
├── requirements.txt           # Project dependencies
└── .github/workflows/         # GitHub Actions workflow
```

## Model Architecture

- Input: 28x28 grayscale images
- Convolutional Layers:
  - Conv1: 4 channels, 3x3 kernel, padding=1
  - Conv2: 8 channels, 3x3 kernel, padding=1
  - MaxPool2d after each conv layer
- Fully Connected Layers:
  - FC1: 8*7*7 -> 32
  - FC2: 32 -> 10 (output)
- Dropout: 0.005 for regularization
- Total Parameters: ~14,410

## Requirements

```
torch==2.2.0
torchvision==0.17.0
tqdm
numpy<2
```

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run model training and testing:

```bash
python test_model.py
```

Run only training:

```bash
python main.py
```

## Model Constraints

- Parameter count < 25,000
- Training accuracy > 95% in one epoch
- Dataset: MNIST
- Single epoch training requirement

## Training Details

- Batch Size: 10
- Optimizer: Adam
- Learning Rate Schedule: OneCycleLR
  - Base LR: 0.0001
  - Max LR: 0.005
  - Steps per epoch: 6000
  - Warmup: 10% of training
  - Div Factor: 1.0
  - Final Div Factor: 10.0
- Gradient Clipping: max_norm=1.0
- Dropout Rate: 0.005
- Data Normalization: ((0.1307,), (0.3081,))
- Workers: 1
- Shuffle: True

## Features

- Efficient CNN architecture
- OneCycleLR for faster convergence
- Gradient clipping for training stability
- Minimal dropout for regularization
- GitHub Actions for automated testing

## Performance

- Parameter Count: 14,410
- Training Accuracy: >95%
- Training Time: ~5 minutes (CPU)
- Memory Usage: <1GB

## CI/CD

Automated testing via GitHub Actions:

- Parameter count verification (<25,000)
- Model accuracy validation (>95%)
- Training completion check
- Python version: 3.8
- Ubuntu latest runner

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- PyTorch framework
- MNIST dataset creators
- GitHub Actions

## Observations

### 1. Batch Size Impact on Accuracy

- As the batch_size reduced from 128 -> 100 -> 50 -> 10, the accuracy increased for the model
- Batch size progression:
  - 128: Lower accuracy
  - 100: Moderate improvement
  - 50: Better performance
  - 10: Best accuracy

**Assumption:** 

* As the model is seeing every image, I am pretty sure it is trying to memorise instead of learning from the image.
* Hence when the inferencing will be done, it will 100% give wrong answers.

### 2. Dropout's Effect on Performance

- As the dropout value decreased, the model's accuracy increased
- Progression:
  - 0.25: Lower accuracy
  - 0.2: Moderate improvement
  - 0.1: Better performance
  - 0.005: Best accuracy
- This suggests a trade-off between regularization and model performance
