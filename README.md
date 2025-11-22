# FairFace - Emotion Recognition using MobileNetV2

This project implements emotion recognition on the FER2013 dataset using a pretrained MobileNetV2 model.

## Dataset

The FER2013 dataset contains facial expression images categorized into 7 emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The dataset is organized in the `FER2013/` directory with `train/` and `test/` subdirectories, each containing folders for each emotion category.

## Model

The model uses **MobileNetV2** as a pretrained backbone, fine-tuned for emotion recognition. MobileNetV2 is a lightweight, efficient model that's perfect for real-time emotion recognition applications.

## Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on the FER2013 dataset:

```bash
python main.py
```

The training script will:
- Load and preprocess the FER2013 dataset
- Split the training data into train/validation (80/20)
- Train the MobileNetV2 model with data augmentation
- Save the best model based on validation accuracy
- Evaluate the final model on the test set

### Training Configuration

You can modify the following parameters in `main.py`:
- `BATCH_SIZE`: Batch size for training (default: 32)
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `IMG_SIZE`: Input image size (default: 224)

### Model Output

The training will save the best model to `best_emotion_model.pth` based on validation accuracy. The script will also print:
- Training and validation loss/accuracy per epoch
- Final test accuracy
- Classification report
- Confusion matrix

## Features

- **Pretrained MobileNetV2**: Uses ImageNet-pretrained weights for transfer learning
- **Data Augmentation**: Random horizontal flip, rotation, and color jitter during training
- **Learning Rate Scheduling**: Reduces learning rate on plateau
- **Comprehensive Evaluation**: Includes accuracy, classification report, and confusion matrix

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA-capable GPU (recommended but not required)
