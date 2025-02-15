# Lung Cancer Detection using Nature-Inspired Algorithm

## Overview

This project focuses on detecting lung cancer using deep learning techniques, specifically Convolutional Neural Networks (CNNs) optimized with a nature-inspired algorithm, Particle Swarm Optimization (PSO). The model is trained and evaluated on the **IQ-OTHNCCD lung cancer dataset**, which consists of **benign, malignant, and normal cases**.

## Features

- **Dataset Preprocessing:** Splits images into training and testing sets.
- **CNN Model:** Deep learning model built using TensorFlow/Keras.
- **PSO Optimization:** Hyperparameter tuning for CNN using Particle Swarm Optimization.
- **Performance Evaluation:** Accuracy, loss, confusion matrix, and classification report.
- **Prediction Pipeline:** Model inference on new lung cancer images.

## Dataset

The dataset consists of:

- **Benign cases:** 120 images (Training: 108, Testing: 12)
- **Malignant cases:** 561 images (Training: 504, Testing: 57)
- **Normal cases:** 416 images (Training: 374, Testing: 42)

## Model Architecture

The CNN architecture consists of:

1. **Three convolutional layers** with ReLU activation.
2. **MaxPooling layers** to downsample feature maps.
3. **Fully connected dense layers** for classification.
4. **Softmax activation** for multi-class classification.

### Hyperparameter Optimization using PSO

- Filters in Conv1 Layer: **~23 filters**
- Filters in Conv2 Layer: **~115 filters**
- Learning Rate: **~0.0028**

## Training and Performance

### Training Results

| Epoch | Training Accuracy | Validation Accuracy |
| ----- | ----------------- | ------------------- |
| 1     | 40.99%            | 68.18%              |
| 2     | 61.04%            | 66.67%              |
| 3     | 73.10%            | 79.29%              |
| 4     | 84.64%            | 94.44%              |
| 5     | 93.53%            | 87.37%              |
| 6     | 95.43%            | 95.45%              |

- **Final Test Accuracy:** **95.45%**
- **Final Train Accuracy:** **96.83%**

## Installation & Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Numpy
- Pandas

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/lung-cancer-detection.git
   cd lung-cancer-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run training:
   ```sh
   python train.py
   ```
4. Run prediction on new images:
   ```sh
   python predict.py --image path_to_image.jpg
   ```

## Results

- The confusion matrix and classification report are included in the results folder.
- Sample predictions are stored in `predictions.csv`.
- Training loss and accuracy plots are saved as `loss_plot.png` and `accuracy_plot.png`.

## Conclusion

This project demonstrates the effectiveness of **CNNs optimized using Particle Swarm Optimization** for lung cancer detection. The achieved accuracy of **95.45%** indicates a promising approach for automated medical diagnosis.

## Acknowledgments

Special thanks to the dataset providers and contributors to TensorFlow/Keras.

## License

MIT License
