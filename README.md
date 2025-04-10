# Pothole Detection Using Computer Vision and Deep Learning

This project aims to detect potholes in road images using a Convolutional Neural Network (CNN) model built on top of MobileNetV2. The system classifies images into two categories: normal roads and pothole-affected roads. This project uses TensorFlow and Keras for model building and training.

## Project Overview

The project involves:
1. **Data Preprocessing**: Image augmentation and normalization.
2. **Model Architecture**: Transfer learning using MobileNetV2, followed by fine-tuning.
3. **Performance Evaluation**: Classification accuracy, loss curves, F1 score, confusion matrix, precision-recall curve, and ROC curve.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

```bash
git clone <repository_url>
cd pothole-detection
```