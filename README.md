# Machine_Learning_Project

# MLP Image Classifier for MNIST and EMNIST

This project implements a Machine Learning pipeline using a **Multilayer Perceptron (MLP)** for image classification on the **MNIST** and **EMNIST** datasets. The goal is to train models capable of recognizing **handwritten digits and characters**.

---

## Project Structure

- `train_mlp.py` — Main script to train the MLP model on MNIST and EMNIST
- `mlp_model.py` — Defines the architecture of the MLP model
- `data_utils.py` — Utility functions for loading and preparing data
- `main.ipynb` — Jupyter notebook for data exploration and experimentation
- `train.csv`, `test.csv`, `sample_submission.csv` — Optional data files for competitions or custom testing

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn

## How to Run

You can train the model and explore the data using the Jupyter notebook: `main.ipynb`.


### 1. Exploring and Visualizing the Data

Open the Jupyter notebook to explore the data, visualize samples, and run experiments. In the notebook, you will find examples of:
- Loading and visualizing images
- Preprocessing the data
- Creating DataLoaders for PyTorch
- Training and evaluating the model

### 2. Model Structure

The implemented MLP has:
- An input layer for 784-pixel vectors (28x28 images)
- Two hidden layers with ReLU activation and Dropout
- An output layer adjustable to the number of classes (10 for MNIST, 47 for EMNIST)