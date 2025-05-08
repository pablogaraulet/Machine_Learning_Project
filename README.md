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
```

## How to Run

You can train the model and explore the data using either the Jupyter notebook: `main.ipynb` or the command line. Below are the steps to run the project.

### 1. Train the MLP and CNN Models

To train the MLP and CNN models on **both MNIST and EMNIST**, run the following commands:

```bash
python train_mlp.py
python train_cnn.py
```

This will:

- Automatically download the MNIST and EMNIST datasets (if not already present).
- Train an MLP model and a CNN model for each dataset.
- Save the trained models as `mlp_mnist.pt`, `mlp_emnist.pt`, `cnn_mnist.pt`, and `cnn_emnist.pt`.

### 2. Testing and Evaluation

You can test the trained models using the provided test scripts:

```bash
python test_mlp.py
python test_cnn.py
```

This will:
- Evaluate the MLP and CNN models on the test data.
- Output classification metrics and save predictions/results for further analysis.

### 3. Exploring and Visualizing the Data

Open the Jupyter notebook to explore the data, visualize samples, and run experiments. In the notebook, you will find examples of:
- Loading and visualizing images
- Preprocessing the data
- Creating DataLoaders for PyTorch
- Training and evaluating both MLP and CNN models
- Comparing model performance

### 4. Model Structure

The implemented MLP has:
- An input layer for 784-pixel vectors (28x28 images)
- Two hidden layers with ReLU activation and Dropout
- An output layer adjustable to the number of classes (10 for MNIST, 47 for EMNIST)

### 5. Utilities

The `data_utils.py` file contains functions to automatically load and prepare the data, making it easy to use the MNIST and EMNIST datasets.

### 6. Model Evaluation and Metrics

After running the test scripts, you can further analyze the predictions and model performance using the provided evaluation script:

```bash
python evaluate.py
```

This script will:

- Load the saved predictions and true labels from the results directory for both MLP and CNN models on MNIST and EMNIST.
- Print detailed classification metrics such as confusion matrices and classification reports.
- Help you compare the performance of different models and datasets in a consistent way.

Use this script after testing to get a comprehensive view of your models' strengths and weaknesses.