# MyMLib 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Framework](https://img.shields.io/badge/Framework-Custom_Deep_Learning_%26_ML-orange.svg)](#mytorch-deep-learning-framework)
**MyMLib** is an end-to-end, from-scratch Machine Learning and Deep Learning library written in Python and NumPy. Built primarily for educational clarity, architectural research, and deep technical understanding, **MyMLib** features a custom automatic differentiation (autograd) engine, PyTorch-like neural network modules, optimizers, and classic machine learning algorithms (Decision Trees, Random Forests, PCA, Linear Models).
---
##  Highlights
-  **`mytorch` Autograd Engine**: Dynamic reverse-mode automatic differentiation engine with DAG-based computational graph tracking.
-  **Modular Neural Network API**: Flexible `Module`, `Linear`, `Con1d/2d/3d`, `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `Flatten`, `ReLU`, `Sigmoid`, and `CrossEntropy` / `MSE` loss functions.
-  **Optimizers**: Standard Stochastic Gradient Descent (`SGD`) and Adaptive Moment Estimation (`Adam`).
-  **Tree Algorithms**: Custom implementations of Gini/Entropy `DecisionTree_Classification`, MSE `DecisionTree_Regression`, and ensemble `RandomForest` classifiers and regressors.
-  **Dimensionality Reduction**: Principal Component Analysis (`PCA`) and Kernel PCA (`KPCA`).
-  **Preprocessing & Utilities**: Cross-validation `KFold`, `Data_split`, and dataset loaders (`load_spambase`).
---
##  Repository 
```text
mymlib/
├── my_ml/
│   ├── datasets/               # Dataset loaders (e.g., spambase)
│   ├── decomposition/          # Dimensionality reduction (PCA, KPCA)
│   ├── linear_models/          # Classification (Logistic) & Regression (Ridge, Bayes)
│   ├── mytorch/                # Deep Learning Framework
│   │   ├── autograd/           # Computational graph DAG, ops, & backprop engine
│   │   ├── nn/                 # Layers (Conv, Pooling, Linear, Activations, Losses)
│   │   ├── optim/              # Optimizers (SGD, Adam)
│   │   └── tensor/             # Tensor data structure & autograd wrapper
│   ├── preprocessing/          # Data splitting & K-Fold cross validation
│   ├── trees/                  # Decision Trees & Random Forests
│   └── notes.md                # Package export design notes
├── train_cnn.py                # CNN training script on synthetic image dataset
├── train_cnn_batched.py        # Batched CNN training implementation
└── test_cnn_fix.py             # CNN unit test and backprop verification
```
---
## 💻 Quickstart & Examples
### 1. Training a Custom CNN (`mytorch`)
`train_cnn.py` demonstrates constructing and training a Convolutional Neural Network from scratch using `mytorch`.
```python
import numpy as np
from my_ml.mytorch.tensor.tensor import tensor
from my_ml.mytorch.nn import Con2d, MaxPool2d, Flatten, Linear, ReLU, CrossEntropy, Module
from my_ml.mytorch.optim import SGD
class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv = Con2d(in_channel=1, out_channel=4, kernel_size=3, padding=1, stride=1)
        self.relu1 = ReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = Flatten()
        self.fc1 = Linear(64, 16)
        self.relu2 = ReLU()
        self.fc2 = Linear(16, 2)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        return self.fc2(x)
# Instantiate model & optimizer
model = SimpleCNN()
optimizer = SGD(model.parameters(), lr=0.02)
criterion = CrossEntropy()
```
Run the demo script:
```bash
python train_cnn.py
```
---
### 2. Autograd & Tensor Computations
```python
from my_ml.mytorch.tensor.tensor import tensor
# Initialize tensors with requires_grad=True
x = tensor([2.0, 3.0], requires_grad=True)
y = tensor([4.0, 5.0], requires_grad=True)
# Forward pass
z = (x ** 2) + (3 * y)
loss = z.sum()
# Backward pass
loss.backward()
print("x gradient:", x.grad)
print("y gradient:", y.grad)
```
---
### 3. Decision Trees & Random Forests
```python
from my_ml.trees import DecisionTree_Classification, RandomForestClassification
from my_ml.preprocessing import Data_split
# Train Decision Tree
tree = DecisionTree_Classification(max_depth=5, criterion="gini")
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
# Train Random Forest Ensemble
rf = RandomForestClassification(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
```
---
### 4. Dimensionality Reduction (PCA & Kernel PCA)
```python
from my_ml.decomposition import PCA, KPCA
# Standard Principal Component Analysis
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
# Kernel PCA with RBF kernel
kpca = KPCA(n_components=2, kernel="rbf", gamma=15.0)
X_kpca = kpca.fit_transform(X)
```
---
## 🛠️ Package Modules Overview
|
 Subpackage 
|
 Modules / Classes 
|
 Description 
|
|
---
|
---
|
---
|
|
**
`mytorch.nn`
**
|
`Linear`
, 
`Con1d/2d/3d`
, 
`MaxPool1d/2d/3d`
, 
`AvgPool1d/2d/3d`
, 
`Flatten`
, 
`ReLU`
, 
`Sigmoid`
, 
`CrossEntropy`
, 
`MSE`
|
 Neural network layers, activations, and loss functions 
|
|
**
`mytorch.optim`
**
|
`SGD`
, 
`Adam`
|
 Gradient-based parameter optimizers 
|
|
**
`mytorch.autograd`
**
|
`engine`
, 
`node`
, 
`ops`
|
 Dynamic Reverse-Mode Autograd engine & node DAG tracking 
|
|
**
`trees`
**
|
`DecisionTree_Classification`
, 
`DecisionTree_Regression`
, 
`RandomForestClassification`
, 
`RandomForestRegression`
|
 Tree-based models supporting Gini, Entropy, and MSE split criteria 
|
|
**
`linear_models`
**
|
`LogisticRegression`
, 
`RidgeRegression`
, 
`BayesianRegression`
|
 Classic supervised linear learning models 
|
|
**
`decomposition`
**
|
`PCA`
, 
`KPCA`
|
 Dimensionality reduction techniques 
|
|
**
`preprocessing`
**
|
`KFold`
, 
`Data_split`
|
 Data splitting and evaluation utilities 
|
|
**
`datasets`
**
|
`load_spambase`
|
 Built-in dataset loading functions 
|
---
