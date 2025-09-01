# Iris Dataset Classification

## 📌 Project Overview

This project demonstrates a simple **classification task using the Iris dataset**.
The Iris dataset is a classic dataset in machine learning containing 150 samples of iris flowers with 4 features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The goal is to classify the flowers into one of **three species**:

* Setosa
* Versicolor
* Virginica

## ⚙️ Workflow

1. **Load the Dataset**

   * Uses `sklearn.datasets.load_iris()` to load the data.

2. **Data Splitting**

   * Divides the dataset into **training (80%)** and **testing (20%)** sets.

3. **Feature Scaling**

   * Standardizes features using `StandardScaler` (important for distance-based models like KNN).

4. **Model Training**

   * Trains a **K-Nearest Neighbors (KNN)** classifier with `k=5`.

5. **Evaluation**

   * Predictions on test data
   * Metrics: Accuracy, Classification Report, Confusion Matrix

## 🛠️ Requirements

Install dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## 🚀 Usage

Run the script:

```bash
python iris_classification.py
```

Expected output:

* Model Accuracy
* Precision, Recall, F1-score per class
* Confusion Matrix

## 📊 Example Output

```
Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

## 📌 Future Improvements

* Try different models (Logistic Regression, SVM, Random Forest).
* Perform hyperparameter tuning for KNN (optimize `k`).
* Add decision boundary visualization.

