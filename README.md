# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project focuses on building an end-to-end Fraud Detection Pipeline to identify fraudulent credit card transactions using supervised machine learning techniques. Since fraud datasets are highly imbalanced, the project applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes and improve model performance.

The pipeline includes data preprocessing, exploratory data analysis, feature scaling, class imbalance handling, model training, hyperparameter tuning, evaluation, model comparison, and model deployment through model serialization.

---

## Objectives

* Detect fraudulent credit card transactions.
* Handle severe class imbalance using SMOTE.
* Train multiple machine learning algorithms.
* Evaluate models using Precision, Recall, F1-Score, and ROC-AUC.
* Compare model performance and select the best model.
* Save the trained model for future predictions.

---

## Dataset

**Dataset Name:** Credit Card Fraud Detection Dataset

**Source:** Kaggle

The dataset contains transactions made by European cardholders over two days.

### Dataset Characteristics

* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Legitimate Transactions: 284,315
* Fraud Rate: 0.172%

### Features

| Feature Type | Description                                 |
| ------------ | ------------------------------------------- |
| Time         | Seconds elapsed between transactions        |
| V1-V28       | PCA-transformed confidential features       |
| Amount       | Transaction amount                          |
| Class        | Target Variable (0 = Legitimate, 1 = Fraud) |

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Joblib

---

## Project Workflow

```text
Credit Card Dataset
        │
        ▼
Data Loading
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Scaling
(Time & Amount)
        │
        ▼
Train-Test Split
        │
        ▼
SMOTE Oversampling
        │
        ▼
Model Training
(Logistic Regression)
(Random Forest)
        │
        ▼
Hyperparameter Tuning
(GridSearchCV)
        │
        ▼
Model Evaluation
        │
        ├── Precision
        ├── Recall
        ├── F1 Score
        └── ROC-AUC
        │
        ▼
Model Comparison
        │
        ▼
Best Model Selection
        │
        ▼
Model Saving (.pkl)
        │
        ▼
Fraud Prediction
```

---

## Data Preprocessing

### Missing Value Analysis

The dataset was checked for missing values to ensure data quality.

### Feature Scaling

The following features were standardized:

* Time
* Amount

StandardScaler was used to normalize the data.

### Class Imbalance Handling

The dataset contains very few fraud cases.

To overcome this issue:

* SMOTE was applied.
* Synthetic fraud samples were generated.
* The training dataset became balanced.

---

## Machine Learning Models

### Logistic Regression

A linear classification algorithm used as a baseline model.

Hyperparameters Tuned:

* C = [0.01, 0.1, 1, 10]
* Penalty = L2

---

### Random Forest Classifier

An ensemble learning algorithm using multiple decision trees.

Hyperparameters Tuned:

* Number of Trees = [100, 200]
* Maximum Depth = [10, 20]
* Minimum Samples Split = [2, 5]

---

## Evaluation Metrics

Accuracy was intentionally avoided because the dataset is highly imbalanced.

Instead, the following metrics were used:

### Precision

Measures how many predicted fraud transactions are actually fraud.

Formula:

Precision = TP / (TP + FP)

---

### Recall

Measures how many actual fraud transactions were detected.

Formula:

Recall = TP / (TP + FN)

---

### F1 Score

Harmonic mean of Precision and Recall.

Formula:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

### ROC-AUC Score

Measures the model's ability to distinguish between classes.

Range:

* 0.5 = Random Prediction
* 1.0 = Perfect Classification

---

## Visualizations

The project generates:

* Fraud vs Non-Fraud Distribution Plot
* Transaction Amount Distribution
* Confusion Matrix
* ROC Curve
* Feature Importance Plot
* Model Comparison Table

---

## Results

The trained models were compared using:

* Precision
* Recall
* F1 Score
* ROC-AUC

The model with the highest ROC-AUC score was selected as the final model.

Expected Performance:

| Model               | Precision   | Recall      | ROC-AUC     |
| ------------------- | ----------- | ----------- | ----------- |
| Logistic Regression | 0.85 - 0.95 | 0.85 - 0.95 | 0.96 - 0.98 |
| Random Forest       | 0.90 - 0.99 | 0.85 - 0.95 | 0.98 - 0.99 |

---

## Model Deployment

The best-performing model is saved using Joblib.

```python
joblib.dump(best_model, "fraud_detection_model.pkl")
```

To load the model:

```python
loaded_model = joblib.load("fraud_detection_model.pkl")
```

---

