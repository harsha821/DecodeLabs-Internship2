
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline



df = pd.read_csv(r"C:\Users\harsh\Downloads\creditcard.csv")
print("="*60)
print("DATASET INFORMATION")
print("="*60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True)*100)


plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()



scaler = StandardScaler()

df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"] = scaler.fit_transform(df[["Time"]])



X = df.drop("Class", axis=1)
y = df["Class"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)


logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=5000))
])

logistic_params = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l2']
}

print("\nTraining Logistic Regression...")

logistic_grid = GridSearchCV(
    logistic_pipeline,
    logistic_params,
    scoring='recall',
    cv=5,
    n_jobs=-1
)

logistic_grid.fit(X_train, y_train)

print("\nBest Logistic Parameters:")
print(logistic_grid.best_params_)



log_pred = logistic_grid.predict(X_test)
log_prob = logistic_grid.predict_proba(X_test)[:, 1]



print("\n" + "="*60)
print("LOGISTIC REGRESSION RESULTS")
print("="*60)

print("Precision :", precision_score(y_test, log_pred))
print("Recall    :", recall_score(y_test, log_pred))
print("F1 Score  :", f1_score(y_test, log_pred))
print("ROC-AUC   :", roc_auc_score(y_test, log_prob))

print("\nClassification Report:\n")
print(classification_report(y_test, log_pred))

cm = confusion_matrix(y_test, log_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

RocCurveDisplay.from_predictions(y_test, log_prob)
plt.title("Logistic Regression ROC Curve")
plt.show()



rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5]
}

print("\nTraining Random Forest...")

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    scoring='recall',
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)



rf_pred = rf_grid.predict(X_test)
rf_prob = rf_grid.predict_proba(X_test)[:, 1]



print("\n" + "="*60)
print("RANDOM FOREST RESULTS")
print("="*60)

print("Precision :", precision_score(y_test, rf_pred))
print("Recall    :", recall_score(y_test, rf_pred))
print("F1 Score  :", f1_score(y_test, rf_pred))
print("ROC-AUC   :", roc_auc_score(y_test, rf_prob))

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

RocCurveDisplay.from_predictions(y_test, rf_prob)
plt.title("Random Forest ROC Curve")
plt.show()



best_rf = rf_grid.best_estimator_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance":
    best_rf.named_steps["classifier"].feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(10)
)
plt.title("Top 10 Important Features")
plt.show()



results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Precision": [
        precision_score(y_test, log_pred),
        precision_score(y_test, rf_pred)
    ],
    "Recall": [
        recall_score(y_test, log_pred),
        recall_score(y_test, rf_pred)
    ],
    "F1 Score": [
        f1_score(y_test, log_pred),
        f1_score(y_test, rf_pred)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, log_prob),
        roc_auc_score(y_test, rf_prob)
    ]
})

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

print(results)



if roc_auc_score(y_test, rf_prob) > roc_auc_score(y_test, log_prob):
    best_model = rf_grid.best_estimator_
    model_name = "Random Forest"
else:
    best_model = logistic_grid.best_estimator_
    model_name = "Logistic Regression"

print("\nBest Model:", model_name)


joblib.dump(best_model, "fraud_detection_model.pkl")

print("\nModel Saved Successfully!")



loaded_model = joblib.load("fraud_detection_model.pkl")

print("Saved Model Loaded Successfully!")


sample = X_test.iloc[[0]]

prediction = loaded_model.predict(sample)

probability = loaded_model.predict_proba(sample)

print("\nPrediction :", prediction[0])

print("Fraud Probability :",
      round(probability[0][1] * 100, 2), "%")

if prediction[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Legitimate Transaction")

