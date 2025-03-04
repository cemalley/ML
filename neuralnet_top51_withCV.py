# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:08:57 2025

@author: malleyce
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

# -------------------------
# Data Loading and Preprocessing
# -------------------------
file_path = "C:/Users/malleyce/Documents/Metabolomics/AREDS/Data_release4_person.csv"
data = pl.read_csv(file_path).to_pandas()
data = data.dropna()

# Read the top 51 metabolites
top51 = pl.read_csv('C:/Users/malleyce/Documents/Metabolomics/AREDS/Release4/AREDS-AREDS2-comparison/AREDS_AREDS2_shared.csv').to_pandas()

# Keep only columns in top51 + relevant covariates
#columns_to_keep = top51["CHEM_ID_NEW"].tolist() + ["PARENT_SAMPLE_NAME", "LateAMD_person", "edu","trt","smoked","BMI","age","male"]

# OR keep only top51...
#columns_to_keep = top51["CHEM_ID_NEW"].tolist() + ["PARENT_SAMPLE_NAME", "LateAMD_person"]

# OR keep only clinical vars:
columns_to_keep = ["PARENT_SAMPLE_NAME", "LateAMD_person", "edu","trt","smoked","BMI","age","male"]



data = data[[col for col in columns_to_keep if col in data.columns]]

# One-hot encode categorical variables (if they exist)
data = pd.get_dummies(data, columns=['edu','trt','smoked'], dtype=int)
data = data.dropna()

# -------------------------
# Prepare Features, Target, and Groups
# -------------------------
target_column = 'LateAMD_person'
X = data.drop(columns=['PARENT_SAMPLE_NAME', target_column])
y = data[target_column]
groups = data["PARENT_SAMPLE_NAME"]

# Initialize the random undersampler (we'll apply it on the training set only)
rus = RandomUnderSampler(random_state=42)

# StandardScaler instance (fit later on training folds)
scaler = StandardScaler()

# -------------------------
# Group K-Fold Cross Validation
# -------------------------
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

accuracy_scores = []
auc_scores = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"Fold {fold+1}")
    
    # Split the data using indices
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Rebalance the training set only
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    
    # Standardize features: fit scaler on training data and transform both training and test data
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the neural network
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    nn.fit(X_train_bal, y_train_bal)
    
    # Evaluate model performance
    y_pred = nn.predict(X_test_scaled)
    y_probs = nn.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    
    accuracy_scores.append(acc)
    auc_scores.append(auc)
    
    print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 40)

print("Average Accuracy:", np.mean(accuracy_scores))
print("Average AUC:", np.mean(auc_scores))

# with both metabolites and demography:
# Average Accuracy: 0.7323869775618379
# Average AUC: 0.8219015101436616

# with only metabolites:
# Average Accuracy: 0.6680177883319615
# Average AUC: 0.742082949992111

# with only demography/clinical variables:
# Average Accuracy: 0.7607581044232876
# Average AUC: 0.8664509105236359
