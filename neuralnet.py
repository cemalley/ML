# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:29:56 2025

@author: Claire Weber
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns


# Load the dataset using Polars and convert to pandas
file_path = "C:/Users/malleyce/Documents/Metabolomics/AREDS/Data_release4_person.csv"
data = pl.read_csv(file_path)
data = data.to_pandas()

# Define features and target variable
target_column = 'LateAMD_person'
X = data.drop(columns=['PARENT_SAMPLE_NAME', 'CLIENT_SAMPLE_ID', 'TIME_POINT', 'age', 'male', 'edu', 'smoked', 'BMI', 'trt', 'trt_ax', 'trt_zc', 'trt_axzc', 'trt_pl','AvgSevScale', 'AnyAMD_person', 'IntAMD_person','yearnum','end', target_column])
y = data[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features (important for Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn.fit(X_train, y_train)

# Compute Permutation Importance
perm_importance = permutation_importance(nn, X_test, y_test, scoring="accuracy", n_repeats=30, random_state=42)

# Convert feature importance results into a DataFrame
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": perm_importance.importances_mean})
importance_df.sort_values(by="Importance", ascending=False, inplace=True)

# merge with the chem info

chemical_names = pl.read_csv('C:/Users/malleyce/Documents/Metabolomics/AREDS/chemical_names.csv').to_pandas()

merged_df = importance_df.merge(chemical_names, left_on='Feature', right_on='CHEM_ID_NEW', how='left')

merged_df.to_csv('NeuralNetwork_permutation_importance.csv',index=False)

# get AUC

y_probs = nn.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc:.4f}") # 0.8669.

#get predicted labels
y_pred = nn.predict(X_test)

# compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# 1754;22
# 98; 55

# Compute ROC curve points
fpr, tpr, _ = roc_curve(y_test, y_probs)


# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Late AMD", "Late AMD"], yticklabels=["No Late AMD", "Late AMD"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
