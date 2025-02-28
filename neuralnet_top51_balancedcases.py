# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:11:35 2025

@author: malleyce
"""

# in this script: randomly select balanced numbers of late amd and non-late in areds and/or areds2 separately and rerun to predict late amd (on the 51 mets)
# doing it with areds here first.



import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler



# Load the dataset using Polars and convert to pandas
file_path = "C:/Users/malleyce/Documents/Metabolomics/AREDS/Data_release4_person.csv"
data = pl.read_csv(file_path)
data = data.to_pandas()

# remove rows with missing values, from missing BMI or edu
data = data.dropna()

# read the 51 mets of interest from cox ph models of areds and areds2 data.

top51 = pl.read_csv('C:/Users/malleyce/Documents/Metabolomics/AREDS/Release4/AREDS-AREDS2-comparison/AREDS_AREDS2_shared.csv')
top51 = top51.to_pandas()

# Keep only columns in top_chemicals + covariates
# Get the list of columns to keep
columns_to_keep = top51["CHEM_ID_NEW"].tolist() + ["PARENT_SAMPLE_NAME", "LateAMD_person"]

# Subset the data
data = data[[col for col in columns_to_keep if col in data.columns]]

# one-hot encode edu, trt, smoked...get_dummies will delete edu in the process.
#data = pd.get_dummies(data, columns=['edu','trt','smoked'], dtype=int)

# Define features and target variable
target_column = 'LateAMD_person'
X = data.drop(columns=['PARENT_SAMPLE_NAME', target_column])
y = data[target_column]

# randomly undersample the majority class to be the same size as minority class.
rus = RandomUnderSampler(random_state=42)
X, y = rus.fit_resample(X, y)

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
perm_importance = permutation_importance(nn, X_test, y_test, scoring="accuracy", n_repeats=30, random_state=42) # quick.

# Convert feature importance results into a DataFrame
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": perm_importance.importances_mean})
importance_df.sort_values(by="Importance", ascending=False, inplace=True)

# merge with the chem info

chemical_names = pl.read_csv('C:/Users/malleyce/Documents/Metabolomics/AREDS/chemical_names.csv').to_pandas()

merged_df = importance_df.merge(chemical_names, left_on='Feature', right_on='CHEM_ID_NEW', how='left')

merged_df.to_csv('C:/Users/malleyce/Documents/Metabolomics/AREDS/Release4/ML_models/NeuralNetwork_permutation_importance_using_ONLY_top51_noclinc_balancedclasses.csv',index=False)

# get AUC
y_probs = nn.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc:.4f}") # 0.7237

#get predicted labels
y_pred = nn.predict(X_test)

# compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
# [[112  59]
# [ 42 108]]


#accuracy
accuracy = accuracy_score(y_test,y_pred)
print(accuracy) # 0.6854


# Compute ROC curve points
fpr, tpr, _ = roc_curve(y_test, y_probs)

# couple plots.........
# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

