#!/usr/bin/env python
# coding: utf-8

# # Notebook 3 - Modeling binding affinity to protein receptors

# By Vincent Blay, November 2021

# This notebook was developed using **RPReactor 3.8** kernel on [jprime.lbl.gov](https://gpu2.ese.lbl.gov/).

# In this notebook, we demonstrate the use of MACAW embeddings to model binding affinity to a protein receptor of pharmacological interest. MACAW embeddings are then applied to identify promising candidate molecules in a custom virtual library.

# In[416]:


import sys
import os

# Completely suppress stderr output
sys.stderr = open(os.devnull, 'w')

# Now import everything
import warnings
warnings.filterwarnings('ignore')


# In[417]:


import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score

#Needed to show molecules
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole 

import sys
sys.path.append('../')

import macaw
print(macaw.__version__)
from macaw import *

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['figure.dpi'] = 96
matplotlib.rcParams['savefig.dpi'] = 600

get_ipython().run_line_magic('run', '"./plotting.py"')


# In[418]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ## 1. Regression Target: pPotency

# ### 1.1 Data preparation

# In[419]:


dataDir = '/mnt/data.ese/nfs/users/sghosh6/DTRA_project/MACAW/DrugDesignData/'
modelBuildingDataDir = os.path.join(dataDir, 'modelBuildingData/')
resultsDir = os.path.join(dataDir, 'Results/')


# In[420]:


allVirusData_chEMBL_wMACAW = pd.read_csv(modelBuildingDataDir + 'allVirusData_chEMBL_uM_wMACAW.csv')


# In[421]:


allVirusData_chEMBL_wMACAW.columns


# ### Filter columns

# In[422]:


allVirusData_chEMBL_wMACAW = allVirusData_chEMBL_wMACAW.filter(
    items=["molecule_chembl_id", "smiles", "value_uM", "pPotency", "Virus"]
)
allVirusData_chEMBL_wMACAW


# ### Add some randomization

# In[423]:


allVirusData_chEMBL_wMACAW = allVirusData_chEMBL_wMACAW.sample(frac=1, random_state=42).reset_index(drop=True)


# ### Rename columns

# In[424]:


allVirusData_chEMBL_wMACAW = allVirusData_chEMBL_wMACAW.rename(columns={
    "molecule_chembl_id": "compound_id",
    "smiles": "Smiles",
    "Virus" : "VirusClassifier",
})


# ### Compound Distribution Across Viruses

# In[425]:


virus_counts = allVirusData_chEMBL_wMACAW['VirusClassifier'].value_counts().sort_values(ascending=False)
total_compounds = len(allVirusData_chEMBL_wMACAW)

print(f"\n{'Virus':<30} {'Count':>10} {'Percentage':>12}")
print("-"*70)

for virus, count in virus_counts.items():
    percentage = (count / total_compounds) * 100
    print(f"{virus:<30} {count:>10} {percentage:>11.2f}%")

print("-"*70)
print(f"{'TOTAL':<30} {total_compounds:>10} {100.0:>11.2f}%")
print(f"\nNumber of unique viruses: {len(virus_counts)}")


# ### Add `ID` column to left

# In[426]:


allVirusData_chEMBL_wMACAW.insert(0, 'ID', range(1, len(allVirusData_chEMBL_wMACAW) + 1))
allVirusData_chEMBL_wMACAW.head()


# ### Add another filer to keep only these four columns: `ID`, `compound_id`,	`Smiles`, `pPotency`, `VirusClassifier`

# In[427]:


allVirusData_chEMBL_wMACAW = allVirusData_chEMBL_wMACAW.filter(
    items=["ID", "compound_id", "Smiles", "pPotency", "VirusClassifier"]
)
allVirusData_chEMBL_wMACAW.head()


# ### Remove NaN values in `pPotency` before cross-validation

# In[428]:


print("Original shape:", allVirusData_chEMBL_wMACAW.shape)

# Drop rows where pPotency is NaN
allVirusData_chEMBL_wMACAW = allVirusData_chEMBL_wMACAW.dropna(subset=['pPotency'])

print("Shape after dropping rows with NaN in 'pPotency':", allVirusData_chEMBL_wMACAW.shape)


# ### Clip first 1000 rows from the data set for fast testing
# Sample 100 random rows for testing
allVirusData_chEMBL_wMACAW = allVirusData_chEMBL_wMACAW.sample(n=1000, random_state=42).reset_index(drop=True)

print("Shape after sampling 100 rows:", allVirusData_chEMBL_wMACAW.shape)
# ### Remove duplicate compounds
# Print shape before removing duplicates
print("Before removing duplicates:", allVirusData_chEMBL_wMACAW.shape)

# Remove duplicates based on 'compound_id'
allVirusData_chEMBL_wMACAW = allVirusData_chEMBL_wMACAW.drop_duplicates(subset='compound_id')

# Print shape after removing duplicates
print("After removing duplicates:", allVirusData_chEMBL_wMACAW.shape)
# In[429]:


# df = allVirusData_chEMBL_wMACAW
Y = allVirusData_chEMBL_wMACAW.pPotency
smiles = allVirusData_chEMBL_wMACAW.Smiles


# In[430]:


print(len(smiles))


# ### Normalize the potency data
# $$
# \begin{align}
# \text{value (M)} &= \text{value}_{\mu M} \times 10^{-6} \\
# p\text{Potency} &= -\log_{10}(\text{value (M)}) \\
# &= 6 - \log_{10}(\text{value}_{\mu M})
# \end{align}
# $$
# 
# so, $$ p\text{Potency} = 6 - \log_{10}(\text{value}_{\mu M}) $$

# In[431]:


plot_histogram(Y, xlabel="pPotency")


# Define the partitions for cross-validation.

# In[432]:


num_of_partitions = 10
kf = KFold(n_splits=num_of_partitions, shuffle=True, random_state=42)


# Define hyperparameters for SVR:

# In[433]:


param_grid = {
    'C': [1, 5, 7, 10, 30, 50, 100, 300, 500], 
    'epsilon': [0.1, 0.3, 1, 3, 5, 10, 20],
    'kernel': ['rbf']
}


# Define MACAW embedding:

# In[516]:


mcw = MACAW(
    type_fp='atompairs', 
    metric='sokal', 
    n_components=15, 
    n_landmarks=200, 
    random_state=39
)


# After cleaning NaN values, reset the indices

# In[435]:


valid_mask = Y.notna()
smiles = smiles[valid_mask].reset_index(drop=True)  
Y = Y[valid_mask].reset_index(drop=True)            

print(f"Cleaned data has: {len(smiles)} samples")
print(f"Y index: {Y.index[:10].tolist()}")  


# ### ML predictions with Test only results

# In[436]:


get_ipython().run_cell_magic('time', '', '\nY_cv_pred = []\nY_obs = []\n\ni = 1\nfor train_index, val_index in kf.split(smiles):\n    print(f"Partition {i}/{num_of_partitions}")\n    i+=1\n    smi_train , smi_val = smiles.iloc[train_index], smiles.iloc[val_index]\n    y_train , y_val = Y[train_index], Y[val_index]\n\n    # Compute MACAW embeddings\n    mcw.fit(smi_train, y_train)\n\n    X_train = mcw.transform(smi_train)\n    X_val = mcw.transform(smi_val)\n\n    # Train the SVR model\n    # Optimize hyperparameters\n    grid = GridSearchCV(SVR(), param_grid, cv=5, refit=True)\n    grid.fit(X_train, y_train)\n#     print(grid.best_params_)\n\n    # Test set predictions\n    y_cv_pred = grid.predict(X_val)\n\n    # Save corresponding validation instances\n    Y_cv_pred.extend(y_cv_pred)\n    Y_obs.extend(y_val)\n')


# In[437]:


# Parity plot
parity_plot(x=Y_obs,
            y=Y_cv_pred, 
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'virus/allVirus_CV_testOnly.svg',
            save_formats=['svg', 'png'])  # Specify both formats


# ### ML predictions with Train + Test results

# In[438]:


import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Storage for predictions
Y_cv_pred_train = []
Y_obs_train = []
Y_cv_pred_test = []
Y_obs_test = []

best_params_per_fold = []

i = 1
for train_index, test_index in kf.split(smiles):
    print(f"Partition {i}/{num_of_partitions}")

    # Split data
    smi_train, smi_test = smiles.iloc[train_index], smiles.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Compute MACAW embeddings
    mcw.fit(smi_train, y_train)

    X_train = mcw.transform(smi_train)
    X_test = mcw.transform(smi_test)

    # Optimize hyperparameters
    grid = GridSearchCV(SVR(), param_grid, cv=5, refit=True)
    grid.fit(X_train, y_train)

    best_params_per_fold.append(grid.best_params_)

    # ===== PREDICT ON BOTH TRAIN AND TEST =====
    # Train set predictions
    y_cv_pred_train = grid.predict(X_train)

    # Test set predictions
    y_cv_pred_test = grid.predict(X_test)

    # Store train predictions
    Y_cv_pred_train.extend(y_cv_pred_train)
    Y_obs_train.extend(y_train)

    # Store test predictions
    Y_cv_pred_test.extend(y_cv_pred_test)
    Y_obs_test.extend(y_test)

    # Print fold metrics
    train_mae = mean_absolute_error(y_train, y_cv_pred_train)
    test_mae = mean_absolute_error(y_test, y_cv_pred_test)
    print(f"  Fold {i} - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")

    i += 1

# Convert to numpy arrays
Y_cv_pred_train = np.array(Y_cv_pred_train)
Y_obs_train = np.array(Y_obs_train)
Y_cv_pred_test = np.array(Y_cv_pred_test)
Y_obs_test = np.array(Y_obs_test)

# ===== OVERALL METRICS =====
print("\n" + "-"*40)
print("Cross Validation Metrics")
print("-"*40)

# Train metrics
train_mae = mean_absolute_error(Y_obs_train, Y_cv_pred_train)
train_rmse = np.sqrt(mean_squared_error(Y_obs_train, Y_cv_pred_train))
train_r2 = r2_score(Y_obs_train, Y_cv_pred_train)

print(f"Train Set:")
print(f"  MAE:  {train_mae:.2f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  R²:   {train_r2:.2f}")

# Test metrics
test_mae = mean_absolute_error(Y_obs_test, Y_cv_pred_test)
test_rmse = np.sqrt(mean_squared_error(Y_obs_test, Y_cv_pred_test))
test_r2 = r2_score(Y_obs_test, Y_cv_pred_test)

print(f"Test Set:")
print(f"  MAE:  {test_mae:.2f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  R²:   {test_r2:.2f}")


# In[439]:


parity_plot(x=Y_obs_train, 
            y=Y_cv_pred_train, 
            x_test=Y_obs_test, 
            y_test=Y_cv_pred_test, 
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'virus/allVirus_CV_TrTs.svg',
            save_formats=['svg', 'png'])


# Generate a model trained on the whole data set, to be used for prediction tasks

# In[440]:


mcw = MACAW(
    type_fp='atompairs', 
    metric='sokal', 
    n_components=15, 
    n_landmarks=200, 
    random_state=39
)
mcw.fit(smiles, Y)


# In[441]:


X_all = mcw.transform(smiles)
X_all.shape


# In[442]:


# Optimize hyperparameters
regr_pred = GridSearchCV(SVR(), param_grid, cv=5, refit=True)
regr_pred.fit(X_all, Y)
print(regr_pred.best_params_)

# Train set predictions
y_pred = regr_pred.predict(X_all)
print(f"R^2 = {r2_score(y_pred, Y):0.2f}")


# ### Using Random Forest Regressor with train + test split

# Define MACAW embedding:

# In[522]:


mcw = MACAW(
    type_fp='atompairs', 
    metric='sokal', 
    n_components=15, 
    n_landmarks=200, 
    random_state=42
)


# In[525]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for RandomForest
param_grid_RandomForest = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Storage results
Y_cv_pred_train_rf = []
Y_obs_train_rf = []
Y_cv_pred_test_rf = []
Y_obs_test_rf = []
best_params_rf_per_fold = []

i = 1
for train_index, test_index in kf.split(smiles):
    print(f"Partition {i}/{num_of_partitions}")

    # Split data
    smi_train, smi_test = smiles.iloc[train_index], smiles.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Compute MACAW embeddings
    mcw.fit(smi_train, y_train)
    X_train = mcw.transform(smi_train)
    X_test = mcw.transform(smi_test)

    # GridSearch for RandomForest
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_RandomForest, cv=5, refit=True, n_jobs=-1)
    #grid_rf = GridSearchCV(SVR(),param_grid=param_grid,scoring='neg_mean_absolute_error',cv=5,n_jobs=-1,refit=True,verbose=0)
    grid_rf.fit(X_train, y_train)

    best_params_rf_per_fold.append(grid_rf.best_params_)

    # ===== PREDICT ON BOTH TRAIN AND TEST =====
    # Train set predictions
    y_cv_pred_train = grid_rf.predict(X_train)

    # Test set predictions
    y_cv_pred_test = grid_rf.predict(X_test)

    # Store train predictions
    Y_cv_pred_train_rf.extend(y_cv_pred_train)
    Y_obs_train_rf.extend(y_train)

    # Store test predictions
    Y_cv_pred_test_rf.extend(y_cv_pred_test)
    Y_obs_test_rf.extend(y_test)

    # Print fold metrics
    train_mae = mean_absolute_error(y_train, y_cv_pred_train)
    test_mae = mean_absolute_error(y_test, y_cv_pred_test)
    print(f"  Fold {i} - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")

    i += 1


# In[526]:


# Convert to numpy arrays
Y_cv_pred_train_rf = np.array(Y_cv_pred_train_rf)
Y_obs_train_rf = np.array(Y_obs_train_rf)
Y_cv_pred_test_rf = np.array(Y_cv_pred_test_rf)
Y_obs_test_rf = np.array(Y_obs_test_rf)

# ===== OVERALL METRICS =====
print("\n" + "-"*40)
print("RandomForest Cross Validation Metrics")
print("-"*40)

# Train metrics
train_mae_rf = mean_absolute_error(Y_obs_train_rf, Y_cv_pred_train_rf)
train_rmse_rf = np.sqrt(mean_squared_error(Y_obs_train_rf, Y_cv_pred_train_rf))
train_r2_rf = r2_score(Y_obs_train_rf, Y_cv_pred_train_rf)

print(f"Train Set:")
print(f"  MAE:  {train_mae_rf:.2f}")
print(f"  RMSE: {train_rmse_rf:.2f}")
print(f"  R²:   {train_r2_rf:.2f}")

# Test metrics
test_mae_rf = mean_absolute_error(Y_obs_test_rf, Y_cv_pred_test_rf)
test_rmse_rf = np.sqrt(mean_squared_error(Y_obs_test_rf, Y_cv_pred_test_rf))
test_r2_rf = r2_score(Y_obs_test_rf, Y_cv_pred_test_rf)

print(f"Test Set:")
print(f"  MAE:  {test_mae_rf:.2f}")
print(f"  RMSE: {test_rmse_rf:.2f}")
print(f"  R²:   {test_r2_rf:.2f}")

print(f"\n[Info] Best params per fold (sample): {best_params_rf_per_fold[:3]}")


# In[527]:


# Parity plot for RandomForest showing both train and test
parity_plot(x=Y_obs_train_rf,
            y=Y_cv_pred_train_rf, 
            x_test=Y_obs_test_rf,
            y_test=Y_cv_pred_test_rf,
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'virus/allVirus_CV_Reg_RandomForest.svg',
            save_formats=['svg', 'png'])


# ### Using Gradient Boosting (XGBoost) with train + test split 

# In[541]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Storage results
Y_cv_pred_train_xgb = []
Y_obs_train_xgb = []
Y_cv_pred_test_xgb = []
Y_obs_test_xgb = []
best_params_xgb_per_fold = []

i = 1
for train_index, test_index in kf.split(smiles):
    print(f"Partition {i}/{num_of_partitions}")

    # Split data
    smi_train, smi_test = smiles.iloc[train_index], smiles.iloc[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Compute MACAW embeddings
    mcw.fit(smi_train, y_train)
    X_train = mcw.transform(smi_train)
    X_test = mcw.transform(smi_test)

    # GridSearch for XGBoost
    grid_xgb = GridSearchCV(
        XGBRegressor(random_state=42, tree_method='hist'),  # 'hist' is faster
        param_grid_xgb, 
        cv=5, 
        refit=True, 
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )
    grid_xgb.fit(X_train, y_train)

    best_params_xgb_per_fold.append(grid_xgb.best_params_)

    # Predict on both train and test
    y_cv_pred_train = grid_xgb.predict(X_train)
    y_cv_pred_test = grid_xgb.predict(X_test)

    # Store predictions
    Y_cv_pred_train_xgb.extend(y_cv_pred_train)
    Y_obs_train_xgb.extend(y_train)
    Y_cv_pred_test_xgb.extend(y_cv_pred_test)
    Y_obs_test_xgb.extend(y_test)

    # Print fold metrics
    train_mae = mean_absolute_error(y_train, y_cv_pred_train)
    test_mae = mean_absolute_error(y_test, y_cv_pred_test)
    print(f"  Fold {i} - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")

    i += 1


# In[542]:


# Convert to numpy arrays
Y_cv_pred_train_xgb = np.array(Y_cv_pred_train_xgb)
Y_obs_train_xgb = np.array(Y_obs_train_xgb)
Y_cv_pred_test_xgb = np.array(Y_cv_pred_test_xgb)
Y_obs_test_xgb = np.array(Y_obs_test_xgb)

# Overall metrics
print("\n" + "-"*40)
print("XGBoost Cross Validation Metrics")
print("-"*40)

train_mae_xgb = mean_absolute_error(Y_obs_train_xgb, Y_cv_pred_train_xgb)
train_rmse_xgb = np.sqrt(mean_squared_error(Y_obs_train_xgb, Y_cv_pred_train_xgb))
train_r2_xgb = r2_score(Y_obs_train_xgb, Y_cv_pred_train_xgb)

print(f"Train Set:")
print(f"  MAE:  {train_mae_xgb:.2f}")
print(f"  RMSE: {train_rmse_xgb:.2f}")
print(f"  R²:   {train_r2_xgb:.2f}")

test_mae_xgb = mean_absolute_error(Y_obs_test_xgb, Y_cv_pred_test_xgb)
test_rmse_xgb = np.sqrt(mean_squared_error(Y_obs_test_xgb, Y_cv_pred_test_xgb))
test_r2_xgb = r2_score(Y_obs_test_xgb, Y_cv_pred_test_xgb)

print(f"Test Set:")
print(f"  MAE:  {test_mae_xgb:.2f}")
print(f"  RMSE: {test_rmse_xgb:.2f}")
print(f"  R²:   {test_r2_xgb:.2f}")


# In[543]:


# Parity plot
parity_plot(x=Y_obs_train_xgb,
            y=Y_cv_pred_train_xgb, 
            x_test=Y_obs_test_xgb,
            y_test=Y_cv_pred_test_xgb,
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'virus/allVirus_CV_Reg_XGBoost.svg',
            save_formats=['svg', 'png'])


# ### Compare performance of different ML model
# 
# - SVR
# - Random Forest
# - XGBoost
# - LightGBM
# - CatBoost
# - Neural Network

# Import all necessary libraries

# In[ ]:


get_ipython().system(' pip install seaborn catboost xgboost lightgbm')


# In[551]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")


# Run model comparison

# In[557]:


use_cpu_cores = 4


# In[ ]:


"""
Multi-Model Comparison for Regression Analysis
Models: SVR, Random Forest, XGBoost, LightGBM, CatBoost, Neural Network
"""

# Parameter grids for hyperparameter tuning
param_grids = {
    'SVR': {
        'C': [1, 10, 100],
        'epsilon': [0.1, 1, 10],
        'kernel': ['rbf']
    },

    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },

    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    },

    'LightGBM': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63]
    },

    'CatBoost': {
        'iterations': [100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1]
    },

    'NeuralNetwork': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500]
    }
}

def train_and_evaluate_models(smiles, Y, kf, mcw, param_grids, verbose=True):
    """
    Train multiple regression models with cross-validation

    Parameters:
    -----------
    smiles : pd.Series
        SMILES strings
    Y : np.array
        Target values
    kf : KFold
        Cross-validation splitter
    mcw : MACAW
        MACAW embedding object
    param_grids : dict
        Parameter grids for each model
    verbose : bool
        Print progress

    Returns:
    --------
    results : dict
        Dictionary containing predictions for each model
    """

    model_names = ['SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNetwork']
    results = {}
    for name in model_names:
        results[name] = {
            'train_pred': [], 'train_obs': [], 
            'test_pred': [], 'test_obs': [], 
            'best_params': []
        }

    num_folds = kf.get_n_splits()

    for fold_id, (train_index, test_index) in enumerate(kf.split(smiles), 1):
        if verbose:
            print(f"\nFold {fold_id}/{num_folds}")
            print("-" * 60)

        smi_train = smiles.iloc[train_index].tolist()
        smi_test = smiles.iloc[test_index].tolist()
        y_train = Y[train_index]
        y_test = Y[test_index]

        mcw_fold = deepcopy(mcw)
        #mcw_fold.n_landmarks = min(mcw_fold.n_landmarks, len(smi_train))
        mcw_fold.fit(smi_train, y_train)

        X_train = mcw_fold.transform(smi_train)
        X_test = mcw_fold.transform(smi_test)

        if verbose:
            print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

        # SVR
        if verbose:
            print("Training SVR...")
        grid_svr = GridSearchCV(
            SVR(), param_grids['SVR'],
            cv=3, refit=True, n_jobs=use_cpu_cores,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid_svr.fit(X_train, y_train)
        results['SVR']['train_pred'].extend(grid_svr.predict(X_train))
        results['SVR']['train_obs'].extend(y_train)
        results['SVR']['test_pred'].extend(grid_svr.predict(X_test))
        results['SVR']['test_obs'].extend(y_test)
        results['SVR']['best_params'].append(grid_svr.best_params_)

        # Random Forest
        if verbose:
            print("Training Random Forest...")
        grid_rf = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=use_cpu_cores),
            param_grids['RandomForest'],
            cv=3, refit=True, n_jobs=use_cpu_cores,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid_rf.fit(X_train, y_train)
        results['RandomForest']['train_pred'].extend(grid_rf.predict(X_train))
        results['RandomForest']['train_obs'].extend(y_train)
        results['RandomForest']['test_pred'].extend(grid_rf.predict(X_test))
        results['RandomForest']['test_obs'].extend(y_test)
        results['RandomForest']['best_params'].append(grid_rf.best_params_)

        # XGBoost
        if verbose:
            print("Training XGBoost...")
        grid_xgb = GridSearchCV(
            XGBRegressor(random_state=42, tree_method='hist', verbosity=0),
            param_grids['XGBoost'],
            cv=3, refit=True, n_jobs=use_cpu_cores,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid_xgb.fit(X_train, y_train)
        results['XGBoost']['train_pred'].extend(grid_xgb.predict(X_train))
        results['XGBoost']['train_obs'].extend(y_train)
        results['XGBoost']['test_pred'].extend(grid_xgb.predict(X_test))
        results['XGBoost']['test_obs'].extend(y_test)
        results['XGBoost']['best_params'].append(grid_xgb.best_params_)

        # LightGBM
        if verbose:
            print("Training LightGBM...")
        grid_lgbm = GridSearchCV(
            LGBMRegressor(random_state=42, verbose=-1),
            param_grids['LightGBM'],
            cv=3, refit=True, n_jobs=use_cpu_cores,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid_lgbm.fit(X_train, y_train)
        results['LightGBM']['train_pred'].extend(grid_lgbm.predict(X_train))
        results['LightGBM']['train_obs'].extend(y_train)
        results['LightGBM']['test_pred'].extend(grid_lgbm.predict(X_test))
        results['LightGBM']['test_obs'].extend(y_test)
        results['LightGBM']['best_params'].append(grid_lgbm.best_params_)

        # CatBoost
        if verbose:
            print("Training CatBoost...")
        grid_cat = GridSearchCV(
            CatBoostRegressor(random_state=42, verbose=False),
            param_grids['CatBoost'],
            cv=3, refit=True, n_jobs=use_cpu_cores,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid_cat.fit(X_train, y_train)
        results['CatBoost']['train_pred'].extend(grid_cat.predict(X_train))
        results['CatBoost']['train_obs'].extend(y_train)
        results['CatBoost']['test_pred'].extend(grid_cat.predict(X_test))
        results['CatBoost']['test_obs'].extend(y_test)
        results['CatBoost']['best_params'].append(grid_cat.best_params_)

        # Neural Network
        if verbose:
            print("Training Neural Network...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        grid_nn = GridSearchCV(
            MLPRegressor(random_state=42, early_stopping=True),
            param_grids['NeuralNetwork'],
            cv=3, refit=True, n_jobs=use_cpu_cores,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid_nn.fit(X_train_scaled, y_train)
        results['NeuralNetwork']['train_pred'].extend(grid_nn.predict(X_train_scaled))
        results['NeuralNetwork']['train_obs'].extend(y_train)
        results['NeuralNetwork']['test_pred'].extend(grid_nn.predict(X_test_scaled))
        results['NeuralNetwork']['test_obs'].extend(y_test)
        results['NeuralNetwork']['best_params'].append(grid_nn.best_params_)

    for model_name in results:
        for key in ['train_pred', 'train_obs', 'test_pred', 'test_obs']:
            results[model_name][key] = np.array(results[model_name][key])

    return results

def calculate_metrics_table(results):
    """Calculate performance metrics for all models"""

    metrics_data = []
    model_order = ['SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNetwork']

    for model_name in model_order:
        data = results[model_name]

        train_mae = mean_absolute_error(data['train_obs'], data['train_pred'])
        train_rmse = np.sqrt(mean_squared_error(data['train_obs'], data['train_pred']))
        train_r2 = r2_score(data['train_obs'], data['train_pred'])

        test_mae = mean_absolute_error(data['test_obs'], data['test_pred'])
        test_rmse = np.sqrt(mean_squared_error(data['test_obs'], data['test_pred']))
        test_r2 = r2_score(data['test_obs'], data['test_pred'])

        metrics_data.append({
            'Model': model_name,
            'Train_R2': train_r2,
            'Train_MAE': train_mae,
            'Train_RMSE': train_rmse,
            'Test_R2': test_r2,
            'Test_MAE': test_mae,
            'Test_RMSE': test_rmse
        })

    df_metrics = pd.DataFrame(metrics_data)
    return df_metrics

def plot_individual_parity(results, save_dir):
    """Create individual parity plots for each model"""

    model_colors = {
        'SVR': '#1f77b4',
        'RandomForest': '#ff7f0e',
        'XGBoost': '#2ca02c',
        'LightGBM': '#d62728',
        'CatBoost': '#9467bd',
        'NeuralNetwork': '#8c564b'
    }

    model_order = ['SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNetwork']

    for model_name in model_order:
        data = results[model_name]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Train plot
        ax = axes[0]
        ax.scatter(data['train_obs'], data['train_pred'], 
                  alpha=0.5, s=30, color=model_colors[model_name], 
                  edgecolors='black', linewidth=0.5)

        min_val = min(data['train_obs'].min(), data['train_pred'].min())
        max_val = max(data['train_obs'].max(), data['train_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')

        r2 = r2_score(data['train_obs'], data['train_pred'])
        mae = mean_absolute_error(data['train_obs'], data['train_pred'])
        rmse = np.sqrt(mean_squared_error(data['train_obs'], data['train_pred']))

        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Observed pPotency', fontsize=12)
        ax.set_ylabel('Predicted pPotency', fontsize=12)
        ax.set_title(f'{model_name} - Training set', fontsize=13)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Test plot
        ax = axes[1]
        ax.scatter(data['test_obs'], data['test_pred'], 
                  alpha=0.5, s=30, color=model_colors[model_name], 
                  edgecolors='black', linewidth=0.5)

        min_val = min(data['test_obs'].min(), data['test_pred'].min())
        max_val = max(data['test_obs'].max(), data['test_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')

        r2 = r2_score(data['test_obs'], data['test_pred'])
        mae = mean_absolute_error(data['test_obs'], data['test_pred'])
        rmse = np.sqrt(mean_squared_error(data['test_obs'], data['test_pred']))

        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Observed pPotency', fontsize=12)
        ax.set_ylabel('Predicted pPotency', fontsize=12)
        ax.set_title(f'{model_name} - Test set', fontsize=13)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}parity_{model_name}.svg', format='svg', bbox_inches='tight', dpi=300)
        plt.savefig(f'{save_dir}parity_{model_name}.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_combined_parity(results, save_dir):
    """Create combined parity plot with all models"""

    model_colors = {
        'SVR': '#1f77b4',
        'RandomForest': '#ff7f0e',
        'XGBoost': '#2ca02c',
        'LightGBM': '#d62728',
        'CatBoost': '#9467bd',
        'NeuralNetwork': '#8c564b'
    }

    model_markers = {
        'SVR': 'o',
        'RandomForest': 's',
        'XGBoost': '^',
        'LightGBM': 'D',
        'CatBoost': 'v',
        'NeuralNetwork': 'p'
    }

    model_order = ['SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNetwork']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Train plot
    ax = axes[0]
    for model_name in model_order:
        data = results[model_name]
        ax.scatter(data['train_obs'], data['train_pred'],
                  alpha=0.4, s=25, 
                  color=model_colors[model_name],
                  marker=model_markers[model_name],
                  label=model_name,
                  edgecolors='black', linewidth=0.3)

    all_train_obs = np.concatenate([results[m]['train_obs'] for m in model_order])
    all_train_pred = np.concatenate([results[m]['train_pred'] for m in model_order])
    min_val = min(all_train_obs.min(), all_train_pred.min())
    max_val = max(all_train_obs.max(), all_train_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction', zorder=10)

    ax.set_xlabel('Observed pPotency', fontsize=13)
    ax.set_ylabel('Predicted pPotency', fontsize=13)
    ax.set_title('Training set - all models', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Test plot
    ax = axes[1]
    for model_name in model_order:
        data = results[model_name]
        r2 = r2_score(data['test_obs'], data['test_pred'])
        ax.scatter(data['test_obs'], data['test_pred'],
                  alpha=0.4, s=25,
                  color=model_colors[model_name],
                  marker=model_markers[model_name],
                  label=f'{model_name} (R²={r2:.3f})',
                  edgecolors='black', linewidth=0.3)

    all_test_obs = np.concatenate([results[m]['test_obs'] for m in model_order])
    all_test_pred = np.concatenate([results[m]['test_pred'] for m in model_order])
    min_val = min(all_test_obs.min(), all_test_pred.min())
    max_val = max(all_test_obs.max(), all_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction', zorder=10)

    ax.set_xlabel('Observed pPotency', fontsize=13)
    ax.set_ylabel('Predicted pPotency', fontsize=13)
    ax.set_title('Test set - all models', fontsize=14)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}parity_combined.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_dir}parity_combined.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_metrics_comparison(df_metrics, save_dir):
    """Create bar plots comparing metrics across models"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    models = df_metrics['Model'].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Train R2
    ax = axes[0, 0]
    bars = ax.bar(models, df_metrics['Train_R2'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Train R²', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Train MAE
    ax = axes[0, 1]
    bars = ax.bar(models, df_metrics['Train_MAE'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('Train MAE', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Train RMSE
    ax = axes[0, 2]
    bars = ax.bar(models, df_metrics['Train_RMSE'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Train RMSE', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Test R2
    ax = axes[1, 0]
    bars = ax.bar(models, df_metrics['Test_R2'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Test R²', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Test MAE
    ax = axes[1, 1]
    bars = ax.bar(models, df_metrics['Test_MAE'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('Test MAE', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Test RMSE
    ax = axes[1, 2]
    bars = ax.bar(models, df_metrics['Test_RMSE'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Test RMSE', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Model performance comparison', fontsize=15, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}metrics_comparison.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_dir}metrics_comparison.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_residuals(results, save_dir):
    """Create residual plots for all models"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    model_colors = {
        'SVR': '#1f77b4',
        'RandomForest': '#ff7f0e',
        'XGBoost': '#2ca02c',
        'LightGBM': '#d62728',
        'CatBoost': '#9467bd',
        'NeuralNetwork': '#8c564b'
    }

    model_order = ['SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNetwork']

    for idx, model_name in enumerate(model_order):
        data = results[model_name]
        ax = axes[idx]

        residuals = data['test_obs'] - data['test_pred']

        ax.scatter(data['test_pred'], residuals, 
                  alpha=0.5, s=30, 
                  color=model_colors[model_name],
                  edgecolors='black', linewidth=0.5)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero residual')

        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        ax.axhline(y=2*std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±2σ')
        ax.axhline(y=-2*std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Predicted pPotency', fontsize=11)
        ax.set_ylabel('Residuals', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        textstr = f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Residual plots - test set', fontsize=15, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}residuals.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_dir}residuals.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_train_test_comparison(df_metrics, save_dir):
    """Plot train vs test performance"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = df_metrics['Model'].values
    x = np.arange(len(models))
    width = 0.35

    colors_train = ['#6baed6', '#fd8d3c', '#74c476', '#fb6a4a', '#bcbddc', '#a1887f']
    colors_test = ['#08519c', '#d94801', '#238b45', '#a50f15', '#54278f', '#5d4037']

    # R2 comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, df_metrics['Train_R2'], width, 
                   label='Train', color=colors_train, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_metrics['Test_R2'], width,
                   label='Test', color=colors_test, alpha=0.8, edgecolor='black')
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('R² comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # MAE comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, df_metrics['Train_MAE'], width,
                   label='Train', color=colors_train, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_metrics['Test_MAE'], width,
                   label='Test', color=colors_test, alpha=0.8, edgecolor='black')
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # RMSE comparison
    ax = axes[2]
    bars1 = ax.bar(x - width/2, df_metrics['Train_RMSE'], width,
                   label='Train', color=colors_train, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_metrics['Test_RMSE'], width,
                   label='Test', color=colors_test, alpha=0.8, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Train vs test performance', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}train_test_comparison.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_dir}train_test_comparison.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()

def save_results(results, df_metrics, save_dir):
    """Save all predictions and metrics to CSV files"""

    df_metrics.to_csv(f'{save_dir}metrics_summary.csv', index=False)

    for model_name, data in results.items():
        df_test = pd.DataFrame({
            'Observed': data['test_obs'],
            'Predicted': data['test_pred'],
            'Residual': data['test_obs'] - data['test_pred'],
            'Absolute_Error': np.abs(data['test_obs'] - data['test_pred'])
        })
        df_test.to_csv(f'{save_dir}predictions_test_{model_name}.csv', index=False)

        df_train = pd.DataFrame({
            'Observed': data['train_obs'],
            'Predicted': data['train_pred'],
            'Residual': data['train_obs'] - data['train_pred'],
            'Absolute_Error': np.abs(data['train_obs'] - data['train_pred'])
        })
        df_train.to_csv(f'{save_dir}predictions_train_{model_name}.csv', index=False)

def generate_report(results, df_metrics, save_path):
    """Generate text summary report"""

    report = []
    report.append("Model comparison summary")
    report.append("=" * 80)
    report.append("")

    best_idx = df_metrics['Test_R2'].idxmax()
    best_model = df_metrics.iloc[best_idx]['Model']
    best_r2 = df_metrics.iloc[best_idx]['Test_R2']

    report.append(f"Best model (test R²): {best_model} (R² = {best_r2:.3f})")
    report.append("")

    report.append("Performance ranking (by test R²):")
    report.append("-" * 80)
    df_sorted = df_metrics.sort_values('Test_R2', ascending=False)
    for idx, row in df_sorted.iterrows():
        report.append(f"{idx+1}. {row['Model']:15s} - R²: {row['Test_R2']:.3f}, MAE: {row['Test_MAE']:.2f}, RMSE: {row['Test_RMSE']:.2f}")
    report.append("")

    report.append("Detailed metrics:")
    report.append("-" * 80)
    report.append(df_metrics.to_string(index=False))
    report.append("")

    report.append("Model analysis:")
    report.append("-" * 80)
    for model_name, data in results.items():
        train_r2 = r2_score(data['train_obs'], data['train_pred'])
        test_r2 = r2_score(data['test_obs'], data['test_pred'])
        overfit_gap = train_r2 - test_r2

        report.append(f"\n{model_name}:")
        report.append(f"  Train R²: {train_r2:.3f}")
        report.append(f"  Test R²: {test_r2:.3f}")
        report.append(f"  Overfitting gap: {overfit_gap:.3f}")

        if overfit_gap < 0.05:
            report.append(f"  Status: Excellent generalization")
        elif overfit_gap < 0.10:
            report.append(f"  Status: Good generalization")
        elif overfit_gap < 0.15:
            report.append(f"  Status: Moderate overfitting")
        else:
            report.append(f"  Status: High overfitting")

    report.append("")
    report.append("=" * 80)

    full_report = "\n".join(report)

    with open(save_path, 'w') as f:
        f.write(full_report)

    return full_report

# Main execution
if __name__ == "__main__":

    print("\nStarting model comparison")
    print("=" * 80)

    # Run model training and evaluation
    model_results = train_and_evaluate_models(
        smiles=smiles,
        Y=Y,
        kf=kf,
        mcw=mcw,
        param_grids=param_grids,
        verbose=True
    )

    print("\nCalculating metrics")
    df_comparison = calculate_metrics_table(model_results)

    print("\nMetrics summary:")
    print(df_comparison.to_string(index=False))

    print("\nGenerating plots")
    plot_individual_parity(model_results, resultsDir + 'virus/')
    plot_combined_parity(model_results, resultsDir + 'virus/')
    plot_metrics_comparison(df_comparison, resultsDir + 'virus/')
    plot_residuals(model_results, resultsDir + 'virus/')
    plot_train_test_comparison(df_comparison, resultsDir + 'virus/')

    print("\nSaving results")
    save_results(model_results, df_comparison, resultsDir + 'virus/')

    print("\nGenerating report")
    report = generate_report(model_results, df_comparison, resultsDir + 'virus/model_comparison_report.txt')
    print(report)

    print("\nAnalysis complete")
    print(f"Results saved to: {resultsDir}virus/")
    print("=" * 80)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 2. Discovery of new hits specific to all viruses (data source Enamine_antiviralsData.csv)

# In this section, we screen a custom virtual library looking for molecules that are promising accoring to the the SVR models `regr` above, which use 15-D MACAW embeddings as their input. The custom library ("Enamine_antiviralsData.csv") compiled from commercial catalogs by Enamine. In particular, we are interested in molecules with high predicted pPotency.

# In[443]:


EnamineAntiviralsData = pd.read_csv( modelBuildingDataDir + "Enamine_antiviralsData.csv")
print(EnamineAntiviralsData.shape)
EnamineAntiviralsData.head()


# In[444]:


smi_lib = EnamineAntiviralsData.Smiles


# Generate predictions for the H1 receptor:

# In[445]:


X1_lib = mcw.transform(smi_lib)

Y1_lib_pred = regr_pred.predict(X1_lib)


# Let us represent the predictions of both models:

# In[446]:


plt.figure(figsize=(4.7, 4.0), dpi=300)
plt.hist(Y1_lib_pred, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Predicted potency")
plt.ylabel("Number of compounds")
plt.title(f"Virtual screening of custom library ({len(smi_lib)} molecules)", pad=35)
plt.axvline(x=5, color='r', linestyle='--', linewidth=2, label='pPotency = 5.0 (minimum)') 
plt.axvline(x=6, color='g', linestyle='--', linewidth=2, label='pPotency = 6.0 (good)')
plt.axvline(x=7, color='b', linestyle='--', linewidth=2, label='pPotency = 7.0 (excellent)')

# Legend on top, outside plot area
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)

plt.grid(True, alpha=0.3)
plt.savefig(resultsDir + 'virus/allVirus_validation.svg', bbox_inches='tight', dpi=300)
plt.savefig(resultsDir + 'virus/allVirus_validation.png', bbox_inches='tight', dpi=300)
plt.show()


# Let us have a look at the compounds:

# In[447]:


# Define multiple priority levels
high_priority_idx = np.where(Y1_lib_pred >= 7)[0]
medium_priority_idx = np.where((Y1_lib_pred >= 6.0) & (Y1_lib_pred < 7.0))[0]
low_priority_idx = np.where((Y1_lib_pred >= 5.0) & (Y1_lib_pred < 6))[0]

print(f"High priority (pPotency ≥ 7.0): {len(high_priority_idx)} compounds")
print(f"Medium priority (6 ≤ pPotency < 7.0): {len(medium_priority_idx)} compounds")
print(f"Low priority (5 ≤ pPotency < 6): {len(low_priority_idx)} compounds")

# Use the one you need
idx = high_priority_idx  # or combine them


# In[448]:


EnamineAntiviralsData_final = EnamineAntiviralsData.iloc[idx].copy()
EnamineAntiviralsData_final['pPotency_prediction'] = Y1_lib_pred[idx]

EnamineAntiviralsData_final


# In[449]:


molecules = [Chem.MolFromSmiles(smi) for smi in EnamineAntiviralsData_final.Smiles[:50]]

Draw.MolsToGridImage(molecules, subImgSize=(200,200), molsPerRow=3, useSVG=True)


# # Build feature matrix X and targets (regression + classification)

# In[465]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score, balanced_accuracy_score
from copy import deepcopy


# In[466]:


yReg = allVirusData_chEMBL_wMACAW["pPotency"].to_numpy(dtype=np.float32)

labelEncoder = LabelEncoder()
yCls = labelEncoder.fit_transform(allVirusData_chEMBL_wMACAW["VirusClassifier"].astype(str).to_numpy())
virusClasses = labelEncoder.classes_
print(f"[Info] Virus classes: {len(virusClasses)}")


# Train/valid split

# In[467]:


XTrain, XValid, yRegTrain, yRegValid, yClsTrain, yClsValid = train_test_split(
    XAll, yReg, yCls, test_size=0.2, random_state=42, stratify=yCls
)


# Define the partitions for cross-validation.

# In[468]:


num_of_partitions = 10
kf = KFold(n_splits=num_of_partitions, shuffle=True, random_state=42)


# Define hyperparameters for SVR:

# In[469]:


param_grid_reg = {
    'C': [1, 5, 7, 10, 30, 50, 100, 300, 500],
    'epsilon': [0.1, 0.3, 1, 3, 5, 10, 20],
    'kernel': ['rbf']
}


# Define MACAW embedding

# In[470]:


mcw = MACAW(
    type_fp='atompairs', 
    metric='sokal', 
    n_components=15, 
    n_landmarks=200, 
    random_state=39
)


# Regression

# In[471]:


smiles = allVirusData_chEMBL_wMACAW["Smiles"].astype(str).reset_index(drop=True)
Y = allVirusData_chEMBL_wMACAW["pPotency"].to_numpy(dtype=np.float32)

# Ensure smiles is a pandas Series
if isinstance(smiles, np.ndarray):
    smiles = pd.Series(smiles)
if isinstance(Y, np.ndarray):
    Y = pd.Series(Y)

# Drop NaN or missing targets
validMaskY = Y.notna() & ~Y.isnull()
smiles = smiles[validMaskY].reset_index(drop=True)
Y = Y[validMaskY].reset_index(drop=True)

# Validate SMILES strings
validIdx = []
for i, s in enumerate(smiles):
    if isinstance(s, str) and len(s) > 0 and Chem.MolFromSmiles(s) is not None:
        validIdx.append(i)

smiles = smiles.iloc[validIdx].reset_index(drop=True)
Y = Y.iloc[validIdx].reset_index(drop=True)

print(f" Cleaned data has: {len(smiles)} valid samples after dropping NaN targets and invalid SMILES.")
print(f"Example indices: {Y.index[:10].tolist()}")


# In[472]:


YcvPred = []
Yobs = []
bestParamsPerFold = []

foldId = 1
for trainIndex, valIndex in kf.split(smiles):
    print(f"Partition {foldId}/{kf.get_n_splits()}")
    foldId += 1

    smiTrain, smiVal = smiles.iloc[trainIndex], smiles.iloc[valIndex]
    yTrain, yVal = Y[trainIndex], Y[valIndex]

    # --- fit MACAW on TRAIN only (clone to avoid contaminating global mcw) ---
    mcwFold = deepcopy(mcw)
    mcwFold.fit(smiTrain, yTrain)

    XTrain = mcwFold.transform(smiTrain)
    XVal = mcwFold.transform(smiVal)

    # --- inner CV for SVR hyperparams on the TRAIN split ---
    grid = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid_reg,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        refit=True
    )
    grid.fit(XTrain, yTrain)
    bestParamsPerFold.append(grid.best_params_)

    # --- predict the validation split ---
    yPred = grid.predict(XVal)

    # accumulate
    YcvPred.extend(yPred)
    Yobs.extend(yVal)


# ### Evaluate cross validation performance

# In[473]:


mae = mean_absolute_error(Yobs, YcvPred)
r2 = r2_score(Yobs, YcvPred)
print(f"[Nested-CV Regression] MAE: {mae:.3f} | R2: {r2:.3f}")
print(f"[Info] Best params per fold (sample): {bestParamsPerFold[:3]}")


# In[474]:


# Parity plot
parity_plot(x=Yobs,
            y=YcvPred, 
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'virus/allVirus_CV_Reg.svg',
            save_formats=['svg', 'png'])  # Specify both formats


# ### Generate a model trained on the whole data set, to be used for prediction tasks

# In[475]:


mcw = MACAW(
    type_fp='atompairs', 
    metric='sokal', 
    n_components=15, 
    n_landmarks=200, 
    random_state=39
)
mcw.fit(smiles, Y)


# In[476]:


X_all = mcw.transform(smiles)
X_all.shape


# In[477]:


# Fit MACAW on ALL data 
mcwAll = deepcopy(mcw)
mcwAll.fit(smiles, Y)
XAll = mcwAll.transform(smiles)

# Hyperparameter search on ALL data (SVR with 5-fold CV), refit best on ALL
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score

regrPred = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid_reg,  # same grid you defined earlier
    cv=5,
    refit=True,
    n_jobs=-1
)
regrPred.fit(XAll, Y)
print(f"[Refit-All] Best params: {regrPred.best_params_}")

# Train-set predictions and R² on ALL
yPredTrain = regrPred.predict(XAll)
print(f"[Refit-All] R^2 (train) = {r2_score(Y, yPredTrain):.2f}")


# In[478]:


# Parity plot
parity_plot(x=Y,
            y=yPredTrain, 
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'virus/allVirus_allData_Reg.svg',
            save_formats=['svg', 'png'])  


# # Classification: Cross Validation loop

# In[479]:


num_of_partitions = 10
kf = KFold(n_splits=num_of_partitions, shuffle=True, random_state=42)


# In[480]:


# For classification we use SVC with RBF; only C is tuned here (gamma='scale' default)
param_grid_cls = {
    'C': [1, 5, 7, 10, 30, 50, 100, 300, 500],
    'kernel': ['rbf']
}


# ### Classification with Morgan (ECFP6) + RandomForest

# In[481]:


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Helpers
# ---------------------------
def morganBits(smilesList, nBits=2048, radius=3):
    """Return (X, validMask) where X is [n_valid, nBits] uint8 array of ECFP bits."""
    fps = []
    validMask = []
    for s in smilesList:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None:
            validMask.append(False)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.uint8)
        # NOTE: RDKit DataStructs is optional here; ConvertToNumpyArray handles directly if imported.
        # To avoid extra import, we iterate bits:
        onBits = list(fp.GetOnBits())
        arr[onBits] = 1
        fps.append(arr)
        validMask.append(True)
    if len(fps) == 0:
        return np.zeros((0, nBits), dtype=np.uint8), np.array(validMask, dtype=bool)
    return np.vstack(fps), np.array(validMask, dtype=bool)

def safeInsert(df, loc, name, values):
    if name in df.columns:
        df.drop(columns=[name], inplace=True)
    df.insert(loc, name, values)

# ---------------------------
# Prepare data
# ---------------------------
DF = allVirusData_chEMBL_wMACAW  
assert {"Smiles", "VirusClassifier"}.issubset(DF.columns), "Expected Smiles and VirusClassifier in DF."

smilesSeries = DF["Smiles"].astype(str).reset_index(drop=True)
labelsStr = DF["VirusClassifier"].astype(str).reset_index(drop=True)

# Encode string labels to ints (for modeling), but we will report strings
labelEncoder = LabelEncoder()
yEncAll = labelEncoder.fit_transform(labelsStr)

# Compute Morgan fingerprints and filter invalids
XAllBits, validMask = morganBits(smilesSeries.tolist(), nBits=2048, radius=3)
yEncAllValid = yEncAll[validMask]
labelsStrValid = labelsStr[validMask].reset_index(drop=True)

if XAllBits.shape[0] < 10:
    raise RuntimeError(f"Too few valid molecules after featurization: {XAllBits.shape[0]}")

print(f"[Info] Valid training molecules: {XAllBits.shape[0]} / {len(smilesSeries)}")

# ---------------------------
# Nested CV (outer stratified K-fold, inner grid search)
# ---------------------------
numOfPartitions = 10
kfOuter = StratifiedKFold(n_splits=numOfPartitions, shuffle=True, random_state=42)

paramGridCls = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"]  # handle imbalance
}

YcvPredCls, YobsCls, bestParamsPerFold = [], [], []
foldId = 1
for trainIndex, valIndex in kfOuter.split(XAllBits, yEncAllValid):
    print(f"Partition {foldId}/{kfOuter.get_n_splits()}")
    foldId += 1

    XTrain, XVal = XAllBits[trainIndex], XAllBits[valIndex]
    yTrainEnc, yValEnc = yEncAllValid[trainIndex], yEncAllValid[valIndex]
    yValStr = labelEncoder.inverse_transform(yValEnc)

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gridCls = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=paramGridCls,
        scoring="f1_macro",
        cv=inner,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    gridCls.fit(XTrain, yTrainEnc)
    bestParamsPerFold.append(gridCls.best_params_)

    yPredEnc = gridCls.predict(XVal)
    yPredStr = labelEncoder.inverse_transform(yPredEnc)

    YcvPredCls.extend(yPredStr.tolist())
    YobsCls.extend(yValStr.tolist())


# ### Cross Validation metrics

# In[482]:


f1Cv  = f1_score(YobsCls, YcvPredCls, average="macro")
baCv  = balanced_accuracy_score(YobsCls, YcvPredCls)
accCv = accuracy_score(YobsCls, YcvPredCls)
print(f"[Nested-CV] Classification | F1-macro: {f1Cv:.3f} | Balanced Acc: {baCv:.3f} | Acc: {accCv:.3f}")
print(f"[Info] Best params (sample): {bestParamsPerFold[:3]}")


# ### Cross Validation Confusion Matrix

# In[486]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

yTrue = YobsCls   # true labels (strings)
yPred = YcvPredCls  # predicted labels (strings)

cm = confusion_matrix(yTrue, yPred, labels=np.unique(yTrue), normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(yTrue))
disp.plot(xticks_rotation='vertical', cmap='Blues', values_format='.2f')
plt.title("Confusion Matrix (Validation Predictions)")
plt.savefig(resultsDir + 'virus/allVirus_CV_ConMat.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig(resultsDir + 'virus/allVirus_CV_ConMat.png', format='png', bbox_inches='tight', dpi=300)
plt.show()


# ### Cross Validation Per-class precision/recall/F1 values

# In[487]:


from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

report = classification_report(yTrue, yPred, output_dict=True)
reportDF = pd.DataFrame(report).T.iloc[:-3]  # remove avg rows
reportDF[['precision','recall','f1-score']].plot(kind='bar', figsize=(10,4))
plt.title("Per-Class Precision/Recall/F1")
plt.ylabel("Score")
plt.ylim(0,1)
plt.savefig(resultsDir + 'virus/allVirus_CV_F1.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig(resultsDir + 'virus/allVirus_CV_F1.png', format='png', bbox_inches='tight', dpi=300)
plt.show()


# ### Refit on ALL valid data

# In[488]:


finalGrid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=paramGridCls,
    scoring="f1_macro",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    refit=True,
    verbose=0
)
finalGrid.fit(XAllBits, yEncAllValid)
finalCls = finalGrid.best_estimator_
print(f"[Refit-All] Best params: {finalGrid.best_params_}")


# ### In-sample metrics on ALL valid

# In[489]:


yAllPredEnc = finalCls.predict(XAllBits)
yAllPredStr = labelEncoder.inverse_transform(yAllPredEnc)
print(f"[Refit-All] F1-macro (train): {f1_score(labelsStrValid, yAllPredStr, average='macro'):.3f}")
print(f"[Refit-All] Balanced Acc (train): {balanced_accuracy_score(labelsStrValid, yAllPredStr):.3f}")
print(f"[Refit-All] Accuracy (train): {accuracy_score(labelsStrValid, yAllPredStr):.3f}")


# ### Confusion Matrix

# In[490]:


# --- Predictions on ALL (in-sample) ---
yPredEnc = finalCls.predict(XAllBits)

# Class labels (strings) for display
if 'labelEncoder' in globals():
    displayLabels = labelEncoder.classes_
    yTrueStr = labelEncoder.inverse_transform(yEncAllValid)
    yPredStr = labelEncoder.inverse_transform(yPredEnc)
else:
    # fallback: show encoded ints if no encoder is available
    displayLabels = finalCls.classes_
    yTrueStr = yEncAllValid
    yPredStr = yPredEnc


cm = confusion_matrix(yTrueStr, yPredStr, labels=displayLabels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=displayLabels)
fig, ax = plt.subplots(figsize=(7,6))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
ax.set_title("Confusion Matrix (Train/All Data)")
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
plt.tight_layout()
plt.show()


# In[491]:


# Confusion matrix (row-normalized by true class)
cm_norm = confusion_matrix(yTrueStr, yPredStr, labels=displayLabels, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=displayLabels)
fig, ax = plt.subplots(figsize=(7,6))
disp_norm.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True, values_format=".2f")
ax.set_title("Confusion Matrix (Row-Normalized)")
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
plt.tight_layout()
plt.savefig(resultsDir + 'virus/allVirus_fullData_ConMat.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig(resultsDir + 'virus/allVirus_fullData_ConMat.png', format='png', bbox_inches='tight', dpi=300)
plt.show()


# ### Per-class precision/recall/F1 values

# In[492]:


report = classification_report(yTrueStr, yPredStr, target_names=displayLabels, output_dict=True)
rep_df = pd.DataFrame(report).T.loc[displayLabels, ["precision", "recall", "f1-score"]]

ax = rep_df.plot(kind="bar", figsize=(10,4))
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Per-Class Precision / Recall / F1 (Train/All Data)")
plt.xticks(rotation=45, ha="right")
plt.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(resultsDir + 'virus/allVirus_fullData_F1.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig(resultsDir + 'virus/allVirus_fullData_F1.png', format='png', bbox_inches='tight', dpi=300)
plt.show()


# # Validation

# In[493]:


EnamineAntiviralsData = pd.read_csv( modelBuildingDataDir + "Enamine_antiviralsData.csv")
print(EnamineAntiviralsData.shape)
EnamineAntiviralsData.head()


# Keep SMILES as strings

# In[494]:


smi_enamine = EnamineAntiviralsData["Smiles"].astype(str).reset_index(drop=False)  # keeps 'index' column


# Validate SMILES with RDKit

# In[495]:


valid_mask = smi_enamine["Smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None).to_numpy()
valid_rows = smi_enamine.loc[valid_mask, "index"].to_numpy()
smi_valid  = smi_enamine.loc[valid_mask, "Smiles"].tolist()

print(f"[Enamine] valid molecular smiles: {len(smi_valid)} / {len(smi_enamine)}")


# Regression on pPotency data

# In[496]:


# MACAW features for Enamine (regression head)
X_enamine_reg = mcw.transform(smi_valid)

# Predict potency
Y_enamine_pred = regrPred.predict(X_enamine_reg)  # regrPred = best GridSearchCV or best estimator


# Plot the results

# In[497]:


plt.figure(figsize=(4.7, 4.0), dpi=300)
plt.hist(Y_enamine_pred, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Predicted potency")
plt.ylabel("Number of compounds")
plt.title(f"Virtual screening of custom library ({len(smi_lib)} molecules)", pad=35)
plt.axvline(x=5, color='r', linestyle='--', linewidth=2, label='pPotency = 5.0 (minimum)') 
plt.axvline(x=6, color='g', linestyle='--', linewidth=2, label='pPotency = 6.0 (good)')
plt.axvline(x=7, color='b', linestyle='--', linewidth=2, label='pPotency = 7.0 (excellent)')

# Legend on top, outside plot area
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)

plt.grid(True, alpha=0.3)
plt.savefig(resultsDir + 'virus/allVirus_validation_Reg.svg', bbox_inches='tight', dpi=300)
plt.savefig(resultsDir + 'virus/allVirus_validation_Reg.png', bbox_inches='tight', dpi=300)
plt.show()


# In[498]:


# Define multiple priority levels
high_priority_idx = np.where(Y_enamine_pred >= 7)[0]
medium_priority_idx = np.where((Y_enamine_pred >= 6.0) & (Y1_lib_pred < 7.0))[0]
low_priority_idx = np.where((Y_enamine_pred >= 5.0) & (Y1_lib_pred < 6))[0]

print(f"High priority (pPotency ≥ 7.0): {len(high_priority_idx)} compounds")
print(f"Medium priority (6 ≤ pPotency < 7.0): {len(medium_priority_idx)} compounds")
print(f"Low priority (5 ≤ pPotency < 6): {len(low_priority_idx)} compounds")

# Use the one you need
idx = high_priority_idx  # or combine them


# In[499]:


EnamineAntiviralsData_final_Reg = EnamineAntiviralsData.iloc[idx].copy()
EnamineAntiviralsData_final_Reg['pPotency_prediction'] = Y_enamine_pred[idx]

EnamineAntiviralsData_final_Reg


# In[500]:


molecules = [Chem.MolFromSmiles(smi) for smi in EnamineAntiviralsData_final_Reg.Smiles[:50]]

Draw.MolsToGridImage(molecules, subImgSize=(200,200), molsPerRow=3, useSVG=True)


# ### Classification task Enamine dataset

# In[501]:


from rdkit.Chem import AllChem

def morganBits(smilesList, nBits=2048, radius=3):
    X = np.zeros((len(smilesList), nBits), dtype=np.uint8)
    for i, s in enumerate(smilesList):
        m = Chem.MolFromSmiles(s)
        if m is None: 
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        X[i, list(fp.GetOnBits())] = 1
    return X

X_enamine_cls = morganBits(smi_valid, nBits=2048, radius=3)


# Predict class probabilities

# In[502]:


# Predict probabilities and top-1 / top-3
proba = finalCls.predict_proba(X_enamine_cls)       # [n_valid, n_classes]
top1 = np.argmax(proba, axis=1)

# classes as strings
classes_str = labelEncoder.classes_ if "labelEncoder" in globals() else finalCls.classes_
predVirus_valid = classes_str[top1]
predVirusConf_valid = proba[np.arange(len(top1)), top1]

# Top-3 labels + probs (nice for triage)
top3_idx = np.argsort(-proba, axis=1)[:, :3]
predVirusTop3_valid = []
predVirusTop3Proba_valid = []
for r in range(top3_idx.shape[0]):
    labels3 = classes_str[top3_idx[r]]
    probs3 = [f"{labels3[j]}:{proba[r, top3_idx[r][j]]:.2f}" for j in range(len(labels3))]
    predVirusTop3_valid.append(", ".join(labels3))
    predVirusTop3Proba_valid.append(", ".join(probs3))


# Assemble results in a table

# In[503]:


def safeInsert(df, loc, name, values):
    if name in df.columns:
        df.drop(columns=[name], inplace=True)
    df.insert(loc, name, values)

# Start with a copy of the valid subset rows
EnamineAntiviralsData_final_Cls = EnamineAntiviralsData.iloc[valid_rows].copy()

# Attach regression (potency)
safeInsert(EnamineAntiviralsData_final_Cls, 0, "pred_pPotency", np.round(Y_enamine_pred, 3))

# Attach classification (virus)
safeInsert(EnamineAntiviralsData_final_Cls, 1, "predVirus", predVirus_valid)
safeInsert(EnamineAntiviralsData_final_Cls, 2, "predVirusConf", np.round(predVirusConf_valid, 3))
safeInsert(EnamineAntiviralsData_final_Cls, 3, "predVirusTop3", predVirusTop3_valid)
safeInsert(EnamineAntiviralsData_final_Cls, 4, "predVirusTop3Proba", predVirusTop3Proba_valid)

# Optional: composite rank for prioritization
alpha, beta = 1.0, 0.5
rankScore = alpha * EnamineAntiviralsData_final_Cls["pred_pPotency"].to_numpy() + \
            beta  * EnamineAntiviralsData_final_Cls["predVirusConf"].to_numpy()
safeInsert(EnamineAntiviralsData_final_Cls, 5, "rankScore", np.round(rankScore, 3))

print("[Enamine] final shape:", EnamineAntiviralsData_final_Cls.shape)
EnamineAntiviralsData_final_Cls.head()


# Filter for pred_pPotency >= 7.0

# In[504]:


EnamineAntiviralsData_final_pPotency = EnamineAntiviralsData_final_Cls[
    EnamineAntiviralsData_final_Cls['pred_pPotency'] >= 7.0
]
EnamineAntiviralsData_final_pPotency


# In[505]:


molecules = [Chem.MolFromSmiles(smi) for smi in EnamineAntiviralsData_final_pPotency.Smiles[:50]]

Draw.MolsToGridImage(molecules, subImgSize=(200,200), molsPerRow=3, useSVG=True)


# Filter for rankScore >= 7.0
#  
#  **rankScore=α×pred_pPotency+β×predVirusConf** (α = 1.0, β = 0.5)
# 
# * ≥ 7.5	--> Very promising (strong potency + good confidence)
# * 7.0–7.5 -->	Potent hits, moderate confidence
# * 6.0–7.0 -->	Potentially interesting, lower confidence or moderate potency
# * < 6.0	--> Weak activity or uncertain classification

# In[506]:


EnamineAntiviralsData_final_rankScore = EnamineAntiviralsData_final_Cls[
    EnamineAntiviralsData_final_Cls['rankScore'] >= 7.0
]
EnamineAntiviralsData_final_rankScore


# In[507]:


molecules = [Chem.MolFromSmiles(smi) for smi in EnamineAntiviralsData_final_rankScore.Smiles[:50]]

Draw.MolsToGridImage(molecules, subImgSize=(200,200), molsPerRow=3, useSVG=True)


# Filter for rankScore >= 7.5 (best predicted molecule)

# In[508]:


EnamineAntiviralsData_final_rankScore_best = EnamineAntiviralsData_final_Cls[
    EnamineAntiviralsData_final_Cls['rankScore'] >= 7.5
]
EnamineAntiviralsData_final_rankScore_best


# In[510]:


EnamineAntiviralsData_final_rankScore_best.shape


# In[509]:


molecules = [Chem.MolFromSmiles(smi) for smi in EnamineAntiviralsData_final_rankScore_best.Smiles[:50]]

Draw.MolsToGridImage(molecules, subImgSize=(200,200), molsPerRow=3, useSVG=True)


# ### Extract molecular properties of the best predicted drug molecule

# In[511]:


bestDrugMoleculeSMILES = EnamineAntiviralsData_final_rankScore_best["Smiles"]

smilesList = bestDrugMoleculeSMILES.tolist()
print(f"Number of top molecules: {len(smilesList)}")


# Convert each SMILES to RDKit molecule objects

# In[512]:


from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

mols = [Chem.MolFromSmiles(smi) for smi in smilesList if Chem.MolFromSmiles(smi) is not None]
print(f"Successfully parsed {len(mols)} valid molecules out of {len(smilesList)}")
bestDrugMoleculeSMILES = Chem.MolToSmiles(mol)

molProps = []
for i, mol in enumerate(mols):
    props = {
        "index": i,
        "SMILES": Chem.MolToSmiles(mol),
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "QED": Descriptors.qed(mol),
    }
    molProps.append(props)

molPropsDF = pd.DataFrame(molProps)
molPropsDF.head()


# Add mordred properties

# In[513]:


from mordred import Calculator, descriptors

# Build a calculator for ALL 2D descriptors (ignore 3D)
calc = Calculator(descriptors, ignore_3D=True)

# This returns a pandas DataFrame aligned to the order of `mols`
mordredDF = calc.pandas(mols, quiet=True, nproc=4)

# Mordred can output non-numeric and NaN/inf
# Replace infs with NaN
mordredDF = mordredDF.replace([np.inf, -np.inf], np.nan)
# Drop columns that are entirely NaN
mordredDF = mordredDF.dropna(axis=1, how="all")
# Keep only numeric columns
numericCols = mordredDF.select_dtypes(include=[np.number]).columns
mordredNumDF = mordredDF[numericCols].copy()
# Optionally fill remaining NaNs with 0 
mordredNumDF = mordredNumDF.fillna(0.0)

print(f"Mordred: {mordredDF.shape[1]} total columns, {mordredNumDF.shape[1]} numeric columns kept.")
mordredNumDF.head()


# Combine all the molecular descriptors

# In[514]:


combinedDF = pd.concat(
    [molPropsDF.reset_index(drop=True), mordredNumDF.reset_index(drop=True)],
    axis=1
)
combinedDF


# In[ ]:




