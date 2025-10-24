#!/usr/bin/env python
# coding: utf-8

# # Notebook - Find best Drug compound for Staphylococcus aureus bacteria

# In this notebook, we demonstrate the use of MACAW embeddings to model binding affinity to a protein receptor of pharmacological interest. MACAW embeddings are then applied to identify promising candidate molecules in a custom virtual library.

# In[2]:


import sys
import os

# Completely suppress stderr output
sys.stderr = open(os.devnull, 'w')

# Now import everything
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import os
import csv
import numpy as np
import pandas as pd


from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score

#Needed to show molecules
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole 
import seaborn as sns

import sys
sys.path.append('../')

import macaw
print(macaw.__version__)
from macaw import *

import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['figure.dpi'] = 96
matplotlib.rcParams['savefig.dpi'] = 600

get_ipython().run_line_magic('run', '"./plotting.py"')


# In[4]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ## 1. Regression Target: pPotency

# ### 1.1 Data preparation

# In[5]:


dataDir = '/mnt/data.ese/nfs/users/sghosh6/DTRA_project/MACAW/DrugDesignData/'
modelBuildingDataDir = os.path.join(dataDir, 'modelBuildingData/')
resultsDir = os.path.join(dataDir, 'Results/')


# In[6]:


allBacteriaData_chEMBL_wMACAW = pd.read_csv(modelBuildingDataDir + 'allBacteriaData_chEMBL_uM_wMACAW.csv')


# In[7]:


allBacteriaData_chEMBL_wMACAW


# ### Filter columns

# In[8]:


allBacteriaData_chEMBL_wMACAW = allBacteriaData_chEMBL_wMACAW.filter(
    items=["molecule_chembl_id", "smiles", "value_uM", "pPotency", "Virus"]
)
allBacteriaData_chEMBL_wMACAW


# ### Remove the trailing 'Data_chEMBL_combined.csv' from each entry in the 'Virus' column

# In[9]:


allBacteriaData_chEMBL_wMACAW['Virus'] = allBacteriaData_chEMBL_wMACAW['Virus'].str.replace('Data_chEMBL_combined.csv', '', regex=False)
allBacteriaData_chEMBL_wMACAW.rename(columns={'Virus': 'Bacteria'}, inplace=True)
allBacteriaData_chEMBL_wMACAW


# ### Keep only data for `Staphylococcus_aureus` bacteria

# In[10]:


StaphylococcusBacteriaData_chEMBL_wMACAW = allBacteriaData_chEMBL_wMACAW[allBacteriaData_chEMBL_wMACAW['Bacteria'] == 'Staphylococcus_aureus']
StaphylococcusBacteriaData_chEMBL_wMACAW


# ### Rename columns

# In[11]:


StaphylococcusBacteriaData_chEMBL_wMACAW = StaphylococcusBacteriaData_chEMBL_wMACAW.rename(columns={
    "molecule_chembl_id": "compound_id",
    "smiles": "Smiles",
    "Bacteria" : "BacteriaClassifier",
})
StaphylococcusBacteriaData_chEMBL_wMACAW.head()


# ### Compound Distribution Across Bacteriaes

# In[12]:


Bacteria_counts = StaphylococcusBacteriaData_chEMBL_wMACAW['BacteriaClassifier'].value_counts().sort_values(ascending=False)
total_compounds = len(StaphylococcusBacteriaData_chEMBL_wMACAW)

print(f"\n{'Bacteria':<30} {'Count':>10} {'Percentage':>12}")
print("-"*70)

for Bacteria, count in Bacteria_counts.items():
    percentage = (count / total_compounds) * 100
    print(f"{Bacteria:<30} {count:>10} {percentage:>11.2f}%")

print("-"*70)
print(f"{'TOTAL':<30} {total_compounds:>10} {100.0:>11.2f}%")
print(f"\nNumber of unique Bacteriaes: {len(Bacteria_counts)}")


# ### Add `ID` column to left

# In[13]:


StaphylococcusBacteriaData_chEMBL_wMACAW.insert(0, 'ID', range(1, len(StaphylococcusBacteriaData_chEMBL_wMACAW) + 1))
StaphylococcusBacteriaData_chEMBL_wMACAW.head()


# ### Add another filer to keep only these columns: `ID`, `compound_id`,	`Smiles`, `pPotency`, `BacteriaClassifier`

# In[14]:


StaphylococcusBacteriaData_chEMBL_wMACAW = StaphylococcusBacteriaData_chEMBL_wMACAW.filter(
    items=["ID", "compound_id", "Smiles", "pPotency", "BacteriaClassifier"]
)
StaphylococcusBacteriaData_chEMBL_wMACAW.head()


# ### Remove NaN, -ve and unphysical (>12) values in `pPotency` before cross-validation
# 
# | Range (pPotency) | IC₅₀/EC₅₀ (in M)          | Interpretation                                | Action              |
# | ---------------- | -------------------- | --------------------------------------------- | ------------------- |
# | **0–3**   | > 1×10⁻³ M (millimolar)     | Very weak or inactive compounds               | Drop         |
# | **3–10**  | 1×10⁻³ – 1×10⁻¹⁰ M          | Normal drug-like activity range               | **Keep**                |
# | **10–12** | 1×10⁻¹⁰ – 1×10⁻¹² M         | Extremely potent but still physically possible | **Keep (with caution)** |
# | **>12**   | < 1×10⁻¹² M (picomolar–femto) | Physically unrealistic / likely data error     | Drop                |
# 
# 

# In[15]:


print("Original shape:", StaphylococcusBacteriaData_chEMBL_wMACAW.shape)

StaphylococcusBacteriaData_chEMBL_wMACAW = StaphylococcusBacteriaData_chEMBL_wMACAW[StaphylococcusBacteriaData_chEMBL_wMACAW['pPotency'].notna() 
    & (StaphylococcusBacteriaData_chEMBL_wMACAW['pPotency'] > 0) & (StaphylococcusBacteriaData_chEMBL_wMACAW['pPotency'] < 12)]

print("Shape after cleaning 'pPotency' value:", StaphylococcusBacteriaData_chEMBL_wMACAW.shape)


# ### Remove duplicate compounds using Compute median potency per (Smiles, Bacteria) pair
# Print shape before removing duplicates
print("Before removing duplicates:", StaphylococcusBacteriaData_chEMBL_wMACAW.shape)

# Aggregate duplicates *within the same bacteria*
aggDF = (
    StaphylococcusBacteriaData_chEMBL_wMACAW.groupby(['Smiles', 'BacteriaClassifier'], as_index=False)
      .agg({'pPotency': 'median'})   # or 'mean' if you prefer
)

# Merge back non-target columns (like MACAW features)
StaphylococcusBacteriaData_chEMBL_wMACAW = (aggDF
      .merge(
          aggDF.drop(columns=['pPotency']),
          on=['Smiles', 'BacteriaClassifier'],
          how='left'
      )
      .drop_duplicates(subset=['Smiles', 'BacteriaClassifier'])
      .reset_index(drop=True)
)

# Print shape after removing duplicates
print("After removing duplicates:", StaphylococcusBacteriaData_chEMBL_wMACAW.shape)
# In[16]:


duplicates = StaphylococcusBacteriaData_chEMBL_wMACAW.duplicated(subset=['Smiles', 'BacteriaClassifier'])
print("Remaining duplicates:", duplicates.sum())


# In[17]:


# Get counts and percentages
counts = StaphylococcusBacteriaData_chEMBL_wMACAW['BacteriaClassifier'].value_counts()
percentages = counts / counts.sum() * 100

# Create bar plot
plt.figure(figsize=(10, 6))
bars = counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Number of Compounds per Bacteria')
plt.xlabel('Bacteria')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Annotate each bar with count and %
for i, (count, pct) in enumerate(zip(counts, percentages)):
    plt.text(
        i, count + 1,                 # position (x, y)
        f'{count}\n({pct:.1f}%)',     # label: count + percentage
        ha='center', va='bottom', fontsize=10
    )

plt.tight_layout()
plt.show()


# In[18]:


StaphylococcusBacteriaData_chEMBL_wMACAW.to_csv(os.path.join(modelBuildingDataDir, "StaphylococcusBacteriaData_chEMBL_wMACAW_MLready.csv"), index=False)


# ### pPotency value distribution across each bacteria

# In[19]:


plt.figure(figsize=(10, 6))

sns.violinplot(
    data=StaphylococcusBacteriaData_chEMBL_wMACAW,
    x='BacteriaClassifier',
    y='pPotency',
    palette='Set2',
    inner='box'
)

plt.xticks(rotation=45, ha='right')
plt.title('pPotency Distribution Across Bacteria')
plt.xlabel('Bacteria')
plt.ylabel('pPotency')
plt.tight_layout()
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.stripplot(
    data=StaphylococcusBacteriaData_chEMBL_wMACAW,
    x='BacteriaClassifier',
    y='pPotency',
    jitter=True,          # spreads the dots to avoid overlap
    palette='Set2',
    alpha=1
)
plt.xticks(rotation=45, ha='right')
plt.title('pPotency values per Bacteria')
plt.xlabel('Bacteria')
plt.ylabel('pPotency')
plt.tight_layout()
plt.show()


# ### Clip first 1000 rows from the data set for fast testing
# Sample 1000 random rows for testing
StaphylococcusBacteriaData_chEMBL_wMACAW = StaphylococcusBacteriaData_chEMBL_wMACAW.sample(n=1000, random_state=42).reset_index(drop=True)

print("Shape after sampling 1000 rows:", StaphylococcusBacteriaData_chEMBL_wMACAW.shape)
# ### Train ML Model

# In[21]:


# df = StaphylococcusBacteriaData_chEMBL_wMACAW
Y = StaphylococcusBacteriaData_chEMBL_wMACAW.pPotency
smiles = StaphylococcusBacteriaData_chEMBL_wMACAW.Smiles


# In[22]:


print(len(smiles))


# In[23]:


plot_histogram(Y, xlabel="pPotency")


# Define the partitions for cross-validation.

# In[24]:


num_of_partitions = 10
kf = KFold(n_splits=num_of_partitions, shuffle=True, random_state=42)


# Define hyperparameters for SVR:

# In[25]:


param_grid = {
    'C': [1, 5, 7, 10, 30, 50, 100, 300, 500], 
    'epsilon': [0.1, 0.3, 1, 3, 5, 10, 20],
    'kernel': ['rbf']
}


# Define MACAW embedding:

# In[26]:


mcw = MACAW(
    type_fp='atompairs', 
    metric='sokal', 
    n_components=15, 
    n_landmarks=200, 
    random_state=39
)


# After cleaning NaN values, reset the indices

# In[27]:


valid_mask = Y.notna()
smiles = smiles[valid_mask].reset_index(drop=True)  
Y = Y[valid_mask].reset_index(drop=True)            

print(f"Cleaned data has: {len(smiles)} samples")
print(f"Y index: {Y.index[:10].tolist()}")  


# ### Create directory to save plot

# In[28]:


save_path = resultsDir + 'StaphylococcusBacteria/StaphylococcusBacteria_CV_testOnly.svg'
os.makedirs(os.path.dirname(save_path), exist_ok=True)


# ### ML predictions with Test only results
%%time

Y_cv_pred = []
Y_obs = []

i = 1
for train_index, val_index in kf.split(smiles):
    print(f"Partition {i}/{num_of_partitions}")
    i+=1
    smi_train , smi_val = smiles.iloc[train_index], smiles.iloc[val_index]
    y_train , y_val = Y[train_index], Y[val_index]
    
    # Compute MACAW embeddings
    mcw.fit(smi_train, y_train)
    
    X_train = mcw.transform(smi_train)
    X_val = mcw.transform(smi_val)
    
    # Train the SVR model
    # Optimize hyperparameters
    grid = GridSearchCV(SVR(), param_grid, cv=5, refit=True, n_jobs=4, verbose=0, scoring='neg_mean_absolute_error', pre_dispatch='2*n_jobs')

    grid.fit(X_train, y_train)
#     print(grid.best_params_)
    
    # Test set predictions
    y_cv_pred = grid.predict(X_val)
    
    # Save corresponding validation instances
    Y_cv_pred.extend(y_cv_pred)
    Y_obs.extend(y_val)from joblib import Parallel, delayed


def process_fold(fold_id, train_index, val_index, smiles, Y, mcw, param_grid):
    """Process a single fold"""
    print(f"Partition {fold_id}/{num_of_partitions}")
    
    smi_train = smiles.iloc[train_index]
    smi_val = smiles.iloc[val_index]
    y_train = Y[train_index]
    y_val = Y[val_index]
    
    # Compute MACAW embeddings (clone mcw to avoid conflicts)
    from copy import deepcopy
    mcw_fold = deepcopy(mcw)
    mcw_fold.fit(smi_train, y_train)
    
    X_train = mcw_fold.transform(smi_train)
    X_val = mcw_fold.transform(smi_val)
    
    # Optimize hyperparameters
    grid = GridSearchCV(SVR(), param_grid, cv=5, refit=True, n_jobs=4, verbose=0, scoring='neg_mean_absolute_error', pre_dispatch='2*n_jobs')
    grid.fit(X_train, y_train)
    
    # Test set predictions
    y_cv_pred = grid.predict(X_val)
    
    return y_cv_pred, y_val

# Parallel execution
results = Parallel(n_jobs=4, verbose=10)(  # Use 10 CPU cores
    delayed(process_fold)(i+1, train_idx, val_idx, smiles, Y, mcw, param_grid)
    for i, (train_idx, val_idx) in enumerate(kf.split(smiles))
)

# Combine results
Y_cv_pred = []
Y_obs = []
for y_pred, y_val in results:
    Y_cv_pred.extend(y_pred)
    Y_obs.extend(y_val)# Parity plot
parity_plot(x=Y_obs,
            y=Y_cv_pred, 
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'StaphylococcusBacteria/StaphylococcusBacteria_CV_testOnly.svg',
            save_formats=['svg', 'png'])  # Specify both formats
# ### ML predictions with Train + Test results
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed
from copy import deepcopy

# ===== DEFINE FUNCTION FOR SINGLE FOLD =====
def process_fold(fold_id, train_index, test_index, smiles, Y, mcw, param_grid, num_partitions):
    """Process a single CV fold"""
    print(f"Partition {fold_id}/{num_partitions}")
    
    # Split data
    smi_train = smiles.iloc[train_index]
    smi_test = smiles.iloc[test_index]
    y_train = Y[train_index]
    y_test = Y[test_index]
    
    # Clone MACAW to avoid conflicts in parallel execution
    mcw_fold = deepcopy(mcw)
    mcw_fold.fit(smi_train, y_train)
    
    X_train = mcw_fold.transform(smi_train)
    X_test = mcw_fold.transform(smi_test)
    
    # Optimize hyperparameters
    grid = GridSearchCV(SVR(), param_grid, cv=5, refit=True, n_jobs=1)  # n_jobs=1 to avoid nested parallelism
    grid.fit(X_train, y_train)
    
    # Predict on both train and test
    y_cv_pred_train = grid.predict(X_train)
    y_cv_pred_test = grid.predict(X_test)
    
    # Calculate fold metrics
    train_mae = mean_absolute_error(y_train, y_cv_pred_train)
    test_mae = mean_absolute_error(y_test, y_cv_pred_test)
    print(f"  Fold {fold_id} - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
    
    # Return all results
    return {
        'train_pred': y_cv_pred_train,
        'train_obs': y_train,
        'test_pred': y_cv_pred_test,
        'test_obs': y_test,
        'best_params': grid.best_params_
    }


# ===== RUN PARALLEL CROSS-VALIDATION =====
print("Starting parallel cross-validation with parallel cores...\n")

results = Parallel(n_jobs=8, verbose=50)(
    delayed(process_fold)(
        fold_id=i+1,
        train_index=train_idx,
        test_index=test_idx,
        smiles=smiles,
        Y=Y,
        mcw=mcw,
        param_grid=param_grid,
        num_partitions=num_of_partitions
    )
    for i, (train_idx, test_idx) in enumerate(kf.split(smiles))
)


# ===== COMBINE RESULTS =====
Y_cv_pred_train = []
Y_obs_train = []
Y_cv_pred_test = []
Y_obs_test = []
best_params_per_fold = []

for fold_result in results:
    Y_cv_pred_train.extend(fold_result['train_pred'])
    Y_obs_train.extend(fold_result['train_obs'])
    Y_cv_pred_test.extend(fold_result['test_pred'])
    Y_obs_test.extend(fold_result['test_obs'])
    best_params_per_fold.append(fold_result['best_params'])

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

print(f"\nBest params per fold (sample): {best_params_per_fold[:3]}")
# In[29]:


commandLineData_train = pd.read_csv( resultsDir + 'StaphylococcusBacteriaCommandline_10fold/predictions_train_SVR.csv')
Y_obs_train = commandLineData_train['Observed'].values
Y_cv_pred_train = commandLineData_train['Predicted'].values


# In[30]:


commandLineData_test = pd.read_csv( resultsDir + 'StaphylococcusBacteriaCommandline_10fold/predictions_test_SVR.csv')
Y_obs_test = commandLineData_test['Observed'].values
Y_cv_pred_test = commandLineData_test['Predicted'].values


# In[31]:


parity_plot(x=Y_obs_train, 
            y=Y_cv_pred_train, 
            x_test=Y_obs_test, 
            y_test=Y_cv_pred_test, 
            xlabel="pPotency observations", 
            ylabel="Cross-validated predictions",
            savetitle=resultsDir + 'StaphylococcusBacteria/StaphylococcusBacteria_CV_TrTs.svg',
            save_formats=['svg', 'png'])


# Generate a model trained on the whole data set, to be used for prediction tasks

# In[32]:


mcw = MACAW(
    type_fp='atompairs', 
    metric='sokal', 
    n_components=15, 
    n_landmarks=200, 
    random_state=39
)
mcw.fit(smiles, Y)


# In[33]:


X_all = mcw.transform(smiles)
X_all.shape


# ### Optimize hyperparameters

# In[ ]:


regr_pred = GridSearchCV(SVR(), param_grid, cv=5, refit=True)
regr_pred.fit(X_all, Y)
print(regr_pred.best_params_)

# Train set predictions
y_pred = regr_pred.predict(X_all)
print(f"R^2 = {r2_score(y_pred, Y):0.2f}")


# ### Train/test split

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.svm import SVR\nfrom sklearn.metrics import r2_score, mean_absolute_error\nimport numpy as np\n\n# ---------------------------\n# 1. TRAIN-TEST SPLIT\n# ---------------------------\n# Since you\'re working with only one virus group, use regular train_test_split\nX_train_smi, X_test_smi, Y_train, Y_test = train_test_split(\n    smiles, Y, \n    test_size=0.2, \n    random_state=42,\n    shuffle=True\n)\n\nprint(f"Training samples: {len(X_train_smi)}")\nprint(f"Test samples: {len(X_test_smi)}")\n\n# ---------------------------\n# 2. FIT MACAW ON TRAINING DATA ONLY\n# ---------------------------\nmcw = MACAW(\n    type_fp=\'atompairs\', \n    metric=\'sokal\', \n    n_components=15, \n    n_landmarks=200, \n    random_state=39\n)\nmcw.fit(X_train_smi, Y_train)\n\n# Transform both train and test\nX_train = mcw.transform(X_train_smi)\nX_test = mcw.transform(X_test_smi)\n\nprint(f"Feature dimensions: {X_train.shape[1]}")\n\n# ---------------------------\n# 3. GRID SEARCH ON TRAINING DATA ONLY\n# ---------------------------\nn_jobs = 8\n\nregr_pred = GridSearchCV(\n    estimator=SVR(),\n    param_grid=param_grid,\n    cv=5,\n    refit=True,\n    n_jobs=n_jobs,\n    verbose=0,\n    scoring=\'neg_mean_absolute_error\',\n    return_train_score=True,\n    pre_dispatch=\'2*n_jobs\'\n)\n\nprint(f"\\nRunning GridSearchCV with {n_jobs} parallel jobs...")\nregr_pred.fit(X_train, Y_train)\n\n# ---------------------------\n# 4. DISPLAY BEST HYPERPARAMETERS\n# ---------------------------\nprint("\\n" + "-"*60)\nprint("Best Hyperparameters:")\nprint("-"*60)\nfor param, value in regr_pred.best_params_.items():\n    print(f"  {param}: {value}")\n\nprint(f"\\nBest CV MAE (on training folds): {-regr_pred.best_score_:.2f}")\n\n# ---------------------------\n# 5. PREDICTIONS AND METRICS\n# ---------------------------\n# Training predictions (in-sample)\ny_train_pred = regr_pred.predict(X_train)\n\n# Test predictions (held-out)\ny_test_pred = regr_pred.predict(X_test)\n\n# Calculate metrics for TRAIN\ntrain_r2 = r2_score(Y_train, y_train_pred)\ntrain_mae = mean_absolute_error(Y_train, y_train_pred)\ntrain_rmse = np.sqrt(np.mean((Y_train - y_train_pred)**2))\n\n# Calculate metrics for TEST\ntest_r2 = r2_score(Y_test, y_test_pred)\ntest_mae = mean_absolute_error(Y_test, y_test_pred)\ntest_rmse = np.sqrt(np.mean((Y_test - y_test_pred)**2))\n\n# ---------------------------\n# 6. DISPLAY RESULTS\n# ---------------------------\nprint("\\n" + "-"*60)\nprint("PERFORMANCE COMPARISON:")\nprint("-"*60)\nprint(f"{\'Metric\':<10} {\'Training\':<15} {\'Test\':<15} {\'Difference\':<15}")\nprint("-"*60)\nprint(f"{\'R²\':<10} {train_r2:<15.2f} {test_r2:<15.2f} {abs(train_r2-test_r2):<15.2f}")\nprint(f"{\'MAE\':<10} {train_mae:<15.2f} {test_mae:<15.2f} {abs(train_mae-test_mae):<15.2f}")\nprint(f"{\'RMSE\':<10} {train_rmse:<15.2f} {test_rmse:<15.2f} {abs(train_rmse-test_rmse):<15.2f}")\nprint("-"*60)\n\n# Check for overfitting\noverfit_gap = train_r2 - test_r2\nif overfit_gap > 0.1:\n    print("\\n  WARNING: Possible overfitting detected (R² gap > 0.1)")\nelif test_r2 >= 0.7:\n    print("\\n Good generalization performance!")\n\nprint("\\n" + "="*60)\n')


# In[ ]:


parity_plot(
    x=Y_train,                      # observed train
    y=y_train_pred,                 # predicted train (in-sample)
    x_test=Y_test,                  # observed test
    y_test=y_test_pred,             # predicted test (held-out)
    xlabel="pPotency observations",
    ylabel="Predictions",
    savetitle=resultsDir + 'StaphylococcusBacteria/StaphylococcusBacteria_fullData_TrTs.svg',
    save_formats=['svg', 'png']
)


# ## 2. Discovery of new hits specific to all Bacteriaes (data source Enamine_antiviralsData.csv)

# In this section, we screen a custom virtual library looking for molecules that are promising accoring to the the SVR models `regr` above, which use 15-D MACAW embeddings as their input. The custom library ("Enamine_antiviralsData.csv") compiled from commercial catalogs by Enamine. In particular, we are interested in molecules with high predicted pPotency.

# In[ ]:


EnamineAntiviralsData = pd.read_csv( modelBuildingDataDir + "Enamine_antiviralsData.csv")
print(EnamineAntiviralsData.shape)
EnamineAntiviralsData.head()


# In[ ]:


smi_lib = EnamineAntiviralsData.Smiles


# Generate predictions for the H1 receptor:

# In[ ]:


X1_lib = mcw.transform(smi_lib)

Y1_lib_pred = regr_pred.predict(X1_lib)


# Let us represent the predictions of both models:

# In[ ]:


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
plt.savefig(resultsDir + 'StaphylococcusBacteria/StaphylococcusBacteria_validation.svg', bbox_inches='tight', dpi=300)
plt.savefig(resultsDir + 'StaphylococcusBacteria/StaphylococcusBacteria_validation.png', bbox_inches='tight', dpi=300)
plt.show()


# Let us have a look at the compounds:

# In[ ]:


# Define multiple priority levels
high_priority_idx = np.where(Y1_lib_pred >= 7)[0]
medium_priority_idx = np.where((Y1_lib_pred >= 6.0) & (Y1_lib_pred < 7.0))[0]
low_priority_idx = np.where((Y1_lib_pred >= 5.0) & (Y1_lib_pred < 6))[0]

print(f"High priority (pPotency ≥ 7.0): {len(high_priority_idx)} compounds")
print(f"Medium priority (6 ≤ pPotency < 7.0): {len(medium_priority_idx)} compounds")
print(f"Low priority (5 ≤ pPotency < 6): {len(low_priority_idx)} compounds")

# Use the one you need
idx = high_priority_idx  # or combine them


# In[ ]:


EnamineAntiviralsData_final = EnamineAntiviralsData.iloc[idx].copy()
EnamineAntiviralsData_final['pPotency_prediction'] = Y1_lib_pred[idx]

EnamineAntiviralsData_final


# In[ ]:


molecules = [Chem.MolFromSmiles(smi) for smi in EnamineAntiviralsData_final.Smiles[:50]]

Draw.MolsToGridImage(molecules, subImgSize=(200,200), molsPerRow=3, useSVG=True)


# In[ ]:




