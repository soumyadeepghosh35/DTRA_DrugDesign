#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MACAW-SVR Model Training for Staphylococcus Bacteria Compound Screening
Performs cross-validation and train/test split evaluation 
* (full) python StaphylococcusBacteria_MLmodelComparison-fullData.py
* (test) python StaphylococcusBacteria_MLmodelComparison-fullData.py --samples 500
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
# Completely suppress stderr output
sys.stderr = open(os.devnull, 'w')
# Now import everything
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from joblib import dump
import argparse
from rdkit import Chem
import time


# Add MACAW to path
sys.path.append('../')
from macaw import MACAW



# ==========================================
# CONFIGURATION - MODIFY THESE PARAMETERS
# ==========================================
dataDir = '/users/sghosh6/DTRA_project/MACAW/DrugDesignData/'
modelBuildingDir = os.path.join(dataDir, 'modelBuildingData/')
DATA_FILE = os.path.join(modelBuildingDir, 'StaphylococcusBacteriaData_chEMBL_wMACAW_MLready.csv')
OUTPUT_DIR = os.path.join(dataDir, 'Results/StaphylococcusBacteriaCommandline')
TEST_SIZE = 0.2
N_CV_FOLDS = 10
RANDOM_STATE = 42
MACAW_RANDOM_STATE = 39

PARAM_GRID = {
    'C': [1, 10, 100],
    'epsilon': [0.1, 1, 10],
    'kernel': ['rbf']
}

MACAW_PARAMS = {
    'type_fp': 'atompairs',
    'metric': 'sokal',
    'n_components': 15,
    'n_landmarks': 200,
    'random_state': MACAW_RANDOM_STATE
}

def train_and_save(data_path, results_dir, n_samples=None):
    
    print("Data Loading")
    print(f"Input file: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Total rows loaded: {len(data)}")
    
    if n_samples:
        print(f"Sampling {n_samples} rows for testing")
        data = data.sample(n=n_samples, random_state=RANDOM_STATE).reset_index(drop=True)
    
    Y = data.pPotency
    smiles = data.Smiles
    
    # Remove NaN values
    valid_mask = Y.notna()
    smiles = smiles[valid_mask].reset_index(drop=True)
    Y = Y[valid_mask].reset_index(drop=True).values
    
    print(f"Valid samples: {len(smiles)}")
    print(f"Train/test split ratio: {1-TEST_SIZE:.1%}/{TEST_SIZE:.1%}")
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"Output directory: {results_dir}")
    
    # Cross-validation
    print("\n--- Cross-Validation Analysis ---")
    print(f"Number of folds: {N_CV_FOLDS}")
    
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mcw_cv = MACAW(**MACAW_PARAMS)
    
    Y_cv_pred = []
    Y_obs = []
    
    for i, (train_index, val_index) in enumerate(kf.split(smiles), 1):
        print(f"Fold {i}/{N_CV_FOLDS}: train_n={len(train_index)}, val_n={len(val_index)}")
        
        smi_train, smi_val = smiles.iloc[train_index], smiles.iloc[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        
        mcw_cv.fit(smi_train, y_train)
        X_train = mcw_cv.transform(smi_train)
        X_val = mcw_cv.transform(smi_val)
        
        grid = GridSearchCV(SVR(), PARAM_GRID, cv=5, refit=True, 
                          n_jobs=4, verbose=0, 
                          scoring='neg_mean_absolute_error')
        grid.fit(X_train, y_train)
        
        y_cv_pred = grid.predict(X_val)
        Y_cv_pred.extend(y_cv_pred)
        Y_obs.extend(y_val)
    
    cv_results = pd.DataFrame({
        'Y_observed': Y_obs,
        'Y_predicted': Y_cv_pred
    })
    cv_file = f'{results_dir}/fullData_cv_predictions.csv'
    cv_results.to_csv(cv_file, index=False)
    
    cv_r2 = r2_score(Y_obs, Y_cv_pred)
    cv_mae = mean_absolute_error(Y_obs, Y_cv_pred)
    cv_rmse = np.sqrt(mean_squared_error(Y_obs, Y_cv_pred))
    
    print(f"\nCross-validation results:")
    print(f"  R2 = {cv_r2:.4f}")
    print(f"  MAE = {cv_mae:.4f}")
    print(f"  RMSE = {cv_rmse:.4f}")
    print(f"  Predictions saved: {cv_file}")
    
    # Train/test split
    print("\n--- Train/Test Split Analysis ---")
    
    smi_train, smi_test, Y_train, Y_test = train_test_split(
        smiles, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set: n={len(smi_train)}")
    print(f"Test set: n={len(smi_test)}")
    
    mcw = MACAW(**MACAW_PARAMS)
    
    print("Computing MACAW embeddings on training set")
    mcw.fit(smi_train, Y_train)
    
    X_train = mcw.transform(smi_train)
    X_test = mcw.transform(smi_test)
    
    print(f"Embedding dimensions: {X_train.shape[1]}")
    
    print("Hyperparameter optimization via grid search")
    regr_pred = GridSearchCV(SVR(), PARAM_GRID, cv=5, refit=True,
                            n_jobs=8, verbose=1,
                            scoring='neg_mean_absolute_error')
    regr_pred.fit(X_train, Y_train)
    
    print(f"Best hyperparameters: {regr_pred.best_params_}")
    
    print("Generating predictions")
    y_train_pred = regr_pred.predict(X_train)
    y_test_pred = regr_pred.predict(X_test)
    
    train_r2 = r2_score(Y_train, y_train_pred)
    test_r2 = r2_score(Y_test, y_test_pred)
    train_mae = mean_absolute_error(Y_train, y_train_pred)
    test_mae = mean_absolute_error(Y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
    
    print(f"\nTraining set performance:")
    print(f"  R2 = {train_r2:.4f}")
    print(f"  MAE = {train_mae:.4f}")
    print(f"  RMSE = {train_rmse:.4f}")
    
    print(f"\nTest set performance:")
    print(f"  R2 = {test_r2:.4f}")
    print(f"  MAE = {test_mae:.4f}")
    print(f"  RMSE = {test_rmse:.4f}")
    
    # Save predictions
    train_results = pd.DataFrame({
        'Smiles': smi_train,
        'Y_observed': Y_train,
        'Y_predicted': y_train_pred,
        'Set': 'Train'
    })
    train_file = f'{results_dir}/fullData_train_predictions.csv'
    train_results.to_csv(train_file, index=False)
    
    test_results = pd.DataFrame({
        'Smiles': smi_test,
        'Y_observed': Y_test,
        'Y_predicted': y_test_pred,
        'Set': 'Test'
    })
    test_file = f'{results_dir}/fullData_test_predictions.csv'
    test_results.to_csv(test_file, index=False)
    
    combined_results = pd.concat([train_results, test_results], ignore_index=True)
    combined_file = f'{results_dir}/fullData_train_test_predictions.csv'
    combined_results.to_csv(combined_file, index=False)
    
    print(f"\nPrediction files saved:")
    print(f"  {train_file}")
    print(f"  {test_file}")
    print(f"  {combined_file}")
    
    # Save models
    regr_file = f'{results_dir}/fullData_regr_pred.joblib'
    macaw_file = f'{results_dir}/fullData_macaw_model.joblib'
    
    dump(regr_pred, regr_file)
    dump(mcw, macaw_file)
    
    print(f"\nModel files saved:")
    print(f"  {regr_file}")
    print(f"  {macaw_file}")
    
    # Save metrics summary
    metrics = pd.DataFrame({
        'Metric': ['R2', 'MAE', 'RMSE', 'N_samples'],
        'Cross_Validation': [cv_r2, cv_mae, cv_rmse, len(Y_obs)],
        'Train': [train_r2, train_mae, train_rmse, len(Y_train)],
        'Test': [test_r2, test_mae, test_rmse, len(Y_test)]
    })
    metrics_file = f'{results_dir}/fullData_metrics_summary.csv'
    metrics.to_csv(metrics_file, index=False)
    print(f"  {metrics_file}")
    
    # Save configuration
    config = pd.DataFrame({
        'Parameter': ['DATA_FILE', 'OUTPUT_DIR', 'TEST_SIZE', 'N_CV_FOLDS', 
                     'RANDOM_STATE', 'MACAW_n_components', 'MACAW_n_landmarks', 
                     'MACAW_type_fp', 'MACAW_metric'],
        'Value': [data_path, results_dir, TEST_SIZE, N_CV_FOLDS, RANDOM_STATE,
                 MACAW_PARAMS['n_components'], MACAW_PARAMS['n_landmarks'],
                 MACAW_PARAMS['type_fp'], MACAW_PARAMS['metric']]
    })
    config_file = f'{results_dir}/fullData_configuration.csv'
    config.to_csv(config_file, index=False)
    print(f"  {config_file}")
    
    print("\nTraining completed")

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--samples', type=int, default=None, 
                       help='Number of samples to use (optional, for testing)')
    
    args = parser.parse_args()
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found: {DATA_FILE}")
        print("Please update DATA_FILE variable at the top of the script")
        sys.exit(1)
    
    try:
        train_and_save(DATA_FILE, OUTPUT_DIR, args.samples)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"Total Runtime: {end_time - start_time: .2f} s")

if __name__ == "__main__":
    main()
