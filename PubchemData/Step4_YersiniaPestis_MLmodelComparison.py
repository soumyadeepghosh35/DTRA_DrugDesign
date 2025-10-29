#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Model Comparison for Antiviral Drug Discovery

Run from command line: 
* (full) python Step4_YersiniaPestis_MLmodelComparison.py --n_folds 10 --n_jobs 4 --random_state 42
* (test) python Step4_YersiniaPestis_MLmodelComparison.py --n_samples 100 --n_folds 5 --n_jobs 4 --random_state 42
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import argparse
# Completely suppress stderr output
sys.stderr = open(os.devnull, 'w')
import warnings
warnings.filterwarnings('ignore')
# Suppress RDKit warnings specifically
os.environ['RDKIT_VERBOSITY'] = '0'
# Suppress RDKit deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from rdkit import Chem
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
import multiprocessing

# Add MACAW to path
sys.path.append('../')
from macaw import MACAW


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Model Comparison for Drug Discovery')
    
    parser.add_argument('--data_dir', type=str, 
                        default='/users/sghosh6/DTRA_project/MACAW/DrugDesignData/',
                        help='Base data directory')
    
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to use (for testing, use smaller number)')
    
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of cross-validation folds')
    
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of CPU cores to use')
    
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir/Results/BacillusAnthracis_Commandline/)')
    
    return parser.parse_args()


def load_and_prepare_data(data_dir, n_samples=None, random_state=42):
    """Load and prepare the dataset"""
    print("\n" + "-"*80)
    print("LOADING AND PREPARING DATA")
    print("-"*80)
    
    model_building_dir = os.path.join(data_dir, 'modelBuildingData/')
    file_path = os.path.join(model_building_dir, 'YersiniaPestisData_chEMBL_MLready.csv')
    output_dir = os.path.join(data_dir, 'Results', 'YersiniaPestisData_Commandline')
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Filter columns
    df = df.filter(items=["Smiles", "bacteriaClassifier", "pPotency"])
    
    # Extract data
    smiles = df['Smiles'].astype(str).reset_index(drop=True)
    Y = df['pPotency'].to_numpy(dtype=np.float32)
    
    # Validate SMILES
    valid_idx = []
    for i, s in enumerate(smiles):
        if isinstance(s, str) and len(s) > 0 and Chem.MolFromSmiles(s) is not None:
            valid_idx.append(i)
    
    smiles = smiles.iloc[valid_idx].reset_index(drop=True)
    Y = Y[valid_idx]
    
    print(f"Valid samples after SMILES validation: {len(smiles)}")
    print("-"*80 + "\n")
    
    return smiles, Y, output_dir


def get_param_grids():
    """Define parameter grids for all models"""
    return {
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


def process_single_fold(fold_id, train_index, test_index, smiles, Y, mcw, param_grids, n_jobs_inner):
    """Process a single CV fold for all models"""
    print(f"\nProcessing Fold {fold_id}")
    
    smi_train = smiles.iloc[train_index].tolist()
    smi_test = smiles.iloc[test_index].tolist()
    y_train = Y[train_index]
    y_test = Y[test_index]
    
    # Deep copy MACAW to avoid conflicts
    mcw_fold = deepcopy(mcw)
    mcw_fold.fit(smi_train, y_train)
    
    X_train = mcw_fold.transform(smi_train)
    X_test = mcw_fold.transform(smi_test)
    
    fold_results = {}
    
    # SVR
    grid_svr = GridSearchCV(
        SVR(), param_grids['SVR'],
        cv=3, refit=True, n_jobs=n_jobs_inner,
        scoring='neg_mean_absolute_error', verbose=0
    )
    grid_svr.fit(X_train, y_train)
    fold_results['SVR'] = {
        'train_pred': grid_svr.predict(X_train),
        'train_obs': y_train,
        'test_pred': grid_svr.predict(X_test),
        'test_obs': y_test,
        'best_params': grid_svr.best_params_
    }
    
    # Random Forest
    grid_rf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),  # Set to 1!
        param_grids['RandomForest'],
        cv=3, refit=True, n_jobs=n_jobs_inner,
        scoring='neg_mean_absolute_error', verbose=0
    )
    grid_rf.fit(X_train, y_train)
    fold_results['RandomForest'] = {
        'train_pred': grid_rf.predict(X_train),
        'train_obs': y_train,
        'test_pred': grid_rf.predict(X_test),
        'test_obs': y_test,
        'best_params': grid_rf.best_params_
    }
    
    # XGBoost
    grid_xgb = GridSearchCV(
        XGBRegressor(random_state=42, tree_method='hist', verbosity=0, n_jobs=1),
        param_grids['XGBoost'],
        cv=3, refit=True, n_jobs=n_jobs_inner,
        scoring='neg_mean_absolute_error', verbose=0
    )
    grid_xgb.fit(X_train, y_train)
    fold_results['XGBoost'] = {
        'train_pred': grid_xgb.predict(X_train),
        'train_obs': y_train,
        'test_pred': grid_xgb.predict(X_test),
        'test_obs': y_test,
        'best_params': grid_xgb.best_params_
    }
    
    # LightGBM
    grid_lgbm = GridSearchCV(
        LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
        param_grids['LightGBM'],
        cv=3, refit=True, n_jobs=n_jobs_inner,
        scoring='neg_mean_absolute_error', verbose=0
    )
    grid_lgbm.fit(X_train, y_train)
    fold_results['LightGBM'] = {
        'train_pred': grid_lgbm.predict(X_train),
        'train_obs': y_train,
        'test_pred': grid_lgbm.predict(X_test),
        'test_obs': y_test,
        'best_params': grid_lgbm.best_params_
    }
    
    # CatBoost
    grid_cat = GridSearchCV(
        CatBoostRegressor(random_state=42, verbose=False, thread_count=1),
        param_grids['CatBoost'],
        cv=3, refit=True, n_jobs=n_jobs_inner,
        scoring='neg_mean_absolute_error', verbose=0
    )
    grid_cat.fit(X_train, y_train)
    fold_results['CatBoost'] = {
        'train_pred': grid_cat.predict(X_train),
        'train_obs': y_train,
        'test_pred': grid_cat.predict(X_test),
        'test_obs': y_test,
        'best_params': grid_cat.best_params_
    }
    
    # Neural Network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    grid_nn = GridSearchCV(
        MLPRegressor(random_state=42, early_stopping=True),
        param_grids['NeuralNetwork'],
        cv=3, refit=True, n_jobs=n_jobs_inner,
        scoring='neg_mean_absolute_error', verbose=0
    )
    grid_nn.fit(X_train_scaled, y_train)
    fold_results['NeuralNetwork'] = {
        'train_pred': grid_nn.predict(X_train_scaled),
        'train_obs': y_train,
        'test_pred': grid_nn.predict(X_test_scaled),
        'test_obs': y_test,
        'best_params': grid_nn.best_params_
    }
    
    return fold_results


def train_and_evaluate_models(smiles, Y, kf, mcw, param_grids, n_jobs=4, verbose=True):
    """Train multiple regression models with PARALLELIZED cross-validation"""
    
    # Configure CPU usage
    n_cores = multiprocessing.cpu_count()
    num_folds = kf.get_n_splits()
    
    # Strategy: Parallelize outer folds
    # n_jobs_outer * n_jobs_inner ≈ n_cores
    n_jobs_outer = min(num_folds, max(1, n_cores // 4))  # Run multiple folds in parallel
    n_jobs_inner = max(1, n_cores // n_jobs_outer)  # Cores per fold for GridSearchCV
    
    print(f"Parallelization strategy:")
    print(f"  Total cores: {n_cores}")
    print(f"  Parallel folds: {n_jobs_outer}")
    print(f"  Cores per fold (GridSearchCV): {n_jobs_inner}")
    print(f"  Estimated speedup: {n_jobs_outer}x")
    
    # Run folds in parallel
    fold_results_list = Parallel(n_jobs=n_jobs_outer, verbose=10)(
        delayed(process_single_fold)(
            fold_id, train_idx, test_idx, smiles, Y, mcw, param_grids, n_jobs_inner
        )
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(smiles), 1)
    )
    
    # Aggregate results across folds
    model_names = ['SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNetwork']
    results = {}
    for name in model_names:
        results[name] = {
            'train_pred': [], 'train_obs': [],
            'test_pred': [], 'test_obs': [],
            'best_params': []
        }
    
    for fold_result in fold_results_list:
        for model_name in model_names:
            results[model_name]['train_pred'].extend(fold_result[model_name]['train_pred'])
            results[model_name]['train_obs'].extend(fold_result[model_name]['train_obs'])
            results[model_name]['test_pred'].extend(fold_result[model_name]['test_pred'])
            results[model_name]['test_obs'].extend(fold_result[model_name]['test_obs'])
            results[model_name]['best_params'].append(fold_result[model_name]['best_params'])
    
    # Convert to numpy arrays
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
            'Train_R2': round(train_r2, 2),
            'Train_MAE': round(train_mae, 2),
            'Train_RMSE': round(train_rmse, 2),
            'Test_R2': round(test_r2, 2),
            'Test_MAE': round(test_mae, 2),
            'Test_RMSE': round(test_rmse, 2)
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    return df_metrics


def save_results(results, df_metrics, save_dir):
    """Save all predictions and metrics to CSV files"""
    os.makedirs(save_dir, exist_ok=True)
    
    df_metrics.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)
    
    for model_name, data in results.items():
        df_test = pd.DataFrame({
            'Observed': data['test_obs'],
            'Predicted': data['test_pred'],
            'Residual': data['test_obs'] - data['test_pred'],
            'Absolute_Error': np.abs(data['test_obs'] - data['test_pred'])
        })
        df_test.to_csv(os.path.join(save_dir, f'predictions_test_{model_name}.csv'), index=False)
        
        df_train = pd.DataFrame({
            'Observed': data['train_obs'],
            'Predicted': data['train_pred'],
            'Residual': data['train_obs'] - data['train_pred'],
            'Absolute_Error': np.abs(data['train_obs'] - data['train_pred'])
        })
        df_train.to_csv(os.path.join(save_dir, f'predictions_train_{model_name}.csv'), index=False)


def generate_report(results, df_metrics, save_path):
    """Generate text summary report"""
    report = []
    report.append("Model Comparison Summary")
    report.append("-" * 80)
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
        report.append(f"{idx+1}. {row['Model']:15s} - R²: {row['Test_R2']:.3f}, "
                     f"MAE: {row['Test_MAE']:.2f}, RMSE: {row['Test_RMSE']:.2f}")
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
    report.append("-" * 80)
    
    full_report = "\n".join(report)
    
    with open(save_path, 'w') as f:
        f.write(full_report)
    
    return full_report

def create_parity_plots(results, save_dir):
    """Generate parity plots for all models"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, data in results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Test set parity plot
        ax1.scatter(data['test_obs'], data['test_pred'], alpha=0.5, s=20)
        min_val = min(data['test_obs'].min(), data['test_pred'].min())
        max_val = max(data['test_obs'].max(), data['test_pred'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Observed pPotency', fontsize=12)
        ax1.set_ylabel('Predicted pPotency', fontsize=12)
        ax1.set_title(f'{model_name} - Test Set', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        test_r2 = r2_score(data['test_obs'], data['test_pred'])
        test_mae = mean_absolute_error(data['test_obs'], data['test_pred'])
        ax1.text(0.05, 0.95, f'R² = {test_r2:.3f}\nMAE = {test_mae:.2f}',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Train set parity plot
        ax2.scatter(data['train_obs'], data['train_pred'], alpha=0.5, s=20)
        min_val = min(data['train_obs'].min(), data['train_pred'].min())
        max_val = max(data['train_obs'].max(), data['train_pred'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax2.set_xlabel('Observed pPotency', fontsize=12)
        ax2.set_ylabel('Predicted pPotency', fontsize=12)
        ax2.set_title(f'{model_name} - Train Set', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        train_r2 = r2_score(data['train_obs'], data['train_pred'])
        train_mae = mean_absolute_error(data['train_obs'], data['train_pred'])
        ax2.text(0.05, 0.95, f'R² = {train_r2:.3f}\nMAE = {train_mae:.2f}',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'parity_plot_bacteria_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'parity_plot_bacteria_{model_name}.svg'), bbox_inches='tight')
        plt.close()
        
    print(f"Parity plots saved to: {save_dir}")
    
def main():
    """Main execution function"""
    start_time = time.time()
    
    args = parse_arguments()
    
    print("\n" + "-"*80)
    print("MULTI-MODEL COMPARISON FOR ANTIVIRAL DRUG DISCOVERY")
    print("-"*80)
    print(f"Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Number of folds: {args.n_folds}")
    print(f"  CPU cores: {args.n_jobs}")
    print(f"  Random state: {args.random_state}")  
    
    # Load data
    smiles, Y, default_output_dir = load_and_prepare_data(args.data_dir, args.n_samples, args.random_state)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = default_output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  Output directory: {args.output_dir}")
    
    # Initialize MACAW
    print("Initializing MACAW embeddings...")
    mcw = MACAW(
        type_fp='atompairs',
        metric='sokal',
        n_components=15,
        n_landmarks=50,
        random_state=args.random_state
    )
    
    # Initialize KFold
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    
    # Get parameter grids
    param_grids = get_param_grids()
    
    # Run model training and evaluation
    print("\n" + "-"*80)
    print("Training and Evaluating Models")
    print("-"*80)
    
    model_results = train_and_evaluate_models(
        smiles=smiles,
        Y=Y,
        kf=kf,
        mcw=mcw,
        param_grids=param_grids,
        n_jobs=args.n_jobs,
        verbose=True
    )
    
    # Calculate metrics
    print("\n" + "-"*80)
    print("Calculating Metrics")
    print("-"*80)
    df_comparison = calculate_metrics_table(model_results)
    print("\nMetrics summary:")
    print(df_comparison.to_string(index=False))
    
    # Save results
    print("\n" + "-"*80)
    print("Saving Results")
    print("-"*80)
    save_results(model_results, df_comparison, args.output_dir)
    print(f"Results saved to: {args.output_dir}")

    # Generate plots
    print("\n" + "="*80)
    print("Plots")
    print("="*80)
    create_parity_plots(model_results, args.output_dir)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'model_comparison_report.txt')
    report = generate_report(model_results, df_comparison, report_path)
    print("\n" + "-"*80)
    print("Report")
    print("-"*80)
    print(report)
    

    print(f"All results saved to: {args.output_dir}")
    end_time = time.time()
    print(f"Total Runtime: {end_time - start_time: .2f} s")


if __name__ == "__main__":
    main()