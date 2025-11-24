#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Model Comparison
Usage: python Step4_bacteria_MLmodelComparison_noDuplicates.py --config config_noDuplicate.yaml
/users/sghosh6/DTRA_project/MACAW/PubchemData/Step4_bacteria_MLmodelComparison_noDuplicates.py --config config_noDuplicate.yaml > noDuplicate_out.txt
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import argparse
import yaml
sys.stderr = open(os.devnull, 'w')
import warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_VERBOSITY'] = '0'
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import time
from datetime import datetime
import gc  # ADDED

import numpy as np
import pandas as pd
# from copy import deepcopy  # REMOVED - don't need this

from rdkit import Chem
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from joblib import Parallel, delayed, dump
import multiprocessing
from sklearn.model_selection import KFold, GridSearchCV, train_test_split  # Add train_test_split

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

sys.path.append('../')
from macaw import MACAW


# ----------------------------------------------------------------------------
# CONFIGURATION SECTION
# ----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Input/Output paths
    'inputFile': '/users/sghosh6/DTRA_project/MACAW/DrugDesignData/modelBuildingData/allBacteriaData_chEMBL_noDuplicates_MLready.csv',
    'outputDir': '/users/sghosh6/DTRA_project/MACAW/DrugDesignData/Results/bacteriaCommandline_10fold/noDuplicates_coarseGrid/',
    'filePrefix': 'noDuplicates',
    
    # Data columns
    'smilesColumn': 'Smiles',
    'targetColumn': 'pPotency',
    'filterColumns': None,
    
    # Model parameters
    'nFolds': 10,
    'randomState': 42,
    'nJobs': 4,
    'nSamples': None,
    
    # MACAW parameters
    'macawTypeFp': 'atompairs',
    'macawMetric': 'sokal',
    'macawNComponents': 15,
    'macawNLandmarks': 200,
}

# ----------------------------------------------------------------------------


def printMemoryUsage(label=""):
    """Print current memory usage"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024**3
        print(f"  [Memory {label}]: {mem_gb:.2f} GB")


def parseYamlConfig(yamlPath):
    """Parse YAML configuration file"""
    with open(yamlPath, 'r') as f:
        yamlConfig = yaml.safe_load(f)
    
    config = {}
    for key, value in yamlConfig.items():
        if value == 'None' or value is None:
            config[key] = None
        elif key in ['nFolds', 'randomState', 'nJobs', 'nSamples', 'macawNComponents', 'macawNLandmarks']:
            config[key] = int(value) if value != 'None' and value is not None else None
        elif key == 'filterColumns':
            if isinstance(value, str):
                if value == 'None':
                    config[key] = None
                else:
                    config[key] = [x.strip() for x in value.split(',')]
            elif isinstance(value, list):
                config[key] = value
            else:
                config[key] = None
        else:
            config[key] = value
    
    return config


def loadAndPrepareData(config):
    """Load and prepare dataset"""
    df = pd.read_csv(config['inputFile'])
    
    print(f"Original columns in file: {df.columns.tolist()}")
    print(f"Requested filterColumns: {config['filterColumns']}")
    
    if config['filterColumns']:
        actualCols = df.columns.tolist()
        colMapping = {col.lower(): col for col in actualCols}
        
        validCols = []
        for reqCol in config['filterColumns']:
            if reqCol in actualCols:
                validCols.append(reqCol)
            elif reqCol.lower() in colMapping:
                actualCol = colMapping[reqCol.lower()]
                validCols.append(actualCol)
                print(f"  Note: Using '{actualCol}' for requested '{reqCol}'")
            else:
                print(f"  Warning: Column '{reqCol}' not found in data!")
        
        if not validCols:
            raise ValueError(f"None of the requested columns found in data!")
        
        df = df[validCols]
        print(f"Columns after filtering: {df.columns.tolist()}")
    
    colsLower = {col.lower(): col for col in df.columns}
    
    smilesColRequested = config['smilesColumn'].lower()
    targetColRequested = config['targetColumn'].lower()
    
    if smilesColRequested not in colsLower:
        raise ValueError(f"SMILES column '{config['smilesColumn']}' not found!")
    if targetColRequested not in colsLower:
        raise ValueError(f"Target column '{config['targetColumn']}' not found!")
    
    smilesColActual = colsLower[smilesColRequested]
    targetColActual = colsLower[targetColRequested]
    
    print(f"Using SMILES column: '{smilesColActual}'")
    print(f"Using Target column: '{targetColActual}'")
    
    smiles = df[smilesColActual].astype(str).reset_index(drop=True)
    Y = df[targetColActual].to_numpy(dtype=np.float32)
    
    validIdx = [i for i, s in enumerate(smiles) 
                if isinstance(s, str) and len(s) > 0 and Chem.MolFromSmiles(s) is not None]
    
    smiles = smiles.iloc[validIdx].reset_index(drop=True)
    Y = Y[validIdx]
    
    print(f"Valid SMILES after validation: {len(smiles)} out of {len(df)}")
    
    if config['nSamples'] and config['nSamples'] < len(smiles):
        np.random.seed(config['randomState'])
        idx = np.random.choice(len(smiles), config['nSamples'], replace=False)
        smiles = smiles.iloc[idx].reset_index(drop=True)
        Y = Y[idx]
        print(f"Sampled down to: {len(smiles)} samples")
    
    printMemoryUsage("after data loading")
    return smiles, Y


def parseArguments():
    """Parse command line arguments or config file"""
    parser = argparse.ArgumentParser(description='Multi-Model Comparison for Drug Discovery')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (YAML format)')
    parser.add_argument('--inputFile', type=str, default=None)
    parser.add_argument('--outputDir', type=str, default=None)
    parser.add_argument('--filePrefix', type=str, default=None)
    parser.add_argument('--nSamples', type=int, default=None)
    parser.add_argument('--nFolds', type=int, default=None)
    parser.add_argument('--nJobs', type=int, default=None)
    parser.add_argument('--randomState', type=int, default=None)
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        yamlConfig = parseYamlConfig(args.config)
        config.update(yamlConfig)
    
    for key in ['inputFile', 'outputDir', 'filePrefix', 'nSamples', 'nFolds', 'nJobs', 'randomState']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    
    return config


def getParamGrids():
    """Define parameter grids for all models"""
    return {
        'SVR': {
            'C': [1, 10, 100],
            'epsilon': [0.1, 1],
            'kernel': ['rbf']
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        'NeuralNetwork': {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500]
        }
    }


'''
def getParamGrids():
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
        'NeuralNetwork': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500]
        }
    }
'''

def processSingleFold(foldId, trainIndex, testIndex, smilesArray, Y, macawParams, paramGrids):
    """
    Process a single CV fold for all models - MEMORY OPTIMIZED
    """
    # Extract data for this fold
    smiTrain = [smilesArray[i] for i in trainIndex]
    smiTest = [smilesArray[i] for i in testIndex]
    yTrain = Y[trainIndex]
    yTest = Y[testIndex]
    
    # Create FRESH MACAW instance (no deepcopy needed)
    mcwFold = MACAW(**macawParams)
    mcwFold.fit(smiTrain, yTrain)
    
    xTrain = mcwFold.transform(smiTrain)
    xTest = mcwFold.transform(smiTest)
    
    # Free SMILES memory immediately
    del smiTrain, smiTest
    gc.collect()
    
    foldResults = {}
    
    # ========== SVR ==========
    gridSvr = GridSearchCV(
        SVR(), 
        paramGrids['SVR'], 
        cv=3, 
        refit=True, 
        n_jobs=1,
        scoring='neg_mean_absolute_error', 
        verbose=0
    )
    gridSvr.fit(xTrain, yTrain)
    
    foldResults['SVR'] = {
        'trainPred': gridSvr.predict(xTrain), 
        'trainObs': yTrain.copy(),  # Make copies to avoid reference issues
        'testPred': gridSvr.predict(xTest), 
        'testObs': yTest.copy(),
        'bestParams': gridSvr.best_params_
    }
    
    # Clean up SVR
    del gridSvr
    gc.collect()
    
    # ========== Random Forest ==========
    gridRf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        paramGrids['RandomForest'], 
        cv=3, 
        refit=True, 
        n_jobs=1,
        scoring='neg_mean_absolute_error', 
        verbose=0
    )
    gridRf.fit(xTrain, yTrain)
    
    foldResults['RandomForest'] = {
        'trainPred': gridRf.predict(xTrain), 
        'trainObs': yTrain.copy(),
        'testPred': gridRf.predict(xTest), 
        'testObs': yTest.copy(),
        'bestParams': gridRf.best_params_
    }
    
    # Clean up RF
    del gridRf
    gc.collect()
    
    # ========== XGBoost ==========
    gridXgb = GridSearchCV(
        XGBRegressor(random_state=42, tree_method='hist', verbosity=0, n_jobs=1),
        paramGrids['XGBoost'], 
        cv=3, 
        refit=True, 
        n_jobs=1,
        scoring='neg_mean_absolute_error', 
        verbose=0
    )
    gridXgb.fit(xTrain, yTrain)
    
    foldResults['XGBoost'] = {
        'trainPred': gridXgb.predict(xTrain), 
        'trainObs': yTrain.copy(),
        'testPred': gridXgb.predict(xTest), 
        'testObs': yTest.copy(),
        'bestParams': gridXgb.best_params_
    }
    
    # Clean up XGB
    del gridXgb
    gc.collect()
    
    # ========== Neural Network ==========
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    
    gridNn = GridSearchCV(
        MLPRegressor(random_state=42, early_stopping=True),
        paramGrids['NeuralNetwork'], 
        cv=3, 
        refit=True, 
        n_jobs=1,
        scoring='neg_mean_absolute_error', 
        verbose=0
    )
    gridNn.fit(xTrainScaled, yTrain)
    
    foldResults['NeuralNetwork'] = {
        'trainPred': gridNn.predict(xTrainScaled), 
        'trainObs': yTrain.copy(),
        'testPred': gridNn.predict(xTestScaled), 
        'testObs': yTest.copy(),
        'bestParams': gridNn.best_params_
    }
    
    # Clean up NN and scaled data
    del gridNn, scaler, xTrainScaled, xTestScaled
    gc.collect()
    
    # Clean up fold data
    del mcwFold, xTrain, xTest, yTrain, yTest
    gc.collect()
    
    return foldResults


def trainAndEvaluateModels(smiles, Y, kf, macawParams, paramGrids, nJobs=4):
    """
    Train models with parallelized cross-validation - MEMORY OPTIMIZED
    """
    numFolds = kf.get_n_splits()
    nJobsToUse = min(nJobs, numFolds)
    
    print(f"Parallelization strategy:")
    print(f"  - Outer (folds): {nJobsToUse} cores")
    print(f"  - Inner (GridSearchCV): 1 core per fold")
    print(f"  - Models (RF/XGB): 1 core per model")
    print(f"  - TOTAL max cores used: {nJobsToUse}")
    
    printMemoryUsage("before CV")
    
    # Convert smiles to list ONCE
    smilesArray = smiles.tolist()
    
    # Parallelize ONLY across folds
    foldResultsList = Parallel(n_jobs=nJobsToUse, verbose=1, backend='loky')(
        delayed(processSingleFold)(
            foldId, trainIdx, testIdx, smilesArray, Y, macawParams, paramGrids
        )
        for foldId, (trainIdx, testIdx) in enumerate(kf.split(smiles), 1)
    )
    
    # Free smiles array
    del smilesArray
    gc.collect()
    
    printMemoryUsage("after CV, before aggregation")
    
    # Aggregate results - MEMORY EFFICIENT VERSION
    modelNames = ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']
    results = {name: {'trainPred': [], 'trainObs': [], 'testPred': [], 'testObs': [], 'bestParams': []}
               for name in modelNames}
    
    # Process fold results one at a time
    for foldResult in foldResultsList:
        for modelName in modelNames:
            for key in ['trainPred', 'trainObs', 'testPred', 'testObs']:
                results[modelName][key].append(foldResult[modelName][key])
            results[modelName]['bestParams'].append(foldResult[modelName]['bestParams'])
        
        # Clear this fold's results from memory
        del foldResult
    
    # Clear all fold results
    del foldResultsList
    gc.collect()
    
    # Convert to numpy arrays (concatenate)
    for modelName in modelNames:
        for key in ['trainPred', 'trainObs', 'testPred', 'testObs']:
            results[modelName][key] = np.concatenate(results[modelName][key])
    
    printMemoryUsage("after aggregation")
    gc.collect()
    
    return results


def calculateMetricsTable(results):
    """Calculate performance metrics"""
    metricsData = []
    modelOrder = ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']
    
    for modelName in modelOrder:
        data = results[modelName]
        metricsData.append({
            'Model': modelName,
            'Train_R2': round(r2_score(data['trainObs'], data['trainPred']), 3),
            'Train_MAE': round(mean_absolute_error(data['trainObs'], data['trainPred']), 3),
            'Train_RMSE': round(np.sqrt(mean_squared_error(data['trainObs'], data['trainPred'])), 3),
            'Test_R2': round(r2_score(data['testObs'], data['testPred']), 3),
            'Test_MAE': round(mean_absolute_error(data['testObs'], data['testPred']), 3),
            'Test_RMSE': round(np.sqrt(mean_squared_error(data['testObs'], data['testPred'])), 3)
        })
    
    return pd.DataFrame(metricsData)


def saveResults(results, dfMetrics, saveDir, prefix):
    """Save predictions and metrics"""
    os.makedirs(saveDir, exist_ok=True)
    
    dfMetrics.to_csv(os.path.join(saveDir, f'{prefix}_metrics_summary.csv'), index=False)
    
    for modelName, data in results.items():
        # Test predictions
        dfTest = pd.DataFrame({
            'Observed': data['testObs'],
            'Predicted': data['testPred'],
            'Residual': data['testObs'] - data['testPred'],
            'Absolute_Error': np.abs(data['testObs'] - data['testPred'])
        })
        dfTest.to_csv(os.path.join(saveDir, f'{prefix}_predictions_test_{modelName}.csv'), index=False)
        
        # Train predictions
        dfTrain = pd.DataFrame({
            'Observed': data['trainObs'],
            'Predicted': data['trainPred'],
            'Residual': data['trainObs'] - data['trainPred'],
            'Absolute_Error': np.abs(data['trainObs'] - data['trainPred'])
        })
        dfTrain.to_csv(os.path.join(saveDir, f'{prefix}_predictions_train_{modelName}.csv'), index=False)


def generateReport(results, dfMetrics, savePath):
    """Generate summary report"""
    report = []
    report.append("-"*80)
    report.append("MODEL COMPARISON SUMMARY")
    report.append("-"*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    bestIdx = dfMetrics['Test_R2'].idxmax()
    bestModel = dfMetrics.iloc[bestIdx]['Model']
    bestR2 = dfMetrics.iloc[bestIdx]['Test_R2']
    
    report.append(f"Best Model: {bestModel} (Test R² = {bestR2:.3f})")
    report.append("")
    
    report.append("Performance Metrics:")
    report.append("-"*80)
    report.append(dfMetrics.to_string(index=False))
    report.append("")
    
    report.append("Model Analysis:")
    report.append("-"*80)
    for modelName, data in results.items():
        trainR2 = r2_score(data['trainObs'], data['trainPred'])
        testR2 = r2_score(data['testObs'], data['testPred'])
        overfitGap = trainR2 - testR2
        
        status = "Excellent" if overfitGap < 0.05 else "Good" if overfitGap < 0.10 else "Moderate" if overfitGap < 0.15 else "High overfitting"
        
        report.append(f"{modelName:15s}: Train R²={trainR2:.3f}, Test R²={testR2:.3f}, Gap={overfitGap:.3f} ({status})")
    
    report.append("-"*80)
    
    fullReport = "\n".join(report)
    with open(savePath, 'w') as f:
        f.write(fullReport)
    
    return fullReport


def trainAndSaveModel(smiles, Y, modelName, bestParams, macawParams, saveDir, prefix, 
                      randomState=42, testSize=0.2):
    """Train a specific model with train/test split and save comprehensive results"""
    from sklearn.model_selection import train_test_split
    
    # Split data into train/test
    indices = np.arange(len(smiles))
    trainIdx, testIdx = train_test_split(
        indices, test_size=testSize, random_state=randomState, shuffle=True
    )
    
    smiTrain = smiles.iloc[trainIdx].tolist()
    smiTest = smiles.iloc[testIdx].tolist()
    yTrain = Y[trainIdx]
    yTest = Y[testIdx]
    
    # Create and fit MACAW on training data only
    mcwFull = MACAW(**macawParams)
    mcwFull.fit(smiTrain, yTrain)
    
    xTrain = mcwFull.transform(smiTrain)
    xTest = mcwFull.transform(smiTest)
    
    # Train model
    if modelName == 'SVR':
        finalModel = SVR(**bestParams)
        finalModel.fit(xTrain, yTrain)
        trainPred = finalModel.predict(xTrain)
        testPred = finalModel.predict(xTest)
        
    elif modelName == 'RandomForest':
        finalModel = RandomForestRegressor(**bestParams, random_state=randomState, n_jobs=1)
        finalModel.fit(xTrain, yTrain)
        trainPred = finalModel.predict(xTrain)
        testPred = finalModel.predict(xTest)
        
    elif modelName == 'XGBoost':
        finalModel = XGBRegressor(**bestParams, random_state=randomState, 
                                 tree_method='hist', verbosity=0, n_jobs=1)
        finalModel.fit(xTrain, yTrain)
        trainPred = finalModel.predict(xTrain)
        testPred = finalModel.predict(xTest)
        
    elif modelName == 'NeuralNetwork':
        scaler = StandardScaler()
        xTrainScaled = scaler.fit_transform(xTrain)
        xTestScaled = scaler.transform(xTest)
        
        finalModel = MLPRegressor(**bestParams, random_state=randomState, early_stopping=True)
        finalModel.fit(xTrainScaled, yTrain)
        trainPred = finalModel.predict(xTrainScaled)
        testPred = finalModel.predict(xTestScaled)
        
        dump(scaler, os.path.join(saveDir, f'{prefix}_{modelName}_scaler.joblib'))
        
        # Clean up scaled data
        del xTrainScaled, xTestScaled
    
    # Calculate metrics
    trainMetrics = {
        'R2': r2_score(yTrain, trainPred),
        'MAE': mean_absolute_error(yTrain, trainPred),
        'RMSE': np.sqrt(mean_squared_error(yTrain, trainPred))
    }
    
    testMetrics = {
        'R2': r2_score(yTest, testPred),
        'MAE': mean_absolute_error(yTest, testPred),
        'RMSE': np.sqrt(mean_squared_error(yTest, testPred))
    }
    
    # Save TRAIN predictions from full data split
    dfTrainPred = pd.DataFrame({
        'SMILES': smiTrain,
        'Observed': yTrain,
        'Predicted': trainPred,
        'Residual': yTrain - trainPred,
        'Absolute_Error': np.abs(yTrain - trainPred)
    })
    dfTrainPred.to_csv(
        os.path.join(saveDir, f'{prefix}_predictions_fullData_train_{modelName}.csv'), 
        index=False
    )
    
    # Save TEST predictions from full data split
    dfTestPred = pd.DataFrame({
        'SMILES': smiTest,
        'Observed': yTest,
        'Predicted': testPred,
        'Residual': yTest - testPred,
        'Absolute_Error': np.abs(yTest - testPred)
    })
    dfTestPred.to_csv(
        os.path.join(saveDir, f'{prefix}_predictions_fullData_test_{modelName}.csv'), 
        index=False
    )
    
    # Prepare metrics summary
    metrics = {
        'Model': modelName,
        'Train_R2': round(trainMetrics['R2'], 4),
        'Train_MAE': round(trainMetrics['MAE'], 4),
        'Train_RMSE': round(trainMetrics['RMSE'], 4),
        'Test_R2': round(testMetrics['R2'], 4),
        'Test_MAE': round(testMetrics['MAE'], 4),
        'Test_RMSE': round(testMetrics['RMSE'], 4),
        'N_train': len(yTrain),
        'N_test': len(yTest),
        'Best_Params': str(bestParams)
    }
    
    # Save model and MACAW
    dump(finalModel, os.path.join(saveDir, f'{prefix}_{modelName}_regr_pred.joblib'))
    dump(mcwFull, os.path.join(saveDir, f'{prefix}_{modelName}_macaw_model.joblib'))
    
    # Clean up
    del xTrain, xTest, trainPred, testPred
    gc.collect()
    
    return finalModel, mcwFull, metrics


def trainAndSaveAllModels(smiles, Y, modelResults, macawParams, saveDir, prefix, 
                         randomState=42, testSize=0.2):
    """Train all models on full data with train/test split and save comprehensive results"""
    modelNames = ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']
    savedModels = {}
    fullDataMetrics = []
    
    print(f"\nTraining on {(1-testSize)*100:.0f}% of data, testing on {testSize*100:.0f}%")
    
    for modelName in modelNames:
        print(f"Training {modelName} on full dataset...")
        printMemoryUsage(f"before {modelName}")
        
        # Use first best params from CV (or implement majority voting)
        bestParams = modelResults[modelName]['bestParams'][0]
        
        finalModel, mcwFull, metrics = trainAndSaveModel(
            smiles, Y, modelName, bestParams, macawParams, saveDir, prefix, 
            randomState, testSize
        )
        
        savedModels[modelName] = {'model': finalModel, 'macaw': mcwFull}
        fullDataMetrics.append(metrics)
        
        # Clean up between models
        del finalModel, mcwFull
        gc.collect()
        
        printMemoryUsage(f"after {modelName}")
    
    # Save full data metrics summary
    dfFullMetrics = pd.DataFrame(fullDataMetrics)
    dfFullMetrics.to_csv(
        os.path.join(saveDir, f'{prefix}_fullData_metrics.csv'), 
        index=False
    )
    
    print("\n" + "="*80)
    print("FULL DATA HOLDOUT PERFORMANCE")
    print("="*80)
    print(dfFullMetrics[['Model', 'Train_R2', 'Train_MAE', 'Test_R2', 'Test_MAE']].to_string(index=False))
    
    return savedModels, dfFullMetrics


def main():
    """Main execution"""
    startTime = time.time()
    
    config = parseArguments()
    
    print(f"{'='*80}")
    print(f"MULTI-MODEL COMPARISON - MEMORY OPTIMIZED")
    print(f"{'='*80}")
    print(f"Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"")
    
    # Validate nJobs
    maxCores = multiprocessing.cpu_count()
    if config['nJobs'] > maxCores:
        print(f"Warning: nJobs ({config['nJobs']}) > available cores ({maxCores}). Using {maxCores}.")
        config['nJobs'] = maxCores
    
    dfInput = pd.read_csv(config['inputFile'])
    print(f"Input file: {config['inputFile']} (shape: {dfInput.shape})")
    print(f"Output Directory: {config['outputDir']}")
    print(f"Prefix: {config['filePrefix']}")
    print(f"Using {config['nJobs']} CPU cores (strict limit)")
    print(f"")
    
    # Load data
    smiles, Y = loadAndPrepareData(config)
    print(f"Loaded {len(smiles)} valid samples after validation")
    
    # MACAW parameters (don't create instance yet)
    macawParams = {
        'type_fp': config['macawTypeFp'],
        'metric': config['macawMetric'],
        'n_components': config['macawNComponents'],
        'n_landmarks': config['macawNLandmarks'],
        'random_state': config['randomState']
    }
    
    # Initialize KFold and parameter grids
    kf = KFold(n_splits=config['nFolds'], shuffle=True, random_state=config['randomState'])
    paramGrids = getParamGrids()
    
    # Train and evaluate models
    print(f"\nTraining models with {config['nFolds']}-fold CV...")
    modelResults = trainAndEvaluateModels(smiles, Y, kf, macawParams, paramGrids, config['nJobs'])
    
    # Calculate metrics
    dfComparison = calculateMetricsTable(modelResults)
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    print(dfComparison.to_string(index=False))
    
    # Save results
    saveResults(modelResults, dfComparison, config['outputDir'], config['filePrefix'])
    
    # Generate and save report
    reportPath = os.path.join(config['outputDir'], f'{config["filePrefix"]}_report.txt')
    report = generateReport(modelResults, dfComparison, reportPath)
    print(f"\n{report}")

    # Train and save ALL models on full dataset 
    print(f"\n{'='*80}")
    print(f"TRAINING FINAL MODELS ON FULL DATASET")
    print(f"{'='*80}")
    savedModels, dfFullMetrics = trainAndSaveAllModels(
        smiles, Y, modelResults, macawParams, 
        config['outputDir'], config['filePrefix'], 
        config['randomState'], testSize=0.2  # Add testSize parameter
    )
    
    
    print(f"\n{'='*80}")
    print("ALL MODELS SAVED")
    print(f"{'='*80}")
    for modelName in ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']:
        print(f"  - {config['filePrefix']}_{modelName}_regr_pred.joblib")
        print(f"  - {config['filePrefix']}_{modelName}_macaw_model.joblib")
        if modelName == 'NeuralNetwork':
            print(f"  - {config['filePrefix']}_{modelName}_scaler.joblib")
    
    elapsed = time.time() - startTime
    print(f"\n{'='*80}")
    print(f"COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: {config['outputDir']}")
    printMemoryUsage("final")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()