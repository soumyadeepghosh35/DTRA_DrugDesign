#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Model Comparison for Drug Discovery
Usage: python /users/sghosh6/DTRA_project/MACAW/PubchemData/Step4_bacteria_MLmodelComparison_noDuplicates.py --config config_noDuplicate.yaml
   or: python /users/sghosh6/DTRA_project/MACAW/PubchemData/Step4_bacteria_MLmodelComparison_noDuplicates.py [command line args]
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

import numpy as np
import pandas as pd
from copy import deepcopy

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

sys.path.append('../')
from macaw import MACAW


# ----------------------------------------------------------------------------
# CONFIGURATION SECTION - MODIFY THESE VALUES
# ----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Input/Output paths
    'inputFile': '/users/sghosh6/DTRA_project/MACAW/DrugDesignData/modelBuildingData/allBacteriaData_chEMBL_noDuplicates_MLready.csv',
    'outputDir': '/users/sghosh6/DTRA_project/MACAW/DrugDesignData/Results/bacteriaCommandline/',
    'filePrefix': 'noDuplicates',
    
    # Data columns
    'smilesColumn': 'Smiles',
    'targetColumn': 'pPotency',
    'filterColumns': None,  # None = auto-detect all columns from input file
    
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


def parseYamlConfig(yamlPath):
    """Parse YAML configuration file"""
    with open(yamlPath, 'r') as f:
        yamlConfig = yaml.safe_load(f)
    
    config = {}
    for key, value in yamlConfig.items():
        # Convert string values to appropriate types
        if value == 'None' or value is None:
            config[key] = None
        elif key in ['nFolds', 'randomState', 'nJobs', 'nSamples', 'macawNComponents', 'macawNLandmarks']:
            config[key] = int(value) if value != 'None' and value is not None else None
        elif key == 'filterColumns':
            # Handle both list and comma-separated string formats
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
    """Load and prepare dataset - works with any columns specified in filterColumns"""
    df = pd.read_csv(config['inputFile'])
    
    print(f"Original columns in file: {df.columns.tolist()}")
    print(f"Requested filterColumns: {config['filterColumns']}")
    
    # If filterColumns specified, filter the dataframe
    if config['filterColumns']:
        # Check which columns actually exist (case-insensitive matching)
        actualCols = df.columns.tolist()
        colMapping = {col.lower(): col for col in actualCols}
        
        validCols = []
        for reqCol in config['filterColumns']:
            if reqCol in actualCols:
                validCols.append(reqCol)
            elif reqCol.lower() in colMapping:
                # Case-insensitive match found
                actualCol = colMapping[reqCol.lower()]
                validCols.append(actualCol)
                print(f"  Note: Using '{actualCol}' for requested '{reqCol}'")
            else:
                print(f"  Warning: Column '{reqCol}' not found in data!")
        
        if not validCols:
            raise ValueError(f"None of the requested columns found in data!")
        
        df = df[validCols]
        print(f"Columns after filtering: {df.columns.tolist()}")
    
    # Find the smiles and target columns (case-insensitive)
    colsLower = {col.lower(): col for col in df.columns}
    
    smilesColRequested = config['smilesColumn'].lower()
    targetColRequested = config['targetColumn'].lower()
    
    if smilesColRequested not in colsLower:
        raise ValueError(f"SMILES column '{config['smilesColumn']}' not found in filtered data!")
    if targetColRequested not in colsLower:
        raise ValueError(f"Target column '{config['targetColumn']}' not found in filtered data!")
    
    smilesColActual = colsLower[smilesColRequested]
    targetColActual = colsLower[targetColRequested]
    
    print(f"Using SMILES column: '{smilesColActual}'")
    print(f"Using Target column: '{targetColActual}'")
    
    smiles = df[smilesColActual].astype(str).reset_index(drop=True)
    Y = df[targetColActual].to_numpy(dtype=np.float32)
    
    # Validate SMILES
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
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Override with YAML config file if provided
    if args.config and os.path.exists(args.config):
        yamlConfig = parseYamlConfig(args.config)
        config.update(yamlConfig)
    
    # Override with command line arguments
    for key in ['inputFile', 'outputDir', 'filePrefix', 'nSamples', 'nFolds', 'nJobs', 'randomState']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    
    return config





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


def processSingleFold(foldId, trainIndex, testIndex, smiles, Y, mcw, paramGrids, nJobsInner):
    """Process a single CV fold for all models"""
    smiTrain = smiles.iloc[trainIndex].tolist()
    smiTest = smiles.iloc[testIndex].tolist()
    yTrain = Y[trainIndex]
    yTest = Y[testIndex]
    
    mcwFold = deepcopy(mcw)
    mcwFold.fit(smiTrain, yTrain)
    
    xTrain = mcwFold.transform(smiTrain)
    xTest = mcwFold.transform(smiTest)
    
    foldResults = {}
    
    # SVR
    gridSvr = GridSearchCV(SVR(), paramGrids['SVR'], cv=3, refit=True, n_jobs=nJobsInner,
                           scoring='neg_mean_absolute_error', verbose=0)
    gridSvr.fit(xTrain, yTrain)
    foldResults['SVR'] = {
        'trainPred': gridSvr.predict(xTrain), 'trainObs': yTrain,
        'testPred': gridSvr.predict(xTest), 'testObs': yTest,
        'bestParams': gridSvr.best_params_
    }
    
    # Random Forest
    gridRf = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1),
                          paramGrids['RandomForest'], cv=3, refit=True, n_jobs=nJobsInner,
                          scoring='neg_mean_absolute_error', verbose=0)
    gridRf.fit(xTrain, yTrain)
    foldResults['RandomForest'] = {
        'trainPred': gridRf.predict(xTrain), 'trainObs': yTrain,
        'testPred': gridRf.predict(xTest), 'testObs': yTest,
        'bestParams': gridRf.best_params_
    }
    
    # XGBoost
    gridXgb = GridSearchCV(XGBRegressor(random_state=42, tree_method='hist', verbosity=0, n_jobs=1),
                           paramGrids['XGBoost'], cv=3, refit=True, n_jobs=nJobsInner,
                           scoring='neg_mean_absolute_error', verbose=0)
    gridXgb.fit(xTrain, yTrain)
    foldResults['XGBoost'] = {
        'trainPred': gridXgb.predict(xTrain), 'trainObs': yTrain,
        'testPred': gridXgb.predict(xTest), 'testObs': yTest,
        'bestParams': gridXgb.best_params_
    }
    
    # Neural Network
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    
    gridNn = GridSearchCV(MLPRegressor(random_state=42, early_stopping=True),
                          paramGrids['NeuralNetwork'], cv=3, refit=True, n_jobs=nJobsInner,
                          scoring='neg_mean_absolute_error', verbose=0)
    gridNn.fit(xTrainScaled, yTrain)
    foldResults['NeuralNetwork'] = {
        'trainPred': gridNn.predict(xTrainScaled), 'trainObs': yTrain,
        'testPred': gridNn.predict(xTestScaled), 'testObs': yTest,
        'bestParams': gridNn.best_params_
    }
    
    return foldResults


def trainAndEvaluateModels(smiles, Y, kf, mcw, paramGrids, nJobs=4):
    """Train models with parallelized cross-validation"""
    nCores = multiprocessing.cpu_count()
    numFolds = kf.get_n_splits()
    nJobsOuter = min(numFolds, max(1, nCores // 4))
    nJobsInner = max(1, nCores // nJobsOuter)
    
    foldResultsList = Parallel(n_jobs=nJobsOuter, verbose=0)(
        delayed(processSingleFold)(
            foldId, trainIdx, testIdx, smiles, Y, mcw, paramGrids, nJobsInner
        )
        for foldId, (trainIdx, testIdx) in enumerate(kf.split(smiles), 1)
    )
    
    # Aggregate results
    modelNames = ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']
    results = {name: {'trainPred': [], 'trainObs': [], 'testPred': [], 'testObs': [], 'bestParams': []}
               for name in modelNames}
    
    for foldResult in foldResultsList:
        for modelName in modelNames:
            for key in ['trainPred', 'trainObs', 'testPred', 'testObs']:
                results[modelName][key].extend(foldResult[modelName][key])
            results[modelName]['bestParams'].append(foldResult[modelName]['bestParams'])
    
    # Convert to numpy arrays
    for modelName in results:
        for key in ['trainPred', 'trainObs', 'testPred', 'testObs']:
            results[modelName][key] = np.array(results[modelName][key])
    
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
    
    report.append(f"Best Model: {bestModel} (Test R² - {bestR2:.3f})")
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
        
        report.append(f"{modelName:15s}: Train R²-{trainR2:.3f}, Test R²-{testR2:.3f}, Gap-{overfitGap:.3f} ({status})")
    
    report.append("-"*80)
    
    fullReport = "\n".join(report)
    with open(savePath, 'w') as f:
        f.write(fullReport)
    
    return fullReport


def trainAndSaveModel(smiles, Y, modelName, bestParams, mcw, saveDir, prefix, randomState=42):
    """Train a specific model on full data and save"""
    mcwFull = deepcopy(mcw)
    mcwFull.fit(smiles.tolist(), Y)
    xFull = mcwFull.transform(smiles.tolist())
    
    if modelName == 'SVR':
        finalModel = SVR(**bestParams)
        finalModel.fit(xFull, Y)
        
    elif modelName == 'RandomForest':
        finalModel = RandomForestRegressor(**bestParams, random_state=randomState, n_jobs=1)
        finalModel.fit(xFull, Y)
        
    elif modelName == 'XGBoost':
        finalModel = XGBRegressor(**bestParams, random_state=randomState, tree_method='hist', verbosity=0, n_jobs=1)
        finalModel.fit(xFull, Y)
        
    elif modelName == 'NeuralNetwork':
        scaler = StandardScaler()
        xFullScaled = scaler.fit_transform(xFull)
        finalModel = MLPRegressor(**bestParams, random_state=randomState, early_stopping=True)
        finalModel.fit(xFullScaled, Y)
        dump(scaler, os.path.join(saveDir, f'{prefix}_{modelName}_scaler.joblib'))
    
    # Save model and MACAW with model name in filename
    dump(finalModel, os.path.join(saveDir, f'{prefix}_{modelName}_regr_pred.joblib'))
    dump(mcwFull, os.path.join(saveDir, f'{prefix}_{modelName}_macaw_model.joblib'))
    
    return finalModel, mcwFull


def trainAndSaveAllModels(smiles, Y, modelResults, mcw, saveDir, prefix, randomState=42):
    """Train all models on full data and save"""
    modelNames = ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']
    savedModels = {}
    
    for modelName in modelNames:
        print(f"Training {modelName} on full dataset...")
        bestParams = modelResults[modelName]['bestParams'][0]
        finalModel, mcwFull = trainAndSaveModel(
            smiles, Y, modelName, bestParams, mcw, saveDir, prefix, randomState
        )
        savedModels[modelName] = {'model': finalModel, 'macaw': mcwFull}
    
    return savedModels


def main():
    """Main execution"""
    startTime = time.time()
    
    # Load configuration
    config = parseArguments()
    
    print(f"Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"")
    
    # Read file to get shape
    dfInput = pd.read_csv(config['inputFile'])
    print(f"Input file: {config['inputFile']} (shape: {dfInput.shape})")
    
    print(f"Output Directory: {config['outputDir']}")
    print(f"Prefix: {config['filePrefix']}")
    
    # Load data
    smiles, Y = loadAndPrepareData(config)
    print(f"Loaded {len(smiles)} valid samples (SMILES with pPotency) after validation")
    
    # Initialize MACAW
    mcw = MACAW(
        type_fp=config['macawTypeFp'],
        metric=config['macawMetric'],
        n_components=config['macawNComponents'],
        n_landmarks=config['macawNLandmarks'],
        random_state=config['randomState']
    )
    
    # Initialize KFold and parameter grids
    kf = KFold(n_splits=config['nFolds'], shuffle=True, random_state=config['randomState'])
    paramGrids = getParamGrids()
    
    # Train and evaluate models
    print("Training models...")
    modelResults = trainAndEvaluateModels(smiles, Y, kf, mcw, paramGrids, config['nJobs'])
    
    # Calculate metrics
    dfComparison = calculateMetricsTable(modelResults)
    print("\nMetrics Summary:")
    print(dfComparison.to_string(index=False))
    
    # Save results
    saveResults(modelResults, dfComparison, config['outputDir'], config['filePrefix'])
    
    # Generate and save report
    reportPath = os.path.join(config['outputDir'], f'{config["filePrefix"]}_report.txt')
    report = generateReport(modelResults, dfComparison, reportPath)
    print(f"\n{report}")
    
    # Train and save ALL models on full dataset
    print(f"\nTraining all models on full dataset...")
    savedModels = trainAndSaveAllModels(smiles, Y, modelResults, mcw, 
                                        config['outputDir'], config['filePrefix'], 
                                        config['randomState'])
    
    print(f"\nAll models saved:")
    for modelName in savedModels.keys():
        print(f"  - {config['filePrefix']}_{modelName}_regr_pred.joblib")
        print(f"  - {config['filePrefix']}_{modelName}_macaw_model.joblib")
        if modelName == 'NeuralNetwork':
            print(f"  - {config['filePrefix']}_{modelName}_scaler.joblib")
    
    print(f"\nCompleted in {time.time() - startTime:.1f}s")
    print(f"Results saved to: {config['outputDir']}")


if __name__ == "__main__":
    main()
