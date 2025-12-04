#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Model Comparison
Usage: python Step4_MLmodelComparison.py --config config.yaml
   or: python Step4_MLmodelComparison.py [command line args]
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
import gc

import numpy as np
import pandas as pd
from copy import deepcopy

from rdkit import Chem
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from joblib import Parallel, delayed, dump
import multiprocessing

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with 'pip install psutil' for memory monitoring")

sys.path.append('../')
from macaw import MACAW


# ----------------------------------------------------------------------------
# CONFIGURATION SECTION
# ----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Input/Output paths
    'inputFile': '/users/sghosh6/DTRA_project/MACAW/DrugDesignData/modelBuildingData/allvirusData_chEMBL_noDuplicates_MLready.csv',
    'outputDir': '/users/sghosh6/DTRA_project/MACAW/DrugDesignData/Results/virusCommandline_10fold/wDuplicates_fineGrid/',
    'filePrefix': 'noDuplicates',
    
    # Data columns
    'smilesColumn': 'Smiles',
    'targetColumn': 'pPotency',
    'filterColumns': ['Smiles', 'virusClassifier', 'pPotency'],
    
    # Model parameters
    'nFolds': 10,
    'randomState': 42,
    'nJobs': 4,
    'nSamples': None,
    'testSize': 0.2,
    
    # MACAW parameters
    'macawTypeFp': 'atompairs',
    'macawMetric': 'sokal',
    'macawNComponents': 15,
    'macawNLandmarks': 200,
    
    # Batch processing parameters
    'batchSize': 10000,  # Process 20K samples at a time
    'enableBatching': True,  # Auto-enable for large datasets
    'batchThreshold': 50000,  # Enable batching if dataset > 50K
    
    # Optimization parameters
    'parallelStrategy': 'folds',
    'memoryEfficient': True,
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
        elif key in ['nFolds', 'randomState', 'nJobs', 'nSamples', 'macawNComponents', 
                     'macawNLandmarks', 'batchSize', 'batchThreshold']:
            config[key] = int(value) if value != 'None' and value is not None else None
        elif key in ['testSize']:
            config[key] = float(value) if value != 'None' and value is not None else 0.2
        elif key == 'filterColumns':
            if isinstance(value, str):
                config[key] = [x.strip() for x in value.split(',')] if value != 'None' else None
            else:
                config[key] = value
        elif key in ['memoryEfficient', 'enableBatching']:
            config[key] = bool(value)
        else:
            config[key] = value
    
    return config


def parseArguments():
    """Parse command line arguments or config file"""
    parser = argparse.ArgumentParser(description='Multi-Model Comparison for Drug Discovery - Batch Processing')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (YAML format)')
    parser.add_argument('--inputFile', type=str, default=None)
    parser.add_argument('--outputDir', type=str, default=None)
    parser.add_argument('--filePrefix', type=str, default=None)
    parser.add_argument('--nSamples', type=int, default=None)
    parser.add_argument('--nFolds', type=int, default=None)
    parser.add_argument('--nJobs', type=int, default=None)
    parser.add_argument('--randomState', type=int, default=None)
    parser.add_argument('--testSize', type=float, default=None)
    parser.add_argument('--batchSize', type=int, default=None)
    parser.add_argument('--parallelStrategy', type=str, choices=['folds', 'models'], default=None)
    parser.add_argument('--memoryEfficient', action='store_true')
    parser.add_argument('--disableBatching', action='store_true')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        yamlConfig = parseYamlConfig(args.config)
        config.update(yamlConfig)
    
    for key in ['inputFile', 'outputDir', 'filePrefix', 'nSamples', 'nFolds', 
                'nJobs', 'randomState', 'testSize', 'parallelStrategy', 'batchSize']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    
    if args.memoryEfficient:
        config['memoryEfficient'] = True
        
    if args.disableBatching:
        config['enableBatching'] = False
    
    return config


def loadAndPrepareData(config):
    """Load and prepare dataset - works with any columns specified in filterColumns"""
    df = pd.read_csv(config['inputFile'])
    
    print(f"Original columns in file: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Filter columns if specified
    if config['filterColumns']:
        print(f"Requested filterColumns: {config['filterColumns']}")
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
        print(f"Shape after filtering: {df.shape}")
    
    # Find smiles and target columns (case-insensitive)
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
    
    print(f"Initial data size: {len(smiles)} rows")
    
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
    
    print(f"Final dataset size: {len(smiles)} samples")
    printMemoryUsage("after data loading")
    
    return smiles, Y



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

def trainSingleModel(modelName, xTrain, yTrain, xTest, yTest, paramGrid, randomState=42):
    """Train a single model with grid search - NO internal parallelism"""
    
    if modelName == 'SVR':
        grid = GridSearchCV(
            SVR(), paramGrid, cv=3, refit=True, n_jobs=1,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid.fit(xTrain, yTrain)
        
    elif modelName == 'RandomForest':
        grid = GridSearchCV(
            RandomForestRegressor(random_state=randomState, n_jobs=1),
            paramGrid, cv=3, refit=True, n_jobs=1,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid.fit(xTrain, yTrain)
        
    elif modelName == 'XGBoost':
        grid = GridSearchCV(
            XGBRegressor(random_state=randomState, tree_method='hist', verbosity=0, n_jobs=1),
            paramGrid, cv=3, refit=True, n_jobs=1,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid.fit(xTrain, yTrain)
        
    elif modelName == 'NeuralNetwork':
        scaler = StandardScaler()
        xTrainScaled = scaler.fit_transform(xTrain)
        xTestScaled = scaler.transform(xTest)
        
        grid = GridSearchCV(
            MLPRegressor(random_state=randomState, early_stopping=True),
            paramGrid, cv=3, refit=True, n_jobs=1,
            scoring='neg_mean_absolute_error', verbose=0
        )
        grid.fit(xTrainScaled, yTrain)
        
        return {
            'trainPred': grid.predict(xTrainScaled),
            'trainObs': yTrain,
            'testPred': grid.predict(xTestScaled),
            'testObs': yTest,
            'bestParams': grid.best_params_
        }
    
    return {
        'trainPred': grid.predict(xTrain),
        'trainObs': yTrain,
        'testPred': grid.predict(xTest),
        'testObs': yTest,
        'bestParams': grid.best_params_
    }


def processSingleFold(foldId, trainIndex, testIndex, smilesArray, Y, macawParams, 
                      paramGrids, modelNames, randomState=42, memoryEfficient=True):
    """Process a single CV fold for all models - memory optimized"""
    
    # Extract data for this fold
    smiTrain = [smilesArray[i] for i in trainIndex]
    smiTest = [smilesArray[i] for i in testIndex]
    yTrain = Y[trainIndex]
    yTest = Y[testIndex]
    
    # Create and fit MACAW for this fold
    mcwFold = MACAW(**macawParams)
    mcwFold.fit(smiTrain, yTrain)
    
    xTrain = mcwFold.transform(smiTrain)
    xTest = mcwFold.transform(smiTest)
    
    # Free memory if in memory-efficient mode
    if memoryEfficient:
        del smiTrain, smiTest
        gc.collect()
    
    foldResults = {}
    
    # Train each model sequentially to save memory
    for modelName in modelNames:
        foldResults[modelName] = trainSingleModel(
            modelName, xTrain, yTrain, xTest, yTest, 
            paramGrids[modelName], randomState
        )
        
        # Free memory after each model
        if memoryEfficient:
            gc.collect()
    
    # Clean up
    del mcwFold, xTrain, xTest
    gc.collect()
    
    return foldResults


def trainAndEvaluateModels_FoldParallel(smiles, Y, kf, macawParams, paramGrids, 
                                         nJobs=4, randomState=42, memoryEfficient=True):
    """Train models with parallelization across folds - OPTIMIZED"""
    
    modelNames = ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']
    
    # Convert smiles to list once
    smilesArray = smiles.tolist()
    
    # Determine number of parallel jobs
    numFolds = kf.get_n_splits()
    nJobsToUse = min(nJobs, numFolds)
    
    print(f"  Using {nJobsToUse} parallel workers for {numFolds} folds")
    
    # Parallelize across folds ONLY
    foldResultsList = Parallel(n_jobs=nJobsToUse, verbose=1, backend='loky')(
        delayed(processSingleFold)(
            foldId, trainIdx, testIdx, smilesArray, Y, macawParams,
            paramGrids, modelNames, randomState, memoryEfficient
        )
        for foldId, (trainIdx, testIdx) in enumerate(kf.split(smiles), 1)
    )
    
    # Aggregate results
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
    
    # Clean up
    del foldResultsList
    gc.collect()
    
    return results


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def processBatch(batchId, smilesBatch, yBatch, kf, macawParams, paramGrids, 
                 nJobs, randomState, memoryEfficient):
    """Process a single batch of data through CV"""
    
    print(f"\n{'*'*80}")
    print(f"BATCH {batchId}: Processing {len(smilesBatch)} samples")
    print(f"{'*'*80}")
    printMemoryUsage(f"Batch {batchId} start")
    
    # Run CV on this batch
    batchResults = trainAndEvaluateModels_FoldParallel(
        smilesBatch, yBatch, kf, macawParams, paramGrids,
        nJobs, randomState, memoryEfficient
    )
    
    printMemoryUsage(f"Batch {batchId} end")
    gc.collect()
    
    return batchResults


def mergeBatchResults(batchResultsList):
    """Merge results from multiple batches"""
    
    print(f"\n{'*'*80}")
    print(f"MERGING RESULTS FROM {len(batchResultsList)} BATCHES")
    print(f"{'*'*80}")
    
    modelNames = list(batchResultsList[0].keys())
    
    # Initialize merged results
    mergedResults = {
        name: {
            'trainPred': [], 
            'trainObs': [], 
            'testPred': [], 
            'testObs': [], 
            'bestParams': []
        }
        for name in modelNames
    }
    
    # Merge predictions from all batches
    for batchResults in batchResultsList:
        for modelName in modelNames:
            for key in ['trainPred', 'trainObs', 'testPred', 'testObs']:
                mergedResults[modelName][key].append(batchResults[modelName][key])
            
            # Collect all best params
            mergedResults[modelName]['bestParams'].extend(batchResults[modelName]['bestParams'])
    
    # Concatenate arrays
    for modelName in modelNames:
        for key in ['trainPred', 'trainObs', 'testPred', 'testObs']:
            mergedResults[modelName][key] = np.concatenate(mergedResults[modelName][key])
    
    # Report merged sizes
    for modelName in modelNames:
        print(f"  {modelName}: {len(mergedResults[modelName]['testPred'])} total predictions")
    
    gc.collect()
    return mergedResults


def trainAndEvaluateModels_Batched(smiles, Y, config, macawParams, paramGrids):
    """
    Train models using batch processing for large datasets
    Splits data into batches, processes each batch independently, then merges results
    """
    
    nSamples = len(smiles)
    batchSize = config['batchSize']
    nBatches = (nSamples + batchSize - 1) // batchSize
    
    print(f"\n{'*'*80}")
    print(f"BATCH PROCESSING MODE")
    print(f"{'*'*80}")
    print(f"Total samples: {nSamples:,}")
    print(f"Batch size: {batchSize:,}")
    print(f"Number of batches: {nBatches}")
    print(f"Parallel jobs per batch: {config['nJobs']}")
    print(f"CV folds: {config['nFolds']}")
    print(f"{'*'*80}")
    
    # Initialize KFold
    kf = KFold(n_splits=config['nFolds'], shuffle=True, random_state=config['randomState'])
    
    # Process each batch
    batchResultsList = []
    
    for batchIdx in range(nBatches):
        startIdx = batchIdx * batchSize
        endIdx = min(startIdx + batchSize, nSamples)
        
        # Extract batch
        smilesBatch = smiles.iloc[startIdx:endIdx].reset_index(drop=True)
        yBatch = Y[startIdx:endIdx]
        
        print(f"\nBatch {batchIdx + 1}/{nBatches}: Samples {startIdx:,} to {endIdx:,}")
        
        # Process this batch
        batchResults = processBatch(
            batchIdx + 1, smilesBatch, yBatch, kf, macawParams, paramGrids,
            config['nJobs'], config['randomState'], config['memoryEfficient']
        )
        
        batchResultsList.append(batchResults)
        
        # Clear memory
        del smilesBatch, yBatch
        gc.collect()
        
        printMemoryUsage(f"after batch {batchIdx + 1}")
    
    # Merge all batch results
    mergedResults = mergeBatchResults(batchResultsList)
    
    # Clear batch results
    del batchResultsList
    gc.collect()
    
    return mergedResults


# ============================================================================
# METRICS AND SAVING FUNCTIONS
# ============================================================================

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
    """Save CV predictions and metrics"""
    os.makedirs(saveDir, exist_ok=True)
    
    dfMetrics.to_csv(os.path.join(saveDir, f'{prefix}_metrics_summary.csv'), index=False)
    
    for modelName, data in results.items():
        # Test predictions (from CV)
        dfTest = pd.DataFrame({
            'Observed': data['testObs'],
            'Predicted': data['testPred'],
            'Residual': data['testObs'] - data['testPred'],
            'Absolute_Error': np.abs(data['testObs'] - data['testPred'])
        })
        dfTest.to_csv(os.path.join(saveDir, f'{prefix}_predictions_test_{modelName}.csv'), index=False)
        
        # Train predictions (from CV)
        dfTrain = pd.DataFrame({
            'Observed': data['trainObs'],
            'Predicted': data['trainPred'],
            'Residual': data['trainObs'] - data['trainPred'],
            'Absolute_Error': np.abs(data['trainObs'] - data['trainPred'])
        })
        dfTrain.to_csv(os.path.join(saveDir, f'{prefix}_predictions_train_{modelName}.csv'), index=False)


def trainAndSaveModel(smiles, Y, modelName, bestParams, macawParams, saveDir, prefix, 
                      randomState=42, testSize=0.2):
    """Train a specific model with train/test split and save comprehensive results"""
    
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
        
        # Save feature importances
        featImportance = pd.DataFrame({
            'Feature': [f'MACAW_{i}' for i in range(xTrain.shape[1])],
            'Importance': finalModel.feature_importances_
        }).sort_values('Importance', ascending=False)
        featImportance.to_csv(
            os.path.join(saveDir, f'{prefix}_{modelName}_feature_importances.csv'), 
            index=False
        )
        
    elif modelName == 'XGBoost':
        finalModel = XGBRegressor(**bestParams, random_state=randomState, 
                                 tree_method='hist', verbosity=0, n_jobs=1)
        finalModel.fit(xTrain, yTrain)
        trainPred = finalModel.predict(xTrain)
        testPred = finalModel.predict(xTest)
        
        # Save feature importances
        featImportance = pd.DataFrame({
            'Feature': [f'MACAW_{i}' for i in range(xTrain.shape[1])],
            'Importance': finalModel.feature_importances_
        }).sort_values('Importance', ascending=False)
        featImportance.to_csv(
            os.path.join(saveDir, f'{prefix}_{modelName}_feature_importances.csv'), 
            index=False
        )
        
    elif modelName == 'NeuralNetwork':
        scaler = StandardScaler()
        xTrainScaled = scaler.fit_transform(xTrain)
        xTestScaled = scaler.transform(xTest)
        
        finalModel = MLPRegressor(**bestParams, random_state=randomState, early_stopping=True)
        finalModel.fit(xTrainScaled, yTrain)
        trainPred = finalModel.predict(xTrainScaled)
        testPred = finalModel.predict(xTestScaled)
        
        dump(scaler, os.path.join(saveDir, f'{prefix}_{modelName}_scaler.joblib'))
    
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
    del xTrain, xTest, mcwFull, trainPred, testPred
    gc.collect()
    
    return finalModel, metrics


def trainAndSaveAllModels(smiles, Y, modelResults, macawParams, saveDir, prefix, 
                         randomState=42, testSize=0.2):
    """Train all models on full data with train/test split and save comprehensive results"""
    modelNames = ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']
    savedModels = {}
    fullDataMetrics = []
    
    print(f"\n{'*'*80}")
    print(f"TRAINING FINAL MODELS ON FULL DATASET")
    print(f"{'*'*80}")
    print(f"Training on {(1-testSize)*100:.0f}% of data, testing on {testSize*100:.0f}%")
    
    for modelName in modelNames:
        print(f"\nTraining {modelName}...")
        printMemoryUsage(f"before {modelName}")
        
        # Use most common best params from CV
        bestParamsList = modelResults[modelName]['bestParams']
        # Simple majority vote for each parameter
        from collections import Counter
        bestParams = {}
        if bestParamsList:
            for key in bestParamsList[0].keys():
                values = [params[key] for params in bestParamsList]
                bestParams[key] = Counter(values).most_common(1)[0][0]
        
        finalModel, metrics = trainAndSaveModel(
            smiles, Y, modelName, bestParams, macawParams, saveDir, prefix, 
            randomState, testSize
        )
        savedModels[modelName] = finalModel
        fullDataMetrics.append(metrics)
        
        printMemoryUsage(f"after {modelName}")
        gc.collect()
    
    # Save full data metrics summary
    dfFullMetrics = pd.DataFrame(fullDataMetrics)
    dfFullMetrics.to_csv(
        os.path.join(saveDir, f'{prefix}_fullData_metrics.csv'), 
        index=False
    )
    
    print("\n" + "="*80)
    print("FULL DATA HOLDOUT PERFORMANCE")
    print("="*80)
    print(dfFullMetrics[['Model', 'Train_R2', 'Train_MAE', 'Test_R2', 'Test_MAE', 'N_train', 'N_test']].to_string(index=False))
    
    return savedModels, dfFullMetrics


def generateComprehensiveReport(results, dfMetrics, dfFullMetrics, savePath):
    """Generate comprehensive summary report including CV and holdout results"""
    report = []
    report.append("="*80)
    report.append("MODEL COMPARISON COMPREHENSIVE REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Cross-validation results
    report.append("-"*80)
    report.append("CROSS-VALIDATION RESULTS")
    report.append("-"*80)
    bestIdx = dfMetrics['Test_R2'].idxmax()
    bestModel = dfMetrics.iloc[bestIdx]['Model']
    bestR2 = dfMetrics.iloc[bestIdx]['Test_R2']
    
    report.append(f"Best Model (CV): {bestModel} (Test R² = {bestR2:.3f})")
    report.append("")
    report.append("CV Performance Metrics:")
    report.append(dfMetrics.to_string(index=False))
    report.append("")
    
    report.append("Overfitting Analysis (CV):")
    for modelName, data in results.items():
        trainR2 = r2_score(data['trainObs'], data['trainPred'])
        testR2 = r2_score(data['testObs'], data['testPred'])
        overfitGap = trainR2 - testR2
        
        status = ("Excellent" if overfitGap < 0.05 else "Good" if overfitGap < 0.10 
                 else "Moderate" if overfitGap < 0.15 else "High overfitting")
        
        report.append(f"  {modelName:15s}: Train R²={trainR2:.3f}, Test R²={testR2:.3f}, Gap={overfitGap:.3f} ({status})")
    
    # Full data holdout results
    report.append("")
    report.append("-"*80)
    report.append("HOLDOUT VALIDATION RESULTS (Full Data Split)")
    report.append("-"*80)
    bestFullIdx = dfFullMetrics['Test_R2'].idxmax()
    bestFullModel = dfFullMetrics.iloc[bestFullIdx]['Model']
    bestFullR2 = dfFullMetrics.iloc[bestFullIdx]['Test_R2']
    
    report.append(f"Best Model (Holdout): {bestFullModel} (Test R² = {bestFullR2:.4f})")
    report.append("")
    report.append("Holdout Performance Metrics:")
    report.append(dfFullMetrics[['Model', 'Train_R2', 'Train_MAE', 'Train_RMSE', 
                                  'Test_R2', 'Test_MAE', 'Test_RMSE']].to_string(index=False))
    report.append("")
    
    report.append("Overfitting Analysis (Holdout):")
    for _, row in dfFullMetrics.iterrows():
        overfitGap = row['Train_R2'] - row['Test_R2']
        status = ("Excellent" if overfitGap < 0.05 else "Good" if overfitGap < 0.10 
                 else "Moderate" if overfitGap < 0.15 else "High overfitting")
        report.append(f"  {row['Model']:15s}: Train R²={row['Train_R2']:.3f}, Test R²={row['Test_R2']:.3f}, Gap={overfitGap:.3f} ({status})")
    
    # Comparison
    report.append("")
    report.append("-"*80)
    report.append("CV vs HOLDOUT COMPARISON")
    report.append("-"*80)
    comparison = []
    for modelName in ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']:
        cvR2 = dfMetrics[dfMetrics['Model'] == modelName]['Test_R2'].values[0]
        holdoutR2 = dfFullMetrics[dfFullMetrics['Model'] == modelName]['Test_R2'].values[0]
        diff = holdoutR2 - cvR2
        comparison.append(f"  {modelName:15s}: CV Test R²={cvR2:.3f}, Holdout Test R²={holdoutR2:.3f}, Diff={diff:+.3f}")
    
    for line in comparison:
        report.append(line)
    
    report.append("")
    report.append("="*80)
    
    fullReport = "\n".join(report)
    with open(savePath, 'w') as f:
        f.write(fullReport)
    
    return fullReport


def main():
    """Main execution with batch processing support"""
    startTime = time.time()
    
    # Load configuration
    config = parseArguments()
    
    print(f"{'*'*80}")
    print(f"MULTI-MODEL COMPARISON - BATCH PROCESSING VERSION")
    print(f"{'*'*80}")
    print(f"Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"")
    
    # Validate nJobs
    maxCores = multiprocessing.cpu_count()
    if config['nJobs'] > maxCores:
        print(f"Warning: nJobs ({config['nJobs']}) > available cores ({maxCores}). Using {maxCores}.")
        config['nJobs'] = maxCores
    
    # Load data
    print(f"{'*'*80}")
    print("DATA LOADING AND PREPARATION")
    print(f"{'*'*80}")
    smiles, Y = loadAndPrepareData(config)
    
    # Determine if batching should be used
    useBatching = config['enableBatching'] and len(smiles) >= config['batchThreshold']
    
    if useBatching:
        print(f"\n  LARGE DATASET DETECTED: {len(smiles):,} samples")
        print(f"   Batch processing ENABLED (threshold: {config['batchThreshold']:,})")
        print(f"   Batch size: {config['batchSize']:,}")
    else:
        print(f"\n  Dataset size: {len(smiles):,} samples")
        print(f"   Using standard processing (no batching)")
    
    # MACAW parameters
    macawParams = {
        'type_fp': config['macawTypeFp'],
        'metric': config['macawMetric'],
        'n_components': config['macawNComponents'],
        'n_landmarks': config['macawNLandmarks'],
        'random_state': config['randomState']
    }
    
    print(f"\nMACAW Parameters:")
    for key, value in macawParams.items():
        print(f"  {key}: {value}")
    
    print(f"\nModel Training Configuration:")
    print(f"  CV Folds: {config['nFolds']}")
    print(f"  Random State: {config['randomState']}")
    print(f"  Parallel Jobs: {config['nJobs']}")
    print(f"  Memory Efficient: {config['memoryEfficient']}")
    
    # Get parameter grids
    paramGrids = getParamGrids()
    
    # Train and evaluate models
    print(f"\n{'*'*80}")
    print(f"CROSS-VALIDATION TRAINING ({config['nFolds']}-FOLD)")
    print(f"{'*'*80}")
    
    if useBatching:
        # Use batch processing
        modelResults = trainAndEvaluateModels_Batched(
            smiles, Y, config, macawParams, paramGrids
        )
    else:
        # Use standard processing
        kf = KFold(n_splits=config['nFolds'], shuffle=True, random_state=config['randomState'])
        modelResults = trainAndEvaluateModels_FoldParallel(
            smiles, Y, kf, macawParams, paramGrids, 
            config['nJobs'], config['randomState'], config['memoryEfficient']
        )
    
    # Calculate CV metrics
    dfComparison = calculateMetricsTable(modelResults)
    print(f"\n{'*'*80}")
    print("CV METRICS SUMMARY")
    print(f"{'*'*80}")
    print(dfComparison.to_string(index=False))
    
    # Save CV results
    saveResults(modelResults, dfComparison, config['outputDir'], config['filePrefix'])
    print(f"\nCV results saved to: {config['outputDir']}")
    
    # Train and save ALL models on full dataset WITH HOLDOUT SPLIT
    savedModels, dfFullMetrics = trainAndSaveAllModels(
        smiles, Y, modelResults, macawParams, 
        config['outputDir'], config['filePrefix'], 
        config['randomState'], config['testSize']
    )
    
    # Generate comprehensive report
    reportPath = os.path.join(config['outputDir'], f'{config["filePrefix"]}_comprehensive_report.txt')
    report = generateComprehensiveReport(modelResults, dfComparison, dfFullMetrics, reportPath)
    print(f"\n{report}")
    
    # Final summary
    print(f"\n{'*'*80}")
    print("SAVED FILES SUMMARY")
    print(f"{'*'*80}")
    
    print("\n1. MODELS (.joblib files):")
    for modelName in ['SVR', 'RandomForest', 'XGBoost', 'NeuralNetwork']:
        print(f"   {config['filePrefix']}_{modelName}_regr_pred.joblib")
        print(f"   {config['filePrefix']}_{modelName}_macaw_model.joblib")
        if modelName == 'NeuralNetwork':
            print(f"   {config['filePrefix']}_{modelName}_scaler.joblib")
    
    print("\n2. CROSS-VALIDATION PREDICTIONS:")
    print(f"   {config['filePrefix']}_predictions_test_*.csv")
    print(f"   {config['filePrefix']}_predictions_train_*.csv")
    
    print("\n3. HOLDOUT VALIDATION PREDICTIONS:")
    print(f"   {config['filePrefix']}_predictions_fullData_train_*.csv")
    print(f"   {config['filePrefix']}_predictions_fullData_test_*.csv")
    
    print("\n4. METRICS:")
    print(f"   {config['filePrefix']}_metrics_summary.csv")
    print(f"   {config['filePrefix']}_fullData_metrics.csv")
    
    print("\n5. FEATURE IMPORTANCES:")
    print(f"   {config['filePrefix']}_RandomForest_feature_importances.csv")
    print(f"   {config['filePrefix']}_XGBoost_feature_importances.csv")
    
    print("\n6. COMPREHENSIVE REPORT:")
    print(f"   {config['filePrefix']}_comprehensive_report.txt")
    
    elapsed = time.time() - startTime
    print(f"\n{'*'*80}")
    print(f"COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"All results saved to: {config['outputDir']}")
    printMemoryUsage("final")
    print(f"{'*'*80}")


if __name__ == "__main__":
    main()