#!/usr/bin/env python3

import os
import time
import argparse
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# Reduce thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def loadConfig(configFile):
    with open(configFile, "r") as fileObj:
        configDict = yaml.safe_load(fileObj)
    if configDict is None:
        raise ValueError("Config file is empty.")
    return configDict


def validateConfig(configDict):
    requiredKeys = [
        "dataDir",
        "artResultsSubdir",
        "inputFile",
        "requiredInputColumns",
        "batchSize",
        "outputSuffix",
    ]
    missingKeys = [key for key in requiredKeys if key not in configDict]
    if missingKeys:
        raise KeyError(f"Missing required config keys: {missingKeys}")


def buildPaths(configDict):
    dataDir = configDict["dataDir"]
    artResultsSubdir = configDict["artResultsSubdir"]
    inputFile = configDict["inputFile"]
    outputSuffix = configDict.get("outputSuffix", "_wToxicity")

    artResultsDir = os.path.join(dataDir, artResultsSubdir)
    inputCsvPath = os.path.join(artResultsDir, inputFile)

    inputStem = Path(inputFile).stem
    outputCsvPath = os.path.join(artResultsDir, f"{inputStem}{outputSuffix}.csv")
    checkpointCsvPath = os.path.join(
        artResultsDir, f"{inputStem}{outputSuffix}_checkpoint.csv"
    )
    diversityCsvPath = os.path.join(
        artResultsDir,
        f"{inputStem}{configDict.get('diversityOutputSuffix', '_diverseInput')}.csv"
    )

    return {
        "artResultsDir": artResultsDir,
        "inputCsvPath": inputCsvPath,
        "outputCsvPath": outputCsvPath,
        "checkpointCsvPath": checkpointCsvPath,
        "diversityCsvPath": diversityCsvPath,
        "inputFile": inputFile,
    }


def loadAdmetModel():
    from admet_ai import ADMETModel
    return ADMETModel()


def readInputCsv(inputCsvPath, requiredInputColumns, selectSMILES=None):
    if not os.path.exists(inputCsvPath):
        raise FileNotFoundError(f"Input file not found: {inputCsvPath}")

    inputDF = pd.read_csv(inputCsvPath)

    missingColumns = [col for col in requiredInputColumns if col not in inputDF.columns]
    if missingColumns:
        raise ValueError(f"Missing required input columns: {missingColumns}")

    inputDF = inputDF[requiredInputColumns].copy()

    if selectSMILES is not None:
        if not isinstance(selectSMILES, int) or selectSMILES <= 0:
            raise ValueError("selectSMILES must be a positive integer or null/None.")
        inputDF = inputDF.head(selectSMILES).copy()

    inputDF = inputDF.reset_index(drop=True)
    return inputDF


def deduplicateForPrediction(inputDF):
    return inputDF[["SMILES"]].dropna().drop_duplicates().reset_index(drop=True)


def canonicalizeSmiles(smi):
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def applyDiversityFilter(
    inputDF,
    similarityThreshold=0.75,
    fingerprintRadius=2,
    fingerprintBits=2048,
    useCanonicalFiltering=True,
    diversityChunkSize=5000,
):
    """
    Chunked, memory-friendlier greedy diversity filter.

    Keeps only selected fingerprints and identifiers in memory.
    """

    uniqueSmilesDF = (
        inputDF[["SMILES"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    totalUniqueInput = len(uniqueSmilesDF)

    selectedSmiles = []
    selectedCanonicalSmiles = []
    selectedFingerprints = []
    seenCanonicalSmiles = set()

    validMolCount = 0

    for startIdx in range(0, totalUniqueInput, diversityChunkSize):
        chunkDF = uniqueSmilesDF.iloc[startIdx:startIdx + diversityChunkSize]

        beforeCount = len(selectedSmiles)

        for smi in chunkDF["SMILES"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            validMolCount += 1
            canonicalSmi = Chem.MolToSmiles(mol, canonical=True)

            if canonicalSmi in seenCanonicalSmiles:
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                fingerprintRadius,
                nBits=fingerprintBits
            )

            if not selectedFingerprints:
                selectedSmiles.append(smi)
                selectedCanonicalSmiles.append(canonicalSmi)
                selectedFingerprints.append(fp)
                seenCanonicalSmiles.add(canonicalSmi)
                continue

            similarities = DataStructs.BulkTanimotoSimilarity(fp, selectedFingerprints)
            maxSimilarity = max(similarities) if similarities else 0.0

            if maxSimilarity < similarityThreshold:
                selectedSmiles.append(smi)
                selectedCanonicalSmiles.append(canonicalSmi)
                selectedFingerprints.append(fp)
                seenCanonicalSmiles.add(canonicalSmi)

        afterCount = len(selectedSmiles)
        print(
            f"Diversity chunk {startIdx // diversityChunkSize + 1} | "
            f"processed unique SMILES: {min(startIdx + diversityChunkSize, totalUniqueInput)}/{totalUniqueInput} | "
            f"new selected: {afterCount - beforeCount} | "
            f"total selected: {afterCount}"
        )

    diverseFilteredDF = pd.DataFrame({
        "SMILES": selectedSmiles,
        "CanonicalSMILES": selectedCanonicalSmiles,
    })

    if useCanonicalFiltering:
        selectedCanonicalSet = set(selectedCanonicalSmiles)

        canonicalSeries = inputDF["SMILES"].apply(canonicalizeSmiles)
        mask = canonicalSeries.isin(selectedCanonicalSet)
        filteredInputDF = inputDF.loc[mask].reset_index(drop=True)
    else:
        selectedSmilesSet = set(selectedSmiles)
        mask = inputDF["SMILES"].isin(selectedSmilesSet)
        filteredInputDF = inputDF.loc[mask].reset_index(drop=True)

    diversityStats = {
        "originalUniqueInputSmiles": totalUniqueInput,
        "originalUniqueValidMolecules": validMolCount,
        "diversityFilteredMolecules": len(diverseFilteredDF),
        "finalFilteredRows": len(filteredInputDF),
    }

    return filteredInputDF, diverseFilteredDF, diversityStats


def splitIntoBatches(itemsList, batchSize):
    for startIdx in range(0, len(itemsList), batchSize):
        yield startIdx, itemsList[startIdx:startIdx + batchSize]


def safePredictBatch(model, smilesBatch):
    predictionsDF = model.predict(smiles=smilesBatch)

    if not isinstance(predictionsDF, pd.DataFrame):
        predictionsDF = pd.DataFrame(predictionsDF)

    predictionsDF = predictionsDF.reset_index(drop=True)

    if len(predictionsDF) != len(smilesBatch):
        raise ValueError(
            f"Prediction batch size mismatch: got {len(predictionsDF)} predictions "
            f"for {len(smilesBatch)} SMILES."
        )

    batchDF = pd.DataFrame({"SMILES": smilesBatch}).reset_index(drop=True)
    batchDF = pd.concat([batchDF, predictionsDF], axis=1)
    return batchDF


def loadCheckpoint(checkpointCsvPath):
    if os.path.exists(checkpointCsvPath):
        checkpointDF = pd.read_csv(checkpointCsvPath)
        if "SMILES" not in checkpointDF.columns:
            raise ValueError("Checkpoint exists but does not contain a SMILES column.")
        checkpointDF = checkpointDF.drop_duplicates(subset="SMILES", keep="last").reset_index(drop=True)
        return checkpointDF
    return None


def writeBatchToCheckpoint(batchPredictionsDF, checkpointCsvPath, writeHeader=False):
    Path(os.path.dirname(checkpointCsvPath)).mkdir(parents=True, exist_ok=True)
    batchPredictionsDF.to_csv(
        checkpointCsvPath,
        mode="a",
        header=writeHeader,
        index=False
    )


def compactCheckpoint(checkpointCsvPath):
    checkpointDF = pd.read_csv(checkpointCsvPath)
    checkpointDF = checkpointDF.drop_duplicates(subset="SMILES", keep="last").reset_index(drop=True)
    checkpointDF.to_csv(checkpointCsvPath, index=False)
    return checkpointDF


def mergePredictionsBack(inputDF, predictionsDF):
    return inputDF.merge(predictionsDF, on="SMILES", how="left")


def main():
    parser = argparse.ArgumentParser(
        description="Run batched ADMET toxicity prediction with optional diversity filtering."
    )
    parser.add_argument(
        "configFile",
        help="Path to YAML config file, e.g. python runToxicity.py toxicityConfig.yaml"
    )
    args = parser.parse_args()

    startTime = time.time()

    configDict = loadConfig(args.configFile)
    validateConfig(configDict)
    paths = buildPaths(configDict)

    selectSMILES = configDict.get("selectSMILES", None)
    batchSize = int(configDict.get("batchSize", 256))
    saveEveryBatch = bool(configDict.get("saveEveryBatch", True))
    resumeFromCheckpoint = bool(configDict.get("resumeFromCheckpoint", True))
    deduplicateSmiles = bool(configDict.get("deduplicateSmiles", True))

    enableDiversityFilter = bool(configDict.get("enableDiversityFilter", False))
    similarityThreshold = float(configDict.get("similarityThreshold", 0.75))
    fingerprintRadius = int(configDict.get("fingerprintRadius", 2))
    fingerprintBits = int(configDict.get("fingerprintBits", 2048))
    diversityChunkSize = int(configDict.get("diversityChunkSize", 5000))
    saveDiversityFilteredInput = bool(configDict.get("saveDiversityFilteredInput", True))
    useCanonicalFiltering = bool(configDict.get("useCanonicalFiltering", True))

    requiredInputColumns = configDict["requiredInputColumns"]

    print("\n================ Toxicity Prediction Run ================")
    print(f"inputFile               : {paths['inputFile']}")
    print(f"ARTresultsDir           : {paths['artResultsDir']}")
    print(f"Input CSV               : {paths['inputCsvPath']}")
    print(f"Output CSV              : {paths['outputCsvPath']}")
    print(f"Checkpoint CSV          : {paths['checkpointCsvPath']}")
    print(f"selectSMILES            : {selectSMILES}")
    print(f"batchSize               : {batchSize}")
    print(f"saveEveryBatch          : {saveEveryBatch}")
    print(f"resumeFromCheckpoint    : {resumeFromCheckpoint}")
    print(f"deduplicateSmiles       : {deduplicateSmiles}")
    print(f"enableDiversityFilter   : {enableDiversityFilter}")
    print(f"similarityThreshold     : {similarityThreshold}")
    print(f"fingerprintRadius       : {fingerprintRadius}")
    print(f"fingerprintBits         : {fingerprintBits}")
    print(f"diversityChunkSize      : {diversityChunkSize}")
    print(f"saveDiversityFiltered   : {saveDiversityFilteredInput}")
    print(f"useCanonicalFiltering   : {useCanonicalFiltering}")
    print("=========================================================\n")

    inputDF = readInputCsv(
        inputCsvPath=paths["inputCsvPath"],
        requiredInputColumns=requiredInputColumns,
        selectSMILES=selectSMILES,
    )
    print(f"Loaded input dataframe shape: {inputDF.shape}")

    if enableDiversityFilter:
        print("\nApplying chemical diversity filter before ADMET prediction...")

        inputDF, diverseFilteredDF, diversityStats = applyDiversityFilter(
            inputDF=inputDF,
            similarityThreshold=similarityThreshold,
            fingerprintRadius=fingerprintRadius,
            fingerprintBits=fingerprintBits,
            useCanonicalFiltering=useCanonicalFiltering,
            diversityChunkSize=diversityChunkSize,
        )

        print(f"Original unique valid molecules: {diversityStats['originalUniqueValidMolecules']}")
        print(f"Diversity-filtered molecules: {diversityStats['diversityFilteredMolecules']}")
        print(f"Filtered input dataframe shape: {inputDF.shape}")

        if saveDiversityFilteredInput:
            Path(os.path.dirname(paths["diversityCsvPath"])).mkdir(parents=True, exist_ok=True)
            inputDF.to_csv(paths["diversityCsvPath"], index=False)
            print(f"Saved diversity-filtered input to: {paths['diversityCsvPath']}")

    if deduplicateSmiles:
        predictionInputDF = deduplicateForPrediction(inputDF)
        print(f"Unique SMILES for prediction: {len(predictionInputDF)}")
    else:
        predictionInputDF = inputDF[["SMILES"]].dropna().reset_index(drop=True)
        print(f"Total SMILES for prediction: {len(predictionInputDF)}")

    allSmiles = predictionInputDF["SMILES"].tolist()
    del predictionInputDF

    processedPredictionsDF = None
    processedSmilesSet = set()

    if resumeFromCheckpoint:
        processedPredictionsDF = loadCheckpoint(paths["checkpointCsvPath"])
        if processedPredictionsDF is not None:
            processedSmilesSet = set(processedPredictionsDF["SMILES"].dropna().tolist())
            print(f"Loaded checkpoint with {len(processedPredictionsDF)} predicted SMILES")

    smilesRemaining = [smi for smi in allSmiles if smi not in processedSmilesSet]
    print(f"Remaining SMILES to process: {len(smilesRemaining)}")

    if len(smilesRemaining) > 0:
        model = loadAdmetModel()

        checkpointExists = os.path.exists(paths["checkpointCsvPath"])
        writeHeader = not checkpointExists or os.path.getsize(paths["checkpointCsvPath"]) == 0

        totalBatches = (len(smilesRemaining) + batchSize - 1) // batchSize

        for batchNum, (_, smilesBatch) in enumerate(
            splitIntoBatches(smilesRemaining, batchSize), start=1
        ):
            batchStartTime = time.time()

            batchPredictionsDF = safePredictBatch(model, smilesBatch)

            if saveEveryBatch:
                writeBatchToCheckpoint(
                    batchPredictionsDF=batchPredictionsDF,
                    checkpointCsvPath=paths["checkpointCsvPath"],
                    writeHeader=writeHeader
                )
                writeHeader = False

            batchElapsed = time.time() - batchStartTime
            print(
                f"Processed batch {batchNum}/{totalBatches} | "
                f"batch size: {len(smilesBatch)} | "
                f"time: {batchElapsed:.2f} s"
            )

        if saveEveryBatch:
            finalPredictionsDF = compactCheckpoint(paths["checkpointCsvPath"])
        else:
            raise RuntimeError(
                "saveEveryBatch=False is not supported in this optimized mode. "
                "Set saveEveryBatch=True."
            )
    else:
        if processedPredictionsDF is None:
            raise RuntimeError("No SMILES processed and no checkpoint found.")
        finalPredictionsDF = processedPredictionsDF.copy()

    print(f"Final unique predictions: {len(finalPredictionsDF)}")

    mergedDF = mergePredictionsBack(inputDF, finalPredictionsDF)
    print(f"Final merged dataframe shape: {mergedDF.shape}")

    Path(os.path.dirname(paths["outputCsvPath"])).mkdir(parents=True, exist_ok=True)
    mergedDF.to_csv(paths["outputCsvPath"], index=False)

    elapsedTime = time.time() - startTime
    print(f"Saved final output to: {paths['outputCsvPath']}")
    print(f"Total runtime: {elapsedTime:.2f} seconds\n")


if __name__ == "__main__":
    main()