#!/usr/bin/env python3
"""
Run MACAW inverse design from a YAML config.

Usage:
    python runMACAW_InvDsg.py config.yaml

This script is designed for shared clusters:
    - conservative BLAS/OpenMP thread defaults
    - loads only required CSV columns unless read_all_columns is true
    - selects a small seed pool before Morgan fingerprint diversity selection
    - writes per-campaign outputs and compact summaries
"""

# Set conservative thread defaults before importing numpy/pandas/sklearn/rdkit.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# Silence noisy RDKit parse warnings by default; controlled again in main.
RDLogger.DisableLog("rdApp.warning")


# -----------------------------------------------------------------------------
# Config and basic utilities
# -----------------------------------------------------------------------------

def loadConfig(configPath):
    with open(configPath, "r") as handle:
        cfg = yaml.safe_load(handle)
    if cfg is None:
        raise ValueError(f"Empty config file: {configPath}")
    return cfg


def requireKeys(mapping, keys, sectionName):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing required key(s) in '{sectionName}': {missing}")


def addPythonPaths(paths):
    for path in paths or []:
        path = str(Path(path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def loadObject(path, label="object"):
    """Load joblib/pickle/cloudpickle-compatible objects."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find {label}: {path}")

    suffix = path.suffix.lower()

    if suffix in [".joblib", ".jl"]:
        import joblib
        return joblib.load(path)

    # ART .cpkl files often require cloudpickle, but fall back to pickle.
    try:
        import cloudpickle
        with open(path, "rb") as handle:
            return cloudpickle.load(handle)
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def canonicalizeSmiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def safeMkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

def readInputCsv(inputPath, columnsCfg, ioCfg):
    inputPath = Path(inputPath).expanduser().resolve()
    if not inputPath.exists():
        raise FileNotFoundError(f"Input CSV not found: {inputPath}")

    smilesCol = columnsCfg["smiles"]
    potencyCol = columnsCfg["potency"]
    stdCol = columnsCfg.get("std")
    sourceCol = columnsCfg.get("source")
    extraCols = columnsCfg.get("additional_seed_columns", []) or []

    maxRows = ioCfg.get("max_input_rows", None)
    readAll = bool(ioCfg.get("read_all_columns", False))

    if readAll:
        return pd.read_csv(inputPath, nrows=maxRows, low_memory=False)

    headerCols = pd.read_csv(inputPath, nrows=0).columns.tolist()
    requestedCols = [smilesCol, potencyCol, stdCol, sourceCol] + extraCols
    useCols = []
    for col in requestedCols:
        if col and col in headerCols and col not in useCols:
            useCols.append(col)

    if smilesCol not in useCols or potencyCol not in useCols:
        raise ValueError(
            f"Input CSV must contain smiles='{smilesCol}' and potency='{potencyCol}'. "
            f"Available columns include: {headerCols[:20]}..."
        )

    return pd.read_csv(inputPath, usecols=useCols, nrows=maxRows, low_memory=False)


# -----------------------------------------------------------------------------
# ART prediction wrapper
# -----------------------------------------------------------------------------

def extractPredictionArray(pred, predictionKeys):
    if isinstance(pred, pd.DataFrame):
        for col in predictionKeys:
            if col in pred.columns:
                return pred[col].to_numpy()
        raise ValueError(f"ART DataFrame output did not contain any prediction column from {predictionKeys}")

    if isinstance(pred, dict):
        for key in predictionKeys:
            if key in pred:
                return np.asarray(pred[key]).ravel()
        raise ValueError(f"ART dict output did not contain any prediction key from {predictionKeys}")

    if isinstance(pred, (tuple, list)):
        if len(pred) == 0:
            raise ValueError("ART prediction returned an empty tuple/list")
        return np.asarray(pred[0]).ravel()

    return np.asarray(pred).ravel()


def buildArtMeanModel(artModel, artCfg):
    featurePrefix = artCfg.get("feature_prefix", "MACAW_")
    predictionKeys = artCfg.get(
        "prediction_keys",
        ["pPotency_prediction", "prediction", "mean", "y_mean"],
    )
    passDataFrame = bool(artCfg.get("pass_dataframe", True))

    def artMeanModelForLibraryEvolver(X):
        X = np.asarray(X)
        if passDataFrame:
            featureDF = pd.DataFrame(
                X,
                columns=[f"{featurePrefix}{i}" for i in range(X.shape[1])],
            )
            pred = artModel.predict(featureDF)
        else:
            pred = artModel.predict(X)
        return extractPredictionArray(pred, predictionKeys)

    return artMeanModelForLibraryEvolver


# -----------------------------------------------------------------------------
# Seed selection
# -----------------------------------------------------------------------------

def makeMorganFp(smiles, radius, nBits):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)


def prepareSeedFrame(inputDF, targetPotency, columnsCfg):
    smilesCol = columnsCfg["smiles"]
    potencyCol = columnsCfg["potency"]
    stdCol = columnsCfg.get("std")

    DF = inputDF.copy()
    DF[smilesCol] = DF[smilesCol].apply(canonicalizeSmiles)
    DF[potencyCol] = pd.to_numeric(DF[potencyCol], errors="coerce")

    if stdCol and stdCol in DF.columns:
        DF[stdCol] = pd.to_numeric(DF[stdCol], errors="coerce")

    DF = DF.dropna(subset=[smilesCol, potencyCol]).copy()
    DF["seedTargetError"] = (DF[potencyCol] - targetPotency).abs()

    # Keep the duplicate SMILES row closest to target potency.
    DF = (
        DF.sort_values("seedTargetError", ascending=True)
        .drop_duplicates(subset=[smilesCol], keep="first")
        .reset_index(drop=True)
    )
    return DF


def selectTargetClosestSeeds(inputDF, cfg):
    columnsCfg = cfg["columns"]
    selectionCfg = cfg["seed_selection"]
    targetPotency = float(cfg["target_potency"])
    nSeeds = int(selectionCfg["n_seeds"])
    smilesCol = columnsCfg["smiles"]

    DF = prepareSeedFrame(inputDF, targetPotency, columnsCfg)
    seedDF = DF.nsmallest(nSeeds, "seedTargetError").reset_index(drop=True)

    seedDF["seedSelectionScore"] = seedDF["seedTargetError"]
    seedDF["seedRank"] = np.arange(1, len(seedDF) + 1)

    return seedDF, seedDF[smilesCol].tolist(), buildStarterDisplay(seedDF, columnsCfg)


def selectDiverseThenTargetClosestSeeds(inputDF, cfg):
    columnsCfg = cfg["columns"]
    selectionCfg = cfg["seed_selection"]
    diversityCfg = selectionCfg.get("diversity", {})

    targetPotency = float(cfg["target_potency"])
    nSeeds = int(selectionCfg["n_seeds"])
    smilesCol = columnsCfg["smiles"]

    candidatePoolFactor = int(diversityCfg.get("candidate_pool_factor", 10))
    diversePoolFactor = int(diversityCfg.get("diverse_pool_factor", 3))
    similarityThreshold = float(diversityCfg.get("similarity_threshold", 0.75))
    diversityWeight = float(diversityCfg.get("diversity_weight", 0.75))
    targetWeight = float(diversityCfg.get("target_weight", 1.0 - diversityWeight))
    radius = int(diversityCfg.get("morgan_radius", 2))
    nBits = int(diversityCfg.get("morgan_bits", 2048))

    DF = prepareSeedFrame(inputDF, targetPotency, columnsCfg)

    # Broad near-target candidate pool first.
    nCandidatePool = min(len(DF), nSeeds * candidatePoolFactor)
    poolDF = DF.head(nCandidatePool).copy().reset_index(drop=True)

    print(f"Candidate pool for diversity selection: {len(poolDF)} molecules")

    poolDF["_fp"] = poolDF[smilesCol].apply(lambda x: makeMorganFp(x, radius, nBits))
    poolDF = poolDF.dropna(subset=["_fp"]).reset_index(drop=True)
    if len(poolDF) == 0:
        raise ValueError("No valid molecules remained after fingerprint generation")

    maxErr = poolDF["seedTargetError"].max()
    poolDF["_normTargetError"] = 0.0 if maxErr == 0 else poolDF["seedTargetError"] / maxErr

    fps = poolDF["_fp"].tolist()
    nDiversePool = min(len(poolDF), max(nSeeds, nSeeds * diversePoolFactor))

    selectedIdx = [0]
    remainingIdx = list(range(1, len(poolDF)))
    maxSimToSelected = np.zeros(len(poolDF), dtype=float)
    selectedMaxSimDict = {0: 0.0}

    while len(selectedIdx) < nDiversePool and remainingIdx:
        lastFp = fps[selectedIdx[-1]]
        simsToLast = np.asarray(DataStructs.BulkTanimotoSimilarity(lastFp, fps), dtype=float)
        maxSimToSelected = np.maximum(maxSimToSelected, simsToLast)

        remainingArray = np.asarray(remainingIdx, dtype=int)
        candidateIdx = remainingArray[maxSimToSelected[remainingArray] <= similarityThreshold]
        if len(candidateIdx) == 0:
            candidateIdx = remainingArray

        candidateScores = (
            targetWeight * poolDF.loc[candidateIdx, "_normTargetError"].to_numpy()
            + diversityWeight * maxSimToSelected[candidateIdx]
        )
        bestIdx = int(candidateIdx[np.argmin(candidateScores)])

        selectedIdx.append(bestIdx)
        selectedMaxSimDict[bestIdx] = float(maxSimToSelected[bestIdx])
        remainingIdx.remove(bestIdx)

    diversePoolDF = poolDF.iloc[selectedIdx].copy().reset_index(drop=True)
    diversePoolDF["_poolIndex"] = selectedIdx
    diversePoolDF["seedMaxSimilarityToPrevious"] = [
        selectedMaxSimDict.get(idx, 0.0) for idx in selectedIdx
    ]
    diversePoolDF["seedDiversityScore"] = 1.0 - diversePoolDF["seedMaxSimilarityToPrevious"]

    # Finally choose target-closest molecules from the diverse intermediate pool.
    seedDF = (
        diversePoolDF.sort_values("seedTargetError", ascending=True)
        .head(nSeeds)
        .reset_index(drop=True)
    )
    seedDF["seedSelectionScore"] = seedDF["seedTargetError"]
    seedDF["seedRank"] = np.arange(1, len(seedDF) + 1)
    seedDF = seedDF.drop(columns=["_fp", "_normTargetError", "_poolIndex"], errors="ignore")

    return seedDF, seedDF[smilesCol].tolist(), buildStarterDisplay(seedDF, columnsCfg)


def buildStarterDisplay(seedDF, columnsCfg):
    smilesCol = columnsCfg["smiles"]
    potencyCol = columnsCfg["potency"]
    stdCol = columnsCfg.get("std")
    sourceCol = columnsCfg.get("source")

    displayCols = [
        col for col in [
            "seedRank",
            smilesCol,
            sourceCol,
            potencyCol,
            stdCol,
            "seedTargetError",
            "seedMaxSimilarityToPrevious",
            "seedDiversityScore",
            "seedSelectionScore",
        ] if col and col in seedDF.columns
    ]
    return seedDF[displayCols].copy()


def selectSeeds(inputDF, cfg):
    method = cfg["seed_selection"].get("method", "diverse_then_target_closest")
    if method == "target_closest":
        return selectTargetClosestSeeds(inputDF, cfg)
    if method == "diverse_then_target_closest":
        return selectDiverseThenTargetClosestSeeds(inputDF, cfg)
    raise ValueError(f"Unsupported seed_selection.method: {method}")


# -----------------------------------------------------------------------------
# Campaign execution and output
# -----------------------------------------------------------------------------

def runOneCampaign(seedSmiles, campaignCfg, randomSeed, cfg, libraryEvolver, mcw, model):
    columnsCfg = cfg["columns"]
    targetPotency = float(cfg["target_potency"])
    smilesCol = columnsCfg["smiles"]
    potencyCol = columnsCfg["potency"]

    label = campaignCfg["label"]
    campaignLabel = f"{label}_seed{randomSeed}"

    print(f"\nRunning {campaignLabel}", flush=True)

    result = libraryEvolver(
        smiles=seedSmiles,
        model=model,
        mcw=mcw,
        spec=targetPotency,
        k1=int(campaignCfg["k1"]),
        k2=int(campaignCfg["k2"]),
        n_rounds=int(campaignCfg["n_rounds"]),
        n_hits=int(campaignCfg["n_hits"]),
        noise_factor=campaignCfg["noise_factor"],
        algorithm=campaignCfg.get("algorithm", "transition"),
        p=campaignCfg.get("p", "empirical"),
        force_new=bool(campaignCfg.get("force_new", True)),
        random_state=int(randomSeed),
    )

    if isinstance(result, tuple) and len(result) >= 2:
        smilesOut, predOut = result[0], result[1]
    else:
        smilesOut = result
        predOut = model(mcw.transform(smilesOut))

    predOut = np.asarray(predOut).ravel()

    outDF = pd.DataFrame({
        smilesCol: smilesOut,
        potencyCol: predOut,
        "target_pPotency": targetPotency,
        "targetError": np.abs(predOut - targetPotency),
        "campaignLabel": campaignLabel,
        "campaignBase": label,
        "randomSeed": int(randomSeed),
        "algorithm": campaignCfg.get("algorithm", "transition"),
        "p": campaignCfg.get("p", "empirical"),
        "k1": int(campaignCfg["k1"]),
        "k2": int(campaignCfg["k2"]),
        "n_rounds": int(campaignCfg["n_rounds"]),
        "n_hits": int(campaignCfg["n_hits"]),
        "noise_factor_schedule": json.dumps(campaignCfg["noise_factor"]),
    })

    outDF[smilesCol] = outDF[smilesCol].apply(canonicalizeSmiles)
    outDF = outDF.dropna(subset=[smilesCol]).drop_duplicates(subset=[smilesCol]).reset_index(drop=True)

    if len(outDF) > 0:
        print(
            f"Finished {campaignLabel}: {len(outDF)} molecules | "
            f"best targetError={outDF['targetError'].min():.4f}",
            flush=True,
        )
    else:
        print(f"Finished {campaignLabel}: no valid molecules returned", flush=True)

    return outDF


def summarizeResults(allResults, cfg):
    columnsCfg = cfg["columns"]
    smilesCol = columnsCfg["smiles"]
    tolerances = cfg.get("output", {}).get("tolerances", [0.10, 0.25, 0.50])

    if len(allResults) == 0:
        return pd.DataFrame(), pd.DataFrame()

    resultDF = (
        pd.concat(allResults, ignore_index=True)
        .sort_values("targetError", ascending=True)
        .drop_duplicates(subset=[smilesCol], keep="first")
        .reset_index(drop=True)
    )
    resultDF["targetMatchRank"] = np.arange(1, len(resultDF) + 1)

    for tolerance in tolerances:
        resultDF[f"within_{str(tolerance).replace('.', 'p')}"] = resultDF["targetError"] <= float(tolerance)

    within025 = "within_0p25"
    if within025 not in resultDF.columns:
        resultDF[within025] = resultDF["targetError"] <= 0.25

    within050 = "within_0p5"
    if within050 not in resultDF.columns:
        resultDF[within050] = resultDF["targetError"] <= 0.50

    summaryDF = (
        resultDF.groupby("campaignBase")
        .agg(
            nUniqueReturned=(smilesCol, "count"),
            bestTargetError=("targetError", "min"),
            medianTargetError=("targetError", "median"),
            nWithin0p25=(within025, "sum"),
        )
        .reset_index()
        .sort_values(["bestTargetError", "nWithin0p25"], ascending=[True, False])
    )

    return resultDF, summaryDF


def saveOutputs(seedDF, starterDF, resultDF, summaryDF, failedDF, cfg):
    outCfg = cfg.get("output", {})
    outputDir = Path(outCfg.get("output_dir", ".")).expanduser().resolve()
    prefix = outCfg.get("prefix", "MACAW_generatedCompound")
    topN = int(outCfg.get("top_n", 500))
    saveCampaigns = bool(outCfg.get("save_outputs", True))

    if not saveCampaigns:
        return

    safeMkdir(outputDir)

    seedDF.to_csv(outputDir / f"{prefix}_diverse_targetClosest_seed_molecules.csv", index=False)
    starterDF.to_csv(outputDir / f"{prefix}_starter_seed_pPotency_values.csv", index=False)

    if len(resultDF) > 0:
        resultDF.to_csv(outputDir / f"{prefix}_multiCampaign_all_results.csv", index=False)
        resultDF.head(topN).to_csv(outputDir / f"{prefix}_multiCampaign_top{topN}.csv", index=False)
        resultDF[resultDF["targetError"] <= 0.25].to_csv(
            outputDir / f"{prefix}_multiCampaign_within_0p25.csv", index=False
        )

    summaryDF.to_csv(outputDir / f"{prefix}_multiCampaign_summary.csv", index=False)

    if len(failedDF) > 0:
        failedDF.to_csv(outputDir / f"{prefix}_failed_campaigns.csv", index=False)

    print(f"\nSaved output files to: {outputDir}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run MACAW inverse design with ART potency scoring")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()

    startTime = time.time()
    cfg = loadConfig(args.config)

    requireKeys(cfg, ["input", "models", "columns", "target_potency", "seed_selection", "campaigns"], "root")
    requireKeys(cfg["input"], ["csv"], "input")
    requireKeys(cfg["models"], ["mcw_path", "art_model_path"], "models")
    requireKeys(cfg["columns"], ["smiles", "potency"], "columns")
    requireKeys(cfg["seed_selection"], ["n_seeds"], "seed_selection")

    if not bool(cfg.get("runtime", {}).get("rdkit_warnings", False)):
        RDLogger.DisableLog("rdApp.*")

    addPythonPaths(cfg.get("runtime", {}).get("extra_python_paths", []))

    from macaw.generators import library_evolver

    inputDF = readInputCsv(
        inputPath=cfg["input"]["csv"],
        columnsCfg=cfg["columns"],
        ioCfg=cfg.get("input", {}),
    )

    print(f"Loaded input rows: {len(inputDF)}", flush=True)

    print("Loading MACAW embedder...", flush=True)
    mcw = loadObject(cfg["models"]["mcw_path"], label="MACAW embedder")

    print("Loading ART model...", flush=True)
    artModel = loadObject(cfg["models"]["art_model_path"], label="ART model")
    artModelFn = buildArtMeanModel(artModel, cfg.get("art", {}))

    seedDF, seedSmiles, starterSeedDF = selectSeeds(inputDF, cfg)
    columnsCfg = cfg["columns"]
    potencyCol = columnsCfg["potency"]

    print(f"targetPotency: {float(cfg['target_potency'])}", flush=True)
    print(f"Selected seed molecules: {len(seedSmiles)}", flush=True)
    print(
        f"Starter seed pPotency range: {seedDF[potencyCol].min():.6f} to {seedDF[potencyCol].max():.6f}",
        flush=True,
    )
    print(
        f"Starter seed targetError range: {seedDF['seedTargetError'].min():.6f} to {seedDF['seedTargetError'].max():.6f}",
        flush=True,
    )
    if "seedDiversityScore" in seedDF.columns:
        print(f"Median seed diversity score: {seedDF['seedDiversityScore'].median():.4f}", flush=True)

    outputDir = Path(cfg.get("output", {}).get("output_dir", ".")).expanduser().resolve()
    prefix = cfg.get("output", {}).get("prefix", "MACAW_generatedCompound")
    safeMkdir(outputDir)
    starterSeedDF.to_csv(outputDir / f"{prefix}_starter_seed_pPotency_values.csv", index=False)
    seedDF.to_csv(outputDir / f"{prefix}_selected_seed_molecules.csv", index=False)

    allResults = []
    failedCampaigns = []
    saveEachCampaign = bool(cfg.get("output", {}).get("save_each_campaign", True))

    for campaignCfg in cfg["campaigns"]:
        for randomSeed in cfg.get("random_seeds", [42]):
            label = campaignCfg["label"]
            campaignLabel = f"{label}_seed{randomSeed}"
            try:
                campaignDF = runOneCampaign(
                    seedSmiles=seedSmiles,
                    campaignCfg=campaignCfg,
                    randomSeed=randomSeed,
                    cfg=cfg,
                    libraryEvolver=library_evolver,
                    mcw=mcw,
                    model=artModelFn,
                )
                if len(campaignDF) > 0:
                    allResults.append(campaignDF)
                    if saveEachCampaign:
                        campaignPath = outputDir / f"{prefix}_{campaignLabel}.csv"
                        campaignDF.to_csv(campaignPath, index=False)
                del campaignDF
                gc.collect()
            except Exception as exc:
                failedCampaigns.append({"campaignLabel": campaignLabel, "errorMessage": str(exc)})
                print(f"FAILED {campaignLabel}: {exc}", flush=True)
                gc.collect()

    resultDF, summaryDF = summarizeResults(allResults, cfg)
    failedDF = pd.DataFrame(failedCampaigns)

    print("\nMULTI-CAMPAIGN SEARCH COMPLETE", flush=True)
    print(f"Total unique generated molecules: {len(resultDF)}", flush=True)
    if len(resultDF) > 0:
        print(f"Molecules within ±0.25 pPotency: {(resultDF['targetError'] <= 0.25).sum()}", flush=True)
    if len(summaryDF) > 0:
        print("\nCampaign summary:", flush=True)
        print(summaryDF.to_string(index=False), flush=True)

    saveOutputs(seedDF, starterSeedDF, resultDF, summaryDF, failedDF, cfg)

    print(f"\nTotal runtime: {(time.time() - startTime) / 60.0:.2f} min", flush=True)


if __name__ == "__main__":
    main()
