#!/usr/bin/env python3
"""
run_trainingSet_similarity.py

General CSV/YAML-driven workflow to compare a training chemical set with one or more
external validation/candidate sets using Morgan fingerprints, nearest-neighbor
Tanimoto similarity, potency distributions, and chemical-space embedding plots.

Example:
    python run_trainingSet_similarity.py -c config_trainingSet_similarity.yaml

HPC example:
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
    python run_trainingSet_similarity.py -c config_trainingSet_similarity.yaml
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import yaml

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a training chemical dataset with one or more external CSV datasets."
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to YAML configuration file."
    )
    return parser.parse_args()


def readYaml(configPath: str) -> Dict[str, Any]:
    with open(configPath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {configPath}")
    return config


def getNested(config: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def sanitizeName(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "unnamed"


def ensureDir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def getBool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def requireColumns(DF: pd.DataFrame, requiredCols: List[str], datasetName: str) -> None:
    missingCols = [col for col in requiredCols if col and col not in DF.columns]
    if missingCols:
        raise ValueError(
            f"Dataset '{datasetName}' is missing required columns: {missingCols}\n"
            f"Available columns: {list(DF.columns)}"
        )


def readCsvDataset(csvPath: str, datasetName: str) -> pd.DataFrame:
    path = Path(csvPath).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found for dataset '{datasetName}': {path}")
    return pd.read_csv(path, low_memory=False)


def selectMolecules(
    DF: pd.DataFrame,
    nToUse: Optional[int] = None,
    sortCol: Optional[str] = None,
    ascending: bool = False,
    randomSeed: int = 42,
) -> pd.DataFrame:
    """Select all molecules, top-N molecules by a column, or random-N molecules."""
    DF = DF.copy()

    if nToUse is None or nToUse >= len(DF):
        return DF

    if sortCol is not None:
        if sortCol not in DF.columns:
            raise ValueError(f"sortCol='{sortCol}' was requested but is not present in dataframe.")
        return DF.sort_values(sortCol, ascending=ascending).head(nToUse).copy()

    return DF.sample(n=nToUse, random_state=randomSeed).copy()


def selectDiverseMoleculesByMaxMin(
    DF: pd.DataFrame,
    fpCol: str = "fp",
    nToUse: Optional[int] = 1000,
    randomSeed: int = 42,
) -> pd.DataFrame:
    """Select a chemically diverse subset using RDKit MaxMinPicker over bit fingerprints."""
    DF = DF.copy()

    if nToUse is None or nToUse >= len(DF):
        return DF

    if fpCol not in DF.columns:
        raise ValueError(f"Fingerprint column '{fpCol}' not found for diversity selection.")

    fps = list(DF[fpCol])
    picker = MaxMinPicker()
    selectedIdx = list(
        picker.LazyBitVectorPick(
            fps,
            len(fps),
            int(nToUse),
            seed=int(randomSeed),
        )
    )

    return DF.iloc[selectedIdx].copy()


def selectExternalCandidates(
    DF: pd.DataFrame,
    mode: str = "diverse",
    nToUse: Optional[int] = 1000,
    sortCol: Optional[str] = "pPotency_prediction",
    ascending: bool = False,
    topPotencyPoolSize: int = 10000,
    randomSeed: int = 42,
) -> pd.DataFrame:
    """Select external candidates using topPotency, random, diverse, or topPotencyDiverse."""
    DF = DF.copy()

    if nToUse is None or nToUse >= len(DF):
        return DF

    mode = str(mode)

    if mode == "topPotency":
        if sortCol not in DF.columns:
            raise ValueError(f"sortCol='{sortCol}' not found in external dataframe.")
        return DF.sort_values(sortCol, ascending=ascending).head(nToUse).copy()

    if mode == "random":
        return DF.sample(n=nToUse, random_state=randomSeed).copy()

    if mode == "diverse":
        return selectDiverseMoleculesByMaxMin(
            DF,
            fpCol="fp",
            nToUse=nToUse,
            randomSeed=randomSeed,
        )

    if mode == "topPotencyDiverse":
        if sortCol not in DF.columns:
            raise ValueError(f"sortCol='{sortCol}' not found in external dataframe.")

        poolSize = min(int(topPotencyPoolSize), len(DF))
        potencyPoolDF = DF.sort_values(sortCol, ascending=ascending).head(poolSize).copy()

        return selectDiverseMoleculesByMaxMin(
            potencyPoolDF,
            fpCol="fp",
            nToUse=nToUse,
            randomSeed=randomSeed,
        )

    raise ValueError(
        f"Unknown external selection mode: {mode}. "
        "Use one of: topPotency, random, diverse, topPotencyDiverse."
    )


def addMoleculesAndFingerprints(
    DF: pd.DataFrame,
    smilesCol: str,
    morganGenerator: Any,
    datasetName: str,
    dropDuplicateSmiles: bool = True,
) -> pd.DataFrame:
    """Create RDKit mol and Morgan fingerprint columns after validating SMILES."""
    requireColumns(DF, [smilesCol], datasetName)

    DF = DF.copy()
    DF["mol"] = [Chem.MolFromSmiles(str(x)) if pd.notna(x) else None for x in DF[smilesCol]]

    beforeRows = len(DF)
    DF = DF.dropna(subset=["mol"]).copy()
    validRows = len(DF)

    if dropDuplicateSmiles:
        DF = DF.drop_duplicates(subset=[smilesCol]).copy()

    DF["fp"] = [morganGenerator.GetFingerprint(mol) for mol in DF["mol"]]

    print(
        f"{datasetName}: rows={beforeRows:,}, RDKit-valid={validRows:,}, "
        f"valid unique={len(DF):,}"
    )

    return DF


def getSelectionConfig(datasetConfig: Dict[str, Any], defaultSelection: Dict[str, Any]) -> Dict[str, Any]:
    selection = dict(defaultSelection or {})
    selection.update(datasetConfig.get("selection", {}) or {})
    return selection


def prepareTrainingDataset(
    config: Dict[str, Any],
    morganGenerator: Any,
    smilesCol: str,
    randomSeed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    trainingConfig = config["training"]
    trainingLabel = trainingConfig.get("label", trainingConfig.get("name", "Training"))
    trainingCsvPath = trainingConfig["csvPath"]

    trainingDF = readCsvDataset(trainingCsvPath, trainingLabel)
    trainingDF["dataSet"] = trainingLabel
    trainingDF["sourceType"] = "training"

    trainingDF = addMoleculesAndFingerprints(
        trainingDF,
        smilesCol=smilesCol,
        morganGenerator=morganGenerator,
        datasetName=trainingLabel,
        dropDuplicateSmiles=getBool(getNested(config, ["preprocessing", "dropDuplicateSmiles"], True), True),
    )

    selection = trainingConfig.get("selection", {}) or {}
    nToUse = selection.get("nToUse", None)
    sortCol = selection.get("sortCol", None)
    ascending = getBool(selection.get("ascending", False), False)

    trainingDF = selectMolecules(
        trainingDF,
        nToUse=nToUse,
        sortCol=sortCol,
        ascending=ascending,
        randomSeed=randomSeed,
    )

    print(f"Selected training molecules: {len(trainingDF):,}")

    return trainingDF, trainingConfig


def prepareExternalDatasets(
    config: Dict[str, Any],
    morganGenerator: Any,
    smilesCol: str,
    randomSeed: int,
    trainingDF: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    externalDatasetConfigs = config.get("externalDatasets", [])
    if not externalDatasetConfigs:
        raise ValueError("Config must contain at least one entry under externalDatasets.")

    defaultExternalSelection = config.get("externalSelectionDefaults", {}) or {}
    selectedExternalDFList = []
    processedExternalConfigs = []

    dropDuplicateSmiles = getBool(getNested(config, ["preprocessing", "dropDuplicateSmiles"], True), True)
    removeTrainingOverlap = getBool(getNested(config, ["preprocessing", "removeExternalTrainingOverlap"], False), False)
    trainingSmilesSet = set(trainingDF[smilesCol])

    for idx, externalConfig in enumerate(externalDatasetConfigs, start=1):
        externalLabel = externalConfig.get("label", externalConfig.get("name", f"External_{idx}"))
        externalCsvPath = externalConfig["csvPath"]

        tmpDF = readCsvDataset(externalCsvPath, externalLabel)
        tmpDF["dataSet"] = externalLabel
        tmpDF["sourceType"] = "external"

        tmpDF = addMoleculesAndFingerprints(
            tmpDF,
            smilesCol=smilesCol,
            morganGenerator=morganGenerator,
            datasetName=externalLabel,
            dropDuplicateSmiles=dropDuplicateSmiles,
        )

        overlapBefore = len(set(tmpDF[smilesCol]).intersection(trainingSmilesSet))
        if removeTrainingOverlap and overlapBefore > 0:
            tmpDF = tmpDF.loc[~tmpDF[smilesCol].isin(trainingSmilesSet)].copy()

        selection = getSelectionConfig(externalConfig, defaultExternalSelection)
        mode = selection.get("mode", "diverse")
        nToUse = selection.get("nToUse", 1000)
        sortCol = selection.get("sortCol", externalConfig.get("potencyCol", "pPotency_prediction"))
        ascending = getBool(selection.get("ascending", False), False)
        topPotencyPoolSize = selection.get("topPotencyPoolSize", 10000)

        tmpDF = selectExternalCandidates(
            tmpDF,
            mode=mode,
            nToUse=nToUse,
            sortCol=sortCol,
            ascending=ascending,
            topPotencyPoolSize=topPotencyPoolSize,
            randomSeed=randomSeed,
        )

        overlapAfter = len(set(tmpDF[smilesCol]).intersection(trainingSmilesSet))

        print(f"{externalLabel}: selected molecules = {len(tmpDF):,}")
        print(f"{externalLabel}: exact overlap with training before selection/removal = {overlapBefore:,}")
        print(f"{externalLabel}: exact overlap with training after selection = {overlapAfter:,}")

        selectedExternalDFList.append(tmpDF)
        processedExternalConfigs.append(externalConfig)

    externalAllDF = pd.concat(selectedExternalDFList, ignore_index=True).copy()
    print(f"Total selected external molecules: {len(externalAllDF):,}")

    return externalAllDF, processedExternalConfigs


def getDatasetPlotInfo(configList: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    plotInfo = {}
    for idx, datasetConfig in enumerate(configList, start=1):
        label = datasetConfig.get("label", datasetConfig.get("name", f"Dataset_{idx}"))
        plotInfo[label] = {
            "color": datasetConfig.get("color", None),
            "marker": datasetConfig.get("marker", "+"),
            "alpha": datasetConfig.get("alpha", 0.75),
            "size": datasetConfig.get("size", 60),
        }
    return plotInfo


def safeNumericSeries(DF: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in DF.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(DF[col], errors="coerce").dropna()


def saveCleanCsv(DF: pd.DataFrame, path: Path) -> None:
    dropCols = [col for col in ["mol", "fp"] if col in DF.columns]
    DF.drop(columns=dropCols, errors="ignore").to_csv(path, index=False)


def plotPotencyDistribution(
    trainingDF: pd.DataFrame,
    externalAllDF: pd.DataFrame,
    trainingConfig: Dict[str, Any],
    externalConfigs: List[Dict[str, Any]],
    outputDir: Path,
    jobName: str,
    config: Dict[str, Any],
) -> None:
    plotsConfig = config.get("plots", {}) or {}
    dpi = int(plotsConfig.get("dpi", 600))
    savePlots = getBool(plotsConfig.get("save", True), True)

    trainingLabel = trainingConfig.get("label", trainingConfig.get("name", "Training"))
    trainingPotencyCol = trainingConfig.get("potencyCol", "pPotency")
    trainingColor = trainingConfig.get("color", "red")

    trainingPotency = safeNumericSeries(trainingDF, trainingPotencyCol)

    allPotencyList = [trainingPotency]
    externalPotencyMap = {}

    for externalConfig in externalConfigs:
        label = externalConfig.get("label", externalConfig.get("name"))
        potencyCol = externalConfig.get("potencyCol", "pPotency_prediction")
        tmpPotency = safeNumericSeries(
            externalAllDF.loc[externalAllDF["dataSet"] == label],
            potencyCol,
        )
        externalPotencyMap[label] = tmpPotency
        allPotencyList.append(tmpPotency)

    allPotency = pd.concat(allPotencyList, ignore_index=True).dropna()
    if len(allPotency) == 0:
        print("Skipping potency distribution plot because no valid potency values were found.")
        return

    potencyBins = np.linspace(
        float(allPotency.min()),
        float(allPotency.max()),
        int(plotsConfig.get("potencyBins", 25)),
    )

    fig, ax = plt.subplots(figsize=tuple(plotsConfig.get("potencyFigSize", [9, 5])))

    ax.hist(
        trainingPotency,
        bins=potencyBins,
        alpha=float(trainingConfig.get("histAlpha", 0.70)),
        color=trainingColor,
        label=f"{trainingLabel}, n={len(trainingPotency)}",
    )

    for externalConfig in externalConfigs:
        label = externalConfig.get("label", externalConfig.get("name"))
        color = externalConfig.get("color", None)
        tmpPotency = externalPotencyMap[label]
        ax.hist(
            tmpPotency,
            bins=potencyBins,
            alpha=float(externalConfig.get("histAlpha", 0.45)),
            color=color,
            label=f"{label}, n={len(tmpPotency)}",
        )

    ax.set_xlabel(plotsConfig.get("potencyXLabel", "Potency / predicted potency"))
    ax.set_ylabel(plotsConfig.get("potencyYLabel", "Number of molecules"))
    ax.set_title(plotsConfig.get("potencyTitle", "Potency distribution: training vs external datasets"))

    addLegend(ax, plotsConfig)
    ax.grid(alpha=float(plotsConfig.get("gridAlpha", 0.2)))

    plt.tight_layout()
    if savePlots:
        fig.savefig(outputDir / f"{jobName}_potency_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def addLegend(ax: Any, plotsConfig: Dict[str, Any]) -> None:
    if getBool(plotsConfig.get("legendOutside", True), True):
        ax.legend(
            frameon=False,
            loc=plotsConfig.get("legendLoc", "center left"),
            bbox_to_anchor=tuple(plotsConfig.get("legendBboxToAnchor", [1.02, 0.5])),
            borderaxespad=0,
        )
    else:
        ax.legend(frameon=False)


def buildNearestNeighborTable(
    trainingDF: pd.DataFrame,
    externalAllDF: pd.DataFrame,
    trainingConfig: Dict[str, Any],
    externalConfigs: List[Dict[str, Any]],
    smilesCol: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    trainingFps = list(trainingDF["fp"])
    if not trainingFps:
        raise ValueError("No training fingerprints available for nearest-neighbor similarity.")

    trainingLabel = trainingConfig.get("label", trainingConfig.get("name", "Training"))
    trainingPotencyCol = trainingConfig.get("potencyCol", "pPotency")
    trainingCompoundIdCol = trainingConfig.get("compoundIdCol", "compound_id")
    trainingStrainCol = trainingConfig.get("strainCol", "StrainClassifier")

    externalConfigByLabel = {
        ec.get("label", ec.get("name")): ec for ec in externalConfigs
    }

    trainingSmilesList = trainingDF[smilesCol].tolist()
    trainingCompoundIdList = (
        trainingDF[trainingCompoundIdCol].tolist()
        if trainingCompoundIdCol in trainingDF.columns else [np.nan] * len(trainingDF)
    )
    trainingPotencyList = (
        trainingDF[trainingPotencyCol].tolist()
        if trainingPotencyCol in trainingDF.columns else [np.nan] * len(trainingDF)
    )
    trainingStrainList = (
        trainingDF[trainingStrainCol].tolist()
        if trainingStrainCol in trainingDF.columns else [np.nan] * len(trainingDF)
    )

    nearestRows = []
    progressEvery = int(getNested(config, ["runtime", "progressEvery"], 10000))

    for i, (_, externalRow) in enumerate(externalAllDF.iterrows(), start=1):
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(externalRow["fp"], trainingFps))
        bestIdx = int(np.argmax(similarities))
        bestSimilarity = float(similarities[bestIdx])
        externalSource = externalRow["dataSet"]
        externalConfig = externalConfigByLabel.get(externalSource, {})

        rankCol = externalConfig.get("rankCol", "Rank")
        potencyCol = externalConfig.get("potencyCol", "pPotency_prediction")
        stdCol = externalConfig.get("stdCol", "pPotency_std")
        ic50Col = externalConfig.get("ic50Col", "IC50 (µM)")

        nearestRows.append({
            "External_Source": externalSource,
            "External_Rank": externalRow.get(rankCol, np.nan),
            "External_SMILES": externalRow[smilesCol],
            "External_pPotency_prediction": externalRow.get(potencyCol, np.nan),
            "External_pPotency_std": externalRow.get(stdCol, np.nan),
            "External_IC50_uM": externalRow.get(ic50Col, np.nan),
            "Nearest_training_source": trainingLabel,
            "Nearest_training_compound_id": trainingCompoundIdList[bestIdx],
            "Nearest_training_SMILES": trainingSmilesList[bestIdx],
            "Nearest_training_pPotency": trainingPotencyList[bestIdx],
            "Nearest_training_StrainClassifier": trainingStrainList[bestIdx],
            "Max_Tanimoto_to_training": bestSimilarity,
            "Tanimoto_distance_to_training": 1.0 - bestSimilarity,
        })

        if progressEvery > 0 and i % progressEvery == 0:
            print(f"Nearest-neighbor search completed for {i:,} external molecules")

    nearestNeighborDF = pd.DataFrame(nearestRows)

    categoryBins = getNested(config, ["similarity", "categoryBins"], [-np.inf, 0.40, 0.70, np.inf])
    categoryLabels = getNested(
        config,
        ["similarity", "categoryLabels"],
        ["Far from training set", "Moderately close", "Close to training set"],
    )

    parsedBins = []
    for value in categoryBins:
        if isinstance(value, str) and value.lower() in {"-inf", "-infinity"}:
            parsedBins.append(-np.inf)
        elif isinstance(value, str) and value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
            parsedBins.append(np.inf)
        else:
            parsedBins.append(float(value))

    nearestNeighborDF["Chemical_space_category"] = pd.cut(
        nearestNeighborDF["Max_Tanimoto_to_training"],
        bins=parsedBins,
        labels=categoryLabels,
    )

    nearestNeighborDF = nearestNeighborDF.sort_values(
        ["External_Source", "Max_Tanimoto_to_training"],
        ascending=[True, False],
    )

    return nearestNeighborDF


def summarizeNearestNeighbors(nearestNeighborDF: pd.DataFrame) -> pd.DataFrame:
    summaryDF = (
        nearestNeighborDF
        .groupby(["External_Source", "Chemical_space_category"], observed=False)
        .size()
        .reset_index(name="Number_of_molecules")
    )

    statsDF = (
        nearestNeighborDF
        .groupby("External_Source")["Max_Tanimoto_to_training"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    return summaryDF, statsDF


def plotNearestNeighborSimilarity(
    nearestNeighborDF: pd.DataFrame,
    externalConfigs: List[Dict[str, Any]],
    outputDir: Path,
    jobName: str,
    config: Dict[str, Any],
) -> None:
    plotsConfig = config.get("plots", {}) or {}
    dpi = int(plotsConfig.get("dpi", 600))
    savePlots = getBool(plotsConfig.get("save", True), True)

    fig, ax = plt.subplots(figsize=tuple(plotsConfig.get("similarityFigSize", [8, 5])))
    similarityBins = np.linspace(0, 1, int(plotsConfig.get("similarityBins", 21)))

    for externalConfig in externalConfigs:
        label = externalConfig.get("label", externalConfig.get("name"))
        color = externalConfig.get("color", None)
        tmpSimilarity = nearestNeighborDF.loc[
            nearestNeighborDF["External_Source"] == label,
            "Max_Tanimoto_to_training",
        ]

        ax.hist(
            tmpSimilarity,
            bins=similarityBins,
            alpha=float(externalConfig.get("histAlpha", 0.45)),
            color=color,
            label=f"{label}, n={len(tmpSimilarity)}",
        )

    for threshold in getNested(config, ["similarity", "thresholdLines"], [0.40, 0.70]):
        ax.axvline(float(threshold), linestyle="--", linewidth=1, color="k")

    ax.set_xlabel(plotsConfig.get("similarityXLabel", "Maximum Tanimoto similarity to selected training set"))
    ax.set_ylabel(plotsConfig.get("similarityYLabel", "Number of external molecules"))
    ax.set_title(plotsConfig.get("similarityTitle", "Nearest training-set similarity by external dataset"))
    addLegend(ax, plotsConfig)
    ax.grid(alpha=float(plotsConfig.get("gridAlpha", 0.2)))

    plt.tight_layout()
    if savePlots:
        fig.savefig(outputDir / f"{jobName}_nearest_neighbor_similarity_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fingerprintsToSparseMatrix(fps: List[Any], nBits: int = 2048):
    from scipy.sparse import csr_matrix

    rowIdx = []
    colIdx = []

    for i, fp in enumerate(fps):
        onBits = list(fp.GetOnBits())
        rowIdx.extend([i] * len(onBits))
        colIdx.extend(onBits)

    data = np.ones(len(colIdx), dtype=np.float32)

    return csr_matrix(
        (data, (rowIdx, colIdx)),
        shape=(len(fps), nBits),
        dtype=np.float32,
    )


def buildMdsEmbedding(plotDF: pd.DataFrame, config: Dict[str, Any], randomSeed: int) -> pd.DataFrame:
    from sklearn.manifold import MDS

    mdsConfig = getNested(config, ["embedding", "mds"], {}) or {}
    nJobs = int(mdsConfig.get("distanceNJobs", getNested(config, ["runtime", "nJobs"], 1)))
    nJobs = max(1, nJobs)

    plotFps = list(plotDF["fp"])
    nMolecules = len(plotFps)

    print(f"Total molecules used in MDS plot: {nMolecules:,}")
    print("MDS warning: all-vs-all distance matrix scales as O(N^2).")

    distanceMatrix = np.empty((nMolecules, nMolecules), dtype=np.float32)

    def fillDistanceRow(i: int) -> None:
        distanceMatrix[i, :] = 1.0 - np.asarray(
            DataStructs.BulkTanimotoSimilarity(plotFps[i], plotFps),
            dtype=np.float32,
        )

    if nJobs > 1:
        try:
            from joblib import Parallel, delayed
            Parallel(n_jobs=nJobs, prefer="threads", require="sharedmem", verbose=0)(
                delayed(fillDistanceRow)(i) for i in range(nMolecules)
            )
        except Exception as exc:
            print(f"Parallel distance construction failed; falling back to serial. Error: {exc}")
            for i in range(nMolecules):
                fillDistanceRow(i)
    else:
        for i in range(nMolecules):
            fillDistanceRow(i)

    mdsKwargs = dict(
        n_components=2,
        dissimilarity="precomputed",
        random_state=randomSeed,
        n_init=int(mdsConfig.get("nInit", 4)),
        max_iter=int(mdsConfig.get("maxIter", 500)),
        eps=float(mdsConfig.get("eps", 1e-3)),
    )

    sklearnMdsNJobs = int(mdsConfig.get("nJobs", getNested(config, ["runtime", "nJobs"], 1)))
    if sklearnMdsNJobs > 1:
        mdsKwargs["n_jobs"] = sklearnMdsNJobs

    try:
        mdsModel = MDS(**mdsKwargs)
    except TypeError:
        # Older sklearn versions may not support n_jobs.
        mdsKwargs.pop("n_jobs", None)
        mdsModel = MDS(**mdsKwargs)

    embedding = mdsModel.fit_transform(distanceMatrix)
    plotDF["Embed1"] = embedding[:, 0]
    plotDF["Embed2"] = embedding[:, 1]
    plotDF["embeddingMethod"] = "MDS_Tanimoto"
    return plotDF


def buildSvdEmbedding(plotDF: pd.DataFrame, config: Dict[str, Any], randomSeed: int, fpBits: int) -> pd.DataFrame:
    from sklearn.decomposition import TruncatedSVD

    svdConfig = getNested(config, ["embedding", "svd"], {}) or {}
    plotFps = list(plotDF["fp"])
    fingerprintMatrix = fingerprintsToSparseMatrix(plotFps, nBits=fpBits)

    svdModel = TruncatedSVD(
        n_components=2,
        random_state=randomSeed,
        n_iter=int(svdConfig.get("nIter", 7)),
    )
    embedding = svdModel.fit_transform(fingerprintMatrix)

    plotDF["Embed1"] = embedding[:, 0]
    plotDF["Embed2"] = embedding[:, 1]
    plotDF["embeddingMethod"] = "SVD_fingerprint_bits"
    print(
        "SVD explained variance ratio:",
        [round(float(x), 4) for x in svdModel.explained_variance_ratio_],
    )
    return plotDF


def buildEmbedding(
    trainingDF: pd.DataFrame,
    externalAllDF: pd.DataFrame,
    config: Dict[str, Any],
    randomSeed: int,
    fpBits: int,
) -> pd.DataFrame:
    plotDF = pd.concat([trainingDF, externalAllDF], ignore_index=True).copy()
    method = str(getNested(config, ["embedding", "method"], "mds")).lower()

    if method == "mds":
        return buildMdsEmbedding(plotDF, config, randomSeed)

    if method == "svd":
        return buildSvdEmbedding(plotDF, config, randomSeed, fpBits)

    raise ValueError("embedding.method must be either 'mds' or 'svd'.")


def plotEmbedding(
    plotDF: pd.DataFrame,
    trainingConfig: Dict[str, Any],
    externalConfigs: List[Dict[str, Any]],
    outputDir: Path,
    jobName: str,
    config: Dict[str, Any],
) -> None:
    plotsConfig = config.get("plots", {}) or {}
    dpi = int(plotsConfig.get("dpi", 600))
    savePlots = getBool(plotsConfig.get("save", True), True)

    method = str(getNested(config, ["embedding", "method"], "mds")).lower()
    defaultXLabel = "MDS1 from Morgan/Tanimoto distance" if method == "mds" else "Fingerprint embedding 1"
    defaultYLabel = "MDS2 from Morgan/Tanimoto distance" if method == "mds" else "Fingerprint embedding 2"

    fig, ax = plt.subplots(figsize=tuple(plotsConfig.get("embeddingFigSize", [10, 7])))

    trainingLabel = trainingConfig.get("label", trainingConfig.get("name", "Training"))
    trainingMask = plotDF["dataSet"] == trainingLabel

    ax.scatter(
        plotDF.loc[trainingMask, "Embed1"],
        plotDF.loc[trainingMask, "Embed2"],
        s=float(trainingConfig.get("size", 22)),
        alpha=float(trainingConfig.get("alpha", 0.80)),
        color=trainingConfig.get("color", "red"),
        marker=trainingConfig.get("marker", "o"),
        label=f"{trainingLabel} (n={trainingMask.sum()})",
    )

    for externalConfig in externalConfigs:
        label = externalConfig.get("label", externalConfig.get("name"))
        externalMask = plotDF["dataSet"] == label
        ax.scatter(
            plotDF.loc[externalMask, "Embed1"],
            plotDF.loc[externalMask, "Embed2"],
            s=float(externalConfig.get("size", 70)),
            alpha=float(externalConfig.get("alpha", 0.85)),
            color=externalConfig.get("color", None),
            marker=externalConfig.get("marker", "+"),
            label=f"{label} (n={externalMask.sum()})",
        )

    ax.set_xlabel(plotsConfig.get("embeddingXLabel", defaultXLabel))
    ax.set_ylabel(plotsConfig.get("embeddingYLabel", defaultYLabel))
    ax.set_title(plotsConfig.get("embeddingTitle", "Training vs external datasets in chemical embedding space"))
    addLegend(ax, plotsConfig)
    ax.grid(alpha=float(plotsConfig.get("gridAlpha", 0.2)))

    if getBool(plotsConfig.get("legendOutside", True), True):
        plt.tight_layout(rect=plotsConfig.get("tightLayoutRect", [0, 0, 0.78, 1]))
    else:
        plt.tight_layout()

    if savePlots:
        fig.savefig(outputDir / f"{jobName}_chemical_space_embedding.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plotPotency3D(
    plotDF: pd.DataFrame,
    trainingConfig: Dict[str, Any],
    externalConfigs: List[Dict[str, Any]],
    outputDir: Path,
    jobName: str,
    config: Dict[str, Any],
) -> None:
    plotsConfig = config.get("plots", {}) or {}
    if not getBool(plotsConfig.get("make3DPlot", False), False):
        return

    dpi = int(plotsConfig.get("dpi", 600))
    savePlots = getBool(plotsConfig.get("save", True), True)

    trainingLabel = trainingConfig.get("label", trainingConfig.get("name", "Training"))
    trainingPotencyCol = trainingConfig.get("potencyCol", "pPotency")

    plotDF = plotDF.copy()
    plotDF["potencyAxis"] = np.nan
    trainingMask = plotDF["dataSet"] == trainingLabel
    if trainingPotencyCol in plotDF.columns:
        plotDF.loc[trainingMask, "potencyAxis"] = pd.to_numeric(
            plotDF.loc[trainingMask, trainingPotencyCol],
            errors="coerce",
        )

    for externalConfig in externalConfigs:
        label = externalConfig.get("label", externalConfig.get("name"))
        potencyCol = externalConfig.get("potencyCol", "pPotency_prediction")
        externalMask = plotDF["dataSet"] == label
        if potencyCol in plotDF.columns:
            plotDF.loc[externalMask, "potencyAxis"] = pd.to_numeric(
                plotDF.loc[externalMask, potencyCol],
                errors="coerce",
            )

    plotDF = plotDF.dropna(subset=["Embed1", "Embed2", "potencyAxis"]).copy()
    if len(plotDF) == 0:
        print("Skipping 3D potency plot because no valid potency values were available.")
        return

    fig = plt.figure(figsize=tuple(plotsConfig.get("potency3DFigSize", [10, 8])))
    ax = fig.add_subplot(111, projection="3d")

    trainingMask = plotDF["dataSet"] == trainingLabel
    ax.scatter(
        plotDF.loc[trainingMask, "Embed1"],
        plotDF.loc[trainingMask, "Embed2"],
        plotDF.loc[trainingMask, "potencyAxis"],
        s=float(trainingConfig.get("size", 22)),
        alpha=float(trainingConfig.get("alpha", 0.70)),
        color=trainingConfig.get("color", "red"),
        marker=trainingConfig.get("marker", "o"),
        label=f"{trainingLabel} (n={trainingMask.sum()})",
    )

    for externalConfig in externalConfigs:
        label = externalConfig.get("label", externalConfig.get("name"))
        externalMask = plotDF["dataSet"] == label
        ax.scatter(
            plotDF.loc[externalMask, "Embed1"],
            plotDF.loc[externalMask, "Embed2"],
            plotDF.loc[externalMask, "potencyAxis"],
            s=float(externalConfig.get("size", 55)),
            alpha=float(externalConfig.get("alpha", 0.65)),
            color=externalConfig.get("color", None),
            marker=externalConfig.get("marker", "+"),
            label=f"{label} (n={externalMask.sum()})",
        )

    ax.set_xlabel(plotsConfig.get("embeddingXLabel", "Embedding 1"))
    ax.set_ylabel(plotsConfig.get("embeddingYLabel", "Embedding 2"))
    ax.set_zlabel(plotsConfig.get("potencyZLabel", "Potency / predicted potency"))
    ax.set_title(plotsConfig.get("potency3DTitle", "Chemical embedding space with potency as third axis"))
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout()
    if savePlots:
        fig.savefig(outputDir / f"{jobName}_chemical_space_embedding_3D_potency.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plotPotencyVsSimilarity(
    nearestNeighborDF: pd.DataFrame,
    externalConfigs: List[Dict[str, Any]],
    outputDir: Path,
    jobName: str,
    config: Dict[str, Any],
) -> None:
    plotsConfig = config.get("plots", {}) or {}
    if not getBool(plotsConfig.get("makePotencyVsSimilarityPlot", True), True):
        return

    dpi = int(plotsConfig.get("dpi", 600))
    savePlots = getBool(plotsConfig.get("save", True), True)

    fig, ax = plt.subplots(figsize=tuple(plotsConfig.get("potencyVsSimilarityFigSize", [8, 5])))

    for externalConfig in externalConfigs:
        label = externalConfig.get("label", externalConfig.get("name"))
        tmpDF = nearestNeighborDF.loc[nearestNeighborDF["External_Source"] == label].copy()
        yValues = pd.to_numeric(tmpDF["External_pPotency_prediction"], errors="coerce")
        validMask = yValues.notna()
        ax.scatter(
            tmpDF.loc[validMask, "Max_Tanimoto_to_training"],
            yValues.loc[validMask],
            s=float(externalConfig.get("similarityScatterSize", 35)),
            alpha=float(externalConfig.get("alpha", 0.65)),
            color=externalConfig.get("color", None),
            marker=externalConfig.get("marker", "+"),
            label=f"{label} (n={validMask.sum()})",
        )

    for threshold in getNested(config, ["similarity", "thresholdLines"], [0.40, 0.70]):
        ax.axvline(float(threshold), linestyle="--", linewidth=1, color="k")

    ax.set_xlabel("Maximum Tanimoto similarity to selected training set")
    ax.set_ylabel("Predicted pPotency")
    ax.set_title("Predicted potency vs chemical-space support")
    addLegend(ax, plotsConfig)
    ax.grid(alpha=float(plotsConfig.get("gridAlpha", 0.2)))

    plt.tight_layout()
    if savePlots:
        fig.savefig(outputDir / f"{jobName}_potency_vs_tanimoto_support.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def writeRunMetadata(outputDir: Path, jobName: str, config: Dict[str, Any]) -> None:
    metadataPath = outputDir / f"{jobName}_run_config_resolved.json"
    with open(metadataPath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    startTime = time.time()
    args = parseArgs()
    configPath = Path(args.config).expanduser().resolve()
    config = readYaml(str(configPath))

    jobName = sanitizeName(getNested(config, ["job", "name"], "trainingSet_similarity"))
    outputDir = ensureDir(Path(getNested(config, ["job", "outputDir"], "trainingSet_similarity_results")).expanduser().resolve())

    # Save the exact YAML used for reproducibility.
    try:
        shutil.copy2(configPath, outputDir / f"{jobName}_config.yaml")
    except Exception:
        pass

    randomSeed = int(getNested(config, ["runtime", "randomSeed"], 42))
    smilesCol = getNested(config, ["columns", "smilesCol"], "Canonical_SMILES")
    fpRadius = int(getNested(config, ["fingerprint", "radius"], 2))
    fpBits = int(getNested(config, ["fingerprint", "nBits"], 2048))

    print(f"Job name: {jobName}")
    print(f"Output directory: {outputDir}")
    print(f"SMILES column: {smilesCol}")
    print(f"Morgan fingerprint: radius={fpRadius}, nBits={fpBits}")
    print(f"Embedding method: {getNested(config, ['embedding', 'method'], 'mds')}")

    morganGenerator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fpRadius,
        fpSize=fpBits,
    )

    trainingDF, trainingConfig = prepareTrainingDataset(
        config=config,
        morganGenerator=morganGenerator,
        smilesCol=smilesCol,
        randomSeed=randomSeed,
    )

    externalAllDF, externalConfigs = prepareExternalDatasets(
        config=config,
        morganGenerator=morganGenerator,
        smilesCol=smilesCol,
        randomSeed=randomSeed,
        trainingDF=trainingDF,
    )

    saveCleanCsv(trainingDF, outputDir / f"{jobName}_selected_training.csv")
    saveCleanCsv(externalAllDF, outputDir / f"{jobName}_selected_external_all.csv")

    plotPotencyDistribution(
        trainingDF=trainingDF,
        externalAllDF=externalAllDF,
        trainingConfig=trainingConfig,
        externalConfigs=externalConfigs,
        outputDir=outputDir,
        jobName=jobName,
        config=config,
    )

    nearestNeighborDF = buildNearestNeighborTable(
        trainingDF=trainingDF,
        externalAllDF=externalAllDF,
        trainingConfig=trainingConfig,
        externalConfigs=externalConfigs,
        smilesCol=smilesCol,
        config=config,
    )

    nearestPath = outputDir / f"{jobName}_nearest_neighbors_all_external.csv"
    nearestNeighborDF.to_csv(nearestPath, index=False)

    summaryDF, statsDF = summarizeNearestNeighbors(nearestNeighborDF)
    summaryDF.to_csv(outputDir / f"{jobName}_nearest_neighbor_category_summary.csv", index=False)
    statsDF.to_csv(outputDir / f"{jobName}_nearest_neighbor_similarity_stats.csv", index=False)

    print("Nearest-neighbor category summary:")
    print(summaryDF.to_string(index=False))
    print("\nNearest-neighbor similarity stats:")
    print(statsDF.to_string(index=False))

    plotNearestNeighborSimilarity(
        nearestNeighborDF=nearestNeighborDF,
        externalConfigs=externalConfigs,
        outputDir=outputDir,
        jobName=jobName,
        config=config,
    )

    plotPotencyVsSimilarity(
        nearestNeighborDF=nearestNeighborDF,
        externalConfigs=externalConfigs,
        outputDir=outputDir,
        jobName=jobName,
        config=config,
    )

    if getBool(getNested(config, ["embedding", "enabled"], True), True):
        plotDF = buildEmbedding(
            trainingDF=trainingDF,
            externalAllDF=externalAllDF,
            config=config,
            randomSeed=randomSeed,
            fpBits=fpBits,
        )

        saveCleanCsv(plotDF, outputDir / f"{jobName}_embedding_coordinates.csv")

        plotEmbedding(
            plotDF=plotDF,
            trainingConfig=trainingConfig,
            externalConfigs=externalConfigs,
            outputDir=outputDir,
            jobName=jobName,
            config=config,
        )

        plotPotency3D(
            plotDF=plotDF,
            trainingConfig=trainingConfig,
            externalConfigs=externalConfigs,
            outputDir=outputDir,
            jobName=jobName,
            config=config,
        )
    else:
        print("Embedding disabled by config: embedding.enabled = false")

    writeRunMetadata(outputDir, jobName, config)
    print(f"\nDone. Results saved in: {outputDir}")
    print(f"Main nearest-neighbor output: {nearestPath}")
    print(f"Total runtime: {time.time() - startTime:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
