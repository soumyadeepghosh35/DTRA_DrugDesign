#!/usr/bin/env python3
"""
Mimics the *statistical + plotting logic* of code-2, while still using
code-1's YAML-driven CLI workflow.

Implements:
  - training selection (sort or random, optional)
  - external selection per dataset (topPotency/random/diverse/topPotencyDiverse)
  - potency distribution plot (hist bins from global min/max, 25 bins)
  - nearest-neighbor similarity to training (max Tanimoto per external mol)
  - MDS embedding on (training + selected externals)
  - nearest-neighbor similarity distribution plot
  - predicted potency vs maximum Tanimoto similarity plot

Saves plots to outputDir with the same filenames as code-2:
  - potency_distribution_all_external_sets.png
  - structural_similarity_embeddings.png
  - nearest_neighbor_similarity_distribution.png
  - pPotency_vs_tanimoto.png
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

from sklearn.manifold import MDS


# -----------------------------
# CLI + YAML utils
# -----------------------------
def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training vs external similarity + plots (code-2 logic mimic).")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args()


def readYaml(configPath: str) -> Dict[str, Any]:
    with open(configPath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {configPath}")
    return config


def getNested(config: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    current: Any = config
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


# -----------------------------
# Selection + fingerprinting (same core idea as code-1)
# -----------------------------
def selectMolecules(
    DF: pd.DataFrame,
    nToUse: Optional[int] = None,
    sortCol: Optional[str] = None,
    ascending: bool = False,
    randomSeed: int = 42,
) -> pd.DataFrame:
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
    """MaxMinPicker over bit fingerprints (code-2 logic)."""
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
    """Select external candidates using the same mode semantics as code-2."""
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
        return selectDiverseMoleculesByMaxMin(DF, fpCol="fp", nToUse=nToUse, randomSeed=randomSeed)

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

    raise ValueError(f"Unknown external selection mode: {mode}")


def addMoleculesAndFingerprints(
    DF: pd.DataFrame,
    smilesCol: str,
    morganGenerator: Any,
    datasetName: str,
    dropDuplicateSmiles: bool = True,
) -> pd.DataFrame:
    """Create mol and fp, mimicking code-2 validity filtering + de-dup."""
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

    # ---- code-2 style selection for training
    selection = trainingConfig.get("selection", {}) or {}
    nToUse = selection.get("nToUse", None)
    sortCol = selection.get("sortCol", trainingConfig.get("potencyCol", "pPotency"))
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
    selectedExternalDFList: List[pd.DataFrame] = []
    processedExternalConfigs: List[Dict[str, Any]] = []

    dropDuplicateSmiles = getBool(getNested(config, ["preprocessing", "dropDuplicateSmiles"], True), True)
    # code-2 does NOT remove overlap by default; keep config option but default False
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

        # code-2 does selection AFTER fingerprinting/validity/de-dup
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


# -----------------------------
# Plotting + stats (code-2 logic)
# -----------------------------
def safeNumericSeries(DF: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in DF.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(DF[col], errors="coerce").dropna()


def plotPotencyDistribution(
    trainingDF: pd.DataFrame,
    externalAllDF: pd.DataFrame,
    trainingConfig: Dict[str, Any],
    externalConfigs: List[Dict[str, Any]],
    outputDir: Path,
    savePlots: bool,
    dpi: int,
) -> None:
    trainingPotencyCol = trainingConfig.get("potencyCol", "pPotency")
    trainingColor = "orange"

    trainingPotency = safeNumericSeries(trainingDF, trainingPotencyCol)
    if trainingPotency.empty:
        print("Skipping potency distribution plot: training potency has no valid values.")
        return

    externalPotencyCol = "pPotency_prediction"  # code-2 fixed logic
    allPotencyList = [trainingPotency]

    externalDatasetDict_byLabel = {}
    for ec in externalConfigs:
        label = ec.get("label", ec.get("name"))
        color = ec.get("color", None)
        marker = ec.get("marker", "+")
        externalDatasetDict_byLabel[label] = {"color": color, "marker": marker}

    for label in externalDatasetDict_byLabel.keys():
        tmpPotency = safeNumericSeries(
            externalAllDF.loc[externalAllDF["dataSet"] == label],
            externalPotencyCol,
        )
        allPotencyList.append(tmpPotency)

    allPotency = pd.concat(allPotencyList, ignore_index=True).dropna()
    if allPotency.empty:
        print("Skipping potency distribution plot: no valid potency values overall.")
        return

    potencyBins = np.linspace(allPotency.min(), allPotency.max(), 25)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        trainingPotency,
        bins=potencyBins,
        alpha=0.8,
        color=trainingColor,
        label=f"Training set, n={len(trainingPotency)}",
    )

    for ec in externalConfigs:
        label = ec.get("label", ec.get("name"))
        color = ec.get("color", None)
        tmpPotency = safeNumericSeries(
            externalAllDF.loc[externalAllDF["dataSet"] == label],
            externalPotencyCol,
        )

        ax.hist(
            tmpPotency,
            bins=potencyBins,
            alpha=0.45,
            color=color,
            label=f"{label}, n={len(tmpPotency)}",
        )

    ax.set_xlabel("Potency / predicted potency")
    ax.set_ylabel("Number of molecules")
    ax.set_title("Potency distribution: training vs external validation datasets")
    ax.legend(frameon=False)

    plt.tight_layout()
    if savePlots:
        fig.savefig(outputDir / "potency_distribution_all_external_sets.png", dpi=dpi)
    plt.close(fig)


def buildNearestNeighborTable(
    trainingDF: pd.DataFrame,
    externalAllDF: pd.DataFrame,
    smilesCol: str,
    externalConfigs: List[Dict[str, Any]],
) -> pd.DataFrame:
    trainingFps = list(trainingDF["fp"])
    if len(trainingFps) == 0:
        raise ValueError("No training fingerprints available.")

    trainingPotencyCol = "pPotency"  # code-2 fixed
    trainingCompoundIdCol = "compound_id"
    trainingStrainCol = "StrainClassifier"

    nearestRows: List[Dict[str, Any]] = []
    for _, externalRow in externalAllDF.iterrows():
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(externalRow["fp"], trainingFps))
        bestIdx = int(np.argmax(similarities))
        bestSimilarity = float(similarities[bestIdx])
        nearestTrainingRow = trainingDF.iloc[bestIdx]

        nearestRows.append({
            "External_Source": externalRow["dataSet"],
            "External_Rank": externalRow.get("Rank", np.nan),
            "External_SMILES": externalRow[smilesCol],
            "External_pPotency_prediction": externalRow.get("pPotency_prediction", np.nan),
            "External_pPotency_std": externalRow.get("pPotency_std", np.nan),
            "External_IC50_uM": externalRow.get("IC50 (µM)", np.nan),
            "Nearest_training_compound_id": nearestTrainingRow.get(trainingCompoundIdCol, np.nan),
            "Nearest_training_SMILES": nearestTrainingRow[smilesCol],
            "Nearest_training_pPotency": nearestTrainingRow.get(trainingPotencyCol, np.nan),
            "Nearest_training_StrainClassifier": nearestTrainingRow.get(trainingStrainCol, np.nan),
            "Max_Tanimoto_to_training": bestSimilarity,
            "Tanimoto_distance_to_training": 1.0 - bestSimilarity,
        })

    nearestNeighborDF = pd.DataFrame(nearestRows).sort_values(
        ["External_Source", "Max_Tanimoto_to_training"],
        ascending=[True, False],
    )

    nearestNeighborDF["Chemical_space_category"] = pd.cut(
        nearestNeighborDF["Max_Tanimoto_to_training"],
        bins=[-np.inf, 0.40, 0.70, np.inf],
        labels=["Far from training set", "Moderately close", "Close to training set"],
    )
    return nearestNeighborDF


def plotMDS_embedding(
    trainingDF: pd.DataFrame,
    externalAllDF: pd.DataFrame,
    externalConfigs: List[Dict[str, Any]],
    resultsDir: Path,
    savePlots: bool,
    dpi: int,
    randomSeed: int,
) -> None:
    plotDF = pd.concat([trainingDF, externalAllDF], ignore_index=True).copy()
    plotFps = list(plotDF["fp"])

    print(f"Total molecules used in MDS plot: {len(plotDF):,}")
    distanceMatrix = np.zeros((len(plotFps), len(plotFps)), dtype=np.float32)

    for i, fp in enumerate(plotFps):
        # same structure as code-2:
        distanceMatrix[i, :] = 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fp, plotFps))

    mdsModel = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=randomSeed,
        n_init=4,
        max_iter=500
    )
    embedding = mdsModel.fit_transform(distanceMatrix)

    plotDF["MDS1"] = embedding[:, 0]
    plotDF["MDS2"] = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(18, 7))

    trainingMask = plotDF["dataSet"] == "Training"
    # NOTE: code-2 hardcodes label "Training" for training set
    # Our YAML training label may differ; if so, this will be empty.
    # To mimic code-2 exactly, we assume trainingDF["dataSet"] == "Training".
    ax.scatter(
        plotDF.loc[trainingMask, "MDS1"],
        plotDF.loc[trainingMask, "MDS2"],
        s=22,
        alpha=0.8,
        color="orange",
        marker="o",
        label=f"Training set (n={trainingMask.sum()})",
    )

    # external scatter
    for externalConfig in externalConfigs:
        externalName = externalConfig.get("label", externalConfig.get("name"))
        externalInfo_color = externalConfig.get("color", None)
        externalInfo_marker = externalConfig.get("marker", "+")

        externalMask = plotDF["dataSet"] == externalName
        ax.scatter(
            plotDF.loc[externalMask, "MDS1"],
            plotDF.loc[externalMask, "MDS2"],
            s=70,
            alpha=0.85,
            color=externalInfo_color,
            marker=externalInfo_marker,
            label=f"{externalName} (n={externalMask.sum()})",
        )

    ax.set_xlabel("MDS1 from Morgan/Tanimoto distance")
    ax.set_ylabel("MDS2 from Morgan/Tanimoto distance")
    ax.set_title("Training vs multiple external validation datasets in chemical embedding space")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    if savePlots:
        fig.savefig(resultsDir / "structural_similarity_embeddings.png", dpi=dpi)
    plt.close(fig)


def plotNearestNeighborSimilarityDistribution(
    nearestNeighborDF: pd.DataFrame,
    externalConfigs: List[Dict[str, Any]],
    resultsDir: Path,
    savePlots: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    similarityBins = np.linspace(0, 1, 21)

    for externalConfig in externalConfigs:
        externalName = externalConfig.get("label", externalConfig.get("name"))
        externalColor = externalConfig.get("color", None)

        tmpSimilarity = nearestNeighborDF.loc[
            nearestNeighborDF["External_Source"] == externalName,
            "Max_Tanimoto_to_training",
        ]

        ax.hist(
            tmpSimilarity,
            bins=similarityBins,
            alpha=0.45,
            color=externalColor,
            label=f"{externalName}, n={len(tmpSimilarity)}",
        )

    ax.axvline(0.50, linestyle="--", linewidth=1, c="k")
    ax.set_xlabel("Maximum Tanimoto similarity to selected training set")
    ax.set_ylabel("Number of external molecules")
    ax.set_title("Nearest training-set similarity by external dataset")
    ax.legend(frameon=False)

    plt.tight_layout()
    if savePlots:
        fig.savefig(resultsDir / "nearest_neighbor_similarity_distribution.png", dpi=dpi)
    plt.close(fig)


def plotPotencyVsSimilarity(
    nearestNeighborDF: pd.DataFrame,
    externalConfigs: List[Dict[str, Any]],
    resultsDir: Path,
    savePlots: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    for externalConfig in externalConfigs:
        externalName = externalConfig.get("label", externalConfig.get("name"))
        externalColor = externalConfig.get("color", None)
        externalMarker = externalConfig.get("marker", "+")

        tmpDF = nearestNeighborDF.loc[nearestNeighborDF["External_Source"] == externalName].copy()

        tmpDF["External_pPotency_prediction"] = pd.to_numeric(
            tmpDF["External_pPotency_prediction"], errors="coerce"
        )
        tmpDF["Max_Tanimoto_to_training"] = pd.to_numeric(
            tmpDF["Max_Tanimoto_to_training"], errors="coerce"
        )

        tmpDF = tmpDF.dropna(subset=["External_pPotency_prediction", "Max_Tanimoto_to_training"])

        ax.scatter(
            tmpDF["External_pPotency_prediction"],
            tmpDF["Max_Tanimoto_to_training"],
            s=45,
            alpha=0.70,
            color=externalColor,
            marker=externalMarker,
            label=f"{externalName} (n={len(tmpDF)})",
        )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted pPotency")
    ax.set_ylabel("Tanimoto similarity with training set")
    ax.set_title("Predicted potency vs chemical-space support")

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        borderaxespad=0,
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    if savePlots:
        fig.savefig(
            resultsDir / "pPotency_vs_tanimoto.png",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


# -----------------------------
# Run
# -----------------------------
def main() -> None:
    startTime = time.time()
    args = parseArgs()
    configPath = Path(args.config).expanduser().resolve()
    config = readYaml(str(configPath))

    jobName = sanitizeName(getNested(config, ["job", "name"], "trainingSet_similarity"))
    outputDir = ensureDir(Path(getNested(config, ["job", "outputDir"], "trainingSet_similarity_results")).expanduser().resolve())

    # Save the exact YAML used for reproducibility (kept from code-1)
    try:
        shutil.copy2(configPath, outputDir / f"{jobName}_config.yaml")
    except Exception:
        pass

    randomSeed = int(getNested(config, ["runtime", "randomSeed"], 42))
    smilesCol = getNested(config, ["columns", "smilesCol"], "Canonical_SMILES")
    fpRadius = int(getNested(config, ["fingerprint", "radius"], 2))
    fpBits = int(getNested(config, ["fingerprint", "nBits"], 2048))

    dpi = int(getNested(config, ["plots", "dpi"], 600))
    savePlots = getBool(getNested(config, ["plots", "save"], True), True)

    print(f"Job name: {jobName}")
    print(f"Output directory: {outputDir}")
    print(f"SMILES column: {smilesCol}")
    print(f"Morgan fingerprint: radius={fpRadius}, nBits={fpBits}")

    morganGenerator = rdFingerprintGenerator.GetMorganGenerator(radius=fpRadius, fpSize=fpBits)

    trainingDF, trainingConfig = prepareTrainingDataset(
        config=config,
        morganGenerator=morganGenerator,
        smilesCol=smilesCol,
        randomSeed=randomSeed,
    )

    # CODE-2 hardcodes training plot mask to dataSet == "Training"
    # So we force trainingDF["dataSet"]="Training" to mimic code-2 exactly.
    trainingDF["dataSet"] = "Training"

    externalAllDF, externalConfigs = prepareExternalDatasets(
        config=config,
        morganGenerator=morganGenerator,
        smilesCol=smilesCol,
        randomSeed=randomSeed,
        trainingDF=trainingDF,
    )

    # -----------------------------
    # Potency distribution plot (code-2)
    # -----------------------------
    plotPotencyDistribution(
        trainingDF=trainingDF,
        externalAllDF=externalAllDF,
        trainingConfig=trainingConfig,
        externalConfigs=externalConfigs,
        outputDir=outputDir,
        savePlots=savePlots,
        dpi=dpi,
    )

    # -----------------------------
    # Nearest neighbors stats (code-2)
    # -----------------------------
    nearestNeighborDF = buildNearestNeighborTable(
        trainingDF=trainingDF,
        externalAllDF=externalAllDF,
        smilesCol=smilesCol,
        externalConfigs=externalConfigs,
    )

    print("Nearest-neighbor summary by external source:")
    summary = (
        nearestNeighborDF
        .groupby(["External_Source", "Chemical_space_category"], observed=False)
        .size()
        .reset_index(name="Number_of_molecules")
    )
    print(summary.to_string(index=False))

    # -----------------------------
    # MDS embedding plot (code-2)
    # -----------------------------
    plotMDS_embedding(
        trainingDF=trainingDF,
        externalAllDF=externalAllDF,
        externalConfigs=externalConfigs,
        resultsDir=outputDir,
        savePlots=savePlots,
        dpi=dpi,
        randomSeed=randomSeed,
    )

    # -----------------------------
    # Nearest neighbor similarity distribution 
    # -----------------------------
    plotNearestNeighborSimilarityDistribution(
        nearestNeighborDF=nearestNeighborDF,
        externalConfigs=externalConfigs,
        resultsDir=outputDir,
        savePlots=savePlots,
        dpi=dpi,
    )

    # -----------------------------
    # Predicted potency vs similarity 
    # -----------------------------
    plotPotencyVsSimilarity(
        nearestNeighborDF=nearestNeighborDF,
        externalConfigs=externalConfigs,
        resultsDir=outputDir,
        savePlots=savePlots,
        dpi=dpi,
    )

    # Minimal run metadata
    metadataPath = outputDir / f"{jobName}_run_config_resolved.json"
    try:
        with open(metadataPath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass

    print(f"\nDone. Plots saved in: {outputDir}")
    print(f"Total runtime: {time.time() - startTime:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise