#!/usr/bin/env python3
"""
runRL_potency_single_gen.py

YAML-driven potency-only reinforcement-learning workflow for
reaction-constrained molecular optimization with DORAnet, MACAW, and ART.

This version keeps ADMET/Toxicity helper classes in the file for later
post-processing, but the default RL training path does not instantiate ADMET-AI,
does not score toxicity/QED during rollouts, and does not penalize uncertainty.

Usage
-----
python runRL_potency_single_gen.py config_gen1.yaml

Recommended first test
----------------------
Set:
    mode.run_smoke_tests: true
    training.enabled: false
    rollout.enabled: false

Then enable training and rollout once the smoke tests pass.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import pickle
import warnings
import hashlib
import cloudpickle
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import yaml


def configureRuntimeEnvironmentFromArgv() -> None:
    """Configure CPU thread settings before NumPy/RDKit are imported.

    The previous version forced OMP/MKL/OpenBLAS/NumExpr to one thread at import
    time. That is safe for laptop debugging but wasteful on an OpenMP-enabled
    CPU node. This function reads the optional `runtime` block from the YAML
    config before heavy scientific libraries are imported.

    If runtime.force_thread_env is false, existing shell/Slurm environment
    variables are respected. If it is true, the YAML values override them.
    """
    configPath = None
    for arg in sys.argv[1:]:
        if not str(arg).startswith("-"):
            configPath = Path(arg)
            break

    runtimeConfig = {}
    if configPath is not None and configPath.exists():
        try:
            with open(configPath, "r") as fileObj:
                configObj = yaml.safe_load(fileObj) or {}
            runtimeConfig = configObj.get("runtime", {}) or {}
        except Exception:
            runtimeConfig = {}

    forceThreadEnv = bool(runtimeConfig.get("force_thread_env", False))

    def setEnvFromConfig(envName: str, configKey: str, fallback: Optional[str] = None) -> None:
        value = runtimeConfig.get(configKey, None)
        if value is None:
            value = fallback
        if value is None:
            return
        if forceThreadEnv or envName not in os.environ:
            os.environ[envName] = str(value)

    slurmCpus = os.environ.get("SLURM_CPUS_PER_TASK")
    setEnvFromConfig("OMP_NUM_THREADS", "omp_num_threads", slurmCpus)
    setEnvFromConfig("MKL_NUM_THREADS", "mkl_num_threads", "1")
    setEnvFromConfig("OPENBLAS_NUM_THREADS", "openblas_num_threads", "1")
    setEnvFromConfig("NUMEXPR_NUM_THREADS", "numexpr_num_threads", "1")


configureRuntimeEnvironmentFromArgv()

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Configuration and general utilities
# =============================================================================

def loadConfig(configPath: str | Path) -> dict:
    with open(configPath, "r") as fileObj:
        config = yaml.safe_load(fileObj)
    if config is None:
        raise ValueError("Config file is empty.")
    return config


def ensureDir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def addProjectPaths(config: dict) -> None:
    paths = config.get("project_paths", {})

    for key in ("doranet_path", "art_path"):
        pathValue = paths.get(key)
        if pathValue and str(pathValue) not in sys.path:
            if key == "doranet_path":
                sys.path.insert(0, str(pathValue))
            else:
                sys.path.append(str(pathValue))

    for pathValue in paths.get("extra_python_paths", []) or []:
        if pathValue and str(pathValue) not in sys.path:
            sys.path.append(str(pathValue))


def normalizeColumns(DF: pd.DataFrame) -> pd.DataFrame:
    DF = DF.copy()
    DF.columns = [str(col).replace("\ufeff", "").strip() for col in DF.columns]
    return DF


@lru_cache(maxsize=500000)
def canonicalizeSmiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES with an LRU cache.

    RL revisits the same seed/product states many times, so caching this small
    RDKit operation removes a large amount of repeated parsing overhead.
    """
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def computeQED(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    return float(QED.qed(mol)) if mol is not None else np.nan


def countAtomsByElement(smiles: str, elements: List[str]) -> Dict[str, int]:
    """Count selected atom types in a canonical or raw SMILES string.

    This is used to derive DORAnet max_atoms directly from the selected seed
    pool. Counts are based on explicit atoms parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    counts = {element: 0 for element in elements}
    if mol is None:
        return counts

    elementSet = set(elements)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in elementSet:
            counts[symbol] += 1
    return counts


def deriveMaxAtomsFromSeeds(seedDF: pd.DataFrame, config: dict) -> Tuple[Dict[str, int], pd.DataFrame]:
    """Derive global DORAnet max_atoms from the selected seed pool.

    For each configured element, the base value is the maximum atom count found
    in any selected seed molecule. The DORAnet limit is then
    ceil(base_count * multiplier).

    Example: if the largest selected seed contains 10 carbon atoms and the
    multiplier is 1.5, max_atoms["C"] becomes 15 for the entire workflow.
    """
    doranetConfig = config.get("doranet", {}) or {}
    mode = str(doranetConfig.get("max_atoms_mode", "from_seed_pool")).lower()

    if mode in {"manual", "fixed", "config"}:
        manualMaxAtoms = doranetConfig.get("max_atoms")
        if not manualMaxAtoms:
            raise ValueError(
                "doranet.max_atoms_mode is manual/fixed, but doranet.max_atoms is not provided."
            )
        cleanManual = {str(k): int(v) for k, v in manualMaxAtoms.items()}
        atomRows = []
        return cleanManual, pd.DataFrame(atomRows)

    elements = doranetConfig.get("max_atoms_elements", ["C", "N", "O", "S"])
    elements = [str(element) for element in elements]
    multiplier = float(doranetConfig.get("max_atoms_multiplier", 1.5))

    smilesCol = "canonicalSmiles" if "canonicalSmiles" in seedDF.columns else "SMILES"
    atomRows = []
    for _, row in seedDF.iterrows():
        smiles = row[smilesCol]
        counts = countAtomsByElement(smiles, elements)
        counts["SMILES"] = smiles
        atomRows.append(counts)

    atomCountDF = pd.DataFrame(atomRows)
    if atomCountDF.empty:
        raise ValueError("Cannot derive DORAnet max_atoms because the selected seed pool is empty.")

    maxAtoms = {}
    baseCounts = {}
    for element in elements:
        baseCount = int(pd.to_numeric(atomCountDF[element], errors="coerce").fillna(0).max())
        baseCounts[element] = baseCount
        maxAtoms[element] = int(np.ceil(baseCount * multiplier))

    print("Derived DORAnet atom-count limits from selected seeds.")
    print("Base max seed atom counts:", baseCounts)
    print(f"Multiplier: {multiplier}")
    print("DORAnet max_atoms used for all generation runs:", maxAtoms)

    return maxAtoms, atomCountDF


def getDoranetGenList(config: dict) -> List[int]:
    """Read one or more DORAnet generation depths from config."""
    doranetConfig = config.get("doranet", {}) or {}

    if "gen_list" in doranetConfig and doranetConfig["gen_list"] is not None:
        genValue = doranetConfig["gen_list"]
    elif "gens" in doranetConfig and doranetConfig["gens"] is not None:
        genValue = doranetConfig["gens"]
    else:
        genValue = doranetConfig.get("gen", 1)

    if isinstance(genValue, (list, tuple)):
        genList = [int(x) for x in genValue]
    else:
        genList = [int(genValue)]

    genList = sorted(dict.fromkeys(genList))
    if not genList:
        raise ValueError("No DORAnet generations were provided. Use doranet.gen_list: [1, 2, 3].")
    return genList


def buildGenerationOutputDir(baseOutputDir: str | Path, gen: int) -> Path:
    """Append _genN to the configured base output directory."""
    basePath = Path(baseOutputDir)
    return basePath.parent / f"{basePath.name}_gen{int(gen)}"


def makeGenerationConfig(config: dict, gen: int, maxAtoms: Dict[str, int], outputDir: Path) -> dict:
    """Create a per-generation config without mutating the original config."""
    import copy

    runConfig = copy.deepcopy(config)
    runConfig.setdefault("output", {})["output_dir"] = str(outputDir)

    doranetConfig = runConfig.setdefault("doranet", {})
    baseJobPrefix = str(doranetConfig.get("job_prefix", "rl_potency_only"))
    doranetConfig["gen"] = int(gen)
    doranetConfig["max_atoms"] = {str(k): int(v) for k, v in maxAtoms.items()}
    doranetConfig["job_prefix"] = f"{baseJobPrefix}_gen{int(gen)}"

    networkSubdir = doranetConfig.get("network_subdir", "doranet_networks")
    doranetConfig["network_output_dir"] = str(outputDir / str(networkSubdir))

    return runConfig


@lru_cache(maxsize=500000)
def smilesToMorganFPTuple(smiles: str, radius: int = 2, nBits: int = 2048) -> Tuple[float, ...]:
    fpArray = np.zeros((nBits,), dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    if mol is None:
        return tuple(fpArray.tolist())

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    DataStructs.ConvertToNumpyArray(fp, fpArray)
    return tuple(fpArray.tolist())


def smilesToMorganFP(smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """Return a Morgan fingerprint as an ndarray using a cached tuple backend."""
    return np.asarray(smilesToMorganFPTuple(str(smiles), radius, nBits), dtype=np.float32)


# =============================================================================
# Seed library
# =============================================================================

def molFingerprintForDiversity(smiles: str, radius: int = 2, nBits: int = 2048):
    """Return an RDKit bit vector fingerprint for greedy diversity selection."""
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)


def greedyDiverseSeedSelection(
    seedDF: pd.DataFrame,
    maxSeeds: int,
    selectionConfig: dict,
) -> pd.DataFrame:
    """Pick high-potency seeds while avoiding near-duplicate starting points."""
    potencyCol = selectionConfig.get("potency_col", "pPotency_prediction")
    uncertaintyCol = selectionConfig.get("uncertainty_col", "pPotency_std")
    similarityThreshold = float(selectionConfig.get("similarity_threshold", 0.75))
    radius = int(selectionConfig.get("fingerprint_radius", 2))
    nBits = int(selectionConfig.get("fingerprint_bits", 2048))
    prefilterMultiplier = int(selectionConfig.get("prefilter_multiplier", 10))
    candidateLimit = max(maxSeeds * max(prefilterMultiplier, 1), maxSeeds)

    sortCols = [potencyCol]
    ascending = [False]
    if uncertaintyCol and uncertaintyCol in seedDF.columns:
        sortCols.append(uncertaintyCol)
        ascending.append(True)

    candidateDF = (
        seedDF.sort_values(sortCols, ascending=ascending)
        .head(candidateLimit)
        .reset_index(drop=True)
    )

    selectedRows = []
    selectedFPs = []
    for _, row in candidateDF.iterrows():
        fp = molFingerprintForDiversity(
            row["canonicalSmiles"],
            radius=radius,
            nBits=nBits,
        )
        if fp is None:
            continue
        if selectedFPs:
            maxSimilarity = max(DataStructs.TanimotoSimilarity(fp, selectedFP) for selectedFP in selectedFPs)
            if maxSimilarity >= similarityThreshold:
                continue
        selectedRows.append(row)
        selectedFPs.append(fp)
        if len(selectedRows) >= maxSeeds:
            break

    if len(selectedRows) < maxSeeds:
        selectedKeys = {row["canonicalSmiles"] for row in selectedRows}
        for _, row in candidateDF.iterrows():
            if row["canonicalSmiles"] in selectedKeys:
                continue
            selectedRows.append(row)
            selectedKeys.add(row["canonicalSmiles"])
            if len(selectedRows) >= maxSeeds:
                break

    return pd.DataFrame(selectedRows).reset_index(drop=True)


def applySeedSelection(seedDF: pd.DataFrame, seedConfig: dict) -> pd.DataFrame:
    """Config-driven seed selection.

    The previous workflow used the first max_seeds rows. For potency-only
    frontier search, the default is to start from high-potency seeds.
    Set seed.selection.uncertainty_col: null to avoid uncertainty-biased
    seed sorting.
    """
    maxSeeds = seedConfig.get("max_seeds")
    if maxSeeds is None:
        return seedDF.reset_index(drop=True)
    maxSeeds = int(maxSeeds)

    selectionConfig = seedConfig.get("selection", {}) or {}
    method = str(selectionConfig.get("method", "top_potency")).lower()
    potencyCol = selectionConfig.get("potency_col", "pPotency_prediction")
    uncertaintyCol = selectionConfig.get("uncertainty_col", "pPotency_std")
    priorityCol = selectionConfig.get("priority_col", seedConfig.get("priority_col", "OverallPriority2D"))

    if method in {"first", "head"}:
        selectedDF = seedDF.head(maxSeeds).copy()
    elif method in {"top_priority", "priority"} and priorityCol in seedDF.columns:
        selectedDF = seedDF.sort_values(priorityCol, ascending=False).head(maxSeeds).copy()
    elif method in {"top_potency_diverse", "potency_diverse", "diverse_potency"}:
        selectedDF = greedyDiverseSeedSelection(seedDF, maxSeeds, selectionConfig)
    elif method in {"top_potency", "potency"}:
        sortCols = [potencyCol]
        ascending = [False]
        if uncertaintyCol and uncertaintyCol in seedDF.columns:
            sortCols.append(uncertaintyCol)
            ascending.append(True)
        selectedDF = seedDF.sort_values(sortCols, ascending=ascending).head(maxSeeds).copy()
    else:
        print(f"Unknown seed.selection.method='{method}'. Falling back to first {maxSeeds} rows.")
        selectedDF = seedDF.head(maxSeeds).copy()

    return selectedDF.reset_index(drop=True)


def assignSeedSamplingWeights(seedDF: pd.DataFrame, seedConfig: dict) -> pd.DataFrame:
    """Assign reset sampling weights for RL episodes."""
    seedDF = seedDF.copy()
    samplingConfig = seedConfig.get("sampling", {}) or {}
    method = str(samplingConfig.get("method", "potency_softmax")).lower()
    potencyCol = samplingConfig.get("potency_col", "pPotency_prediction")
    priorityCol = samplingConfig.get("priority_col", seedConfig.get("priority_col", "OverallPriority2D"))

    if method in {"uniform", "equal"}:
        weights = np.ones(len(seedDF), dtype=float)
    elif method in {"priority_rank", "rank"} and priorityCol in seedDF.columns:
        ranks = seedDF[priorityCol].rank(ascending=False, method="first").to_numpy(dtype=float)
        weights = 1.0 / np.maximum(ranks, 1.0)
    elif method in {"potency_rank", "p_potency_rank"} and potencyCol in seedDF.columns:
        ranks = seedDF[potencyCol].rank(ascending=False, method="first").to_numpy(dtype=float)
        weights = 1.0 / np.maximum(ranks, 1.0)
    elif method in {"potency_softmax", "softmax"} and potencyCol in seedDF.columns:
        temperature = float(samplingConfig.get("temperature", 0.15))
        temperature = max(temperature, 1e-6)
        values = pd.to_numeric(seedDF[potencyCol], errors="coerce").fillna(seedDF[potencyCol].median()).to_numpy(dtype=float)
        logits = (values - np.nanmax(values)) / temperature
        weights = np.exp(np.clip(logits, -60.0, 0.0))
    elif priorityCol in seedDF.columns:
        ranks = seedDF[priorityCol].rank(ascending=False, method="first").to_numpy(dtype=float)
        weights = 1.0 / np.maximum(ranks, 1.0)
    else:
        weights = np.ones(len(seedDF), dtype=float)

    minWeight = float(samplingConfig.get("min_weight", 1e-6))
    weights = np.asarray(weights, dtype=float)
    weights[~np.isfinite(weights)] = 0.0
    weights = np.maximum(weights, minWeight)
    weights = weights / weights.sum()
    seedDF["seedWeight"] = weights
    seedDF["seedRank"] = np.arange(1, len(seedDF) + 1)
    return seedDF


def loadSeedDF(config: dict) -> Tuple[pd.DataFrame, dict]:
    """Load seed molecules and select a potency-enriched starting pool."""
    seedConfig = config["seed"]
    seedCsv = seedConfig["seed_csv"]
    smilesCol = seedConfig.get("smiles_col", "SMILES")

    seedDF = renameSmilesColumn(pd.read_csv(seedCsv), smilesCol)
    minimumCols = seedConfig.get("minimum_columns", ["SMILES", "pPotency_prediction"])
    missingMinimum = [col for col in minimumCols if col not in seedDF.columns]
    if missingMinimum:
        raise ValueError(f"Seed CSV is missing minimum required columns: {missingMinimum}")

    requestedCols = seedConfig.get("required_columns")
    if requestedCols:
        requestedCols = ["SMILES" if col == smilesCol else col for col in requestedCols]
        keepCols = [col for col in requestedCols if col in seedDF.columns]
        missingRequested = [col for col in requestedCols if col not in seedDF.columns]
        if missingRequested:
            print(f"Seed CSV does not contain these optional requested columns; continuing without them: {missingRequested}")
        if "SMILES" not in keepCols:
            keepCols = ["SMILES"] + keepCols
        for col in minimumCols:
            if col in seedDF.columns and col not in keepCols:
                keepCols.append(col)
        seedDF = seedDF[keepCols].copy()

    seedDF["canonicalSmiles"] = seedDF["SMILES"].apply(canonicalizeSmiles)
    seedDF = seedDF.dropna(subset=["canonicalSmiles"]).drop_duplicates("canonicalSmiles").reset_index(drop=True)
    if seedDF.empty:
        raise ValueError("No valid seed molecules remain after SMILES canonicalization.")

    if "pPotency_std" not in seedDF.columns:
        seedDF["pPotency_std"] = np.nan
    if "pPotency_lower_95CI" not in seedDF.columns:
        seedDF["pPotency_lower_95CI"] = seedDF["pPotency_prediction"] - 1.96 * seedDF["pPotency_std"]
    if "pPotency_upper_95CI" not in seedDF.columns:
        seedDF["pPotency_upper_95CI"] = seedDF["pPotency_prediction"] + 1.96 * seedDF["pPotency_std"]

    nBeforeSelection = len(seedDF)
    seedDF = applySeedSelection(seedDF, seedConfig)
    seedDF = assignSeedSamplingWeights(seedDF, seedConfig)
    print(f"Seed selection retained {len(seedDF)} / {nBeforeSelection} molecules.")
    if "pPotency_prediction" in seedDF.columns:
        print(
            "Selected seed potency range: "
            f"min={seedDF['pPotency_prediction'].min():.3f}, "
            f"median={seedDF['pPotency_prediction'].median():.3f}, "
            f"max={seedDF['pPotency_prediction'].max():.3f}"
        )

    rewardConfig = config.get("reward", {}).copy()
    rewardConfig.setdefault("potencyTarget", float(seedDF["pPotency_prediction"].quantile(0.75)))
    rewardConfig.setdefault("potencyScale", 0.25)
    rewardConfig.setdefault("deltaPotencyTarget", 0.10)
    rewardConfig.setdefault("deltaPotencyScale", 0.08)
    rewardConfig.setdefault("weights", {"potency": 0.70, "deltaPotency": 0.30})
    return seedDF, rewardConfig


# =============================================================================
# MACAW and ART
# =============================================================================

class MacawFeatureBuilder:
    """Build ART-ready MACAW features for candidate SMILES."""

    def __init__(self, macawTransformerPath: str | Path):
        self.macawTransformerPath = str(macawTransformerPath)
        with open(macawTransformerPath, "rb") as fileObj:
            self.mcw = pickle.load(fileObj)

    def transformSmilesList(self, smilesList: List[str]) -> pd.DataFrame:
        canonicalList = []
        for smiles in smilesList:
            canonical = canonicalizeSmiles(smiles)
            if canonical is not None:
                canonicalList.append(canonical)

        if not canonicalList:
            return pd.DataFrame(columns=["SMILES"])

        smilesSeries = pd.Series(canonicalList, name="SMILES")
        macawArray = self.mcw.transform(smilesSeries)

        macawCols = [f"MACAW_{idx + 1}" for idx in range(macawArray.shape[1])]
        macawDF = pd.DataFrame(macawArray, columns=macawCols)
        macawDF.insert(0, "SMILES", canonicalList)
        return macawDF


class ArtPotencyOracle:
    """ART potency predictor using post_pred_stats on MACAW features."""

    def __init__(
        self,
        artModelPath: str | Path,
        macawFeatureBuilder: MacawFeatureBuilder,
        artOutputDir: Optional[str | Path] = None,
        inputFeaturePrefix: str = "MACAW_",
    ):
        self.artModelPath = str(artModelPath)
        self.macawFeatureBuilder = macawFeatureBuilder
        self.inputFeaturePrefix = inputFeaturePrefix

        with open(artModelPath, "rb") as fileObj:
            self.artModel = cloudpickle.load(fileObj)

        if artOutputDir is not None:
            self.artModel.output_dir = str(artOutputDir)
            os.makedirs(self.artModel.output_dir, exist_ok=True)

    def predictBatch(self, smilesList: List[str]) -> pd.DataFrame:
        featureDF = self.macawFeatureBuilder.transformSmilesList(smilesList)

        outputCols = [
            "SMILES",
            "pPotency_prediction",
            "pPotency_std",
            "pPotency_lower_95CI",
            "pPotency_upper_95CI",
            "IC50(M)_prediction",
            "IC50(M)_lower_95CI",
            "IC50(M)_upper_95CI",
        ]

        if featureDF.empty:
            return pd.DataFrame(columns=outputCols)

        macawCols = [
            col for col in featureDF.columns
            if col.startswith(self.inputFeaturePrefix)
        ]
        if not macawCols:
            raise ValueError(
                f"No MACAW feature columns found with prefix '{self.inputFeaturePrefix}'."
            )

        featureMatrix = featureDF[macawCols].to_numpy(dtype=float)
        meanArray, stdArray = self.artModel.post_pred_stats(featureMatrix)

        meanArray = np.asarray(meanArray, dtype=float).ravel()
        stdArray = np.asarray(stdArray, dtype=float).ravel()

        predDF = featureDF[["SMILES"]].copy()
        predDF["pPotency_prediction"] = meanArray
        predDF["pPotency_std"] = stdArray
        predDF["pPotency_lower_95CI"] = meanArray - 1.96 * stdArray
        predDF["pPotency_upper_95CI"] = meanArray + 1.96 * stdArray
        predDF["IC50(M)_prediction"] = 10 ** (-predDF["pPotency_prediction"])
        predDF["IC50(M)_lower_95CI"] = 10 ** (-predDF["pPotency_upper_95CI"])
        predDF["IC50(M)_upper_95CI"] = 10 ** (-predDF["pPotency_lower_95CI"])

        return predDF[outputCols]

    def predictOne(self, smiles: str) -> dict:
        predDF = self.predictBatch([smiles])
        if predDF.empty:
            raise ValueError(f"ART prediction failed for SMILES: {smiles}")
        return predDF.iloc[0].to_dict()


# =============================================================================
# ADMET-AI and toxicity scoring
# =============================================================================

class AdmetOracle:
    """ADMET-AI predictor."""

    def __init__(self):
        from admet_ai import ADMETModel
        self.model = ADMETModel()

    def predictBatch(self, smilesList: List[str]) -> pd.DataFrame:
        predDF = self.model.predict(smiles=smilesList)
        if not isinstance(predDF, pd.DataFrame):
            predDF = pd.DataFrame(predDF)

        predDF = normalizeColumns(predDF.reset_index(drop=True))

        if len(predDF) != len(smilesList):
            raise ValueError(
                f"ADMET prediction mismatch: {len(predDF)} rows for {len(smilesList)} SMILES."
            )

        outDF = pd.DataFrame({"SMILES": smilesList}).reset_index(drop=True)
        outDF = pd.concat([outDF, predDF], axis=1)
        outDF = outDF.loc[:, ~outDF.columns.duplicated()]
        return outDF

    def predictOne(self, smiles: str) -> dict:
        return self.predictBatch([smiles]).iloc[0].to_dict()


def renameSmilesColumn(DF, smilesCol="SMILES"):
    """Normalize common SMILES column variants to a single SMILES column."""
    DF = normalizeColumns(DF)
    smilesCol = str(smilesCol).replace("\ufeff", "").strip()
    if smilesCol in DF.columns and smilesCol != "SMILES":
        return DF.rename(columns={smilesCol: "SMILES"})
    if "SMILES" in DF.columns:
        return DF
    lowerMap = {str(col).lower(): col for col in DF.columns}
    for candidate in ("smiles", "canonicalsmiles", "canonical_smiles"):
        if candidate in lowerMap:
            return DF.rename(columns={lowerMap[candidate]: "SMILES"})
    return DF


def getAdmetEndpointList():
    return defaultEndpointMetaDF()["endpoint"].drop_duplicates().tolist()


def countAdmetEndpoints(DF):
    return sum(endpoint in DF.columns for endpoint in getAdmetEndpointList())


def ensureAdmetPredictions(inputDF, admetOracle, smilesCol="SMILES", minExistingEndpoints=5, outputCsvPath=None, label="dataframe"):
    """Use existing ADMET endpoint columns when available; otherwise compute them."""
    DF = renameSmilesColumn(inputDF, smilesCol)
    if "SMILES" not in DF.columns:
        print(f"WARNING: {label} has no SMILES column. Cannot compute ADMET; using CSV unchanged.")
        return DF
    existingCount = countAdmetEndpoints(DF)
    if existingCount >= int(minExistingEndpoints):
        print(f"{label}: found {existingCount} ADMET endpoint columns. Using existing CSV values.")
        return DF
    canonicalList = (DF["SMILES"].dropna().astype(str).map(canonicalizeSmiles).dropna().drop_duplicates().tolist())
    if not canonicalList:
        print(f"WARNING: {label} has no valid SMILES for ADMET prediction. Using CSV unchanged.")
        return DF
    print(f"{label}: found only {existingCount} ADMET endpoint columns. Running ADMET-AI for {len(canonicalList)} unique molecules.")
    predDF = renameSmilesColumn(admetOracle.predictBatch(canonicalList), "SMILES")
    DF = DF.copy()
    DF["_canonicalSmilesForMerge"] = DF["SMILES"].apply(canonicalizeSmiles)
    predDF["_canonicalSmilesForMerge"] = predDF["SMILES"].apply(canonicalizeSmiles)
    predDF = predDF.drop(columns=["SMILES"], errors="ignore").drop_duplicates("_canonicalSmilesForMerge", keep="last")
    predCols = [col for col in predDF.columns if col != "_canonicalSmilesForMerge"]
    DF = DF.drop(columns=[col for col in predCols if col in DF.columns], errors="ignore")
    mergedDF = DF.merge(predDF, on="_canonicalSmilesForMerge", how="left")
    mergedDF = mergedDF.drop(columns=["_canonicalSmilesForMerge"], errors="ignore")
    mergedDF = mergedDF.loc[:, ~mergedDF.columns.duplicated()]
    if outputCsvPath is not None:
        outputCsvPath = Path(outputCsvPath)
        outputCsvPath.parent.mkdir(parents=True, exist_ok=True)
        mergedDF.to_csv(outputCsvPath, index=False)
        print(f"Saved ADMET-scored {label} to: {outputCsvPath}")
    return mergedDF


def finalizeSeedScores(seedDF, toxicityOracle, outputDir):
    seedDF = seedDF.copy()
    hasSummary = ("coreToxicityScore" in seedDF.columns and "ToxicitySafety" in seedDF.columns and seedDF["coreToxicityScore"].notna().any() and seedDF["ToxicitySafety"].notna().any())
    if hasSummary:
        print("Seed CSV already contains toxicity summary columns. Using existing seed toxicity scores.")
        if "ADMEFeasibility" not in seedDF.columns:
            seedDF["ADMEFeasibility"] = np.nan
        return seedDF
    print("Seed CSV is missing toxicity summaries. Computing seed toxicity scores from ADMET-AI.")
    scoredSeedDF = toxicityOracle.scoreBatch(seedDF["canonicalSmiles"].tolist())
    keepCols = [col for col in ["SMILES", "ToxicitySafety", "coreToxicityScore", "ADMEFeasibility"] if col in scoredSeedDF.columns]
    scoredSeedDF = scoredSeedDF[keepCols].copy()
    scoredSeedDF["canonicalSmiles"] = scoredSeedDF["SMILES"].apply(canonicalizeSmiles)
    scoredSeedDF = scoredSeedDF.drop(columns=["SMILES"], errors="ignore")
    seedDF = seedDF.drop(columns=["ToxicitySafety", "coreToxicityScore", "ADMEFeasibility"], errors="ignore")
    seedDF = seedDF.merge(scoredSeedDF, on="canonicalSmiles", how="left")
    outPath = Path(outputDir) / "seedDF_admet_scored.csv"
    seedDF.to_csv(outPath, index=False)
    print(f"Saved ADMET-scored seed dataframe to: {outPath}")
    return seedDF


def finalizeRewardConfig(config, seedDF, rewardConfig=None):
    """Finalize the potency-only reward configuration.

    This function intentionally does not add toxicity, QED, route-depth,
    potency-uncertainty, or potency-loss defaults. The RL reward should depend
    only on absolute potency and potency improvement over the selected seed.
    """
    rewardConfig = (rewardConfig or config.get("reward", {})).copy()
    rewardConfig.setdefault("potencyTarget", float(seedDF["pPotency_prediction"].median()))
    rewardConfig.setdefault("potencyScale", 0.25)
    rewardConfig.setdefault("deltaPotencyTarget", 0.10)
    rewardConfig.setdefault("deltaPotencyScale", 0.08)
    rewardConfig.setdefault("invalidActionPenalty", -0.50)
    rewardConfig.setdefault("invalidActionTerminates", True)
    rewardConfig.setdefault("weights", {"potency": 0.70, "deltaPotency": 0.30})
    return rewardConfig


def defaultEndpointMetaDF() -> pd.DataFrame:
    """Endpoint metadata used when no endpoint metadata CSV is supplied."""
    records = [
        ("AMES", "toxicity", "lowerBetter"),
        ("Carcinogens_Lagunin", "toxicity", "lowerBetter"),
        ("ClinTox", "toxicity", "lowerBetter"),
        ("DILI", "toxicity", "lowerBetter"),
        ("Skin_Reaction", "toxicity", "lowerBetter"),
        ("hERG", "toxicity", "lowerBetter"),
        ("LD50_Zhu", "toxicity", "higherBetter"),
        ("NR-AR-LBD", "toxicity", "lowerBetter"),
        ("NR-AR", "toxicity", "lowerBetter"),
        ("NR-AhR", "toxicity", "lowerBetter"),
        ("NR-Aromatase", "toxicity", "lowerBetter"),
        ("NR-ER-LBD", "toxicity", "lowerBetter"),
        ("NR-ER", "toxicity", "lowerBetter"),
        ("NR-PPAR-gamma", "toxicity", "lowerBetter"),
        ("SR-ARE", "toxicity", "lowerBetter"),
        ("SR-ATAD5", "toxicity", "lowerBetter"),
        ("SR-HSE", "toxicity", "lowerBetter"),
        ("SR-MMP", "toxicity", "lowerBetter"),
        ("SR-p53", "toxicity", "lowerBetter"),
        ("Bioavailability_Ma", "adme", "higherBetter"),
        ("HIA_Hou", "adme", "higherBetter"),
        ("PAMPA_NCATS", "adme", "higherBetter"),
        ("Caco2_Wang", "adme", "higherBetter"),
        ("Solubility_AqSolDB", "adme", "higherBetter"),
        ("CYP1A2_Veith", "adme", "lowerBetter"),
        ("CYP2C19_Veith", "adme", "lowerBetter"),
        ("CYP2C9_Substrate_CarbonMangels", "adme", "referenceMatch"),
        ("CYP2C9_Veith", "adme", "lowerBetter"),
        ("CYP2D6_Substrate_CarbonMangels", "adme", "referenceMatch"),
        ("CYP2D6_Veith", "adme", "lowerBetter"),
        ("CYP3A4_Substrate_CarbonMangels", "adme", "referenceMatch"),
        ("CYP3A4_Veith", "adme", "lowerBetter"),
        ("Pgp_Broccatelli", "adme", "referenceMatch"),
        ("Clearance_Hepatocyte_AZ", "adme", "referenceMatch"),
        ("Clearance_Microsome_AZ", "adme", "referenceMatch"),
        ("Half_Life_Obach", "adme", "referenceMatch"),
        ("PPBR_AZ", "adme", "referenceMatch"),
        ("VDss_Lombardo", "adme", "referenceMatch"),
    ]
    return pd.DataFrame(records, columns=["endpoint", "majorGroup", "direction"])


def robustIQR(series: pd.Series) -> float:
    cleanSeries = pd.to_numeric(series, errors="coerce").dropna()
    if len(cleanSeries) < 5:
        return np.nan
    q25, q75 = np.percentile(cleanSeries, [25, 75])
    return float(q75 - q25)


def buildWeightDF(refScoreDF: pd.DataFrame, endpointMetaDF: pd.DataFrame) -> pd.DataFrame:
    """Build data-driven endpoint weights from reference ADMET distributions."""
    weightRows = []

    for majorGroupName in endpointMetaDF["majorGroup"].unique():
        groupMetaDF = endpointMetaDF[endpointMetaDF["majorGroup"] == majorGroupName].copy()
        endpoints = [e for e in groupMetaDF["endpoint"].tolist() if e in refScoreDF.columns]

        if not endpoints:
            continue

        groupRefDF = refScoreDF[endpoints].apply(pd.to_numeric, errors="coerce")
        groupCorrDF = groupRefDF.corr().abs()

        for endpointName in endpoints:
            series = groupRefDF[endpointName]
            iqrValue = robustIQR(series)

            informativeness = 1.0 / (iqrValue + 1e-6) if pd.notna(iqrValue) else 1.0
            otherEndpoints = [x for x in endpoints if x != endpointName]

            if otherEndpoints and endpointName in groupCorrDF.index:
                meanAbsCorr = groupCorrDF.loc[endpointName, otherEndpoints].dropna().mean()
                uniqueness = max(1.0 - meanAbsCorr, 1e-3) if pd.notna(meanAbsCorr) else 1.0
            else:
                uniqueness = 1.0

            direction = groupMetaDF.loc[
                groupMetaDF["endpoint"] == endpointName,
                "direction",
            ].iloc[0]

            weightRows.append(
                {
                    "endpoint": endpointName,
                    "majorGroup": majorGroupName,
                    "direction": direction,
                    "iqrValue": iqrValue,
                    "informativeness": informativeness,
                    "uniqueness": uniqueness,
                    "rawWeight": informativeness * uniqueness,
                }
            )

    weightDF = pd.DataFrame(weightRows)
    if weightDF.empty:
        raise ValueError(
            "No endpoint weights could be built. Check reference ADMET columns."
        )

    weightDF["finalWeight"] = weightDF.groupby("majorGroup")["rawWeight"].transform(
        lambda x: x / x.sum()
    )
    return weightDF


def getSortedRefArray(refSeries: pd.Series) -> np.ndarray:
    refArray = pd.to_numeric(refSeries, errors="coerce").dropna().to_numpy(dtype=float)
    refArray = refArray[np.isfinite(refArray)]
    return np.sort(refArray)


def higherBetterDes(refSeries: pd.Series, valueSeries: pd.Series) -> pd.Series:
    refArray = getSortedRefArray(refSeries)
    valueArray = pd.to_numeric(valueSeries, errors="coerce").to_numpy(dtype=float)
    outArray = np.full(len(valueArray), np.nan, dtype=float)

    if len(refArray) == 0:
        return pd.Series(outArray, index=valueSeries.index)

    finiteMask = np.isfinite(valueArray)
    percentArray = np.searchsorted(refArray, valueArray[finiteMask], side="right") / len(refArray)
    outArray[finiteMask] = np.clip(percentArray, 0.0, 1.0)
    return pd.Series(outArray, index=valueSeries.index)


def lowerBetterDes(refSeries: pd.Series, valueSeries: pd.Series) -> pd.Series:
    refArray = getSortedRefArray(refSeries)
    valueArray = pd.to_numeric(valueSeries, errors="coerce").to_numpy(dtype=float)
    outArray = np.full(len(valueArray), np.nan, dtype=float)

    if len(refArray) == 0:
        return pd.Series(outArray, index=valueSeries.index)

    finiteMask = np.isfinite(valueArray)
    percentArray = np.searchsorted(refArray, valueArray[finiteMask], side="right") / len(refArray)
    outArray[finiteMask] = np.clip(1.0 - percentArray, 0.0, 1.0)
    return pd.Series(outArray, index=valueSeries.index)


def referenceMatchDes(refSeries: pd.Series, valueSeries: pd.Series) -> pd.Series:
    refArray = getSortedRefArray(refSeries)
    valueArray = pd.to_numeric(valueSeries, errors="coerce").to_numpy(dtype=float)
    outArray = np.full(len(valueArray), np.nan, dtype=float)

    if len(refArray) == 0:
        return pd.Series(outArray, index=valueSeries.index)

    refMedian = np.nanmedian(refArray)
    q25, q75 = np.nanpercentile(refArray, [25, 75])
    robustSigma = (q75 - q25) / 1.349 if q75 > q25 else np.nanstd(refArray)
    robustSigma = robustSigma if np.isfinite(robustSigma) and robustSigma > 0 else 1e-6

    finiteMask = np.isfinite(valueArray)
    zArray = (valueArray[finiteMask] - refMedian) / robustSigma
    outArray[finiteMask] = np.exp(-0.5 * zArray**2)
    return pd.Series(np.clip(outArray, 0.0, 1.0), index=valueSeries.index)


def weightedGeoMeanSeries(
    DF: pd.DataFrame,
    valueColList: List[str],
    weightSeries: pd.Series,
) -> pd.Series:
    valueMatrix = DF[valueColList].to_numpy(dtype=float)
    weightArray = np.asarray(weightSeries, dtype=float)
    outArray = np.full(valueMatrix.shape[0], np.nan, dtype=float)

    for rowIdx, rowArray in enumerate(valueMatrix):
        finiteMask = np.isfinite(rowArray)
        if finiteMask.sum() == 0:
            continue

        rowValues = np.clip(rowArray[finiteMask], 1e-6, 1.0)
        rowWeights = weightArray[finiteMask]
        outArray[rowIdx] = np.exp(np.sum(rowWeights * np.log(rowValues)) / np.sum(rowWeights))

    return pd.Series(outArray, index=DF.index)


class ToxicityScoringOracle:
    """Convert ADMET-AI endpoint predictions into toxicity and ADME scores."""

    def __init__(self, admetOracle: AdmetOracle, refScoreDF: pd.DataFrame, weightDF: pd.DataFrame):
        self.admetOracle = admetOracle
        self.refScoreDF = refScoreDF.copy()
        self.weightDF = weightDF.copy()
        self.toxWeightDF = self.weightDF[self.weightDF["majorGroup"] == "toxicity"].copy()
        self.admeWeightDF = self.weightDF[self.weightDF["majorGroup"] == "adme"].copy()

    def scoreBatch(self, smilesList: List[str]) -> pd.DataFrame:
        admetDF = self.admetOracle.predictBatch(smilesList)
        candScoreDF = admetDF.copy()

        for _, row in self.weightDF.iterrows():
            endpoint = row["endpoint"]
            direction = row["direction"]
            desCol = f"{endpoint}_des"

            if endpoint not in candScoreDF.columns or endpoint not in self.refScoreDF.columns:
                continue

            if direction == "higherBetter":
                candScoreDF[desCol] = higherBetterDes(self.refScoreDF[endpoint], candScoreDF[endpoint])
            elif direction == "lowerBetter":
                candScoreDF[desCol] = lowerBetterDes(self.refScoreDF[endpoint], candScoreDF[endpoint])
            elif direction == "referenceMatch":
                candScoreDF[desCol] = referenceMatchDes(self.refScoreDF[endpoint], candScoreDF[endpoint])

        toxDesCols = [f"{e}_des" for e in self.toxWeightDF["endpoint"] if f"{e}_des" in candScoreDF.columns]
        admeDesCols = [f"{e}_des" for e in self.admeWeightDF["endpoint"] if f"{e}_des" in candScoreDF.columns]

        if toxDesCols:
            toxMask = self.toxWeightDF["endpoint"].apply(lambda e: f"{e}_des" in toxDesCols)
            toxWeights = self.toxWeightDF.loc[toxMask, "finalWeight"]
            candScoreDF["ToxicitySafety"] = weightedGeoMeanSeries(candScoreDF, toxDesCols, toxWeights)
            candScoreDF["coreToxicityScore"] = 1.0 - candScoreDF["ToxicitySafety"]
        else:
            candScoreDF["ToxicitySafety"] = np.nan
            candScoreDF["coreToxicityScore"] = np.nan

        if admeDesCols:
            admeMask = self.admeWeightDF["endpoint"].apply(lambda e: f"{e}_des" in admeDesCols)
            admeWeights = self.admeWeightDF.loc[admeMask, "finalWeight"]
            candScoreDF["ADMEFeasibility"] = weightedGeoMeanSeries(candScoreDF, admeDesCols, admeWeights)
        else:
            candScoreDF["ADMEFeasibility"] = np.nan

        return candScoreDF

    def scoreOne(self, smiles: str) -> dict:
        scoreDF = self.scoreBatch([smiles])
        if scoreDF.empty:
            raise ValueError(f"Toxicity scoring failed for SMILES: {smiles}")
        return scoreDF.iloc[0].to_dict()


def loadOrBuildReferenceAndWeights(config: dict, admetOracle: AdmetOracle, outputDir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load or build ADMET-scored reference data and endpoint weights."""
    toxConfig = config.get("toxicity", {})
    referenceScoredCsv = toxConfig.get("reference_scored_csv")
    referenceInputCsv = toxConfig.get("reference_input_csv")
    referenceSmilesCol = toxConfig.get("reference_smiles_col", "SMILES")
    maxReferenceRows = toxConfig.get("max_reference_rows")
    minExistingEndpoints = int(toxConfig.get("min_existing_admet_endpoints", 5))

    if referenceScoredCsv and Path(referenceScoredCsv).exists():
        print(f"Loading reference CSV: {referenceScoredCsv}")
        refDF = renameSmilesColumn(pd.read_csv(referenceScoredCsv), referenceSmilesCol)
    elif referenceInputCsv and Path(referenceInputCsv).exists():
        print(f"Loading reference input CSV: {referenceInputCsv}")
        refDF = renameSmilesColumn(pd.read_csv(referenceInputCsv), referenceSmilesCol)
    else:
        raise ValueError("No valid toxicity.reference_scored_csv or toxicity.reference_input_csv was found.")
    if "SMILES" not in refDF.columns:
        raise ValueError(f"Reference CSV must contain a SMILES-like column. Available columns: {list(refDF.columns)}")
    if maxReferenceRows is not None:
        refDF = refDF.head(int(maxReferenceRows)).copy()

    refScoreDF = ensureAdmetPredictions(refDF, admetOracle, smilesCol="SMILES", minExistingEndpoints=minExistingEndpoints, outputCsvPath=outputDir / "reference_admet_scored.csv", label="reference dataframe")

    endpointMetaPath = toxConfig.get("endpoint_meta_csv")
    if endpointMetaPath and Path(endpointMetaPath).exists():
        endpointMetaDF = normalizeColumns(pd.read_csv(endpointMetaPath))
    else:
        endpointMetaDF = defaultEndpointMetaDF()

    weightCsv = toxConfig.get("weight_csv")
    if weightCsv and Path(weightCsv).exists():
        weightDF = normalizeColumns(pd.read_csv(weightCsv))
        print(f"Loaded endpoint weights from: {weightCsv}")
    else:
        weightDF = buildWeightDF(refScoreDF, endpointMetaDF)
        weightOut = outputDir / "endpoint_weights.csv"
        weightDF.to_csv(weightOut, index=False)
        print(f"Saved endpoint weights to: {weightOut}")

    requiredWeightCols = ["endpoint", "direction", "majorGroup", "finalWeight"]
    missingCols = [col for col in requiredWeightCols if col not in weightDF.columns]
    if missingCols:
        raise ValueError(f"weightDF is missing required columns: {missingCols}")
    return refScoreDF, weightDF


# =============================================================================
# DORAnet adapter
# =============================================================================

@dataclass
class DoranetAction:
    actionIndex: int
    productSmiles: str
    sourceSmiles: str
    generation: int
    metadata: Dict[str, Any]


class DoranetAdapter:
    """Generate reaction-constrained candidate actions with DORAnet.

    This adapter uses a stable SHA1-based job name instead of Python's built-in
    hash(). The same source SMILES therefore maps to the same DORAnet job prefix
    across Python sessions. It also records source→product→job metadata so final
    RL candidates can be traced back to the DORAnet reaction-network JSON files
    used to generate them.
    """

    def __init__(self, config: dict):
        import doranet.modules.enzymatic as enzymatic

        self.enzymatic = enzymatic
        dconf = config.get("doranet", {})

        self.helpers = set(
            dconf.get(
                "helpers",
                [
                    "O", "O=O", "[H][H]", "O=C=O", "C=O", "[C-]#[O+]",
                    "Br", "[Br][Br]", "CO", "C=C", "O=S(O)O", "N",
                    "O=S(=O)(O)O", "O=NO", "N#N", "O=[N+]([O-])O",
                    "NO", "C#N", "S", "O=S=O", "N#CO",
                ],
            )
        )

        self.ruleset = dconf.get("ruleset", "JN3604IMT")
        self.maxAtoms = {str(k): int(v) for k, v in dconf.get("max_atoms", {"C": 41, "N": 9, "O": 12, "S": 0}).items()}
        self.gen = int(dconf.get("gen", 1))
        self.maxActions = int(dconf.get("max_actions", 16))
        self.jobPrefix = dconf.get("job_prefix", "rl_tmp")

        # Each single-generation job writes DORAnet network JSON files into its
        # own output directory, e.g. ./_gen1/doranet_networks.
        self.networkOutputDir = ensureDir(Path(dconf.get("network_output_dir", ".")).resolve())

        self.actionCacheEnabled = bool(dconf.get("action_cache_enabled", True))
        self.actionCacheDir = ensureDir(
            self.networkOutputDir / str(dconf.get("action_cache_subdir", "action_cache"))
        )

        self.cache: Dict[str, List[DoranetAction]] = {}
        self.jobMapRows: List[dict] = []
        self.actionMapRows: List[dict] = []
        self.loggedJobs: set[str] = set()
        self.loggedActions: set[tuple[str, str]] = set()

    def stableJobName(self, canonicalSmiles: str) -> str:
        stableHash = hashlib.sha1(canonicalSmiles.encode("utf-8")).hexdigest()[:12]
        return f"{self.jobPrefix}_{stableHash}"

    def findGeneratedJsonFiles(self, jobName: str) -> List[str]:
        """Return candidate DORAnet JSON files associated with a job name."""
        candidates = list(Path.cwd().glob(f"{jobName}*.json"))
        candidates += list(self.networkOutputDir.glob(f"{jobName}*.json"))

        unique = []
        seen = set()
        for pathObj in candidates:
            resolved = str(pathObj.resolve())
            if resolved not in seen:
                unique.append(resolved)
                seen.add(resolved)

        if not unique:
            unique = [str((self.networkOutputDir / f"{jobName}.json").resolve())]

        return unique

    def getActionCachePath(self, jobName: str) -> Path:
        return self.actionCacheDir / f"{jobName}_actions.csv"

    def buildActionsFromProducts(
        self,
        canonical: str,
        jobName: str,
        productSmilesList: List[str],
    ) -> List[DoranetAction]:
        """Create DoranetAction objects and trace rows from product SMILES."""
        jsonFiles = self.findGeneratedJsonFiles(jobName)
        actions = [
            DoranetAction(
                actionIndex=idx,
                productSmiles=product,
                sourceSmiles=canonical,
                generation=self.gen,
                metadata={
                    "ruleset": self.ruleset,
                    "jobName": jobName,
                    "jsonFiles": jsonFiles,
                    "networkOutputDir": str(self.networkOutputDir),
                },
            )
            for idx, product in enumerate(productSmilesList)
        ]

        if jobName not in self.loggedJobs:
            self.jobMapRows.append(
                {
                    "jobName": jobName,
                    "sourceSmiles": canonical,
                    "generation": self.gen,
                    "ruleset": self.ruleset,
                    "maxActions": self.maxActions,
                    "maxAtoms": "|".join(f"{k}:{v}" for k, v in self.maxAtoms.items()),
                    "numProductsKept": len(productSmilesList),
                    "jsonFiles": "|".join(jsonFiles),
                    "productSmilesList": "|".join(productSmilesList),
                }
            )
            self.loggedJobs.add(jobName)

        for action in actions:
            key = (jobName, action.productSmiles)
            if key in self.loggedActions:
                continue

            self.actionMapRows.append(
                {
                    "jobName": jobName,
                    "sourceSmiles": canonical,
                    "productSmiles": action.productSmiles,
                    "actionIndex": action.actionIndex,
                    "generation": self.gen,
                    "ruleset": self.ruleset,
                    "jsonFiles": "|".join(jsonFiles),
                }
            )
            self.loggedActions.add(key)

        return actions

    def enumerateActions(self, smiles: str) -> List[DoranetAction]:
        canonical = canonicalizeSmiles(smiles)
        if canonical is None:
            return []

        if canonical in self.cache:
            return self.cache[canonical]

        jobName = self.stableJobName(canonical)
        starters = {canonical}
        cachePath = self.getActionCachePath(jobName)

        if self.actionCacheEnabled and cachePath.exists():
            try:
                cacheDF = pd.read_csv(cachePath)
                productSmilesList = (
                    cacheDF.get("productSmiles", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .map(canonicalizeSmiles)
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )[: self.maxActions]
                actions = self.buildActionsFromProducts(canonical, jobName, productSmilesList)
                self.cache[canonical] = actions
                return actions
            except Exception as exc:
                print(f"WARNING: could not read DORAnet action cache {cachePath}: {exc}")

        try:
            # DORAnet writes network JSON files relative to the active working
            # directory in this workflow. Temporarily run generation inside the
            # configured per-generation network directory so outputs are physically separated.
            oldCwd = Path.cwd()
            os.chdir(self.networkOutputDir)
            try:
                forwardNetwork = self.enzymatic.generate_network(
                    job_name=jobName,
                    starters=starters,
                    gen=self.gen,
                    max_atoms=self.maxAtoms,
                    direction="forward",
                    ruleset=self.ruleset,
                )
            finally:
                os.chdir(oldCwd)
        except Exception as exc:
            print(f"DORAnet failed for {canonical}: {exc}")
            self.cache[canonical] = []
            return []

        productSmilesList = []
        for mol in forwardNetwork.mols:
            product = canonicalizeSmiles(mol.uid)
            if product is None or product == canonical or product in self.helpers:
                continue
            productSmilesList.append(product)

        productSmilesList = list(dict.fromkeys(productSmilesList))[: self.maxActions]

        if self.actionCacheEnabled:
            pd.DataFrame({"productSmiles": productSmilesList}).to_csv(cachePath, index=False)

        actions = self.buildActionsFromProducts(canonical, jobName, productSmilesList)
        self.cache[canonical] = actions
        return actions

    def applyAction(self, smiles: str, actionObj: DoranetAction) -> str:
        return actionObj.productSmiles

    def saveTraceTables(self, outputDir: str | Path) -> None:
        outputDir = ensureDir(outputDir)

        jobMapPath = outputDir / "doranet_job_map.csv"
        actionMapPath = outputDir / "doranet_action_map.csv"

        pd.DataFrame(self.jobMapRows).drop_duplicates().to_csv(jobMapPath, index=False)
        pd.DataFrame(self.actionMapRows).drop_duplicates().to_csv(actionMapPath, index=False)

        print("Saved DORAnet job map to:", jobMapPath)
        print("Saved DORAnet action map to:", actionMapPath)


# =============================================================================
# Combined scoring and RL
# =============================================================================

class CombinedMoleculeScorer:
    """Score molecules with ART potency only.

    ADMET/Toxicity/QED are deliberately excluded from the default RL scoring
    path so that training and rollout are driven only by pPotency_prediction
    and deltaPotency. The ART predictive standard deviation is retained in the
    output for reference, but it is not used in the reward.
    """

    def __init__(
        self,
        artOracle: ArtPotencyOracle,
        toxicityOracle: Optional[ToxicityScoringOracle] = None,
        computeQEDForOutput: bool = False,
        scoreToxicityForOutput: bool = False,
    ):
        self.artOracle = artOracle
        self.toxicityOracle = toxicityOracle
        self.computeQEDForOutput = bool(computeQEDForOutput)
        self.scoreToxicityForOutput = bool(scoreToxicityForOutput and toxicityOracle is not None)
        self.cache: Dict[str, dict] = {}

    def scoreBatch(self, smilesList: List[str]) -> pd.DataFrame:
        canonicalList = []
        for smiles in smilesList:
            canonical = canonicalizeSmiles(smiles)
            if canonical is not None:
                canonicalList.append(canonical)

        canonicalList = list(dict.fromkeys(canonicalList))
        if not canonicalList:
            return pd.DataFrame()

        cachedRows = [self.cache[smiles] for smiles in canonicalList if smiles in self.cache]
        uncachedList = [smiles for smiles in canonicalList if smiles not in self.cache]

        newRows = []
        if uncachedList:
            scoredDF = self.artOracle.predictBatch(uncachedList)

            if self.scoreToxicityForOutput:
                toxDF = self.toxicityOracle.scoreBatch(uncachedList)
                keepToxCols = [
                    col for col in toxDF.columns
                    if col not in scoredDF.columns or col == "SMILES"
                ]
                scoredDF = scoredDF.merge(toxDF[keepToxCols], on="SMILES", how="left")

            if self.computeQEDForOutput:
                scoredDF["QED"] = scoredDF["SMILES"].apply(computeQED)

            newRows = scoredDF.to_dict(orient="records")
            for row in newRows:
                self.cache[row["SMILES"]] = row

        return pd.DataFrame(cachedRows + newRows)

    def scoreOne(self, smiles: str) -> dict:
        scoreDF = self.scoreBatch([smiles])
        if scoreDF.empty:
            raise ValueError(f"Scoring failed for SMILES: {smiles}")
        return scoreDF.iloc[0].to_dict()

def sigmoidScaled(x: float, center: float, scale: float) -> float:
    if pd.isna(x):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-(float(x) - center) / scale)))


def computeRewardVector(scoreDict: dict, rewardConfig: dict, routeDepth: int) -> dict:
    """Return potency-only reward components.

    Reward terms used:
      1. potency: absolute ART-predicted pPotency
      2. deltaPotency: improvement over the exact seed molecule

    Reward terms intentionally not used:
      toxicity, deltaToxicity, QED, route depth, ART uncertainty, potency-loss penalty.
    """
    potencyVal = scoreDict.get("pPotency_prediction", np.nan)
    seedPotency = scoreDict.get("seedPotency", np.nan)

    if pd.isna(seedPotency) or pd.isna(potencyVal):
        deltaPotency = 0.0
    else:
        deltaPotency = float(potencyVal) - float(seedPotency)

    potencyReward = sigmoidScaled(
        potencyVal,
        float(rewardConfig["potencyTarget"]),
        float(rewardConfig.get("potencyScale", 0.25)),
    )

    deltaPotencyReward = sigmoidScaled(
        deltaPotency,
        float(rewardConfig.get("deltaPotencyTarget", 0.10)),
        float(rewardConfig.get("deltaPotencyScale", 0.08)),
    )

    return {
        "potency": float(potencyReward),
        "deltaPotency": float(deltaPotencyReward),
    }


def getDefaultPreferenceWeights(rewardConfig: dict) -> dict:
    """Map scalar reward weights into potency-only preferences."""
    legacy = rewardConfig.get("weights", {})
    return {
        "potency": float(legacy.get("potency", 0.70)),
        "deltaPotency": float(legacy.get("deltaPotency", 0.30)),
    }
def normalizePreferenceWeights(preferenceWeights: dict) -> dict:
    cleaned = {str(k): float(v) for k, v in preferenceWeights.items()}
    denom = sum(abs(v) for v in cleaned.values())
    if denom <= 0:
        raise ValueError(f"Invalid preference weights: {preferenceWeights}")
    return {k: v / denom for k, v in cleaned.items()}


def scalarizeReward(rewardVector: dict, preferenceWeights: dict) -> float:
    weights = normalizePreferenceWeights(preferenceWeights)
    total = 0.0
    for objectiveName, weight in weights.items():
        total += float(weight) * float(rewardVector.get(objectiveName, 0.0))
    return float(total)


def computeMultiObjectiveReward(
    scoreDict: dict,
    rewardConfig: dict,
    routeDepth: int,
    preferenceWeights: dict,
) -> dict:
    rewardVector = computeRewardVector(scoreDict, rewardConfig, routeDepth)
    scalarReward = scalarizeReward(rewardVector, preferenceWeights)

    out = {
        "totalReward": float(scalarReward),
        "rewardVector": rewardVector,
    }
    for key, value in rewardVector.items():
        out[f"{key}RewardComponent"] = float(value)
    return out


def computeRlReward(scoreDict: dict, rewardConfig: dict, routeDepth: int) -> dict:
    """Backward-compatible scalar reward wrapper."""
    return computeMultiObjectiveReward(
        scoreDict=scoreDict,
        rewardConfig=rewardConfig,
        routeDepth=routeDepth,
        preferenceWeights=getDefaultPreferenceWeights(rewardConfig),
    )


# =============================================================================
# Pareto archive and post-rollout Pareto selection
# =============================================================================

DEFAULT_PARETO_OBJECTIVES = {
    "pPotency_prediction": "maximize",
    "deltaPotency": "maximize",
}


def objectiveVector(row, objectives):
    values = []
    for col, direction in objectives.items():
        try:
            value = row[col]
        except Exception:
            return None
        if pd.isna(value):
            return None
        value = float(value)
        values.append(value if direction == "maximize" else -value)
    return np.asarray(values, dtype=float)


def dominates(rowA, rowB, objectives):
    vecA = objectiveVector(rowA, objectives)
    vecB = objectiveVector(rowB, objectives)
    if vecA is None or vecB is None:
        return False
    return bool(np.all(vecA >= vecB) and np.any(vecA > vecB))


class ParetoArchive:
    """Small in-memory Pareto archive used to reward non-dominated molecules."""

    def __init__(self, config: dict, seedDF: Optional[pd.DataFrame] = None):
        paretoConfig = config.get("pareto", {})
        self.enabled = bool(paretoConfig.get("enabled", False))
        self.objectives = paretoConfig.get("objectives", DEFAULT_PARETO_OBJECTIVES).copy()
        self.nonDominatedBonus = float(paretoConfig.get("non_dominated_bonus", 0.20))
        self.dominatedPenalty = float(paretoConfig.get("dominated_penalty", 0.05))
        self.archiveRows: List[dict] = []

        if seedDF is not None and bool(paretoConfig.get("initialize_with_seeds", True)):
            for row in seedDF.to_dict(orient="records"):
                self.add(row, returnBonus=False)

    def add(self, candidate: dict, returnBonus: bool = True) -> float:
        if not self.enabled:
            return 0.0
        if objectiveVector(candidate, self.objectives) is None:
            return 0.0

        if any(dominates(row, candidate, self.objectives) for row in self.archiveRows):
            return -self.dominatedPenalty if returnBonus else 0.0

        self.archiveRows = [row for row in self.archiveRows if not dominates(candidate, row, self.objectives)]
        self.archiveRows.append(dict(candidate))
        return self.nonDominatedBonus if returnBonus else 0.0

    def toDataFrame(self) -> pd.DataFrame:
        return pd.DataFrame(self.archiveRows)


def useParetoArchiveDuringEnv(config: dict) -> bool:
    """Return whether Pareto archive should affect environment rewards.

    Pareto post-processing can remain enabled while the in-environment archive is
    disabled. This avoids pairwise dominance checks during PPO training/rollout
    when non_dominated_bonus and dominated_penalty are both zero.
    """
    paretoConfig = config.get("pareto", {}) or {}
    if not bool(paretoConfig.get("enabled", False)):
        return False
    if not bool(paretoConfig.get("use_during_training", False)):
        return False
    bonus = float(paretoConfig.get("non_dominated_bonus", 0.0))
    penalty = float(paretoConfig.get("dominated_penalty", 0.0))
    return bool(abs(bonus) > 0.0 or abs(penalty) > 0.0)


def addParetoFrontColumns(DF: pd.DataFrame, objectives: Optional[dict] = None) -> pd.DataFrame:
    """Annotate a dataframe with Pareto-front membership and dominance counts."""
    if DF is None or DF.empty:
        return pd.DataFrame() if DF is None else DF.copy()

    objectives = (objectives or DEFAULT_PARETO_OBJECTIVES).copy()
    workDF = DF.copy().reset_index(drop=True)
    nRows = len(workDF)
    isPareto = np.ones(nRows, dtype=bool)
    dominatedByCount = np.zeros(nRows, dtype=int)

    for i in range(nRows):
        rowI = workDF.iloc[i]
        for j in range(nRows):
            if i == j:
                continue
            rowJ = workDF.iloc[j]
            if dominates(rowJ, rowI, objectives):
                dominatedByCount[i] += 1
                isPareto[i] = False

    workDF["isParetoFront"] = isPareto
    workDF["dominatedByCount"] = dominatedByCount
    return workDF


def rankParetoFront(paretoDF: pd.DataFrame, rankingConfig: dict) -> pd.DataFrame:
    """Rank Pareto candidates by continuous percentile preferences.

    This is not a hard filter. It preserves the full front, then ranks candidates
    according to the current scientific direction: higher potency and higher
    potency improvement over the starting seed.
    """
    if paretoDF is None or paretoDF.empty or not rankingConfig.get("enabled", False):
        return pd.DataFrame() if paretoDF is None else paretoDF.copy()

    rankedDF = paretoDF.copy().reset_index(drop=True)
    score = np.zeros(len(rankedDF), dtype=float)
    weightSum = 0.0

    rankingWeights = {k: v for k, v in rankingConfig.items() if k not in {"enabled", "method"}}
    for col, weight in rankingWeights.items():
        if col not in rankedDF.columns:
            continue
        values = pd.to_numeric(rankedDF[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        pct = values.rank(pct=True, method="average").to_numpy(dtype=float)
        # In the potency-only workflow all configured ranking objectives are maximize.
        pct = np.nan_to_num(pct, nan=0.0)
        rankedDF[f"rank_component_{col}"] = pct
        score += float(weight) * pct
        weightSum += abs(float(weight))

    rankedDF["paretoRankScore"] = score / weightSum if weightSum > 0 else score
    rankedDF = rankedDF.sort_values("paretoRankScore", ascending=False).reset_index(drop=True)
    rankedDF["paretoRank"] = np.arange(1, len(rankedDF) + 1)
    return rankedDF


def saveParetoOutputs(generatedDF: pd.DataFrame, config: dict, outputDir: Path):
    paretoConfig = config.get("pareto", {})
    if not bool(paretoConfig.get("enabled", False)) or generatedDF is None or generatedDF.empty:
        return generatedDF, pd.DataFrame()

    objectives = paretoConfig.get("objectives", DEFAULT_PARETO_OBJECTIVES).copy()
    annotatedDF = addParetoFrontColumns(generatedDF, objectives=objectives)
    paretoDF = annotatedDF.loc[annotatedDF["isParetoFront"]].copy()

    sortCols = [col for col in ["pPotency_prediction", "deltaPotency"] if col in paretoDF.columns]
    if sortCols:
        ascending = [False for _ in sortCols]
        paretoDF = paretoDF.sort_values(sortCols, ascending=ascending).reset_index(drop=True)

    rankedDF = rankParetoFront(paretoDF, paretoConfig.get("ranking", {}))

    annotatedPath = outputDir / paretoConfig.get("annotated_output_csv", "rl_generated_candidates_with_pareto.csv")
    paretoPath = outputDir / paretoConfig.get("front_output_csv", "rl_generated_candidates_pareto_front.csv")
    rankedPath = outputDir / paretoConfig.get("ranked_front_output_csv", "rl_generated_candidates_pareto_front_ranked.csv")
    annotatedDF.to_csv(annotatedPath, index=False)
    paretoDF.to_csv(paretoPath, index=False)
    if not rankedDF.empty:
        rankedDF.to_csv(rankedPath, index=False)
    print("Saved Pareto-annotated candidates to:", annotatedPath)
    print("Saved Pareto front candidates to:", paretoPath)
    if not rankedDF.empty:
        print("Saved ranked Pareto front candidates to:", rankedPath)
    return annotatedDF, paretoDF


def buildObservation(smiles: str, stepIndex: int, seedRow: pd.Series, nBits: int = 2048) -> np.ndarray:
    """Build a potency-only observation.

    The policy sees molecular fingerprint, normalized route step, and seed potency.
    It does not receive toxicity, QED, or uncertainty features.
    """
    fpVec = smilesToMorganFP(smiles, nBits=nBits)
    scalarVec = np.array(
        [
            float(stepIndex) / 5.0,
            float(seedRow["pPotency_prediction"]) / 10.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([fpVec, scalarVec]).astype(np.float32)

def buildEnvClass():
    import gymnasium as gym
    from gymnasium import spaces

    class MoleculeRlEnv(gym.Env):
        """Gymnasium-compatible molecule optimization environment."""

        def __init__(
            self,
            seedDF: pd.DataFrame,
            doranetAdapter: DoranetAdapter,
            combinedScorer: CombinedMoleculeScorer,
            rewardConfig: dict,
            maxSteps: int = 3,
            maxActions: int = 16,
            fpBits: int = 2048,
            paretoArchive: Optional[ParetoArchive] = None,
            preferenceWeights: Optional[dict] = None,
        ):
            super().__init__()
            self.seedDF = seedDF.reset_index(drop=True)
            self.doranetAdapter = doranetAdapter
            self.combinedScorer = combinedScorer
            self.rewardConfig = rewardConfig
            self.maxSteps = int(maxSteps)
            self.maxActions = int(maxActions)
            self.fpBits = int(fpBits)
            self.paretoArchive = paretoArchive
            self.preferenceWeights = preferenceWeights or getDefaultPreferenceWeights(rewardConfig)

            self.action_space = spaces.Discrete(self.maxActions + 1)  # 0 = STOP
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.fpBits + 2,),
                dtype=np.float32,
            )

            self.currentSmiles = None
            self.seedRow = None
            self.stepIndex = 0
            self.routeHistory = []
            self.actionList = []

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            choiceIndex = np.random.choice(
                self.seedDF.index,
                p=self.seedDF["seedWeight"].values,
            )

            self.seedRow = self.seedDF.loc[choiceIndex].copy()
            self.currentSmiles = self.seedRow["canonicalSmiles"]
            self.stepIndex = 0
            self.routeHistory = []
            self.actionList = self.doranetAdapter.enumerateActions(self.currentSmiles)

            obs = buildObservation(self.currentSmiles, self.stepIndex, self.seedRow, self.fpBits)
            info = {
                "seedSmiles": self.currentSmiles,
                "numActions": len(self.actionList),
                "seedPotency": float(self.seedRow["pPotency_prediction"]),
            }
            return obs, info

        def step(self, actionId):
            actionId = int(actionId)

            if actionId == 0:
                return self._terminalStep()

            chosenIndex = actionId - 1

            if chosenIndex >= len(self.actionList):
                obs = buildObservation(self.currentSmiles, self.stepIndex, self.seedRow, self.fpBits)
                invalidPenalty = float(self.rewardConfig.get("invalidActionPenalty", -0.50))
                invalidTerminates = bool(self.rewardConfig.get("invalidActionTerminates", True))
                return obs, invalidPenalty, invalidTerminates, False, {
                    "invalidAction": True,
                    "numActions": len(self.actionList),
                }

            actionObj = self.actionList[chosenIndex]
            nextSmiles = canonicalizeSmiles(
                self.doranetAdapter.applyAction(self.currentSmiles, actionObj)
            )

            if nextSmiles is None:
                obs = buildObservation(self.currentSmiles, self.stepIndex, self.seedRow, self.fpBits)
                return obs, -1.0, True, False, {"invalidProduct": True}

            self.currentSmiles = nextSmiles
            self.routeHistory.append(actionObj)
            self.stepIndex += 1
            self.actionList = self.doranetAdapter.enumerateActions(self.currentSmiles)

            if self.stepIndex >= self.maxSteps or len(self.actionList) == 0:
                return self._terminalStep()

            obs = buildObservation(self.currentSmiles, self.stepIndex, self.seedRow, self.fpBits)
            return obs, -0.01, False, False, {
                "intermediate": True,
                "numActions": len(self.actionList),
            }

        def _terminalStep(self):
            scoreDict = self.combinedScorer.scoreOne(self.currentSmiles)

            seedPotency = float(self.seedRow.get("pPotency_prediction", np.nan))

            scoreDict["seedSmiles"] = self.seedRow.get("canonicalSmiles", "")
            scoreDict["seedPotency"] = seedPotency
            scoreDict["deltaPotency"] = (
                float(scoreDict.get("pPotency_prediction", np.nan)) - seedPotency
                if np.isfinite(seedPotency) and pd.notna(scoreDict.get("pPotency_prediction", np.nan))
                else np.nan
            )

            rewardDict = computeMultiObjectiveReward(
                scoreDict=scoreDict,
                rewardConfig=self.rewardConfig,
                routeDepth=len(self.routeHistory),
                preferenceWeights=self.preferenceWeights,
            )

            paretoBonus = 0.0
            if self.paretoArchive is not None:
                paretoBonus = self.paretoArchive.add(scoreDict)
                rewardDict["totalRewardBeforePareto"] = float(rewardDict["totalReward"])
                rewardDict["paretoBonus"] = float(paretoBonus)
                rewardDict["totalReward"] = float(rewardDict["totalReward"] + paretoBonus)
            else:
                rewardDict["paretoBonus"] = 0.0

            obs = buildObservation(self.currentSmiles, self.stepIndex, self.seedRow, self.fpBits)
            info = {
                "scoreDict": scoreDict,
                "rewardDict": rewardDict,
                "routeHistory": self.routeHistory,
            }
            return obs, rewardDict["totalReward"], True, False, info

    return MoleculeRlEnv



# =============================================================================
# Route tracing utilities
# =============================================================================

def summarizeRoute(routeHistory: List[DoranetAction]) -> dict:
    """Convert a list of DoranetAction objects into CSV-friendly route metadata."""
    if not routeHistory:
        return {
            "routeSourceSmiles": "",
            "routeProductSmiles": "",
            "routeJobNames": "",
            "routeJsonFiles": "",
            "routeActionIndices": "",
        }

    sourceList = [action.sourceSmiles for action in routeHistory]
    productList = [action.productSmiles for action in routeHistory]
    jobList = [str(action.metadata.get("jobName", "")) for action in routeHistory]
    actionIndexList = [str(action.actionIndex) for action in routeHistory]

    jsonFileList = []
    for action in routeHistory:
        jsonFiles = action.metadata.get("jsonFiles", [])
        if isinstance(jsonFiles, str):
            jsonFiles = [jsonFiles]
        jsonFileList.extend([str(x) for x in jsonFiles])
    jsonFileList = list(dict.fromkeys(jsonFileList))

    return {
        "routeSourceSmiles": "|".join(sourceList),
        "routeProductSmiles": "|".join(productList),
        "routeJobNames": "|".join(jobList),
        "routeJsonFiles": "|".join(jsonFileList),
        "routeActionIndices": "|".join(actionIndexList),
    }

# =============================================================================
# Execution helpers
# =============================================================================

def runSmokeTests(
    seedDF: pd.DataFrame,
    macawFeatureBuilder: MacawFeatureBuilder,
    artOracle: ArtPotencyOracle,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    rlConfig: dict,
    outputDir: Path,
    nTest: int = 2,
) -> None:
    print("\n================ Potency-only smoke tests ================")

    smilesTest = seedDF["canonicalSmiles"].head(nTest).tolist()

    macawDF = macawFeatureBuilder.transformSmilesList(smilesTest)
    artDF = artOracle.predictBatch(smilesTest)
    combinedDF = combinedScorer.scoreBatch(smilesTest)

    macawDF.to_csv(outputDir / "smoke_macaw_features.csv", index=False)
    artDF.to_csv(outputDir / "smoke_art_predictions.csv", index=False)
    combinedDF.to_csv(outputDir / "smoke_potency_only_scores.csv", index=False)

    actions = doranetAdapter.enumerateActions(smilesTest[0])
    print(f"MACAW features       : {macawDF.shape}")
    print(f"ART predictions      : {artDF.shape}")
    print(f"Potency-only scores  : {combinedDF.shape}")
    print(f"DORAnet first actions: {len(actions)}")

    runManualEnvTest(seedDF, doranetAdapter, combinedScorer, rewardConfig, rlConfig)
    print("Potency-only smoke tests complete. Output written to:", outputDir)

def runManualEnvTest(
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    rlConfig: dict,
):
    print("\n================ Manual environment test ================")

    EnvClass = buildEnvClass()
    env = EnvClass(
        seedDF=seedDF,
        doranetAdapter=doranetAdapter,
        combinedScorer=combinedScorer,
        rewardConfig=rewardConfig,
        maxSteps=int(rlConfig.get("max_steps", 2)),
        maxActions=int(rlConfig.get("max_actions", 16)),
        fpBits=int(rlConfig.get("fingerprint_bits", 2048)),
        preferenceWeights=getDefaultPreferenceWeights(rewardConfig),
    )

    obs, info = env.reset()
    print("Reset info:", info)

    if len(env.actionList) > 0:
        obs, reward, terminated, truncated, stepInfo = env.step(1)
        print(
            "After one DORAnet action | "
            f"reward={reward:.4f}, terminated={terminated}, keys={list(stepInfo.keys())}"
        )

    obs, reward, terminated, truncated, stepInfo = env.step(0)
    print(f"STOP reward: {reward:.4f}")
    print("Reward dict:", stepInfo.get("rewardDict"))
    return env


def _precomputeSeedActionWorker(task: dict) -> dict:
    """Worker used by parallel seed-action precomputation.

    Each worker owns an independent DoranetAdapter instance. This is important
    because DORAnet generation temporarily changes the process working directory
    and writes JSON/cache files. Separate processes avoid shared-current-directory
    race conditions while still using the same per-generation cache folder.
    """
    workerOmpThreads = str(task.get("workerOmpThreads", "1"))
    os.environ["OMP_NUM_THREADS"] = workerOmpThreads
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    config = task["config"]
    seedIndex = int(task["seedIndex"])
    smiles = str(task["smiles"])
    seedPotency = float(task.get("seedPotency", np.nan))

    startTime = time.time()
    try:
        adapter = DoranetAdapter(config)
        actionList = adapter.enumerateActions(smiles)
        jobName = adapter.stableJobName(canonicalizeSmiles(smiles) or smiles)
        cachePath = adapter.getActionCachePath(jobName)
        return {
            "seedIndex": seedIndex,
            "seedSmiles": smiles,
            "seedPotency": seedPotency,
            "numActions": int(len(actionList)),
            "jobName": jobName,
            "cachePath": str(cachePath),
            "status": "ok",
            "error": "",
            "runtimeSeconds": float(time.time() - startTime),
        }
    except Exception as exc:
        return {
            "seedIndex": seedIndex,
            "seedSmiles": smiles,
            "seedPotency": seedPotency,
            "numActions": 0,
            "jobName": "",
            "cachePath": "",
            "status": "failed",
            "error": repr(exc),
            "runtimeSeconds": float(time.time() - startTime),
        }


def precomputeSeedDoranetActions(
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    config: dict,
    outputDir: Path,
) -> None:
    """Optionally precompute/cache DORAnet actions for selected seed states.

    This is Option A: parallelize only the environment action-generation cache
    for the selected seed molecules before PPO starts. It does not score or rank
    molecules, and it does not choose an action path. PPO still learns a policy
    from potency rewards during training.

    The cache is per generation because the config contains one `doranet.gen`,
    one `doranet.job_prefix`, and one per-generation `network_output_dir`.
    Therefore gen=1, gen=2, and gen=3 jobs never share action-cache files.
    """
    doranetConfig = config.get("doranet", {}) or {}
    if not bool(doranetConfig.get("precompute_seed_actions", True)):
        print("Seed-action precompute disabled by config.")
        return

    outputDir = ensureDir(outputDir)
    startTime = time.time()
    seedWorkDF = seedDF.reset_index(drop=True).copy()

    requestedWorkers = int(doranetConfig.get("precompute_num_workers", 1))
    maxWorkers = max(1, min(requestedWorkers, len(seedWorkDF)))
    useParallel = bool(doranetConfig.get("precompute_parallel", maxWorkers > 1)) and maxWorkers > 1
    workerOmpThreads = int(doranetConfig.get("precompute_worker_omp_threads", 1))

    print("\n================ DORAnet seed-action precompute ================")
    print(f"Generation: gen={config.get('doranet', {}).get('gen')}")
    print(f"Selected seed states: {len(seedWorkDF)}")
    print(f"Parallel precompute: {useParallel}")
    print(f"Precompute workers: {maxWorkers}")
    print(f"OMP threads per precompute worker: {workerOmpThreads}")
    print("Action cache directory:", doranetAdapter.actionCacheDir)

    tasks = []
    for seedIndex, row in seedWorkDF.iterrows():
        tasks.append(
            {
                "config": config,
                "seedIndex": int(seedIndex),
                "smiles": str(row["canonicalSmiles"]),
                "seedPotency": float(row.get("pPotency_prediction", np.nan)),
                "workerOmpThreads": int(workerOmpThreads),
            }
        )

    rows = []
    if useParallel:
        import concurrent.futures as futures

        with futures.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futureList = [executor.submit(_precomputeSeedActionWorker, task) for task in tasks]
            for doneIdx, futureObj in enumerate(futures.as_completed(futureList), start=1):
                row = futureObj.result()
                rows.append(row)
                print(
                    f"[{doneIdx:03d}/{len(futureList):03d}] "
                    f"seedIndex={row['seedIndex']} status={row['status']} "
                    f"numActions={row['numActions']} time={row['runtimeSeconds']:.2f}s"
                )
    else:
        # Serial fallback is useful for debugging and avoids multiprocessing
        # overhead on very small runs. This path also populates the main
        # adapter's in-memory cache.
        for task in tasks:
            smiles = task["smiles"]
            actionList = doranetAdapter.enumerateActions(smiles)
            jobName = doranetAdapter.stableJobName(canonicalizeSmiles(smiles) or smiles)
            cachePath = doranetAdapter.getActionCachePath(jobName)
            rows.append(
                {
                    "seedIndex": task["seedIndex"],
                    "seedSmiles": smiles,
                    "seedPotency": task["seedPotency"],
                    "numActions": len(actionList),
                    "jobName": jobName,
                    "cachePath": str(cachePath),
                    "status": "ok",
                    "error": "",
                    "runtimeSeconds": np.nan,
                }
            )

    rows = sorted(rows, key=lambda x: int(x["seedIndex"]))
    outPath = outputDir / "seed_doranet_action_precompute_summary.csv"
    summaryDF = pd.DataFrame(rows)
    summaryDF.to_csv(outPath, index=False)

    nOk = int((summaryDF["status"] == "ok").sum()) if not summaryDF.empty else 0
    nFailed = int((summaryDF["status"] != "ok").sum()) if not summaryDF.empty else 0
    totalActions = int(pd.to_numeric(summaryDF.get("numActions", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    print(
        "Precomputed DORAnet seed actions in "
        f"{time.time() - startTime:.2f} seconds | "
        f"ok={nOk}, failed={nFailed}, totalCachedActions={totalActions}"
    )
    print("Saved seed-action precompute summary to:", outPath)

    if nFailed > 0 and bool(doranetConfig.get("precompute_fail_fast", False)):
        failedDF = summaryDF.loc[summaryDF["status"] != "ok"]
        raise RuntimeError(
            "One or more DORAnet seed-action precompute workers failed. "
            f"See {outPath}. First failure: {failedDF.iloc[0]['error']}"
        )


def trainPolicy(
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    config: dict,
    outputDir: Path,
    paretoArchive: Optional[ParetoArchive] = None,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    EnvClass = buildEnvClass()
    rlConfig = config.get("rl", {})
    trainConfig = config.get("training", {})

    env = DummyVecEnv(
        [
            lambda: EnvClass(
                seedDF=seedDF,
                doranetAdapter=doranetAdapter,
                combinedScorer=combinedScorer,
                rewardConfig=rewardConfig,
                maxSteps=int(rlConfig.get("max_steps", 2)),
                maxActions=int(rlConfig.get("max_actions", 16)),
                fpBits=int(rlConfig.get("fingerprint_bits", 2048)),
                paretoArchive=paretoArchive,
            )
        ]
    )

    model = PPO(
        policy=trainConfig.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(trainConfig.get("learning_rate", 3e-4)),
        n_steps=int(trainConfig.get("n_steps", 256)),
        batch_size=int(trainConfig.get("batch_size", 64)),
        gamma=float(trainConfig.get("gamma", 0.95)),
        gae_lambda=float(trainConfig.get("gae_lambda", 0.95)),
        clip_range=float(trainConfig.get("clip_range", 0.2)),
        ent_coef=float(trainConfig.get("ent_coef", 0.02)),
        verbose=int(trainConfig.get("verbose", 1)),
    )

    totalTimesteps = int(trainConfig.get("total_timesteps", 10000))
    print(f"\nTraining PPO for {totalTimesteps} timesteps...")
    model.learn(total_timesteps=totalTimesteps)

    modelPath = outputDir / trainConfig.get("model_filename", "ppo_doranet_art_admet.zip")
    model.save(modelPath)
    print("Saved PPO model to:", modelPath)
    return model


def rolloutPolicy(
    model,
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    config: dict,
    outputDir: Path,
    paretoArchive: Optional[ParetoArchive] = None,
) -> pd.DataFrame:
    EnvClass = buildEnvClass()
    rlConfig = config.get("rl", {})
    rolloutConfig = config.get("rollout", {})

    env = EnvClass(
        seedDF=seedDF,
        doranetAdapter=doranetAdapter,
        combinedScorer=combinedScorer,
        rewardConfig=rewardConfig,
        maxSteps=int(rlConfig.get("max_steps", 2)),
        maxActions=int(rlConfig.get("max_actions", 16)),
        fpBits=int(rlConfig.get("fingerprint_bits", 2048)),
        paretoArchive=paretoArchive,
    )

    numEpisodes = int(rolloutConfig.get("num_episodes", 100))
    deterministic = bool(rolloutConfig.get("deterministic", True))

    generatedRows = []

    for episodeIndex in range(numEpisodes):
        obs, _ = env.reset()
        done = False
        stepInfo = {}
        reward = np.nan

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, stepInfo = env.step(int(action))
            done = terminated or truncated

        if "scoreDict" in stepInfo:
            row = dict(stepInfo["scoreDict"])
            row["episodeIndex"] = episodeIndex
            row["reward"] = reward
            routeHistory = stepInfo.get("routeHistory", [])
            row["routeLength"] = len(routeHistory)
            row.update(summarizeRoute(routeHistory))
            generatedRows.append(row)

    generatedDF = pd.DataFrame(generatedRows)
    if not generatedDF.empty:
        generatedDF = (
            generatedDF.drop_duplicates(subset=["SMILES"])
            .sort_values("reward", ascending=False)
            .reset_index(drop=True)
        )

    outCsv = outputDir / rolloutConfig.get("output_csv", "rl_generated_candidates.csv")
    generatedDF.to_csv(outCsv, index=False)
    print("Saved generated candidates to:", outCsv)
    return generatedDF




def trainMultiObjectivePolicies(
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    config: dict,
    outputDir: Path,
) -> Dict[str, Any]:
    """Train one PPO policy per preference vector.

    Each policy uses the same vector reward components but a different scalarization.
    The union of their rollouts gives a practical approximation to a Pareto RL search.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    EnvClass = buildEnvClass()
    rlConfig = config.get("rl", {})
    trainConfig = config.get("training", {})
    moConfig = config.get("multi_objective", {})
    preferenceSets = moConfig.get("preference_sets", {})
    if not preferenceSets:
        raise ValueError("multi_objective.preference_sets must contain at least one preference policy.")

    trainedModels = {}
    totalTimesteps = int(trainConfig.get("total_timesteps", 50000))

    for preferenceName, preferenceWeights in preferenceSets.items():
        preferenceWeights = normalizePreferenceWeights(preferenceWeights)
        print(f"\n================ Training preference policy: {preferenceName} ================")
        print("Preference weights:", preferenceWeights)

        paretoArchive = ParetoArchive(config, seedDF=seedDF) if useParetoArchiveDuringEnv(config) else None
        env = DummyVecEnv([
            lambda pref=preferenceWeights, archive=paretoArchive: EnvClass(
                seedDF=seedDF,
                doranetAdapter=doranetAdapter,
                combinedScorer=combinedScorer,
                rewardConfig=rewardConfig,
                maxSteps=int(rlConfig.get("max_steps", 2)),
                maxActions=int(rlConfig.get("max_actions", 16)),
                fpBits=int(rlConfig.get("fingerprint_bits", 2048)),
                paretoArchive=archive,
                preferenceWeights=pref,
            )
        ])

        model = PPO(
            policy=trainConfig.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=float(trainConfig.get("learning_rate", 3e-4)),
            n_steps=int(trainConfig.get("n_steps", 512)),
            batch_size=int(trainConfig.get("batch_size", 64)),
            gamma=float(trainConfig.get("gamma", 0.95)),
            gae_lambda=float(trainConfig.get("gae_lambda", 0.95)),
            clip_range=float(trainConfig.get("clip_range", 0.2)),
            ent_coef=float(trainConfig.get("ent_coef", 0.04)),
            verbose=int(trainConfig.get("verbose", 1)),
        )

        model.learn(total_timesteps=totalTimesteps)
        modelPath = outputDir / f"ppo_{preferenceName}.zip"
        model.save(modelPath)
        trainedModels[preferenceName] = model
        print("Saved preference policy to:", modelPath)

    return trainedModels


def rolloutMultiObjectivePolicies(
    trainedModels: Dict[str, Any],
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    config: dict,
    outputDir: Path,
) -> pd.DataFrame:
    EnvClass = buildEnvClass()
    rlConfig = config.get("rl", {})
    rolloutConfig = config.get("rollout", {})
    moConfig = config.get("multi_objective", {})
    preferenceSets = moConfig.get("preference_sets", {})

    numEpisodesPerPolicy = int(rolloutConfig.get("num_episodes_per_policy", rolloutConfig.get("num_episodes", 1000)))
    deterministic = bool(rolloutConfig.get("deterministic", False))
    rows = []

    for preferenceName, model in trainedModels.items():
        preferenceWeights = normalizePreferenceWeights(preferenceSets[preferenceName])
        paretoArchive = ParetoArchive(config, seedDF=seedDF) if useParetoArchiveDuringEnv(config) else None
        env = EnvClass(
            seedDF=seedDF,
            doranetAdapter=doranetAdapter,
            combinedScorer=combinedScorer,
            rewardConfig=rewardConfig,
            maxSteps=int(rlConfig.get("max_steps", 2)),
            maxActions=int(rlConfig.get("max_actions", 16)),
            fpBits=int(rlConfig.get("fingerprint_bits", 2048)),
            paretoArchive=paretoArchive,
            preferenceWeights=preferenceWeights,
        )

        print(f"\nRolling out {numEpisodesPerPolicy} episodes for policy: {preferenceName}")
        for episodeIndex in range(numEpisodesPerPolicy):
            obs, _ = env.reset()
            done = False
            stepInfo = {}
            reward = np.nan

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, stepInfo = env.step(int(action))
                done = terminated or truncated

            if "scoreDict" in stepInfo:
                row = dict(stepInfo["scoreDict"])
                row["preferencePolicy"] = preferenceName
                row["episodeIndex"] = episodeIndex
                row["reward"] = reward
                routeHistory = stepInfo.get("routeHistory", [])
                row["routeLength"] = len(routeHistory)
                row.update(summarizeRoute(routeHistory))
                rewardDict = stepInfo.get("rewardDict", {})
                for key, value in rewardDict.items():
                    if key == "rewardVector":
                        for objName, objVal in value.items():
                            row[f"mo_{objName}"] = objVal
                    else:
                        row[key] = value
                rows.append(row)

    generatedDF = pd.DataFrame(rows)
    if not generatedDF.empty:
        sortCols = [col for col in ["pPotency_prediction", "deltaPotency", "reward"] if col in generatedDF.columns]
        ascending = [False for _ in sortCols]
        generatedDF = generatedDF.drop_duplicates(subset=["SMILES"])
        if sortCols:
            generatedDF = generatedDF.sort_values(sortCols, ascending=ascending)
        generatedDF = generatedDF.reset_index(drop=True)

    outCsv = outputDir / rolloutConfig.get("output_csv", "rl_multiobjective_generated_candidates.csv")
    generatedDF.to_csv(outCsv, index=False)
    print("Saved multi-objective generated candidates to:", outCsv)
    return generatedDF

def loadGeneratedDFIfAvailable(config: dict, outputDir: Path) -> pd.DataFrame:
    """Load rollout results if present. This allows plotting after a previous run."""
    rolloutConfig = config.get("rollout", {})
    outputCsv = outputDir / rolloutConfig.get("output_csv", "rl_generated_candidates.csv")

    if outputCsv.exists():
        generatedDF = pd.read_csv(outputCsv)
        print(f"Loaded existing generated candidates for plotting: {outputCsv}")
        return generatedDF

    return pd.DataFrame()


def saveComparisonPlot(seedDF: pd.DataFrame, generatedDF: pd.DataFrame, outputDir: Path, config: Optional[dict] = None) -> None:
    """Save a potency-only comparison plot for seed and RL-generated molecules."""
    import matplotlib.pyplot as plt

    plottingConfig = (config or {}).get("plotting", {})
    plotPath = outputDir / plottingConfig.get("output_png", "rl_potency_only_generated_vs_seed_potency.png")

    if generatedDF is None or generatedDF.empty:
        print(
            "No generated molecules are available. "
            "Saving seed-only potency plot. Enable rollout.enabled=true to add RL-generated points."
        )
        generatedDF = pd.DataFrame()

    plt.figure(figsize=(7.5, 5.5))

    seedY = np.zeros(len(seedDF), dtype=float)
    plt.scatter(
        seedDF["pPotency_prediction"],
        seedY,
        label="Seed compounds",
        marker="o",
        alpha=0.85,
    )

    if not generatedDF.empty and "pPotency_prediction" in generatedDF.columns:
        generatedY = np.ones(len(generatedDF), dtype=float)
        plt.scatter(
            generatedDF["pPotency_prediction"],
            generatedY,
            label="RL-generated candidates",
            marker="x",
            alpha=0.75,
        )
        if "isParetoFront" in generatedDF.columns:
            paretoPlotDF = generatedDF.loc[generatedDF["isParetoFront"] == True]
            if not paretoPlotDF.empty:
                plt.scatter(
                    paretoPlotDF["pPotency_prediction"],
                    np.full(len(paretoPlotDF), 1.08),
                    label="Potency Pareto front",
                    marker="*",
                    s=120,
                    alpha=0.95,
                )
    elif not generatedDF.empty:
        print("Generated dataframe is missing pPotency_prediction; skipping generated points.")

    plt.yticks([0, 1], ["Seeds", "Generated"])
    plt.xlabel("Predicted pPotency")
    plt.title("Potency-only RL candidates compared with seed compounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plotPath, dpi=300)
    plt.close()

    print("Saved potency-only comparison plot to:", plotPath)


def runPotencyBeamSearchDiagnostic(
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    config: dict,
    outputDir: Path,
) -> pd.DataFrame:
    """Estimate the DORAnet/ART upper bound before PPO tuning."""
    diagnosticConfig = (
        config.get("diagnostics", {})
        .get("potency_beam_search", {})
    )
    if not bool(diagnosticConfig.get("enabled", False)):
        return pd.DataFrame()

    topSeedCount = int(diagnosticConfig.get("top_seed_count", 25))
    maxDepth = int(diagnosticConfig.get("max_depth", 2))
    beamWidth = int(diagnosticConfig.get("beam_width", 32))
    topOutputRows = int(diagnosticConfig.get("top_output_rows", 500))
    outputCsv = diagnosticConfig.get("output_csv", "diagnostic_doranet_beam_potency.csv")

    seedPoolDF = (
        seedDF.sort_values("pPotency_prediction", ascending=False)
        .head(topSeedCount)
        .reset_index(drop=True)
    )

    allRows = []
    globalSeedBest = float(seedPoolDF["pPotency_prediction"].max()) if not seedPoolDF.empty else np.nan
    print("\n================ Potency beam-search diagnostic ================")
    print(f"Top seed count={len(seedPoolDF)}, max_depth={maxDepth}, beam_width={beamWidth}")
    print(f"Best selected seed pPotency={globalSeedBest:.4f}")

    for seedIndex, seedRow in seedPoolDF.iterrows():
        seedSmiles = seedRow["canonicalSmiles"]
        seedPotency = float(seedRow["pPotency_prediction"])
        frontier = [{"SMILES": seedSmiles, "routeHistory": []}]
        seen = {seedSmiles}

        for depth in range(1, maxDepth + 1):
            productRouteMap = {}
            for state in frontier:
                actionList = doranetAdapter.enumerateActions(state["SMILES"])
                for actionObj in actionList:
                    productSmiles = canonicalizeSmiles(
                        doranetAdapter.applyAction(state["SMILES"], actionObj)
                    )
                    if productSmiles is None or productSmiles in seen:
                        continue
                    if productSmiles not in productRouteMap:
                        productRouteMap[productSmiles] = state["routeHistory"] + [actionObj]

            if not productRouteMap:
                break

            productList = list(productRouteMap.keys())
            scoredDF = combinedScorer.scoreBatch(productList)
            if scoredDF.empty:
                break

            scoredDF["seedIndex"] = seedIndex
            scoredDF["seedSmiles"] = seedSmiles
            scoredDF["seedPotency"] = seedPotency
            scoredDF["depth"] = depth
            scoredDF["deltaPotency"] = scoredDF["pPotency_prediction"] - seedPotency
            scoredDF = scoredDF.sort_values(
                ["pPotency_prediction", "deltaPotency"],
                ascending=[False, False],
            ).reset_index(drop=True)

            for _, scoredRow in scoredDF.iterrows():
                row = scoredRow.to_dict()
                routeHistory = productRouteMap.get(row["SMILES"], [])
                row.update(summarizeRoute(routeHistory))
                allRows.append(row)

            selectedSmiles = scoredDF.head(beamWidth)["SMILES"].tolist()
            frontier = [
                {"SMILES": smiles, "routeHistory": productRouteMap[smiles]}
                for smiles in selectedSmiles
                if smiles in productRouteMap
            ]
            seen.update(productList)

    diagnosticDF = pd.DataFrame(allRows)
    if not diagnosticDF.empty:
        diagnosticDF = (
            diagnosticDF.drop_duplicates(subset=["SMILES"])
            .sort_values(["pPotency_prediction", "deltaPotency"], ascending=[False, False])
            .head(topOutputRows)
            .reset_index(drop=True)
        )
        diagnosticDF.to_csv(outputDir / outputCsv, index=False)
        bestGenerated = float(diagnosticDF["pPotency_prediction"].max())
        bestDelta = float(diagnosticDF["deltaPotency"].max())
        print(f"Best beam-search generated pPotency={bestGenerated:.4f}")
        print(f"Best per-seed deltaPotency={bestDelta:.4f}")
        print("Saved potency beam-search diagnostic to:", outputDir / outputCsv)
    else:
        diagnosticDF.to_csv(outputDir / outputCsv, index=False)
        print("No DORAnet products were found during potency beam-search diagnostic.")

    return diagnosticDF


# =============================================================================
# Main execution
# =============================================================================

def runSingleGenerationWorkflow(
    config: dict,
    seedDF: pd.DataFrame,
    initialRewardConfig: dict,
    macawFeatureBuilder: MacawFeatureBuilder,
    artOracle: ArtPotencyOracle,
    outputDir: Path,
) -> pd.DataFrame:
    """Run the complete potency-only RL workflow for one DORAnet generation depth."""
    outputDir = ensureDir(outputDir)
    gen = int(config.get("doranet", {}).get("gen", 1))

    print("\n" + "=" * 78)
    print(f"Starting DORAnet generation run: gen={gen}")
    print("Output directory:", outputDir.resolve())
    print("DORAnet max_atoms:", config.get("doranet", {}).get("max_atoms"))
    print("DORAnet network directory:", config.get("doranet", {}).get("network_output_dir"))
    print("=" * 78)

    rewardConfig = finalizeRewardConfig(config, seedDF, initialRewardConfig)
    seedDF.to_csv(outputDir / "seedDF_used.csv", index=False)
    print("Seed DF shape:", seedDF.shape)
    print("Reward config:", rewardConfig)

    doranetAdapter = DoranetAdapter(config)
    combinedScorer = CombinedMoleculeScorer(
        artOracle=artOracle,
        toxicityOracle=None,
        computeQEDForOutput=False,
        scoreToxicityForOutput=False,
    )
    paretoArchive = ParetoArchive(config, seedDF=seedDF) if useParetoArchiveDuringEnv(config) else None

    precomputeSeedDoranetActions(
        seedDF=seedDF,
        doranetAdapter=doranetAdapter,
        config=config,
        outputDir=outputDir,
    )

    runPotencyBeamSearchDiagnostic(
        seedDF=seedDF,
        doranetAdapter=doranetAdapter,
        combinedScorer=combinedScorer,
        config=config,
        outputDir=outputDir,
    )

    mode = config.get("mode", {})
    rlConfig = config.get("rl", {})

    if bool(mode.get("run_smoke_tests", True)):
        runSmokeTests(
            seedDF=seedDF,
            macawFeatureBuilder=macawFeatureBuilder,
            artOracle=artOracle,
            doranetAdapter=doranetAdapter,
            combinedScorer=combinedScorer,
            rewardConfig=rewardConfig,
            rlConfig=rlConfig,
            outputDir=outputDir,
            nTest=int(mode.get("smoke_test_n", 2)),
        )

    models = {}
    useMultiObjective = bool(config.get("multi_objective", {}).get("enabled", False))

    if bool(config.get("training", {}).get("enabled", False)):
        if useMultiObjective:
            models = trainMultiObjectivePolicies(
                seedDF,
                doranetAdapter,
                combinedScorer,
                rewardConfig,
                config,
                outputDir,
            )
        else:
            model = trainPolicy(
                seedDF,
                doranetAdapter,
                combinedScorer,
                rewardConfig,
                config,
                outputDir,
                paretoArchive=paretoArchive,
            )
            models = {"single_policy": model}
    else:
        print("Training disabled by config.")

    generatedDF = pd.DataFrame()

    if bool(config.get("rollout", {}).get("enabled", False)):
        if useMultiObjective:
            if not models:
                raise ValueError("Multi-objective rollout requires training.enabled=true in this version.")
            generatedDF = rolloutMultiObjectivePolicies(
                models,
                seedDF,
                doranetAdapter,
                combinedScorer,
                rewardConfig,
                config,
                outputDir,
            )
        else:
            model = models.get("single_policy")
            if model is None:
                modelPath = config.get("rollout", {}).get("model_path")
                if not modelPath:
                    raise ValueError(
                        "Rollout is enabled, but no model was trained in this run and rollout.model_path is not set."
                    )
                from stable_baselines3 import PPO
                model = PPO.load(modelPath)
            rolloutArchive = ParetoArchive(config, seedDF=seedDF) if useParetoArchiveDuringEnv(config) else None
            generatedDF = rolloutPolicy(
                model,
                seedDF,
                doranetAdapter,
                combinedScorer,
                rewardConfig,
                config,
                outputDir,
                paretoArchive=rolloutArchive,
            )
    else:
        print("Rollout disabled by config.")
        generatedDF = loadGeneratedDFIfAvailable(config, outputDir)

    generatedDF, paretoDF = saveParetoOutputs(generatedDF, config, outputDir)

    doranetAdapter.saveTraceTables(outputDir)

    if bool(config.get("plotting", {}).get("enabled", True)):
        saveComparisonPlot(seedDF, generatedDF, outputDir, config=config)

    return generatedDF


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one DORAnet generation + ART potency-only RL workflow."
    )
    parser.add_argument("config", help="Path to YAML config file for a single DORAnet gen job.")
    args = parser.parse_args()

    startTime = time.time()

    config = loadConfig(args.config)
    addProjectPaths(config)

    genList = getDoranetGenList(config)
    if len(genList) != 1:
        raise ValueError(
            "This optimized script is intentionally single-generation. "
            "Use one config per job, for example doranet.gen: 1, doranet.gen: 2, "
            "and doranet.gen: 3, then submit the jobs separately."
        )
    gen = int(genList[0])
    config.setdefault("doranet", {})["gen"] = gen

    outputDir = ensureDir(Path(config.get("output", {}).get("output_dir", f"./_gen{gen}")))
    print("Output directory:", outputDir.resolve())
    print("Single DORAnet generation:", gen)
    print(
        "Runtime thread environment:",
        {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    )

    seedDF, initialRewardConfig = loadSeedDF(config)
    maxAtoms, seedAtomCountDF = deriveMaxAtomsFromSeeds(seedDF, config)

    networkSubdir = config.get("doranet", {}).get("network_subdir", "doranet_networks")
    config.setdefault("doranet", {})["max_atoms"] = {str(k): int(v) for k, v in maxAtoms.items()}
    config.setdefault("doranet", {})["network_output_dir"] = str(outputDir / str(networkSubdir))

    seedAtomCountDF.to_csv(outputDir / "seed_atom_counts_used_for_max_atoms.csv", index=False)
    pd.DataFrame(
        [
            {
                "gen": int(gen),
                "multiplier": float((config.get("doranet", {}) or {}).get("max_atoms_multiplier", 1.5)),
                **{f"max_atoms_{k}": v for k, v in maxAtoms.items()},
            }
        ]
    ).to_csv(outputDir / "doranet_dynamic_max_atoms.csv", index=False)

    macawFeatureBuilder = MacawFeatureBuilder(config["macaw"]["transformer_path"])

    artOracle = ArtPotencyOracle(
        artModelPath=config["art"]["model_path"],
        macawFeatureBuilder=macawFeatureBuilder,
        artOutputDir=config["art"].get("output_dir"),
        inputFeaturePrefix=config["art"].get("input_feature_prefix", "MACAW_"),
    )

    generatedDF = runSingleGenerationWorkflow(
        config=config,
        seedDF=seedDF,
        initialRewardConfig=initialRewardConfig,
        macawFeatureBuilder=macawFeatureBuilder,
        artOracle=artOracle,
        outputDir=outputDir,
    )

    nGenerated = 0 if generatedDF is None or generatedDF.empty else int(len(generatedDF))
    bestPotency = np.nan
    if generatedDF is not None and not generatedDF.empty and "pPotency_prediction" in generatedDF.columns:
        bestPotency = pd.to_numeric(generatedDF["pPotency_prediction"], errors="coerce").max()

    summaryDF = pd.DataFrame(
        [
            {
                "doranetGen": int(gen),
                "outputDir": str(outputDir),
                "nGeneratedUnique": nGenerated,
                "bestGeneratedPotency": bestPotency,
                "maxAtoms": "|".join(f"{k}:{v}" for k, v in maxAtoms.items()),
                "runtimeSeconds": time.time() - startTime,
            }
        ]
    )
    summaryPath = outputDir / "generation_run_summary.csv"
    summaryDF.to_csv(summaryPath, index=False)
    print("Saved generation run summary to:", summaryPath)
    print(f"Total runtime for DORAnet gen={gen}: {time.time() - startTime:.2f} seconds")


if __name__ == "__main__":
    main()

