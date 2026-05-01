#!/usr/bin/env python3
"""
runRL.py

YAML-driven multi-objective reinforcement-learning workflow for
reaction-constrained molecular optimization with DORAnet, MACAW, ART, and ADMET-AI.

Usage
-----
python runRL.py config.yaml

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
import cloudpickle
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Reduce numerical-library thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import yaml
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


def canonicalizeSmiles(smiles: str) -> Optional[str]:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def computeQED(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    return float(QED.qed(mol)) if mol is not None else np.nan


def smilesToMorganFP(smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    fpArray = np.zeros((nBits,), dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    if mol is None:
        return fpArray

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    DataStructs.ConvertToNumpyArray(fp, fpArray)
    return fpArray


# =============================================================================
# Seed library
# =============================================================================

def loadSeedDF(config: dict) -> Tuple[pd.DataFrame, dict]:
    seedConfig = config["seed"]
    seedCsv = seedConfig["seed_csv"]
    smilesCol = seedConfig.get("smiles_col", "SMILES")
    maxSeeds = seedConfig.get("max_seeds")

    seedDF = normalizeColumns(pd.read_csv(seedCsv))

    if smilesCol in seedDF.columns and smilesCol != "SMILES":
        seedDF = seedDF.rename(columns={smilesCol: "SMILES"})
    elif "SMILES" not in seedDF.columns and "smiles" in seedDF.columns:
        seedDF = seedDF.rename(columns={"smiles": "SMILES"})

    requiredCols = seedConfig.get(
        "required_columns",
        [
            "SMILES",
            "pPotency_prediction",
            "pPotency_std",
            "pPotency_lower_95CI",
            "pPotency_upper_95CI",
            "coreToxicityScore",
            "ToxicitySafety",
            "OverallPriority2D",
            "QED",
        ],
    )

    missingCols = [col for col in requiredCols if col not in seedDF.columns]
    if missingCols:
        raise ValueError(f"Seed CSV is missing required columns: {missingCols}")

    seedDF = seedDF[requiredCols].copy()
    seedDF["canonicalSmiles"] = seedDF["SMILES"].apply(canonicalizeSmiles)
    seedDF = (
        seedDF.dropna(subset=["canonicalSmiles"])
        .drop_duplicates("canonicalSmiles")
        .reset_index(drop=True)
    )

    if maxSeeds is not None:
        seedDF = seedDF.head(int(maxSeeds)).copy().reset_index(drop=True)

    if seedDF.empty:
        raise ValueError("No valid seed molecules remain after SMILES canonicalization.")

    priorityCol = seedConfig.get("priority_col", "OverallPriority2D")
    if priorityCol in seedDF.columns:
        seedDF["seedRank"] = seedDF[priorityCol].rank(ascending=False, method="first")
        seedDF["seedWeight"] = 1.0 / seedDF["seedRank"]
        seedDF["seedWeight"] = seedDF["seedWeight"] / seedDF["seedWeight"].sum()
    else:
        seedDF["seedWeight"] = 1.0 / len(seedDF)

    rewardConfig = config.get("reward", {}).copy()
    rewardConfig.setdefault("potencyTarget", float(seedDF["pPotency_prediction"].median()))
    rewardConfig.setdefault("toxicityTarget", float(seedDF["coreToxicityScore"].median()))
    rewardConfig.setdefault("qedTarget", float(seedDF["QED"].median()))
    rewardConfig.setdefault("potencyStdRef", float(max(seedDF["pPotency_std"].median(), 0.05)))

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


def loadOrBuildReferenceAndWeights(
    config: dict,
    admetOracle: AdmetOracle,
    outputDir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    toxConfig = config.get("toxicity", {})
    referenceScoredCsv = toxConfig.get("reference_scored_csv")
    referenceInputCsv = toxConfig.get("reference_input_csv")
    referenceSmilesCol = toxConfig.get("reference_smiles_col", "SMILES")
    maxReferenceRows = toxConfig.get("max_reference_rows")

    if referenceScoredCsv:
        refScoreDF = normalizeColumns(pd.read_csv(referenceScoredCsv))
        if referenceSmilesCol in refScoreDF.columns and referenceSmilesCol != "SMILES":
            refScoreDF = refScoreDF.rename(columns={referenceSmilesCol: "SMILES"})
        elif "SMILES" not in refScoreDF.columns and "smiles" in refScoreDF.columns:
            refScoreDF = refScoreDF.rename(columns={"smiles": "SMILES"})

    elif referenceInputCsv:
        refInputDF = normalizeColumns(pd.read_csv(referenceInputCsv))

        if referenceSmilesCol in refInputDF.columns and referenceSmilesCol != "SMILES":
            refInputDF = refInputDF.rename(columns={referenceSmilesCol: "SMILES"})
        elif "SMILES" not in refInputDF.columns and "smiles" in refInputDF.columns:
            refInputDF = refInputDF.rename(columns={"smiles": "SMILES"})

        if "SMILES" not in refInputDF.columns:
            raise ValueError(
                "Reference input file must contain a SMILES column. "
                f"Available columns: {list(refInputDF.columns)}"
            )

        if maxReferenceRows is not None:
            refInputDF = refInputDF.head(int(maxReferenceRows)).copy()

        smilesList = refInputDF["SMILES"].dropna().astype(str).tolist()
        predDF = admetOracle.predictBatch(smilesList)
        refScoreDF = refInputDF.merge(predDF, on="SMILES", how="left")

        refOut = outputDir / "reference_admet_scored.csv"
        refScoreDF.to_csv(refOut, index=False)
        print(f"Saved reference ADMET predictions to: {refOut}")

    else:
        raise ValueError("Set toxicity.reference_scored_csv or toxicity.reference_input_csv in config.")

    endpointMetaPath = toxConfig.get("endpoint_meta_csv")
    endpointMetaDF = normalizeColumns(pd.read_csv(endpointMetaPath)) if endpointMetaPath else defaultEndpointMetaDF()

    weightCsv = toxConfig.get("weight_csv")
    if weightCsv:
        weightDF = normalizeColumns(pd.read_csv(weightCsv))
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
    """Generate reaction-constrained candidate actions with DORAnet."""

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
        self.maxAtoms = dconf.get("max_atoms", {"C": 41, "N": 9, "O": 12, "S": 0})
        self.gen = int(dconf.get("gen", 1))
        self.maxActions = int(dconf.get("max_actions", 16))
        self.jobPrefix = dconf.get("job_prefix", "rl_tmp")
        self.cache: Dict[str, List[DoranetAction]] = {}

    def enumerateActions(self, smiles: str) -> List[DoranetAction]:
        canonical = canonicalizeSmiles(smiles)
        if canonical is None:
            return []

        if canonical in self.cache:
            return self.cache[canonical]

        jobName = f"{self.jobPrefix}_{abs(hash(canonical)) % 10**10}"
        starters = {canonical}

        try:
            forwardNetwork = self.enzymatic.generate_network(
                job_name=jobName,
                starters=starters,
                gen=self.gen,
                max_atoms=self.maxAtoms,
                direction="forward",
                ruleset=self.ruleset,
            )
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

        actions = [
            DoranetAction(
                actionIndex=idx,
                productSmiles=product,
                sourceSmiles=canonical,
                generation=self.gen,
                metadata={"ruleset": self.ruleset},
            )
            for idx, product in enumerate(productSmilesList)
        ]

        self.cache[canonical] = actions
        return actions

    def applyAction(self, smiles: str, actionObj: DoranetAction) -> str:
        return actionObj.productSmiles


# =============================================================================
# Combined scoring and RL
# =============================================================================

class CombinedMoleculeScorer:
    """Score molecules with ART potency, ADMET toxicity, and QED."""

    def __init__(self, artOracle: ArtPotencyOracle, toxicityOracle: ToxicityScoringOracle):
        self.artOracle = artOracle
        self.toxicityOracle = toxicityOracle
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
            artDF = self.artOracle.predictBatch(uncachedList)
            toxDF = self.toxicityOracle.scoreBatch(uncachedList)

            keepToxCols = [
                col for col in toxDF.columns
                if col not in artDF.columns or col == "SMILES"
            ]

            scoredDF = artDF.merge(toxDF[keepToxCols], on="SMILES", how="left")
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
    """Return vector-valued reward components for multi-objective PPO.

    Stable-Baselines3 PPO consumes a scalar reward, so this script trains several
    PPO policies with different scalarizations of this common reward vector.
    """
    potencyVal = scoreDict.get("pPotency_prediction", np.nan)
    potencyStd = scoreDict.get("pPotency_std", np.nan)
    toxicityVal = scoreDict.get("coreToxicityScore", np.nan)
    qedVal = scoreDict.get("QED", np.nan)

    potencyReward = sigmoidScaled(
        potencyVal,
        float(rewardConfig["potencyTarget"]),
        float(rewardConfig.get("potencyScale", 0.25)),
    )
    toxicityReward = sigmoidScaled(
        float(rewardConfig["toxicityTarget"]) - toxicityVal,
        0.0,
        float(rewardConfig.get("toxicityScale", 0.10)),
    )
    qedReward = sigmoidScaled(
        qedVal,
        float(rewardConfig["qedTarget"]),
        float(rewardConfig.get("qedScale", 0.05)),
    )
    routeReward = max(
        0.0,
        1.0 - float(rewardConfig.get("routeDepthPenalty", 0.15)) * routeDepth,
    )
    uncertaintyPenalty = (
        1.0
        if pd.isna(potencyStd)
        else min(float(potencyStd) / float(rewardConfig["potencyStdRef"]), 2.0)
    )

    # Positive values are desirable. Uncertainty is represented as a negative
    # component so a positive preference weight penalizes uncertain ART outputs.
    return {
        "potency": float(potencyReward),
        "toxicity": float(toxicityReward),
        "qed": float(qedReward),
        "route": float(routeReward),
        "uncertainty": float(-uncertaintyPenalty),
    }


def getDefaultPreferenceWeights(rewardConfig: dict) -> dict:
    """Map legacy scalar reward weights into multi-objective preferences."""
    legacy = rewardConfig.get("weights", {})
    return {
        "potency": float(legacy.get("potency", 0.40)),
        "toxicity": float(legacy.get("toxicity", 0.40)),
        "qed": float(legacy.get("qed", 0.10)),
        "route": float(legacy.get("route", 0.02)),
        "uncertainty": float(legacy.get("uncertainty_penalty", 0.08)),
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
    "coreToxicityScore": "minimize",
    "QED": "maximize",
    "pPotency_std": "minimize",
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
    according to the scientific direction: higher potency, lower toxicity, higher
    QED, and lower ART uncertainty.
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
        if col in {"coreToxicityScore", "pPotency_std"}:
            pct = 1.0 - pct
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

    sortCols = [col for col in ["pPotency_prediction", "coreToxicityScore", "QED", "pPotency_std"] if col in paretoDF.columns]
    if sortCols:
        ascending = [False if col in ["pPotency_prediction", "QED"] else True for col in sortCols]
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
    fpVec = smilesToMorganFP(smiles, nBits=nBits)
    scalarVec = np.array(
        [
            float(stepIndex) / 5.0,
            float(seedRow["pPotency_prediction"]) / 10.0,
            float(seedRow["coreToxicityScore"]),
            float(seedRow["QED"]),
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
                shape=(self.fpBits + 4,),
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
                "seedToxicity": float(self.seedRow["coreToxicityScore"]),
            }
            return obs, info

        def step(self, actionId):
            actionId = int(actionId)

            if actionId == 0:
                return self._terminalStep()

            chosenIndex = actionId - 1

            if chosenIndex >= len(self.actionList):
                obs = buildObservation(self.currentSmiles, self.stepIndex, self.seedRow, self.fpBits)
                return obs, -0.10, False, False, {"invalidAction": True}

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
# Execution helpers
# =============================================================================

def runSmokeTests(
    seedDF: pd.DataFrame,
    macawFeatureBuilder: MacawFeatureBuilder,
    artOracle: ArtPotencyOracle,
    admetOracle: AdmetOracle,
    toxicityOracle: ToxicityScoringOracle,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    rlConfig: dict,
    outputDir: Path,
    nTest: int = 2,
) -> None:
    print("\n================ Smoke tests ================")

    smilesTest = seedDF["canonicalSmiles"].head(nTest).tolist()

    macawDF = macawFeatureBuilder.transformSmilesList(smilesTest)
    artDF = artOracle.predictBatch(smilesTest)
    admetDF = admetOracle.predictBatch(smilesTest)
    toxDF = toxicityOracle.scoreBatch(smilesTest)
    combinedDF = combinedScorer.scoreBatch(smilesTest)

    macawDF.to_csv(outputDir / "smoke_macaw_features.csv", index=False)
    artDF.to_csv(outputDir / "smoke_art_predictions.csv", index=False)
    admetDF.to_csv(outputDir / "smoke_admet_predictions.csv", index=False)
    toxDF.to_csv(outputDir / "smoke_toxicity_scored.csv", index=False)
    combinedDF.to_csv(outputDir / "smoke_combined_scores.csv", index=False)

    actions = doranetAdapter.enumerateActions(smilesTest[0])
    print(f"MACAW features       : {macawDF.shape}")
    print(f"ART predictions      : {artDF.shape}")
    print(f"ADMET predictions    : {admetDF.shape}")
    print(f"Toxicity scored      : {toxDF.shape}")
    print(f"Combined scores      : {combinedDF.shape}")
    print(f"DORAnet first actions: {len(actions)}")

    runManualEnvTest(seedDF, doranetAdapter, combinedScorer, rewardConfig, rlConfig)
    print("Smoke tests complete. Output written to:", outputDir)


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
            row["routeLength"] = len(stepInfo.get("routeHistory", []))
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

        paretoArchive = ParetoArchive(config, seedDF=seedDF)
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
        paretoArchive = ParetoArchive(config, seedDF=seedDF)
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
                row["routeLength"] = len(stepInfo.get("routeHistory", []))
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
        generatedDF = generatedDF.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

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


def saveComparisonPlot(seedDF: pd.DataFrame, generatedDF: pd.DataFrame, outputDir: Path) -> None:
    """Save potency-toxicity comparison plot for seed and RL-generated molecules."""
    import matplotlib.pyplot as plt

    plotPath = outputDir / "rl_generated_vs_seed_potency_toxicity.png"

    if generatedDF is None or generatedDF.empty:
        print(
            "No generated molecules are available. "
            "Saving seed-only plot. Enable rollout.enabled=true to add RL-generated points."
        )
        generatedDF = pd.DataFrame()

    plt.figure(figsize=(7.5, 5.5))

    plt.scatter(
        seedDF["pPotency_prediction"],
        seedDF["coreToxicityScore"],
        label="Seed compounds",
        marker="o",
        alpha=0.85,
    )

    if not generatedDF.empty:
        requiredCols = {"pPotency_prediction", "coreToxicityScore"}
        if requiredCols.issubset(generatedDF.columns):
            plt.scatter(
                generatedDF["pPotency_prediction"],
                generatedDF["coreToxicityScore"],
                label="RL-generated candidates",
                marker="x",
                alpha=0.75,
            )
            if "isParetoFront" in generatedDF.columns:
                paretoPlotDF = generatedDF.loc[generatedDF["isParetoFront"] == True]
                if not paretoPlotDF.empty:
                    plt.scatter(
                        paretoPlotDF["pPotency_prediction"],
                        paretoPlotDF["coreToxicityScore"],
                        label="RL Pareto front",
                        marker="*",
                        s=120,
                        alpha=0.95,
                    )
        else:
            print(f"Generated dataframe is missing required plot columns: {requiredCols}")

    plt.xlabel("Predicted pPotency")
    plt.ylabel("Core Toxicity Score")
    plt.title("RL-generated candidates compared with seed compounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plotPath, dpi=300)
    plt.close()

    print("Saved comparison plot to:", plotPath)


# =============================================================================
# Main execution
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DORAnet + ART + ADMET-AI reinforcement-learning workflow."
    )
    parser.add_argument("config", help="Path to YAML config file.")
    args = parser.parse_args()

    startTime = time.time()

    config = loadConfig(args.config)
    addProjectPaths(config)

    outputDir = ensureDir(config.get("output", {}).get("output_dir", "RL_outputs"))
    print("Output directory:", outputDir.resolve())

    seedDF, rewardConfig = loadSeedDF(config)
    seedDF.to_csv(outputDir / "seedDF_used.csv", index=False)

    print("Seed DF shape:", seedDF.shape)
    print("Reward config:", rewardConfig)

    macawFeatureBuilder = MacawFeatureBuilder(config["macaw"]["transformer_path"])

    artOracle = ArtPotencyOracle(
        artModelPath=config["art"]["model_path"],
        macawFeatureBuilder=macawFeatureBuilder,
        artOutputDir=config["art"].get("output_dir"),
        inputFeaturePrefix=config["art"].get("input_feature_prefix", "MACAW_"),
    )

    admetOracle = AdmetOracle()
    refScoreDF, weightDF = loadOrBuildReferenceAndWeights(config, admetOracle, outputDir)
    refScoreDF.to_csv(outputDir / "reference_score_df_used.csv", index=False)
    weightDF.to_csv(outputDir / "weight_df_used.csv", index=False)

    toxicityOracle = ToxicityScoringOracle(admetOracle, refScoreDF, weightDF)
    doranetAdapter = DoranetAdapter(config)
    combinedScorer = CombinedMoleculeScorer(artOracle, toxicityOracle)
    paretoArchive = ParetoArchive(config, seedDF=seedDF)

    mode = config.get("mode", {})
    rlConfig = config.get("rl", {})

    if bool(mode.get("run_smoke_tests", True)):
        runSmokeTests(
            seedDF=seedDF,
            macawFeatureBuilder=macawFeatureBuilder,
            artOracle=artOracle,
            admetOracle=admetOracle,
            toxicityOracle=toxicityOracle,
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
            models = trainMultiObjectivePolicies(seedDF, doranetAdapter, combinedScorer, rewardConfig, config, outputDir)
        else:
            model = trainPolicy(seedDF, doranetAdapter, combinedScorer, rewardConfig, config, outputDir, paretoArchive=paretoArchive)
            models = {"single_policy": model}
    else:
        print("Training disabled by config.")

    generatedDF = pd.DataFrame()

    if bool(config.get("rollout", {}).get("enabled", False)):
        if useMultiObjective:
            if not models:
                raise ValueError("Multi-objective rollout requires training.enabled=true in this version.")
            generatedDF = rolloutMultiObjectivePolicies(models, seedDF, doranetAdapter, combinedScorer, rewardConfig, config, outputDir)
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
            rolloutArchive = ParetoArchive(config, seedDF=seedDF)
            generatedDF = rolloutPolicy(model, seedDF, doranetAdapter, combinedScorer, rewardConfig, config, outputDir, paretoArchive=rolloutArchive)
    else:
        print("Rollout disabled by config.")
        generatedDF = loadGeneratedDFIfAvailable(config, outputDir)

    generatedDF, paretoDF = saveParetoOutputs(generatedDF, config, outputDir)

    if bool(config.get("plotting", {}).get("enabled", True)):
        saveComparisonPlot(seedDF, generatedDF, outputDir)

    print(f"Total runtime: {time.time() - startTime:.2f} seconds")


if __name__ == "__main__":
    main()
