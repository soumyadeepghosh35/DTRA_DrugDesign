#!/usr/bin/env python3
"""
Integrated DTRA antiviral drug-design pipeline.

Run:
    python runIntegratedDrugDesignPipeline.py integrated_drug_design_config.yaml

Workflow:
    training data -> MACAW embeddings -> ART model -> DORAnet molecules
    -> MACAW embeddings for candidates -> ART potency prediction
    -> ADMET-AI prediction -> FDA-reference scoring -> Pareto final list
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cloudpickle
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def loadConfig(configPath: str) -> Dict[str, Any]:
    with open(configPath, "r") as fileObj:
        configDict = yaml.safe_load(fileObj)
    if configDict is None:
        raise ValueError("Config file is empty.")
    return configDict


def makeDir(pathValue: str | Path) -> Path:
    pathObj = Path(pathValue).expanduser()
    pathObj.mkdir(parents=True, exist_ok=True)
    return pathObj


def resolvePath(pathValue: Optional[str], baseDir: Optional[str | Path] = None) -> Optional[Path]:
    if pathValue in [None, ""]:
        return None
    pathObj = Path(str(pathValue)).expanduser()
    if not pathObj.is_absolute() and baseDir is not None:
        pathObj = Path(baseDir).expanduser() / pathObj
    return pathObj


def saveCsv(DF: pd.DataFrame, pathValue: str | Path) -> None:
    pathObj = Path(pathValue).expanduser()
    pathObj.parent.mkdir(parents=True, exist_ok=True)
    DF.to_csv(pathObj, index=False)
    print(f"Saved: {pathObj}")


def canonicalizeSmiles(smilesValue: Any, includeStereo: bool = True) -> Optional[str]:
    if pd.isna(smilesValue):
        return None
    smilesText = str(smilesValue).strip()
    if smilesText == "":
        return None
    mol = Chem.MolFromSmiles(smilesText)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=includeStereo)


def normalizeSmilesColumn(DF: pd.DataFrame, smilesCol: str, outputCol: str = "SMILES") -> pd.DataFrame:
    if smilesCol not in DF.columns:
        raise KeyError(f"Missing SMILES column: {smilesCol}")
    outDF = DF.copy()
    outDF[outputCol] = outDF[smilesCol].apply(canonicalizeSmiles)
    outDF = outDF.dropna(subset=[outputCol]).drop_duplicates(subset=[outputCol], keep="first")
    return outDF.reset_index(drop=True)


def computeRdkitProperties(DF: pd.DataFrame, smilesCol: str = "SMILES") -> pd.DataFrame:
    outDF = DF.copy()

    values = {
        "QED": [],
        "molecular_weight": [],
        "logP": [],
        "hydrogen_bond_acceptors": [],
        "hydrogen_bond_donors": [],
        "tpsa": [],
        "rotatable_bonds": [],
    }

    for smi in outDF[smilesCol].tolist():
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            for key in values:
                values[key].append(np.nan)
            continue

        values["QED"].append(float(QED.qed(mol)))
        values["molecular_weight"].append(float(Descriptors.MolWt(mol)))
        values["logP"].append(float(Descriptors.MolLogP(mol)))
        values["hydrogen_bond_acceptors"].append(float(Descriptors.NumHAcceptors(mol)))
        values["hydrogen_bond_donors"].append(float(Descriptors.NumHDonors(mol)))
        values["tpsa"].append(float(Descriptors.TPSA(mol)))
        values["rotatable_bonds"].append(float(Descriptors.NumRotatableBonds(mol)))

    for colName, colValues in values.items():
        if colName not in outDF.columns:
            outDF[colName] = colValues

    return outDF


def splitIntoBatches(items: Sequence[Any], batchSize: int) -> Iterable[Tuple[int, Sequence[Any]]]:
    for startIdx in range(0, len(items), batchSize):
        yield startIdx, items[startIdx:startIdx + batchSize]


# -----------------------------------------------------------------------------
# Step 1: Training data
# -----------------------------------------------------------------------------

def loadTrainingData(configDict: Dict[str, Any]) -> pd.DataFrame:
    cfg = configDict["trainingData"]
    inputPath = resolvePath(cfg["inputCsv"])

    if inputPath is None or not inputPath.exists():
        raise FileNotFoundError(f"Training CSV not found: {inputPath}")

    DF = pd.read_csv(inputPath)

    smilesCol = cfg.get("smilesCol", "Smiles")
    responseCol = cfg.get("responseCol", "pPotency")

    keepCols = [smilesCol, responseCol]
    for col in [cfg.get("compoundIdCol"), cfg.get("strainCol")]:
        if col and col in DF.columns:
            keepCols.append(col)

    DF = DF[keepCols].copy()
    DF = normalizeSmilesColumn(DF, smilesCol=smilesCol, outputCol="SMILES")
    DF[responseCol] = pd.to_numeric(DF[responseCol], errors="coerce")

    minPotency = float(cfg.get("minPotency", 3.0))
    maxPotency = float(cfg.get("maxPotency", 12.0))

    DF = DF[
        DF[responseCol].notna()
        & DF[responseCol].between(minPotency, maxPotency, inclusive="both")
    ].copy()

    DF = DF.rename(columns={responseCol: "pPotency"})

    limitRows = cfg.get("limitRows")
    if limitRows is not None:
        DF = DF.sample(
            n=min(int(limitRows), len(DF)),
            random_state=int(configDict.get("randomSeed", 42)),
        ).reset_index(drop=True)

    if "ID" not in DF.columns:
        DF.insert(0, "ID", range(1, len(DF) + 1))

    print(f"Training dataframe shape: {DF.shape}")
    return DF.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Step 2: MACAW embeddings
# -----------------------------------------------------------------------------

def loadMacawClass(configDict: Dict[str, Any]):
    cfg = configDict["macaw"]

    for extraPath in cfg.get("pythonPaths", []):
        if extraPath and str(extraPath) not in sys.path:
            sys.path.insert(0, str(extraPath))

    try:
        from macaw import MACAW
    except Exception as exc:
        raise ImportError(
            "Could not import MACAW. Add the package path to macaw.pythonPaths."
        ) from exc

    return MACAW


def fitOrLoadMacawTransformer(
    trainingDF: pd.DataFrame,
    configDict: Dict[str, Any],
    outputDir: Path,
):
    cfg = configDict["macaw"]
    transformerPath = resolvePath(cfg.get("transformerPath"), outputDir)

    if bool(cfg.get("useExistingTransformer", False)) and transformerPath and transformerPath.exists():
        with open(transformerPath, "rb") as fileObj:
            macawModel = cloudpickle.load(fileObj)
        print(f"Loaded MACAW transformer: {transformerPath}")
        return macawModel

    MACAW = loadMacawClass(configDict)

    macawModel = MACAW(
        type_fp=cfg.get("typeFp", "atompairs"),
        metric=cfg.get("metric", "sokal"),
        n_components=int(cfg.get("nComponents", 60)),
        n_landmarks=int(cfg.get("nLandmarks", 1000)),
        random_state=int(configDict.get("randomSeed", 42)),
    )

    macawModel.fit(trainingDF["SMILES"], trainingDF["pPotency"])

    if transformerPath is not None:
        transformerPath.parent.mkdir(parents=True, exist_ok=True)
        with open(transformerPath, "wb") as fileObj:
            cloudpickle.dump(macawModel, fileObj)
        print(f"Saved MACAW transformer: {transformerPath}")

    return macawModel


def addMacawEmbeddings(
    DF: pd.DataFrame,
    macawModel: Any,
    smilesCol: str = "SMILES",
    prefix: str = "MACAW_",
) -> Tuple[pd.DataFrame, List[str]]:

    print(f"Generating MACAW embeddings for {len(DF)} molecules")

    embeddingArray = np.asarray(macawModel.transform(DF[smilesCol]), dtype=float)
    featureCols = [f"{prefix}{idx + 1}" for idx in range(embeddingArray.shape[1])]

    embeddingDF = pd.DataFrame(embeddingArray, columns=featureCols, index=DF.index)
    outDF = pd.concat(
        [DF.reset_index(drop=True), embeddingDF.reset_index(drop=True)],
        axis=1,
    )

    print(f"MACAW embedding shape: {embeddingArray.shape}")
    return outDF, featureCols


# -----------------------------------------------------------------------------
# Step 3: ART model and pPotency prediction
# -----------------------------------------------------------------------------

def buildOrLoadArtModel(
    trainingWithMacawDF: pd.DataFrame,
    featureCols: List[str],
    configDict: Dict[str, Any],
    outputDir: Path,
):
    cfg = configDict["art"]
    modelPath = resolvePath(cfg.get("modelFile", "art.cpkl"), outputDir)

    for extraPath in cfg.get("pythonPaths", []):
        if extraPath and str(extraPath) not in sys.path:
            sys.path.insert(0, str(extraPath))

    if bool(cfg.get("useExistingModel", False)) and modelPath and modelPath.exists():
        with open(modelPath, "rb") as fileObj:
            artModel = cloudpickle.load(fileObj)
        print(f"Loaded ART model: {modelPath}")
        return artModel

    try:
        from art.core import RecommendationEngine
        import art.utility as artUtils
    except Exception as exc:
        raise ImportError(
            "Could not import ART. Add AutomatedRecommendationTool path to art.pythonPaths."
        ) from exc

    artReadyCsv = resolvePath(cfg.get("artReadyCsv", "training_wMACAW_ARTready.csv"), outputDir)
    responseCols = ["pPotency"]

    features = trainingWithMacawDF[featureCols].to_numpy(dtype=float)
    responses = trainingWithMacawDF[responseCols].to_numpy(dtype=float)

    artReadyCsv.parent.mkdir(parents=True, exist_ok=True)
    artUtils.save_edd_csv(features, responses, featureCols, str(artReadyCsv), responseCols)

    artReadyDF = pd.read_csv(artReadyCsv)
    print(f"Saved ART-ready CSV: {artReadyCsv}")

    artModel = RecommendationEngine(
        df=artReadyDF,
        input_vars=featureCols,
        response_vars=responseCols,
        objective=cfg.get("objective", "maximize"),
        threshold=float(cfg.get("threshold", 0.2)),
        alpha=float(cfg.get("alpha", 0.25)),
        num_recommendations=int(cfg.get("numRecommendations", 10)),
        max_mcmc_cores=int(cfg.get("maxMcmcCores", 16)),
        seed=int(configDict.get("randomSeed", 42)),
        output_dir=str(outputDir),
        recommend=bool(cfg.get("generateRecommendations", False)),
        cross_val=bool(cfg.get("runCrossValidation", True)),
        num_tpot_models=int(cfg.get("numTpotModels", 2)),
    )

    if bool(cfg.get("saveModel", True)):
        modelPath.parent.mkdir(parents=True, exist_ok=True)
        with open(modelPath, "wb") as fileObj:
            cloudpickle.dump(artModel, fileObj)
        print(f"Saved ART model: {modelPath}")

    return artModel


def predictPotencyWithArt(
    artModel: Any,
    candidateWithMacawDF: pd.DataFrame,
    featureCols: List[str],
    configDict: Dict[str, Any],
) -> pd.DataFrame:
    """
    Predict pPotency for a new candidate batch using the trained ART model.

    This follows the Step5 ART pattern:
      - use MACAW_ columns as ART inputs,
      - use pPotency as the ART response,
      - call meanVals, stdVals = artModel.post_pred_stats(chunkX),
      - write pPotency_prediction and pPotency_std.
    """

    cfg = configDict["art"]

    predictionMethod = cfg.get("predictionMethod", "post_pred_stats")
    predictionChunkSize = int(cfg.get("predictionChunkSize", 100000))

    if predictionMethod != "post_pred_stats":
        raise ValueError("This updated pipeline expects art.predictionMethod: post_pred_stats")

    if not hasattr(artModel, "post_pred_stats"):
        raise AttributeError("Loaded ART model does not expose post_pred_stats(...).")

    missingFeatures = [col for col in featureCols if col not in candidateWithMacawDF.columns]
    if missingFeatures:
        raise KeyError(
            f"Candidate dataframe is missing MACAW feature columns: {missingFeatures[:10]}"
        )

    predictionChunks = []
    totalRows = len(candidateWithMacawDF)

    for startIdx in range(0, totalRows, predictionChunkSize):
        endIdx = min(startIdx + predictionChunkSize, totalRows)

        chunkX = candidateWithMacawDF.iloc[startIdx:endIdx][featureCols].to_numpy(dtype=float)

        meanVals, stdVals = artModel.post_pred_stats(chunkX)

        batchPredictionDF = pd.DataFrame({
            "pPotency_prediction": np.asarray(meanVals, dtype=float).reshape(-1),
            "pPotency_std": np.asarray(stdVals, dtype=float).reshape(-1),
        })

        batchPredictionDF["pPotency_lower_95CI"] = (
            batchPredictionDF["pPotency_prediction"]
            - 1.96 * batchPredictionDF["pPotency_std"]
        )

        batchPredictionDF["pPotency_upper_95CI"] = (
            batchPredictionDF["pPotency_prediction"]
            + 1.96 * batchPredictionDF["pPotency_std"]
        )

        predictionChunks.append(batchPredictionDF)
        print(f"ART potency predicted: {endIdx}/{totalRows}")

    predictionDF = pd.concat(predictionChunks, axis=0, ignore_index=True)

    outDF = candidateWithMacawDF.drop(
        columns=[col for col in candidateWithMacawDF.columns if col.startswith("MACAW_")],
        errors="ignore",
    ).reset_index(drop=True)

    outDF = pd.concat([outDF, predictionDF], axis=1)

    outDF["IC50(M)_prediction"] = 10.0 ** (-outDF["pPotency_prediction"])
    outDF["IC50(M)_lower_95CI"] = 10.0 ** (-outDF["pPotency_upper_95CI"])
    outDF["IC50(M)_upper_95CI"] = 10.0 ** (-outDF["pPotency_lower_95CI"])

    return outDF


# -----------------------------------------------------------------------------
# Step 4: DORAnet candidates
# -----------------------------------------------------------------------------

def runDoranetGeneration(configDict: Dict[str, Any], outputDir: Path) -> pd.DataFrame:
    cfg = configDict["doranet"]

    if cfg.get("pythonPath") and str(cfg["pythonPath"]) not in sys.path:
        sys.path.insert(0, str(cfg["pythonPath"]))

    import doranet.modules.enzymatic as enzymatic
    import doranet.modules.post_processing as post_processing

    allRows = []
    helpers = set(cfg.get("helpers", []))
    gen = int(cfg.get("gen", 3))

    for starterSet in cfg.get("starterSets", []):
        starterName = starterSet["name"]
        starters = set(starterSet["starters"])
        jobName = f"{configDict['jobName']}_{starterName}_gen{gen}"

        print(f"Running DORAnet: {jobName}")

        network = enzymatic.generate_network(
            job_name=jobName,
            starters=starters,
            gen=gen,
            max_atoms=cfg.get("maxAtoms", {"C": 41, "N": 9, "O": 12, "S": 0}),
            direction=cfg.get("direction", "forward"),
            ruleset=cfg.get("ruleset", "JN3604IMT"),
        )

        smilesList = list(starters) + [mol.uid for mol in network.mols if mol.uid not in starters]
        targets = set(smilesList) - starters - helpers

        if bool(cfg.get("runPostProcessing", False)) and targets:
            post_processing.one_step(
                networks={network},
                total_generations=gen,
                starters=starters,
                helpers=helpers,
                target=targets,
                job_name=jobName,
            )

        for smi in smilesList:
            mol = Chem.MolFromSmiles(smi)

            allRows.append({
                "SMILES": smi,
                "DORAnetStarterSet": starterName,
                "Is_Starter": smi in starters,
                "MolFormula": rdMolDescriptors.CalcMolFormula(mol) if mol else np.nan,
                "MolWeight": rdMolDescriptors.CalcExactMolWt(mol) if mol else np.nan,
                "NumHeavyAtoms": mol.GetNumHeavyAtoms() if mol else np.nan,
            })

    outDF = pd.DataFrame(allRows)
    outDF = normalizeSmilesColumn(outDF, smilesCol="SMILES", outputCol="SMILES")

    saveCsv(outDF, outputDir / f"{configDict['jobName']}_doranet_molecules.csv")
    return outDF


def loadOrGenerateCandidates(configDict: Dict[str, Any], outputDir: Path) -> pd.DataFrame:
    cfg = configDict["doranet"]

    if bool(cfg.get("enabled", False)):
        candidateDF = runDoranetGeneration(configDict, outputDir)
    else:
        candidatePath = resolvePath(cfg.get("generatedMoleculesCsv"))

        if candidatePath is None or not candidatePath.exists():
            raise FileNotFoundError(f"DORAnet generatedMoleculesCsv not found: {candidatePath}")

        candidateDF = pd.read_csv(candidatePath)
        candidateDF = normalizeSmilesColumn(
            candidateDF,
            smilesCol=cfg.get("smilesCol", "SMILES"),
            outputCol="SMILES",
        )

    limitRows = configDict.get("candidateFiltering", {}).get("limitGeneratedRows")
    if limitRows is not None:
        candidateDF = candidateDF.head(int(limitRows)).copy()

    candidateDF = computeRdkitProperties(candidateDF, smilesCol="SMILES")

    print(f"Candidate dataframe shape: {candidateDF.shape}")
    return candidateDF


def applyCandidateFilters(
    candidateDF: pd.DataFrame,
    trainingDF: pd.DataFrame,
    configDict: Dict[str, Any],
) -> pd.DataFrame:

    cfg = configDict.get("candidateFiltering", {})
    outDF = candidateDF.copy()

    if bool(cfg.get("removeTrainingSmiles", True)):
        trainingSmiles = set(trainingDF["SMILES"].dropna().astype(str))

        beforeCount = len(outDF)
        outDF = outDF[~outDF["SMILES"].astype(str).isin(trainingSmiles)].copy()

        print(f"Removed training-set overlap: {beforeCount - len(outDF)} molecules")

    feasibilityPath = resolvePath(cfg.get("doraXgbFeasibilityCsv"))

    if feasibilityPath is not None and feasibilityPath.exists():
        feasibilityDF = pd.read_csv(feasibilityPath)

        feasibilityDF = normalizeSmilesColumn(
            feasibilityDF,
            smilesCol=cfg.get("feasibilitySmilesCol", "SMILES"),
            outputCol="SMILES",
        )

        feasibilityCol = cfg.get("feasibilityColumn", "feasibilityScore_rule1")
        threshold = float(cfg.get("feasibilityThreshold", 1.0))

        keepCols = ["SMILES", feasibilityCol]
        outDF = outDF.merge(feasibilityDF[keepCols], on="SMILES", how="left")

        beforeCount = len(outDF)
        outDF = outDF[pd.to_numeric(outDF[feasibilityCol], errors="coerce") >= threshold].copy()

        print(f"Applied DORA-XGB feasibility filter: {beforeCount} -> {len(outDF)}")

    return outDF.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Step 5: ADMET-AI prediction
# -----------------------------------------------------------------------------

def loadAdmetModel():
    from admet_ai import ADMETModel
    return ADMETModel()


def predictAdmetWithCheckpoint(
    potencyDF: pd.DataFrame,
    configDict: Dict[str, Any],
    outputDir: Path,
) -> pd.DataFrame:

    cfg = configDict["admet"]

    existingPath = resolvePath(cfg.get("existingAdmetCsv"), outputDir)

    if bool(cfg.get("reuseExistingAdmetCsv", False)) and existingPath and existingPath.exists():
        admetMergedDF = pd.read_csv(existingPath)
        print(f"Loaded existing ADMET file: {existingPath}")
        return admetMergedDF

    checkpointPath = resolvePath(cfg.get("checkpointCsv", "generated_admet_ai_checkpoint.csv"), outputDir)
    outputPath = resolvePath(cfg.get("outputCsv", "generated_wPotency_wADMET.csv"), outputDir)

    batchSize = int(cfg.get("batchSize", 500))
    smilesList = potencyDF["SMILES"].dropna().drop_duplicates().tolist()

    processedSmiles = set()

    if bool(cfg.get("resumeFromCheckpoint", True)) and checkpointPath.exists():
        checkpointDF = pd.read_csv(checkpointPath)

        if "SMILES" in checkpointDF.columns:
            processedSmiles = set(checkpointDF["SMILES"].dropna().astype(str))
            print(f"Loaded ADMET checkpoint with {len(processedSmiles)} SMILES")

    remainingSmiles = [smi for smi in smilesList if smi not in processedSmiles]

    print(f"Remaining ADMET-AI predictions: {len(remainingSmiles)}")

    if remainingSmiles:
        admetModel = loadAdmetModel()

        checkpointPath.parent.mkdir(parents=True, exist_ok=True)
        writeHeader = not checkpointPath.exists() or checkpointPath.stat().st_size == 0

        for batchIdx, (_, smilesBatch) in enumerate(splitIntoBatches(remainingSmiles, batchSize), start=1):
            startTime = time.time()

            predictionDF = admetModel.predict(smiles=list(smilesBatch))

            if not isinstance(predictionDF, pd.DataFrame):
                predictionDF = pd.DataFrame(predictionDF)

            batchDF = pd.concat(
                [
                    pd.DataFrame({"SMILES": list(smilesBatch)}).reset_index(drop=True),
                    predictionDF.reset_index(drop=True),
                ],
                axis=1,
            )

            batchDF.to_csv(checkpointPath, mode="a", header=writeHeader, index=False)
            writeHeader = False

            print(
                f"ADMET batch {batchIdx}: {len(smilesBatch)} molecules "
                f"in {time.time() - startTime:.2f} s"
            )

    finalAdmetDF = (
        pd.read_csv(checkpointPath)
        .drop_duplicates(subset="SMILES", keep="last")
        .reset_index(drop=True)
    )

    finalAdmetDF.to_csv(checkpointPath, index=False)

    mergedDF = potencyDF.merge(finalAdmetDF, on="SMILES", how="left")
    mergedDF = computeRdkitProperties(mergedDF, smilesCol="SMILES")

    saveCsv(mergedDF, outputPath)
    return mergedDF


# -----------------------------------------------------------------------------
# Step 6: ADMET/potency desirability scoring
# -----------------------------------------------------------------------------

DEFAULT_ENDPOINT_META = [
    {"endpoint": "AMES", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "hERG", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "DILI", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "ClinTox", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "Carcinogens_Lagunin", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "Skin_Reaction", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "LD50_Zhu", "majorGroup": "toxicity", "direction": "higherBetter"},
    {"endpoint": "SR-ARE", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "SR-ATAD5", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "SR-HSE", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "SR-MMP", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "SR-p53", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "NR-AR-LBD", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "NR-AR", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "NR-AhR", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "NR-Aromatase", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "NR-ER-LBD", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "NR-ER", "majorGroup": "toxicity", "direction": "lowerBetter"},
    {"endpoint": "NR-PPAR-gamma", "majorGroup": "toxicity", "direction": "lowerBetter"},

    {"endpoint": "Bioavailability_Ma", "majorGroup": "adme", "direction": "higherBetter"},
    {"endpoint": "HIA_Hou", "majorGroup": "adme", "direction": "higherBetter"},
    {"endpoint": "PAMPA_NCATS", "majorGroup": "adme", "direction": "higherBetter"},
    {"endpoint": "Caco2_Wang", "majorGroup": "adme", "direction": "higherBetter"},
    {"endpoint": "Solubility_AqSolDB", "majorGroup": "adme", "direction": "higherBetter"},
    {"endpoint": "CYP1A2_Veith", "majorGroup": "adme", "direction": "lowerBetter"},
    {"endpoint": "CYP2C19_Veith", "majorGroup": "adme", "direction": "lowerBetter"},
    {"endpoint": "CYP2C9_Veith", "majorGroup": "adme", "direction": "lowerBetter"},
    {"endpoint": "CYP2D6_Veith", "majorGroup": "adme", "direction": "lowerBetter"},
    {"endpoint": "CYP3A4_Veith", "majorGroup": "adme", "direction": "lowerBetter"},
    {"endpoint": "CYP2C9_Substrate_CarbonMangels", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "CYP2D6_Substrate_CarbonMangels", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "CYP3A4_Substrate_CarbonMangels", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "Pgp_Broccatelli", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "Clearance_Hepatocyte_AZ", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "Clearance_Microsome_AZ", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "Half_Life_Obach", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "PPBR_AZ", "majorGroup": "adme", "direction": "referenceMatch"},
    {"endpoint": "VDss_Lombardo", "majorGroup": "adme", "direction": "referenceMatch"},
]


def empiricalPercentile(candidateValues: pd.Series, referenceValues: pd.Series) -> np.ndarray:
    ref = (
        pd.to_numeric(referenceValues, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    x = pd.to_numeric(candidateValues, errors="coerce").to_numpy(dtype=float)

    if len(ref) == 0:
        return np.full(len(x), np.nan)

    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / len(ref)


def referenceMatchDesirability(candidateValues: pd.Series, referenceValues: pd.Series) -> np.ndarray:
    ref = (
        pd.to_numeric(referenceValues, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    x = pd.to_numeric(candidateValues, errors="coerce").to_numpy(dtype=float)

    if len(ref) < 5:
        return np.full(len(x), np.nan)

    median = np.nanmedian(ref)
    q25, q75 = np.nanpercentile(ref, [25, 75])
    iqr = q75 - q25

    scale = iqr / 1.349 if iqr > 1e-9 else np.nanstd(ref)

    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0

    return np.exp(-0.5 * ((x - median) / scale) ** 2)


def weightedGeometricMean(
    DF: pd.DataFrame,
    cols: List[str],
    weights: Optional[List[float]] = None,
) -> pd.Series:

    if not cols:
        return pd.Series(np.nan, index=DF.index)

    values = DF[cols].apply(pd.to_numeric, errors="coerce").clip(lower=1e-6, upper=1.0)

    if weights is None:
        weightsArray = np.ones(len(cols), dtype=float)
    else:
        weightsArray = np.asarray(weights, dtype=float)

    weightsArray = weightsArray / weightsArray.sum()

    return np.exp(np.nansum(np.log(values.to_numpy()) * weightsArray, axis=1))


def loadReferenceDF(configDict: Dict[str, Any]) -> pd.DataFrame:
    cfg = configDict["scoring"]

    referencePath = resolvePath(cfg.get("referenceCsv"))

    if referencePath is None or not referencePath.exists():
        raise FileNotFoundError(f"scoring.referenceCsv not found: {referencePath}")

    refDF = pd.read_csv(referencePath)

    refDF = normalizeSmilesColumn(
        refDF,
        smilesCol=cfg.get("referenceSmilesCol", "SMILES"),
        outputCol="SMILES",
    )

    refDF = computeRdkitProperties(refDF, smilesCol="SMILES")

    print(f"Reference dataframe shape: {refDF.shape}")
    return refDF


def scoreCandidates(
    candidateDF: pd.DataFrame,
    referenceDF: pd.DataFrame,
    configDict: Dict[str, Any],
    outputDir: Path,
) -> pd.DataFrame:

    endpointMeta = configDict["scoring"].get("endpointMeta") or DEFAULT_ENDPOINT_META
    endpointMetaDF = pd.DataFrame(endpointMeta)

    outDF = candidateDF.copy()

    if "pPotency_prediction" in referenceDF.columns and "pPotency_prediction" in outDF.columns:
        outDF["PotencyDesRaw"] = empiricalPercentile(
            outDF["pPotency_prediction"],
            referenceDF["pPotency_prediction"],
        )
    else:
        outDF["PotencyDesRaw"] = empiricalPercentile(
            outDF["pPotency_prediction"],
            outDF["pPotency_prediction"],
        )

    toxDesCols = []
    admeDesCols = []

    for _, row in endpointMetaDF.iterrows():
        endpoint = row["endpoint"]

        if endpoint not in outDF.columns or endpoint not in referenceDF.columns:
            continue

        direction = row.get("direction", "lowerBetter")
        majorGroup = row.get("majorGroup", "toxicity")

        desCol = f"{endpoint}_des"

        if direction == "lowerBetter":
            outDF[desCol] = 1.0 - empiricalPercentile(outDF[endpoint], referenceDF[endpoint])
        elif direction == "higherBetter":
            outDF[desCol] = empiricalPercentile(outDF[endpoint], referenceDF[endpoint])
        elif direction == "referenceMatch":
            outDF[desCol] = referenceMatchDesirability(outDF[endpoint], referenceDF[endpoint])
        else:
            raise ValueError(f"Unsupported endpoint direction: {direction}")

        outDF[desCol] = pd.to_numeric(outDF[desCol], errors="coerce").clip(lower=1e-6, upper=1.0)

        if majorGroup == "toxicity":
            toxDesCols.append(desCol)
        elif majorGroup == "adme":
            admeDesCols.append(desCol)

    outDF["ToxicitySafety"] = weightedGeometricMean(outDF, toxDesCols)
    outDF["CoreToxicityScore"] = 1.0 - outDF["ToxicitySafety"]
    outDF["ADMEFeasibility"] = weightedGeometricMean(outDF, admeDesCols)

    outDF["OverallPriority2D"] = weightedGeometricMean(
        outDF,
        ["PotencyDesRaw", "ToxicitySafety"],
    )

    outDF["OverallPriority3D"] = weightedGeometricMean(
        outDF,
        ["PotencyDesRaw", "ToxicitySafety", "ADMEFeasibility"],
    )

    scorePath = outputDir / f"{configDict['jobName']}_generated_scored.csv"
    saveCsv(outDF, scorePath)

    return outDF


# -----------------------------------------------------------------------------
# Step 7: Pareto optimization
# -----------------------------------------------------------------------------

def getParetoFrontMask(values: np.ndarray) -> np.ndarray:
    """
    Returns True for non-dominated points.
    All columns are assumed to be maximized before this function is called.
    """

    nRows = values.shape[0]
    isEfficient = np.ones(nRows, dtype=bool)

    for i in range(nRows):
        if not isEfficient[i]:
            continue

        dominatesI = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        dominatesI[i] = False

        if np.any(dominatesI):
            isEfficient[i] = False

    return isEfficient


def assignParetoRanks(
    DF: pd.DataFrame,
    objectiveCols: List[str],
    maximizeFlags: List[bool],
) -> pd.DataFrame:

    workDF = DF.dropna(subset=objectiveCols).copy().reset_index(drop=True)

    if workDF.empty:
        return DF.assign(ParetoRank=np.nan)

    objectiveDF = workDF[objectiveCols].apply(pd.to_numeric, errors="coerce")
    values = objectiveDF.to_numpy(dtype=float)

    for colIdx, maximizeFlag in enumerate(maximizeFlags):
        if not maximizeFlag:
            values[:, colIdx] = -values[:, colIdx]

    ranks = np.full(len(workDF), np.nan)
    remaining = np.arange(len(workDF))
    rank = 1

    while len(remaining) > 0:
        frontMask = getParetoFrontMask(values[remaining])
        frontIdx = remaining[frontMask]

        ranks[frontIdx] = rank
        remaining = remaining[~frontMask]
        rank += 1

    workDF["ParetoRank"] = ranks.astype(int)

    return workDF


def selectFinalParetoCompounds(
    scoredDF: pd.DataFrame,
    configDict: Dict[str, Any],
    outputDir: Path,
) -> pd.DataFrame:

    cfg = configDict["pareto"]

    objectiveCols = cfg.get(
        "objectiveCols",
        ["pPotency_prediction", "CoreToxicityScore", "ADMEFeasibility"],
    )

    maximizeFlags = cfg.get("maximize", [True, False, True])
    topN = int(cfg.get("topN", 50))

    candidateLimit = cfg.get("candidateLimitBeforePareto", 5000)

    workDF = scoredDF.copy()

    if candidateLimit is not None and len(workDF) > int(candidateLimit):
        workDF = (
            workDF.sort_values("OverallPriority3D", ascending=False)
            .head(int(candidateLimit))
            .copy()
        )
        print(f"Restricted Pareto sorting to top {len(workDF)} by OverallPriority3D")

    paretoDF = assignParetoRanks(
        workDF,
        objectiveCols=objectiveCols,
        maximizeFlags=maximizeFlags,
    )

    paretoDF = paretoDF.sort_values(
        ["ParetoRank", "OverallPriority3D", "pPotency_prediction", "ADMEFeasibility", "CoreToxicityScore"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    paretoDF.insert(0, "FinalRank", range(1, len(paretoDF) + 1))

    finalDF = paretoDF.head(topN).copy()

    saveCsv(paretoDF, outputDir / f"{configDict['jobName']}_pareto_all_ranked.csv")
    saveCsv(finalDF, outputDir / f"{configDict['jobName']}_pareto_final_compounds.csv")

    return finalDF


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def runPipeline(configPath: str) -> pd.DataFrame:
    startTime = time.time()

    configDict = loadConfig(configPath)
    outputDir = makeDir(configDict["outputDir"])

    trainingDF = loadTrainingData(configDict)
    saveCsv(trainingDF, outputDir / f"{configDict['jobName']}_training_clean.csv")

    macawModel = fitOrLoadMacawTransformer(trainingDF, configDict, outputDir)

    trainingWithMacawDF, featureCols = addMacawEmbeddings(
        trainingDF,
        macawModel,
        smilesCol="SMILES",
    )

    saveCsv(trainingWithMacawDF, outputDir / f"{configDict['jobName']}_training_wMACAW.csv")

    artModel = buildOrLoadArtModel(
        trainingWithMacawDF,
        featureCols,
        configDict,
        outputDir,
    )

    candidateDF = loadOrGenerateCandidates(configDict, outputDir)

    candidateDF = applyCandidateFilters(
        candidateDF,
        trainingDF,
        configDict,
    )

    saveCsv(candidateDF, outputDir / f"{configDict['jobName']}_candidate_molecules_clean.csv")

    candidateWithMacawDF, candidateFeatureCols = addMacawEmbeddings(
        candidateDF,
        macawModel,
        smilesCol="SMILES",
    )

    if candidateFeatureCols != featureCols:
        raise ValueError("Candidate MACAW feature columns do not match training MACAW feature columns.")

    potencyDF = predictPotencyWithArt(
        artModel,
        candidateWithMacawDF,
        featureCols,
        configDict,
    )

    saveCsv(potencyDF, outputDir / f"{configDict['jobName']}_generated_wART.csv")

    potencyAdmetDF = predictAdmetWithCheckpoint(
        potencyDF,
        configDict,
        outputDir,
    )

    referenceDF = loadReferenceDF(configDict)

    scoredDF = scoreCandidates(
        potencyAdmetDF,
        referenceDF,
        configDict,
        outputDir,
    )

    finalDF = selectFinalParetoCompounds(
        scoredDF,
        configDict,
        outputDir,
    )

    elapsed = time.time() - startTime

    print(f"Pipeline complete in {elapsed:.2f} seconds")

    previewCols = [
        "FinalRank",
        "SMILES",
        "pPotency_prediction",
        "pPotency_std",
        "CoreToxicityScore",
        "ADMEFeasibility",
        "ParetoRank",
    ]

    existingPreviewCols = [col for col in previewCols if col in finalDF.columns]
    print(finalDF[existingPreviewCols].head())

    return finalDF


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run integrated DTRA antiviral drug-design pipeline."
    )

    parser.add_argument(
        "configFile",
        help="YAML config path",
    )

    args = parser.parse_args()
    runPipeline(args.configFile)


if __name__ == "__main__":
    main()
