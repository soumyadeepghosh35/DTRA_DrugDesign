#!/usr/bin/env python
"""
Run one DORAnet generation from a starter CSV.

Designed workflow:
  python runDORAnet.py -c config_gen1.yaml > DORAnet_gen1.out
  python runDORAnet.py -c config_gen2.yaml > DORAnet_gen2.out
  python runDORAnet.py -c config_gen3.yaml > DORAnet_gen3.out

Key behavior:
  - Each config runs exactly one DORAnet generation.
  - Every run starts fresh from the input starter CSV.
  - Every starter gets its own starter_* directory.
  - Every starter gets its own max_atoms dictionary computed from that starter only:
        max_atoms[element] = ceil(multiplier * count(element in starter))
    for elements C, N, O, S by default.
  - The starter-specific resolved config is saved inside each starter_* directory.
"""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import sys
import time
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path

import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Update this path only if your DORAnet checkout is elsewhere.
# You can also override it without editing the script:
#   export DORANET_PATH=/path/to/doranet
DORANET_PATH = Path(os.environ.get("DORANET_PATH", "/users/sghosh6/DTRA_project/MACAW/doranet"))
sys.path.insert(0, str(DORANET_PATH))

import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as post_processing


HELPERS = {
    "O", "O=O", "[H][H]", "O=C=O", "C=O", "[C-]#[O+]", "Br", "[Br][Br]",
    "CO", "C=C", "O=S(O)O", "N", "O=S(=O)(O)O", "O=NO", "N#N",
    "O=[N+]([O-])O", "NO", "C#N", "S", "O=S=O", "N#CO", "[H+]", "OO",
    "Cl", "I", "O=C(O)O", "O=P(O)(O)O", "O=P(O)(O)OP(=O)(O)O", "C",
    "CC", "CC=O", "CC(=O)O", "CCC(=O)O",
}


# -----------------------------
# Configuration helpers
# -----------------------------

def deepUpdate(defaultDict, userDict):
    """Recursively merge user config into defaults."""
    outputDict = copy.deepcopy(defaultDict)
    if userDict is None:
        return outputDict

    for key, value in userDict.items():
        if (
            key in outputDict
            and isinstance(outputDict[key], dict)
            and isinstance(value, dict)
        ):
            outputDict[key] = deepUpdate(outputDict[key], value)
        else:
            outputDict[key] = value
    return outputDict


def loadConfig(configPath):
    with open(configPath, "r") as f:
        userConfig = yaml.safe_load(f) or {}

    defaults = {
        "input": {
            "SMILESfile": None,
            "smiles_column": "Canonical_SMILES",
            "smiles_column_candidates": [
                "Canonical_SMILES",
                "canonical_smiles",
                "SMILES",
                "Smiles",
                "smiles",
            ],
            "start_index": 0,
            "num_smiles": None,
            "deduplicate_smiles": False,
        },
        "output": {
            "directory": "doranet_output",
            "overwrite_existing_output_dir": True,
            "overwrite_existing_starter_dirs": True,
            "save_generation_config": True,
            "save_starter_config": True,
        },
        "parallel": {
            "num_workers": 4,
        },
        "network": {
            # Single integer only. Use config_gen1, config_gen2, config_gen3 separately.
            "generations": 1,
            "ruleset": "JN3604IMT",
            "direction": "forward",
        },
        "max_atoms": {
            "mode": "per_starter",
            "elements": ["C", "N", "O", "S"],
            "multiplier": 1.5,
            "rounding": "ceil",
            # Keep all default minima at zero so no extra atom type is allowed unless
            # it is present in the starter molecule. This follows the requested logic.
            "minimum": {"C": 0, "N": 0, "O": 0, "S": 0},
        },
        "validation": {
            "fail_on_invalid_smiles": True,
            "expected_num_starters": None,
        },
        "post_processing": {
            "run_one_step": True,
        },
    }

    return deepUpdate(defaults, userConfig)


def validateConfig(config):
    errors = []

    inputFile = config["input"].get("SMILESfile")
    if not inputFile:
        errors.append("input.SMILESfile is required")
    elif not os.path.exists(inputFile):
        errors.append(f"Input file not found: {inputFile}")

    try:
        config["network"]["generations"] = int(config["network"]["generations"])
        if config["network"]["generations"] < 1:
            errors.append("network.generations must be >= 1")
    except Exception:
        errors.append("network.generations must be a single integer, not a list")

    try:
        config["parallel"]["num_workers"] = int(config["parallel"]["num_workers"])
        if config["parallel"]["num_workers"] < 1:
            errors.append("parallel.num_workers must be at least 1")
    except Exception:
        errors.append("parallel.num_workers must be an integer")

    if config["max_atoms"].get("mode") != "per_starter":
        errors.append("max_atoms.mode must be 'per_starter' for this workflow")

    try:
        multiplier = float(config["max_atoms"].get("multiplier", 1.5))
        if multiplier <= 0:
            errors.append("max_atoms.multiplier must be > 0")
        config["max_atoms"]["multiplier"] = multiplier
    except Exception:
        errors.append("max_atoms.multiplier must be numeric")

    rounding = config["max_atoms"].get("rounding", "ceil")
    if rounding not in {"ceil", "floor", "round"}:
        errors.append("max_atoms.rounding must be one of: ceil, floor, round")

    elements = config["max_atoms"].get("elements", ["C", "N", "O", "S"])
    if not isinstance(elements, list) or not elements:
        errors.append("max_atoms.elements must be a non-empty list")

    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    config["parallel"]["num_workers"] = min(config["parallel"]["num_workers"], cpu_count())
    return config


# -----------------------------
# Input and chemistry helpers
# -----------------------------

def chooseSmilesColumn(fieldnames, preferredColumn, candidateColumns):
    if preferredColumn in fieldnames:
        return preferredColumn

    for col in candidateColumns:
        if col in fieldnames:
            print(
                f"Warning: configured smiles_column '{preferredColumn}' was not found. "
                f"Using detected column '{col}' instead."
            )
            return col

    available = ", ".join(fieldnames or [])
    raise ValueError(
        f"Could not find a SMILES column. Requested '{preferredColumn}'. "
        f"Available columns: {available}"
    )


def readSmilesFromCsv(csvPath, smilesColumn, startIdx, numSmiles, candidateColumns, deduplicate=False):
    smilesList = []
    seen = set()

    with open(csvPath, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        resolvedColumn = chooseSmilesColumn(fieldnames, smilesColumn, candidateColumns)

        for rowIdx, row in enumerate(reader):
            if rowIdx < int(startIdx):
                continue
            if numSmiles is not None and len(smilesList) >= int(numSmiles):
                break

            smi = (row.get(resolvedColumn, "") or "").strip()
            if not smi:
                continue

            if deduplicate:
                if smi in seen:
                    continue
                seen.add(smi)

            smilesList.append({
                "row_index": rowIdx,
                "starter_idx": len(smilesList),
                "starter_smiles": smi,
            })

    return smilesList, resolvedColumn


def countAtomsForElements(smiles, elements):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    counts = {element: 0 for element in elements}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in counts:
            counts[symbol] += 1
    return counts


def applyRounding(value, roundingMode):
    if roundingMode == "ceil":
        return int(math.ceil(value))
    if roundingMode == "floor":
        return int(math.floor(value))
    if roundingMode == "round":
        return int(round(value))
    raise ValueError(f"Unsupported rounding mode: {roundingMode}")


def computeStarterMaxAtoms(starterSmiles, maxAtomsConfig):
    elements = list(maxAtomsConfig.get("elements", ["C", "N", "O", "S"]))
    multiplier = float(maxAtomsConfig.get("multiplier", 1.5))
    roundingMode = maxAtomsConfig.get("rounding", "ceil")
    minimum = maxAtomsConfig.get("minimum", {}) or {}

    atomCounts = countAtomsForElements(starterSmiles, elements)
    if atomCounts is None:
        return None, None

    maxAtoms = {}
    for element in elements:
        scaled = applyRounding(atomCounts[element] * multiplier, roundingMode)
        minVal = int(minimum.get(element, 0))
        maxAtoms[element] = max(scaled, minVal)

    return maxAtoms, atomCounts


# -----------------------------
# Output preparation
# -----------------------------

def prepareOutputDirectory(outputDir, overwriteExisting):
    outputPath = Path(outputDir)
    if outputPath.exists() and overwriteExisting:
        shutil.rmtree(outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)
    return outputPath


def saveYaml(data, outputPath):
    with open(outputPath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def buildStarterConfig(baseConfig, starterRecord):
    starterConfig = copy.deepcopy(baseConfig)
    starterConfig["starter"] = {
        "starter_idx": starterRecord["starter_idx"],
        "input_csv_row_index": starterRecord["row_index"],
        "starter_smiles": starterRecord["starter_smiles"],
        "job_name": starterRecord["job_name"],
        "starter_output_directory": starterRecord["starter_dir"],
        "atom_counts": starterRecord["atom_counts"],
        "resolved_max_atoms": starterRecord["max_atoms"],
    }
    starterConfig["resolved_max_atoms"] = starterRecord["max_atoms"]
    return starterConfig


def prepareStarterDirectories(smilesRecords, config):
    outputDir = Path(config["output"]["directory"])
    gen = int(config["network"]["generations"])
    starterRecords = []
    invalidRecords = []

    for record in smilesRecords:
        starterIdx = int(record["starter_idx"])
        starterSmiles = record["starter_smiles"]
        starterName = f"starter_{starterIdx:05d}"
        jobName = f"{starterName}_gen{gen}"
        starterDir = outputDir / starterName

        if starterDir.exists() and config["output"].get("overwrite_existing_starter_dirs", True):
            shutil.rmtree(starterDir)
        starterDir.mkdir(parents=True, exist_ok=True)

        maxAtoms, atomCounts = computeStarterMaxAtoms(starterSmiles, config["max_atoms"])
        if maxAtoms is None:
            invalidRecord = {
                **record,
                "starter_dir": str(starterDir),
                "job_name": jobName,
                "error": "RDKit failed to parse starter SMILES; cannot compute per-starter max_atoms",
            }
            invalidRecords.append(invalidRecord)
            continue

        starterRecord = {
            **record,
            "starter_dir": str(starterDir),
            "job_name": jobName,
            "atom_counts": atomCounts,
            "max_atoms": maxAtoms,
        }

        if config["output"].get("save_starter_config", True):
            starterConfig = buildStarterConfig(config, starterRecord)
            configPath = starterDir / "config_used.yaml"
            saveYaml(starterConfig, configPath)
            starterRecord["starter_config_path"] = str(configPath)

        starterRecords.append(starterRecord)

    if invalidRecords and config["validation"].get("fail_on_invalid_smiles", True):
        invalidPath = outputDir / "invalid_starters.csv"
        with open(invalidPath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["starter_idx", "row_index", "starter_smiles", "starter_dir", "job_name", "error"])
            for r in invalidRecords:
                writer.writerow([
                    r["starter_idx"],
                    r["row_index"],
                    r["starter_smiles"],
                    r["starter_dir"],
                    r["job_name"],
                    r["error"],
                ])
        raise ValueError(
            f"Found {len(invalidRecords)} invalid starter SMILES. Details saved to {invalidPath}"
        )

    return starterRecords, invalidRecords


def saveStarterPreparationSummary(starterRecords, outputDir):
    summaryPath = Path(outputDir) / "starter_preparation_summary.csv"
    with open(summaryPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "starter_idx",
            "input_csv_row_index",
            "starter_smiles",
            "starter_directory",
            "job_name",
            "atom_counts_json",
            "max_atoms_json",
            "starter_config_path",
        ])
        for r in starterRecords:
            writer.writerow([
                r.get("generation"),
                r["starter_idx"],
                r["row_index"],
                r["starter_smiles"],
                r["starter_dir"],
                r["job_name"],
                json.dumps(r["atom_counts"], sort_keys=True),
                json.dumps(r["max_atoms"], sort_keys=True),
                r.get("starter_config_path", ""),
            ])
    return summaryPath


def saveGenerationConfig(config, outputDir):
    generationConfigPath = Path(outputDir) / "config_generation_used.yaml"
    saveYaml(config, generationConfigPath)
    return generationConfigPath


# -----------------------------
# DORAnet execution
# -----------------------------

def processSingleStarter(starterRecord, config):
    starterIdx = int(starterRecord["starter_idx"])
    starterSmiles = starterRecord["starter_smiles"]
    starterDir = Path(starterRecord["starter_dir"])
    jobName = starterRecord["job_name"]
    gen = int(config["network"]["generations"])
    maxAtoms = starterRecord["max_atoms"]

    result = {
        "generation": gen,
        "starter_idx": starterIdx,
        "input_csv_row_index": starterRecord["row_index"],
        "starter_smiles": starterSmiles,
        "starter_directory": str(starterDir),
        "job_name": jobName,
        "atom_counts": starterRecord["atom_counts"],
        "max_atoms": maxAtoms,
        "status": "failed",
        "num_molecules": 0,
        "num_targets": 0,
        "time_seconds": 0,
        "error": None,
    }

    startTime = time.time()
    originalDir = os.getcwd()

    try:
        os.chdir(starterDir)
        userStarters = {starterSmiles}

        forwardNetwork = enzymatic.generate_network(
            job_name=jobName,
            starters=userStarters,
            gen=gen,
            max_atoms=maxAtoms,
            direction=config["network"]["direction"],
            ruleset=config["network"]["ruleset"],
        )

        generatedSmilesList = [mol.uid for mol in forwardNetwork.mols]
        result["num_molecules"] = len(generatedSmilesList)

        allTargets = set(generatedSmilesList) - userStarters - HELPERS
        result["num_targets"] = len(allTargets)

        outputPath = Path(f"{jobName}_molecules.csv")
        with open(outputPath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "Is_Starter", "MolFormula", "MolWeight", "NumHeavyAtoms"])
            for smi in generatedSmilesList:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    molWeight = round(rdMolDescriptors.CalcExactMolWt(mol), 4)
                    numHeavy = mol.GetNumHeavyAtoms()
                else:
                    formula, molWeight, numHeavy = "N/A", 0, 0
                writer.writerow([smi, smi in userStarters, formula, molWeight, numHeavy])

        if allTargets and config.get("post_processing", {}).get("run_one_step", True):
            post_processing.one_step(
                networks={forwardNetwork},
                total_generations=gen,
                starters=userStarters,
                helpers=HELPERS,
                target=allTargets,
                job_name=jobName,
            )

        result["status"] = "success"

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    finally:
        os.chdir(originalDir)

    result["time_seconds"] = round(time.time() - startTime, 2)
    return result


def workerProcess(workerId, taskQueue, resultQueue, config, totalTasks):
    while True:
        starterRecord = taskQueue.get()
        if starterRecord is None:
            break

        result = processSingleStarter(starterRecord, config)

        displayStarterIdx = int(starterRecord["starter_idx"]) + 1
        displayWorkerId = workerId + 1
        gen = int(config["network"]["generations"])

        print(
            f"[Gen {gen}] [Worker {displayWorkerId}] "
            f"[{displayStarterIdx:05d}/{totalTasks:05d}] "
            f"{result['status'].upper()} | "
            f"max_atoms={result['max_atoms']} | "
            f"Molecules: {result['num_molecules']} | "
            f"Targets: {result['num_targets']} | "
            f"Time: {result['time_seconds']}s | "
            f"SMILES: {result['starter_smiles'][:40]}...",
            flush=True,
        )

        resultQueue.put(result)


def runParallelProcessing(starterRecords, config):
    numWorkers = int(config["parallel"]["num_workers"])
    totalTasks = len(starterRecords)
    gen = int(config["network"]["generations"])

    print("\nStarting parallel DORAnet processing")
    print(f"  Generation: {gen}")
    print(f"  Output directory: {config['output']['directory']}")
    print(f"  Total starters: {totalTasks}")
    print(f"  Workers: {numWorkers}")
    print("  Result collection: waits for all starters; no runtime cutoff is applied by this script")
    print("-" * 70, flush=True)

    taskQueue = Queue()
    resultQueue = Queue()

    for starterRecord in starterRecords:
        taskQueue.put(starterRecord)

    for _ in range(numWorkers):
        taskQueue.put(None)

    workers = []
    for workerId in range(numWorkers):
        p = Process(
            target=workerProcess,
            args=(workerId, taskQueue, resultQueue, config, totalTasks),
        )
        p.daemon = False
        workers.append(p)
        p.start()

    results = []
    # No timeout here. This intentionally allows long gen=3 DORAnet jobs to run.
    for _ in range(totalTasks):
        result = resultQueue.get()
        results.append(result)

    for p in workers:
        p.join()

    results.sort(key=lambda x: x["starter_idx"])
    print(f"\nCollected {len(results)} / {totalTasks} results for gen {gen}", flush=True)
    return results


# -----------------------------
# Summary reporting
# -----------------------------

def saveSummary(results, outputDir):
    summaryPath = Path(outputDir) / "processing_summary.csv"
    with open(summaryPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "starter_idx",
            "input_csv_row_index",
            "starter_smiles",
            "starter_directory",
            "job_name",
            "atom_counts_json",
            "max_atoms_json",
            "status",
            "num_molecules",
            "num_targets",
            "time_seconds",
            "error",
        ])
        for r in results:
            writer.writerow([
                r["generation"],
                r["starter_idx"],
                r["input_csv_row_index"],
                r["starter_smiles"],
                r["starter_directory"],
                r["job_name"],
                json.dumps(r["atom_counts"], sort_keys=True),
                json.dumps(r["max_atoms"], sort_keys=True),
                r["status"],
                r["num_molecules"],
                r["num_targets"],
                r["time_seconds"],
                r["error"] or "",
            ])
    return summaryPath


def printConfig(config, resolvedSmilesColumn):
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"DORANET_PATH: {DORANET_PATH}")
    print(f"Input file: {config['input']['SMILESfile']}")
    print(f"SMILES column used: {resolvedSmilesColumn}")
    print(f"Start index: {config['input']['start_index']}")
    print(f"Number to process: {config['input']['num_smiles'] or 'all'}")
    print(f"Deduplicate SMILES: {config['input'].get('deduplicate_smiles', False)}")
    print(f"Output directory: {config['output']['directory']}")
    print(f"Overwrite output directory: {config['output'].get('overwrite_existing_output_dir', True)}")
    print(f"Workers: {config['parallel']['num_workers']}")
    print(f"Generation: {config['network']['generations']}")
    print(f"Ruleset: {config['network']['ruleset']}")
    print(f"Direction: {config['network']['direction']}")
    print(f"Max atoms mode: {config['max_atoms']['mode']}")
    print(f"Max atoms elements: {config['max_atoms']['elements']}")
    print(f"Max atoms multiplier: {config['max_atoms']['multiplier']}")
    print(f"Max atoms rounding: {config['max_atoms']['rounding']}")
    print("=" * 70, flush=True)


def printSummary(results, totalTime):
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    totalMolecules = sum(r["num_molecules"] for r in successful)
    totalTargets = sum(r["num_targets"] for r in successful)
    avgTime = sum(r["time_seconds"] for r in results) / len(results) if results else 0
    gen = results[0]["generation"] if results else "N/A"

    print("\n" + "=" * 70)
    print(f"PROCESSING SUMMARY | GEN {gen}")
    print("=" * 70)
    print(f"Total starters processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total molecules generated: {totalMolecules}")
    print(f"Total targets processed: {totalTargets}")
    print(f"Average time per starter: {avgTime:.2f}s")
    print(f"Total wall time: {totalTime:.2f}s")

    if failed:
        print("\nFailed starters:")
        for r in failed[:20]:
            print(
                f"  [{r['starter_idx']}] {r['starter_smiles'][:50]}... | "
                f"max_atoms={r['max_atoms']} | Error: {r['error']}"
            )
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Run one DORAnet generation with per-starter max_atoms from starter CSV"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)

    print(f"Loading configuration from: {args.config}")
    config = loadConfig(args.config)
    config = validateConfig(config)

    outputDir = prepareOutputDirectory(
        config["output"]["directory"],
        config["output"].get("overwrite_existing_output_dir", True),
    )

    print("\nReading starter SMILES from CSV...")
    smilesRecords, resolvedSmilesColumn = readSmilesFromCsv(
        csvPath=config["input"]["SMILESfile"],
        smilesColumn=config["input"].get("smiles_column", "Canonical_SMILES"),
        startIdx=config["input"].get("start_index", 0),
        numSmiles=config["input"].get("num_smiles", None),
        candidateColumns=config["input"].get("smiles_column_candidates", []),
        deduplicate=bool(config["input"].get("deduplicate_smiles", False)),
    )

    if not smilesRecords:
        print("Error: No non-empty starter SMILES found in input file")
        sys.exit(1)

    expectedNumStarters = config.get("validation", {}).get("expected_num_starters")
    if expectedNumStarters is not None and len(smilesRecords) != int(expectedNumStarters):
        print(
            f"Error: expected {expectedNumStarters} starters, but loaded {len(smilesRecords)}. "
            "Check the CSV path, SMILES column, blank rows, and deduplicate_smiles setting."
        )
        sys.exit(1)

    config["input"]["resolved_smiles_column"] = resolvedSmilesColumn
    printConfig(config, resolvedSmilesColumn)
    print(f"Loaded {len(smilesRecords)} starters from CSV")

    if config["output"].get("save_generation_config", True):
        generationConfigPath = saveGenerationConfig(config, outputDir)
        print(f"Generation-level config saved to: {generationConfigPath}")

    print("\nPreparing starter directories and starter-specific max_atoms configs...")
    try:
        starterRecords, invalidRecords = prepareStarterDirectories(smilesRecords, config)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    gen = int(config["network"]["generations"])
    for r in starterRecords:
        r["generation"] = gen

    prepSummaryPath = saveStarterPreparationSummary(starterRecords, outputDir)
    print(f"Starter preparation summary saved to: {prepSummaryPath}")
    print(f"Prepared {len(starterRecords)} starter directories")

    if invalidRecords:
        print(f"Warning: {len(invalidRecords)} invalid starters were skipped because fail_on_invalid_smiles=false")

    totalStartTime = time.time()
    results = runParallelProcessing(starterRecords, config)
    totalTime = time.time() - totalStartTime

    summaryPath = saveSummary(results, outputDir)
    print(f"\nProcessing summary saved to: {summaryPath}")
    printSummary(results, totalTime)


if __name__ == "__main__":
    main()
