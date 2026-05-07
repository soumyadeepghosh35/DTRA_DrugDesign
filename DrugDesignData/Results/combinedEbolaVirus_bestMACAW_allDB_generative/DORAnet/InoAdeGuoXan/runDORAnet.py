#!/usr/bin/env python

import sys
import os
import argparse
import time
import csv
import copy
import math
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
import yaml

RDLogger.DisableLog('rdApp.*')

# Update this path only if your DORAnet checkout is elsewhere.
DORANET_PATH = Path("/users/sghosh6/DTRA_project/MACAW/doranet")
sys.path.insert(0, str(DORANET_PATH))

import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as post_processing

HELPERS = {
    'O', 'O=O', '[H][H]', 'O=C=O', 'C=O', '[C-]#[O+]', 'Br', '[Br][Br]',
    'CO', 'C=C', 'O=S(O)O', 'N', 'O=S(=O)(O)O', 'O=NO', 'N#N',
    'O=[N+]([O-])O', 'NO', 'C#N', 'S', 'O=S=O', 'N#CO', '[H+]', 'OO',
    'Cl', 'I', 'O=C(O)O', 'O=P(O)(O)O', 'O=P(O)(O)OP(=O)(O)O', 'C',
    'CC', 'CC=O', 'CC(=O)O', 'CCC(=O)O'
}


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
            "start_index": 0,
            "num_smiles": None,
        },
        "output": {
            "directory": "doranet_output",
            # Final folders become, for example: doranet_output_gen1, doranet_output_gen2, ...
            "generation_suffix_template": "_gen{gen}",
            "write_combined_summary": True,
        },
        "parallel": {
            "num_workers": 4,
            "result_timeout_seconds": 600,
            "join_timeout_seconds": 60,
        },
        "network": {
            # Can be either a single integer, e.g. 2, or a list, e.g. [1, 2, 3].
            "generations": [1, 2, 3],
            "ruleset": "JN3604IMT",
            "direction": "forward",
        },
        "max_atoms": {
            # auto_from_starters computes per-element max atoms from the starter SMILES set.
            # static uses explicit values from max_atoms.static_values.
            "mode": "auto_from_starters",
            "elements": ["C", "N", "O", "S"],
            "multiplier": 1.5,
            "rounding": "ceil",
            "minimum": {"C": 1, "N": 0, "O": 0, "S": 0},
            "static_values": {"C": 12, "N": 3, "O": 5, "S": 3},
        },
    }

    return deepUpdate(defaults, userConfig)


def normalizeGenerations(generations):
    if isinstance(generations, int):
        generationsList = [generations]
    elif isinstance(generations, list):
        generationsList = generations
    else:
        raise ValueError("network.generations must be an integer or a list of integers")

    cleaned = []
    for gen in generationsList:
        try:
            genInt = int(gen)
        except Exception as exc:
            raise ValueError(f"Invalid generation value: {gen}") from exc

        if genInt < 1:
            raise ValueError("All network.generations values must be >= 1")
        cleaned.append(genInt)

    return sorted(set(cleaned))


def validateConfig(config):
    errors = []

    if not config["input"]["SMILESfile"]:
        errors.append("input.SMILESfile is required")
    elif not os.path.exists(config["input"]["SMILESfile"]):
        errors.append(f"Input file not found: {config['input']['SMILESfile']}")

    if int(config["parallel"]["num_workers"]) < 1:
        errors.append("parallel.num_workers must be at least 1")

    try:
        normalizeGenerations(config["network"]["generations"])
    except ValueError as exc:
        errors.append(str(exc))

    maxAtomsMode = config["max_atoms"].get("mode", "auto_from_starters")
    if maxAtomsMode not in {"auto_from_starters", "static"}:
        errors.append("max_atoms.mode must be either 'auto_from_starters' or 'static'")

    multiplier = float(config["max_atoms"].get("multiplier", 1.5))
    if multiplier <= 0:
        errors.append("max_atoms.multiplier must be > 0")

    rounding = config["max_atoms"].get("rounding", "ceil")
    if rounding not in {"ceil", "floor", "round"}:
        errors.append("max_atoms.rounding must be one of: ceil, floor, round")

    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    config["parallel"]["num_workers"] = min(
        int(config["parallel"]["num_workers"]),
        cpu_count(),
    )
    config["parallel"]["result_timeout_seconds"] = int(
        config["parallel"].get("result_timeout_seconds", 600)
    )
    config["parallel"]["join_timeout_seconds"] = int(
        config["parallel"].get("join_timeout_seconds", 60)
    )

    config["network"]["generations"] = normalizeGenerations(config["network"]["generations"])

    return config


def readSmilesFromCsv(csvPath, smilesColumn, startIdx, numSmiles):
    smilesList = []

    with open(csvPath, "r") as f:
        reader = csv.DictReader(f)

        if smilesColumn not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"Column '{smilesColumn}' not found. Available: {available}")

        for idx, row in enumerate(reader):
            if idx < int(startIdx):
                continue
            if numSmiles is not None and len(smilesList) >= int(numSmiles):
                break

            smi = row.get(smilesColumn, "")
            if smi and smi.strip():
                smilesList.append(smi.strip())

    return smilesList


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


def computeMaxAtomsFromStarters(smilesList, maxAtomsConfig):
    """
    Compute DORAnet max_atoms from the highest C/N/O/S atom counts in the starter set.

    Example:
      If the largest carbon count across all starter molecules is 10 and multiplier is 1.5,
      max_atoms['C'] becomes ceil(10 * 1.5) = 15.
    """
    elements = list(maxAtomsConfig.get("elements", ["C", "N", "O", "S"]))
    multiplier = float(maxAtomsConfig.get("multiplier", 1.5))
    roundingMode = maxAtomsConfig.get("rounding", "ceil")
    minimum = maxAtomsConfig.get("minimum", {}) or {}

    maxStarterCounts = {element: 0 for element in elements}
    invalidSmiles = []

    for smi in smilesList:
        counts = countAtomsForElements(smi, elements)
        if counts is None:
            invalidSmiles.append(smi)
            continue

        for element in elements:
            maxStarterCounts[element] = max(maxStarterCounts[element], counts[element])

    if len(invalidSmiles) == len(smilesList):
        raise ValueError("All starter SMILES failed RDKit parsing; cannot compute auto max_atoms")

    maxAtoms = {}
    for element in elements:
        scaled = applyRounding(maxStarterCounts[element] * multiplier, roundingMode)
        minVal = int(minimum.get(element, 0))
        maxAtoms[element] = max(scaled, minVal)

    return maxAtoms, maxStarterCounts, invalidSmiles


def resolveMaxAtoms(smilesList, config):
    maxAtomsConfig = config["max_atoms"]
    mode = maxAtomsConfig.get("mode", "auto_from_starters")

    if mode == "static":
        staticValues = maxAtomsConfig.get("static_values", {})
        return {key: int(value) for key, value in staticValues.items()}, None, []

    return computeMaxAtomsFromStarters(smilesList, maxAtomsConfig)


def makeGenerationOutputDir(baseOutputDir, suffixTemplate, gen):
    baseOutputDir = str(baseOutputDir).rstrip("/")
    suffix = suffixTemplate.format(gen=gen)
    return Path(f"{baseOutputDir}{suffix}")


def printConfig(config, maxStarterCounts=None):
    print("-" * 70)
    print("CONFIGURATION")
    print("-" * 70)
    print(f"Input file: {config['input']['SMILESfile']}")
    print(f"SMILES column: {config['input']['smiles_column']}")
    print(f"Start index: {config['input']['start_index']}")
    print(f"Number to process: {config['input']['num_smiles'] or 'all'}")
    print(f"Base output directory: {config['output']['directory']}")
    print(f"Generation suffix template: {config['output']['generation_suffix_template']}")
    print(f"Workers: {config['parallel']['num_workers']}")
    print(f"Generations to run: {config['network']['generations']}")
    print(f"Ruleset: {config['network']['ruleset']}")
    print(f"Direction: {config['network']['direction']}")
    print(f"Max atoms mode: {config['max_atoms'].get('mode')}")
    print(f"Resolved max atoms: {config['resolved_max_atoms']}")
    if maxStarterCounts is not None:
        print(f"Max starter atom counts: {maxStarterCounts}")
    print("-" * 70)


def processSingleStarter(starterIdx, starterSmiles, config):
    """
    Process one starter molecule for one generation setting.
    """
    gen = int(config["network"]["active_generation"])
    jobName = f"starter_{starterIdx:05d}_gen{gen}"
    jobOutputDir = Path(config["output"]["directory"]) / f"starter_{starterIdx:05d}"
    jobOutputDir.mkdir(parents=True, exist_ok=True)

    originalDir = os.getcwd()
    os.chdir(jobOutputDir)

    result = {
        "generation": gen,
        "starter_idx": starterIdx,
        "starter_smiles": starterSmiles,
        "status": "failed",
        "num_molecules": 0,
        "num_targets": 0,
        "time_seconds": 0,
        "error": None,
    }

    startTime = time.time()

    try:
        userStarters = {starterSmiles}

        forwardNetwork = enzymatic.generate_network(
            job_name=jobName,
            starters=userStarters,
            gen=gen,
            max_atoms=config["resolved_max_atoms"],
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
            writer.writerow([
                "SMILES",
                "Is_Starter",
                "MolFormula",
                "MolWeight",
                "NumHeavyAtoms",
            ])
            for smi in generatedSmilesList:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    molWeight = round(rdMolDescriptors.CalcExactMolWt(mol), 4)
                    numHeavy = mol.GetNumHeavyAtoms()
                else:
                    formula, molWeight, numHeavy = "N/A", 0, 0
                writer.writerow([smi, smi in userStarters, formula, molWeight, numHeavy])

        if allTargets:
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
        task = taskQueue.get()
        if task is None:
            break

        starterIdx, starterSmiles = task
        result = processSingleStarter(starterIdx, starterSmiles, config)

        displayStarterIdx = starterIdx + 1
        displayWorkerId = workerId + 1
        gen = config["network"]["active_generation"]

        print(
            f"[Gen {gen}] [Worker {displayWorkerId}] "
            f"[{displayStarterIdx:05d}/{totalTasks:05d}] "
            f"{result['status'].upper()} | "
            f"Molecules: {result['num_molecules']} | "
            f"Targets: {result['num_targets']} | "
            f"Time: {result['time_seconds']}s | "
            f"SMILES: {starterSmiles[:40]}..."
        )

        resultQueue.put(result)


def runParallelProcessing(smilesList, config):
    numWorkers = config["parallel"]["num_workers"]
    totalTasks = len(smilesList)
    gen = config["network"]["active_generation"]

    print("\nStarting parallel processing")
    print(f"  Generation: {gen}")
    print(f"  Output directory: {config['output']['directory']}")
    print(f"  Total SMILES: {totalTasks}")
    print(f"  Workers: {numWorkers}")
    print(f"  Max atoms: {config['resolved_max_atoms']}")
    print("-" * 70)

    taskQueue = Queue()
    for idx, smi in enumerate(smilesList):
        taskQueue.put((idx, smi))

    for _ in range(numWorkers):
        taskQueue.put(None)

    resultQueue = Queue()
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
    timeoutSeconds = int(config["parallel"].get("result_timeout_seconds", 600))
    for _ in range(totalTasks):
        try:
            result = resultQueue.get(timeout=timeoutSeconds)
            results.append(result)
        except Exception as exc:
            print(f"Warning: Timeout or error collecting result: {exc}")
            break

    joinTimeout = int(config["parallel"].get("join_timeout_seconds", 60))
    for p in workers:
        p.join(timeout=joinTimeout)
        if p.is_alive():
            print(f"Warning: Worker {p.pid} did not exit cleanly, terminating...")
            p.terminate()
            p.join(timeout=10)

    results.sort(key=lambda x: x["starter_idx"])
    print(f"\nCollected {len(results)} / {totalTasks} results for gen {gen}")

    return results


def saveSummary(results, outputDir):
    summaryPath = Path(outputDir) / "processing_summary.csv"

    with open(summaryPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "starter_idx",
            "starter_smiles",
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
                r["starter_smiles"],
                r["status"],
                r["num_molecules"],
                r["num_targets"],
                r["time_seconds"],
                r["error"] or "",
            ])

    return summaryPath


def saveCombinedSummary(allResults, outputDir):
    combinedPath = Path(outputDir) / "processing_summary_all_generations.csv"
    with open(combinedPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "starter_idx",
            "starter_smiles",
            "status",
            "num_molecules",
            "num_targets",
            "time_seconds",
            "error",
        ])
        for result in allResults:
            writer.writerow([
                result["generation"],
                result["starter_idx"],
                result["starter_smiles"],
                result["status"],
                result["num_molecules"],
                result["num_targets"],
                result["time_seconds"],
                result["error"] or "",
            ])
    return combinedPath


def printSummary(results, totalTime):
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    totalMolecules = sum(r["num_molecules"] for r in successful)
    totalTargets = sum(r["num_targets"] for r in successful)
    avgTime = sum(r["time_seconds"] for r in results) / len(results) if results else 0
    gen = results[0]["generation"] if results else "N/A"

    print("\n" + "-" * 70)
    print(f"PROCESSING SUMMARY | GEN {gen}")
    print("-" * 70)
    print(f"Total starters processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total molecules generated: {totalMolecules}")
    print(f"Total targets processed: {totalTargets}")
    print(f"Average time per starter: {avgTime:.2f}s")
    print(f"Total wall time: {totalTime:.2f}s")

    if failed:
        print("\nFailed starters:")
        for r in failed[:10]:
            print(f"  [{r['starter_idx']}] {r['starter_smiles'][:50]}... | Error: {r['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


def saveConfigCopy(config, outputDir):
    configCopyPath = Path(outputDir) / "config_used.yaml"
    with open(configCopyPath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return configCopyPath


def main():
    parser = argparse.ArgumentParser(
        description="Parallel DORAnet generation loop with auto max_atoms from starter SMILES"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)

    print(f"Loading configuration from: {args.config}")
    config = loadConfig(args.config)
    config = validateConfig(config)

    print("\nReading SMILES from CSV...")
    smilesList = readSmilesFromCsv(
        config["input"]["SMILESfile"],
        config["input"]["smiles_column"],
        config["input"]["start_index"],
        config["input"]["num_smiles"],
    )

    if not smilesList:
        print("Error: No valid SMILES found in input file")
        sys.exit(1)

    print(f"Loaded {len(smilesList)} SMILES")

    try:
        resolvedMaxAtoms, maxStarterCounts, invalidSmiles = resolveMaxAtoms(smilesList, config)
    except ValueError as exc:
        print(f"Error while resolving max_atoms: {exc}")
        sys.exit(1)

    config["resolved_max_atoms"] = resolvedMaxAtoms
    printConfig(config, maxStarterCounts=maxStarterCounts)

    if invalidSmiles:
        print(f"Warning: {len(invalidSmiles)} starter SMILES could not be parsed by RDKit for max_atoms calculation.")
        print("They will still be sent to DORAnet, but they were excluded from the max_atoms baseline.")
        for smi in invalidSmiles[:10]:
            print(f"  Invalid for atom counting: {smi}")
        if len(invalidSmiles) > 10:
            print(f"  ... and {len(invalidSmiles) - 10} more")

    baseOutputDir = Path(config["output"]["directory"])
    baseOutputDir.mkdir(parents=True, exist_ok=True)

    allResults = []
    allStartTime = time.time()

    for gen in config["network"]["generations"]:
        genConfig = copy.deepcopy(config)
        genConfig["network"]["active_generation"] = int(gen)

        genOutputDir = makeGenerationOutputDir(
            baseOutputDir,
            config["output"].get("generation_suffix_template", "_gen{gen}"),
            gen,
        )
        genConfig["output"]["directory"] = str(genOutputDir)
        genOutputDir.mkdir(parents=True, exist_ok=True)

        configCopy = saveConfigCopy(genConfig, genOutputDir)
        print(f"\nConfiguration for gen {gen} saved to: {configCopy}")

        genStartTime = time.time()
        results = runParallelProcessing(smilesList, genConfig)
        genTotalTime = time.time() - genStartTime

        summaryPath = saveSummary(results, genOutputDir)
        print(f"\nSummary for gen {gen} saved to: {summaryPath}")
        printSummary(results, genTotalTime)

        allResults.extend(results)

    allTotalTime = time.time() - allStartTime
    if config["output"].get("write_combined_summary", True):
        combinedPath = saveCombinedSummary(allResults, baseOutputDir)
        print(f"\nCombined summary saved to: {combinedPath}")

    print("\n" + "-" * 70)
    print("ALL GENERATIONS COMPLETE")
    print("-" * 70)
    print(f"Generations run: {config['network']['generations']}")
    print(f"Resolved max_atoms used for every generation: {resolvedMaxAtoms}")
    print(f"Total generation-starter jobs: {len(allResults)}")
    print(f"Total wall time across all generations: {allTotalTime:.2f}s")


if __name__ == "__main__":
    main()
