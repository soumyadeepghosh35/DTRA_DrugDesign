#!/usr/bin/env python3
"""
Run DORAnet pathway jobs from YAML config for multiple generations.

Example:
    python doranet_pathways_target_multi_gen.py config_doranet_target_multi_gen.yaml

Key behavior:
    1. Searches pathways specifically to the user-requested target, not all generated molecules.
    2. Runs separate jobs for each generation listed in generationsToRun, e.g. [1, 2, 3].
    3. Adds a suffix to each job/output folder, e.g. _wGen1, _wGen2, _wGen3.
    4. Uses search_depth = generation and max_num_rxns = generation by default.
    5. Optionally filters DORAnet's <=N-step pathways to exactly N-step pathways.
"""

from __future__ import annotations

import ast
import csv
import os
import shutil
import sys
import time
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


SCRIPT_VERSION = "doranet_pathways_target_multi_gen_v1"


def setThreadEnv() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def loadYaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def asList(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def loadTargetSmilesFromCsv(csvPath: Path, columnName: str | None = "SMILES") -> list[str]:
    df = pd.read_csv(csvPath, dtype=str)
    series = df[columnName] if (columnName and columnName in df.columns) else df.iloc[:, 0]
    smiles = [s.strip() for s in series.dropna().astype(str) if s.strip()]
    seen: set[str] = set()
    uniqueSmiles: list[str] = []
    for s in smiles:
        if s not in seen:
            seen.add(s)
            uniqueSmiles.append(s)
    return uniqueSmiles


def loadTargets(config: dict[str, Any]) -> list[str]:
    """Load user-requested target molecules from config.

    Priority:
      1. targetSmiles: "..."
      2. targetSmilesList: ["...", "..."]
      3. targetsCsvPath + targetsColumn
    """
    if config.get("targetSmiles"):
        return [str(config["targetSmiles"]).strip()]

    if config.get("targetSmilesList"):
        targets = [str(x).strip() for x in asList(config["targetSmilesList"]) if str(x).strip()]
        return list(dict.fromkeys(targets))

    if config.get("targetsCsvPath"):
        return loadTargetSmilesFromCsv(
            Path(config["targetsCsvPath"]).expanduser(),
            config.get("targetsColumn", "SMILES"),
        )

    raise ValueError("Provide one of: targetSmiles, targetSmilesList, or targetsCsvPath in config.")


def resolveGenerationRuns(config: dict[str, Any]) -> list[int]:
    raw = config.get("generationsToRun", None)
    if raw is None:
        raw = [config.get("generations", 3)]
    if isinstance(raw, int):
        raw = [raw]
    generations: list[int] = []
    for g in raw:
        gi = int(g)
        if gi <= 0:
            raise ValueError(f"Generation values must be positive integers, got {gi}")
        generations.append(gi)
    return sorted(set(generations))


def getSmilesProps(smiles: str) -> tuple[str, float, int]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "N/A", 0.0, 0
    return (
        rdMolDescriptors.CalcMolFormula(mol),
        round(rdMolDescriptors.CalcExactMolWt(mol), 4),
        mol.GetNumHeavyAtoms(),
    )


def canonicalSmiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def countPathwayBlockSteps(blockLines: list[str]) -> int | None:
    """Count number of reaction steps in one DORAnet *_pathways.txt block.

    DORAnet writes a line like:
        reaction SMILES stoichiometry ['...', '...', '...']
    The list length equals the number of reactions in the pathway.
    """
    prefix = "reaction SMILES stoichiometry "
    for line in blockLines:
        if line.startswith(prefix):
            payload = line[len(prefix):].strip()
            try:
                parsed = ast.literal_eval(payload)
                if isinstance(parsed, list):
                    return len(parsed)
            except Exception:
                return None
    return None


def splitPathwayBlocks(pathwaysTxtPath: Path) -> list[list[str]]:
    if not pathwaysTxtPath.exists():
        return []
    lines = pathwaysTxtPath.read_text(encoding="utf-8").splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("pathway number ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def filterPathwaysTxtByExactSteps(jobName: str, exactSteps: int) -> dict[str, Any]:
    """Overwrite {jobName}_pathways.txt with exactly N-step pathway blocks.

    A backup is saved as {jobName}_pathways_unfiltered.txt.
    """
    pathwaysPath = Path(f"{jobName}_pathways.txt")
    backupPath = Path(f"{jobName}_pathways_unfiltered.txt")
    exactPath = Path(f"{jobName}_pathways_exact{exactSteps}.txt")

    if not pathwaysPath.exists():
        return {
            "pathwaysTxtFound": False,
            "numPathwayBlocksOriginal": 0,
            "numPathwayBlocksExact": 0,
            "exactPathwaysTxt": str(exactPath),
        }

    blocks = splitPathwayBlocks(pathwaysPath)
    exactBlocks = [b for b in blocks if countPathwayBlockSteps(b) == exactSteps]

    if not backupPath.exists():
        shutil.copy2(pathwaysPath, backupPath)

    exactText = "\n\n".join("\n".join(block) for block in exactBlocks)
    if exactText:
        exactText += "\n"
    exactPath.write_text(exactText, encoding="utf-8")

    # Overwrite default pathway file so ranking uses only exact-N pathways.
    pathwaysPath.write_text(exactText, encoding="utf-8")

    return {
        "pathwaysTxtFound": True,
        "numPathwayBlocksOriginal": len(blocks),
        "numPathwayBlocksExact": len(exactBlocks),
        "exactPathwaysTxt": str(exactPath),
        "backupPathwaysTxt": str(backupPath),
    }


def maybeRunPostProcessing(
    network: Any,
    jobName: str,
    userTarget: set[str],
    starters: set[str],
    helpers: set[str],
    config: dict[str, Any],
    generationRun: int,
) -> dict[str, Any]:
    """Run DORAnet post-processing specifically to userTarget."""
    import doranet.modules.post_processing as postProcessing

    # For Gen1/2/3 jobs, default search depth and max route length equal the generation.
    searchDepth = int(config.get("searchDepth", generationRun))
    maxNumRxns = int(config.get("maxNumRxns", generationRun))

    if bool(config.get("searchDepthEqualsGeneration", True)):
        searchDepth = generationRun
    if bool(config.get("maxNumRxnsEqualsGeneration", True)):
        maxNumRxns = generationRun

    minRxnAtomEconomy = float(config.get("minRxnAtomEconomy", 0.0))

    removePureHelpersRxns = bool(config.get("removePureHelpersRxns", False))
    sanitize = bool(config.get("sanitize", True))
    transformEnolsFlag = bool(config.get("transformEnolsFlag", False))
    considerNameDifference = bool(config.get("considerNameDifference", True))

    runRanking = bool(config.get("runRanking", True))
    runVisualization = bool(config.get("runVisualization", False))
    numProcess = int(config.get("numProcess", 1))
    weights = config.get("weights", None)
    coolReactions = config.get("coolReactions", None)

    # 1. Pretreat network into {jobName}_network_pretreated.json.
    postProcessing.pretreat_networks(
        networks={network},
        total_generations=generationRun,
        starters=starters,
        helpers=helpers,
        job_name=jobName,
        remove_pure_helpers_rxns=removePureHelpersRxns,
        sanitize=sanitize,
        transform_enols_flag=transformEnolsFlag,
    )

    # 2. Search pathways specifically to userTarget, never allTargets.
    postProcessing.pathway_finder(
        starters=starters,
        helpers=helpers,
        target=userTarget,
        search_depth=searchDepth,
        max_num_rxns=maxNumRxns,
        min_rxn_atom_economy=minRxnAtomEconomy,
        job_name=jobName,
        consider_name_difference=considerNameDifference,
    )

    filterInfo: dict[str, Any] = {
        "pathwaysTxtFound": Path(f"{jobName}_pathways.txt").exists(),
        "numPathwayBlocksOriginal": None,
        "numPathwayBlocksExact": None,
    }

    # DORAnet max_num_rxns=N means up to N reactions. This optional filter keeps exactly N.
    if bool(config.get("filterExactNumRxns", True)):
        exactSteps = int(config.get("exactNumRxns", generationRun))
        if bool(config.get("exactNumRxnsEqualsGeneration", True)):
            exactSteps = generationRun
        filterInfo = filterPathwaysTxtByExactSteps(jobName, exactSteps)

    # 3. Optional ranking. If exact filtering was enabled, ranking uses filtered pathways.txt.
    if runRanking:
        postProcessing.pathway_ranking(
            starters=starters,
            helpers=helpers,
            target=userTarget,
            weights=weights,
            num_process=numProcess,
            reaxys_result_name=config.get("reaxysResultName", None),
            job_name=jobName,
            cool_reactions=coolReactions,
        )

    # 4. Optional visualization. This can be slow; default false.
    if runVisualization:
        postProcessing.pathway_visualization(
            starters=starters,
            helpers=helpers,
            num_process=numProcess,
            reaxys_result_name=config.get("reaxysResultName", "default"),
            job_name=jobName,
            exclude_smiles=config.get("excludeSmiles", None),
            reaxys_rxn_color=config.get("reaxysRxnColor", "blue"),
            normal_rxn_color=config.get("normalRxnColor", "black"),
        )

    return {
        "generationRun": generationRun,
        "searchDepth": searchDepth,
        "maxNumRxns": maxNumRxns,
        "minRxnAtomEconomy": minRxnAtomEconomy,
        "targetUsedForPathwaySearch": ";".join(sorted(userTarget)),
        "runRanking": runRanking,
        "runVisualization": runVisualization,
        **filterInfo,
    }


def saveReproScript(
    jobDir: Path,
    config: dict[str, Any],
    jobName: str,
    targetSmiles: str,
    generationRun: int,
) -> None:
    """Save a standalone reproducer for one target and one generation."""
    searchDepth = generationRun if bool(config.get("searchDepthEqualsGeneration", True)) else int(config.get("searchDepth", generationRun))
    maxNumRxns = generationRun if bool(config.get("maxNumRxnsEqualsGeneration", True)) else int(config.get("maxNumRxns", generationRun))
    exactNumRxns = generationRun if bool(config.get("exactNumRxnsEqualsGeneration", True)) else int(config.get("exactNumRxns", generationRun))

    scriptText = f'''\
#!/usr/bin/env python3
import ast
import csv
import os
import shutil
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

doranetPath = Path(r"{config['doranetPath']}")
sys.path.insert(0, str(doranetPath))

import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as postProcessing

jobName = "{jobName}"
starters = {repr(set(config['starters']))}
helpers = {repr(set(config['helpers']))}
target = {{"{targetSmiles}"}}
maxAtoms = {repr(config['maxAtoms'])}
generations = {generationRun}
ruleset = "{config['ruleset']}"
searchDepth = {searchDepth}
maxNumRxns = {maxNumRxns}
minRxnAtomEconomy = {float(config.get('minRxnAtomEconomy', 0.0))}
filterExactNumRxns = {bool(config.get('filterExactNumRxns', True))}
exactNumRxns = {exactNumRxns}
runRanking = {bool(config.get('runRanking', True))}
runVisualization = {bool(config.get('runVisualization', False))}


def getSmilesProps(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "N/A", 0.0, 0
    return rdMolDescriptors.CalcMolFormula(mol), round(rdMolDescriptors.CalcExactMolWt(mol), 4), mol.GetNumHeavyAtoms()


def countPathwayBlockSteps(blockLines):
    prefix = "reaction SMILES stoichiometry "
    for line in blockLines:
        if line.startswith(prefix):
            payload = line[len(prefix):].strip()
            try:
                parsed = ast.literal_eval(payload)
                if isinstance(parsed, list):
                    return len(parsed)
            except Exception:
                return None
    return None


def splitPathwayBlocks(pathwaysTxtPath):
    if not pathwaysTxtPath.exists():
        return []
    lines = pathwaysTxtPath.read_text(encoding="utf-8").splitlines()
    blocks, current = [], []
    for line in lines:
        if line.startswith("pathway number ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def filterPathwaysTxtByExactSteps(jobName, exactSteps):
    pathwaysPath = Path(f"{{jobName}}_pathways.txt")
    backupPath = Path(f"{{jobName}}_pathways_unfiltered.txt")
    exactPath = Path(f"{{jobName}}_pathways_exact{{exactSteps}}.txt")
    if not pathwaysPath.exists():
        return 0, 0
    blocks = splitPathwayBlocks(pathwaysPath)
    exactBlocks = [b for b in blocks if countPathwayBlockSteps(b) == exactSteps]
    if not backupPath.exists():
        shutil.copy2(pathwaysPath, backupPath)
    exactText = "\n\n".join("\n".join(block) for block in exactBlocks)
    if exactText:
        exactText += "\n"
    exactPath.write_text(exactText, encoding="utf-8")
    pathwaysPath.write_text(exactText, encoding="utf-8")
    return len(blocks), len(exactBlocks)


t0 = time.time()
network = enzymatic.generate_network(
    job_name=jobName,
    starters=starters,
    gen=generations,
    max_atoms=maxAtoms,
    direction="{config.get('direction', 'forward')}",
    targets=target,
    ruleset=ruleset,
)

smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]
with Path(f"{{jobName}}_molecules.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES", "isStarter", "molFormula", "molWeight", "numHeavyAtoms"])
    for s in smilesList:
        formula, weight, heavyAtoms = getSmilesProps(s)
        writer.writerow([s, s in starters, formula, weight, heavyAtoms])

# Important: target=target, not all generated molecules.
postProcessing.pretreat_networks(
    networks={{network}},
    total_generations=generations,
    starters=starters,
    helpers=helpers,
    job_name=jobName,
)
postProcessing.pathway_finder(
    starters=starters,
    helpers=helpers,
    target=target,
    search_depth=searchDepth,
    max_num_rxns=maxNumRxns,
    min_rxn_atom_economy=minRxnAtomEconomy,
    job_name=jobName,
)
if filterExactNumRxns:
    original, exact = filterPathwaysTxtByExactSteps(jobName, exactNumRxns)
    print(f"Exact-step filter Gen{{generations}}: {{exact}}/{{original}} pathways retained")
if runRanking:
    postProcessing.pathway_ranking(
        starters=starters,
        helpers=helpers,
        target=target,
        job_name=jobName,
        num_process=1,
    )
if runVisualization:
    postProcessing.pathway_visualization(
        starters=starters,
        helpers=helpers,
        job_name=jobName,
        num_process=1,
    )

print(f"Done {{jobName}} in {{time.time() - t0:.2f}} s")
'''
    scriptPath = jobDir / "reproDoranetJob.py"
    scriptPath.write_text(textwrap.dedent(scriptText), encoding="utf-8")
    scriptPath.chmod(0o755)


def runOneJob(targetIndex: int, targetSmiles: str, generationRun: int, config: dict[str, Any], outputDir: Path) -> dict[str, Any]:
    startTime = time.time()
    genSuffix = str(config.get("generationSuffixTemplate", "_wGen{generation}")).format(generation=generationRun)
    jobName = f"{config['baseJobPrefix']}{targetIndex}{genSuffix}"
    jobDir = outputDir / jobName
    jobDir.mkdir(parents=True, exist_ok=True)
    saveReproScript(jobDir, config, jobName, targetSmiles, generationRun)

    oldCwd = Path.cwd()
    os.chdir(jobDir)

    try:
        doranetPath = Path(config["doranetPath"]).expanduser().resolve()
        sys.path.insert(0, str(doranetPath))
        import doranet.modules.enzymatic as enzymatic

        starters = set(config["starters"])
        helpers = set(config["helpers"])
        userTarget = {targetSmiles}

        network = enzymatic.generate_network(
            job_name=jobName,
            starters=starters,
            gen=generationRun,
            max_atoms=config["maxAtoms"],
            direction=config.get("direction", "forward"),
            targets=userTarget,
            ruleset=config["ruleset"],
        )

        smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]

        with Path(f"{jobName}_molecules.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "isStarter", "molFormula", "molWeight", "numHeavyAtoms"])
            for s in smilesList:
                formula, weight, heavyAtoms = getSmilesProps(s)
                writer.writerow([s, s in starters, formula, weight, heavyAtoms])

        targetCanonical = canonicalSmiles(targetSmiles)
        moleculeCanonicals = {canonicalSmiles(s) for s in smilesList}
        targetInGeneratedMolecules = targetCanonical in moleculeCanonicals if targetCanonical else False

        postInfo = maybeRunPostProcessing(network, jobName, userTarget, starters, helpers, config, generationRun)

        return {
            "jobName": jobName,
            "jobDir": str(jobDir),
            "targetIndex": targetIndex,
            "targetSmiles": targetSmiles,
            "targetCanonical": targetCanonical,
            "generationRun": generationRun,
            "targetInGeneratedMolecules": targetInGeneratedMolecules,
            "numMolecules": len(smilesList),
            "seconds": round(time.time() - startTime, 2),
            "status": "ok",
            **postInfo,
        }

    except Exception as exc:
        return {
            "jobName": jobName,
            "jobDir": str(jobDir),
            "targetIndex": targetIndex,
            "targetSmiles": targetSmiles,
            "generationRun": generationRun,
            "seconds": round(time.time() - startTime, 2),
            "status": "fail",
            "error": str(exc),
        }

    finally:
        os.chdir(oldCwd)


def validateConfig(config: dict[str, Any]) -> None:
    requiredKeys = [
        "doranetPath",
        "baseJobPrefix",
        "numParallelJobs",
        "starters",
        "helpers",
        "maxAtoms",
        "ruleset",
        "summaryCsv",
    ]
    missing = [k for k in requiredKeys if k not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    generations = resolveGenerationRuns(config)
    if not generations:
        raise ValueError("No generations to run. Set generationsToRun, e.g. [1, 2, 3].")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python doranet_pathways_target_multi_gen.py config_doranet_target_multi_gen.yaml")
        sys.exit(1)

    setThreadEnv()
    mainStart = time.time()
    configPath = Path(sys.argv[1]).expanduser().resolve()
    config = loadYaml(configPath)
    validateConfig(config)

    outputDir = Path(config.get("outputDir", ".")).expanduser().resolve()
    outputDir.mkdir(parents=True, exist_ok=True)

    targets = loadTargets(config)
    generations = resolveGenerationRuns(config)

    targetsToTest = config.get("targetsToTest", None)
    if targetsToTest is None:
        selectedTargets = list(enumerate(targets, start=1))
    else:
        keep = set(int(x) for x in targetsToTest)
        selectedTargets = [(i, s) for i, s in enumerate(targets, start=1) if i in keep]

    workItems = [
        (targetIndex, targetSmiles, generationRun)
        for targetIndex, targetSmiles in selectedTargets
        for generationRun in generations
    ]

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Config: {configPath}")
    print(f"Output dir: {outputDir}")
    print(f"Loaded {len(targets)} requested targets; selected {len(selectedTargets)} targets")
    print(f"Generations to run: {generations}")
    print(f"Total DORAnet jobs: {len(workItems)} with nJobs={config['numParallelJobs']}")
    print("Pathway target mode: userTarget only, never allTargets")
    print("Job suffix template:", config.get("generationSuffixTemplate", "_wGen{generation}"))

    results = Parallel(n_jobs=int(config["numParallelJobs"]), backend="loky", verbose=10)(
        delayed(runOneJob)(targetIndex, targetSmiles, generationRun, config, outputDir)
        for targetIndex, targetSmiles, generationRun in workItems
    )

    resultsDf = pd.DataFrame(results)
    summaryPath = outputDir / config["summaryCsv"]
    resultsDf.to_csv(summaryPath, index=False)

    print(f"Saved summary: {summaryPath}")
    if "status" in resultsDf.columns:
        print(resultsDf["status"].value_counts(dropna=False))
    if {"generationRun", "status"}.issubset(resultsDf.columns):
        print("\nStatus by generation:")
        print(resultsDf.groupby(["generationRun", "status"]).size().unstack(fill_value=0).to_string())
    if {"generationRun", "numPathwayBlocksExact"}.issubset(resultsDf.columns):
        print("\nExact pathway blocks by generation:")
        print(resultsDf.groupby("generationRun")["numPathwayBlocksExact"].sum().to_string())
    print(f"Total time: {time.time() - mainStart:.2f} s")


if __name__ == "__main__":
    main()
