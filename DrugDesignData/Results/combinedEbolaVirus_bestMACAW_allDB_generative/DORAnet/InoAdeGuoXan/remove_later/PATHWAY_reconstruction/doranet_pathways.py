#!/usr/bin/env python3
"""
Run DORAnet pathway jobs from YAML config:
    python doranet_pathways.py config.yaml
"""

from __future__ import annotations
import os
import sys
import csv
import time
import textwrap
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def setThreadEnv() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def loadYaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def loadTargetSmiles(csvPath: Path, columnName: str | None = "SMILES") -> list[str]:
    df = pd.read_csv(csvPath, dtype=str)
    series = df[columnName] if (columnName and columnName in df.columns) else df.iloc[:, 0]
    smiles = [s.strip() for s in series.dropna().astype(str) if s.strip()]
    seen, uniqueSmiles = set(), []
    for s in smiles:
        if s not in seen:
            seen.add(s)
            uniqueSmiles.append(s)
    return uniqueSmiles


def getSmilesProps(smiles: str) -> tuple[str, float, int]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "N/A", 0.0, 0
    return (
        rdMolDescriptors.CalcMolFormula(mol),
        round(rdMolDescriptors.CalcExactMolWt(mol), 4),
        mol.GetNumHeavyAtoms(),
    )


def saveReproScript(jobDir: Path, config: dict, jobName: str, targetSmiles: str) -> None:
    scriptText = f"""\
#!/usr/bin/env python3
import os, sys, csv, time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

doranetPath = Path(r"{config['doranetPath']}")
sys.path.insert(0, str(doranetPath))
import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as postProcessing

jobName = "{jobName}"
starters = {set(config["starters"])}
helpers = {set(config["helpers"])}
target = {{"{targetSmiles}"}}
maxAtoms = {config["maxAtoms"]}
generations = {config["generations"]}
ruleset = "{config["ruleset"]}"

t0 = time.time()
network = enzymatic.generate_network(
    job_name=jobName, starters=starters, gen=generations, max_atoms=maxAtoms,
    direction="forward", targets=target, ruleset=ruleset
)

smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]
allTargets = set(smilesList) - starters - helpers

outPath = Path(f"{{jobName}}_molecules.csv")
with outPath.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["SMILES","isStarter","molFormula","molWeight","numHeavyAtoms"])
    for s in smilesList:
        mol = Chem.MolFromSmiles(s)
        if mol:
            w.writerow([s, s in starters, rdMolDescriptors.CalcMolFormula(mol),
                        round(rdMolDescriptors.CalcExactMolWt(mol),4), mol.GetNumHeavyAtoms()])
        else:
            w.writerow([s, s in starters, "N/A", 0, 0])

if allTargets:
    postProcessing.one_step(
        networks={{network}}, total_generations=generations, starters=starters,
        helpers=helpers, target=allTargets, job_name=jobName
    )

print(f"Done {{jobName}} in {{time.time()-t0:.2f}} s")
"""
    scriptPath = jobDir / "reproDoranetJob.py"
    scriptPath.write_text(textwrap.dedent(scriptText))
    scriptPath.chmod(0o755)


def runOneJob(jobIndex: int, targetSmiles: str, config: dict, outputDir: Path) -> dict:
    startTime = time.time()
    jobName = f"{config['baseJobPrefix']}{jobIndex}"
    jobDir = outputDir / jobName
    jobDir.mkdir(parents=True, exist_ok=True)
    saveReproScript(jobDir, config, jobName, targetSmiles)

    oldCwd = Path.cwd()
    os.chdir(jobDir)

    try:
        sys.path.insert(0, str(Path(config["doranetPath"])))
        import doranet.modules.enzymatic as enzymatic
        import doranet.modules.post_processing as postProcessing

        starters, helpers = set(config["starters"]), set(config["helpers"])
        userTarget = {targetSmiles}

        network = enzymatic.generate_network(
            job_name=jobName,
            starters=starters,
            gen=config["generations"],
            max_atoms=config["maxAtoms"],
            direction="forward",
            targets=userTarget,
            ruleset=config["ruleset"],
        )

        smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]
        allTargets = set(smilesList) - starters - helpers

        with Path(f"{jobName}_molecules.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "isStarter", "molFormula", "molWeight", "numHeavyAtoms"])
            for s in smilesList:
                formula, weight, heavyAtoms = getSmilesProps(s)
                writer.writerow([s, s in starters, formula, weight, heavyAtoms])

        if allTargets:
            postProcessing.one_step(
                networks={network},
                total_generations=config["generations"],
                starters=starters,
                helpers=helpers,
                target=allTargets,
                job_name=jobName,
            )

        return {
            "jobName": jobName,
            "jobDir": str(jobDir),
            "targetSmiles": targetSmiles,
            "numMolecules": len(smilesList),
            "numPathwayTargets": len(allTargets),
            "seconds": round(time.time() - startTime, 2),
            "status": "ok",
        }

    except Exception as e:
        return {
            "jobName": jobName,
            "jobDir": str(jobDir),
            "targetSmiles": targetSmiles,
            "seconds": round(time.time() - startTime, 2),
            "status": "fail",
            "error": str(e),
        }

    finally:
        os.chdir(oldCwd)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python doranetPathways.py config.yaml")
        sys.exit(1)

    setThreadEnv()
    mainStart = time.time()
    config = loadYaml(Path(sys.argv[1]))

    requiredKeys = [
        "doranetPath", "targetsCsvPath", "baseJobPrefix", "numParallelJobs",
        "starters", "helpers", "maxAtoms", "generations", "ruleset", "summaryCsv"
    ]
    missingKeys = [k for k in requiredKeys if k not in config]
    if missingKeys:
        raise ValueError(f"Missing config keys: {missingKeys}")

    outputDir = Path(config.get("outputDir", ".")).resolve()
    outputDir.mkdir(parents=True, exist_ok=True)

    targets = loadTargetSmiles(Path(config["targetsCsvPath"]), config.get("targetsColumn", "SMILES"))
    targetsToTest = config.get("targetsToTest", None)
    selected = list(enumerate(targets, start=1)) if targetsToTest is None else [
        (i, s) for i, s in enumerate(targets, start=1) if i in set(targetsToTest)
    ]

    print(f"Loaded {len(targets)} targets; running {len(selected)} with nJobs={config['numParallelJobs']}")

    results = Parallel(n_jobs=config["numParallelJobs"], backend="loky", verbose=10)(
        delayed(runOneJob)(i, s, config, outputDir) for i, s in selected
    )

    resultsDf = pd.DataFrame(results)
    summaryPath = outputDir / config["summaryCsv"]
    resultsDf.to_csv(summaryPath, index=False)

    print(f"Saved summary: {summaryPath}")
    print(resultsDf["status"].value_counts(dropna=False))
    print(f"Total time: {time.time() - mainStart:.2f} s")


if __name__ == "__main__":
    main()