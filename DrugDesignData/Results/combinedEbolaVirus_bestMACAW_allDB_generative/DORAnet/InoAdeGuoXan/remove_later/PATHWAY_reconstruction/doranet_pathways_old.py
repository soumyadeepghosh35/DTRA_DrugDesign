#!/usr/bin/env python

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import csv
from pathlib import Path
import textwrap

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from joblib import Parallel, delayed


# -------------------------
# User settings
# -------------------------
DORANET_PATH = Path("/users/sghosh6/DTRA_project/MACAW/doranet")
TARGETS_CSV_PATH = Path("../../../combinedEbola_allDORAnetGenerated_Antivirals_wARTprediction_top20.csv")

BASE_JOB_PREFIX = "high_pPotency_molecule_pathway"
NUM_PARALLEL_JOBS = 8

# If None -> run all targets
# If list -> run only these indices (e.g., [1,2,3,4] runs 4 compounds)
TARGETS_TO_TEST = 4

# Starter molecule stays constant
USER_STARTERS = {
    "C1=NC2=C(C(=O)N1)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O"
}

# Helpers stay constant
USER_HELPERS = {
    'O', 'O=O', '[H][H]', 'O=C=O', 'C=O', '[C-]#[O+]', 'Br', '[Br][Br]', 'CO',
    'C=C', 'O=S(O)O', 'N', 'O=S(=O)(O)O', 'O=NO', 'N#N', 'O=[N+]([O-])O', 'NO',
    'C#N', 'S', 'O=S=O', 'N#CO'
}

# Your fixed constraints
MAX_ATOMS = {'C': 15, 'N': 6, 'O': 8, 'S': 3}
GENERATIONS = 3
RULESET = "JN3604IMT"


# -------------------------
# Helpers
# -------------------------
def loadTargetSmilesList(csvPath: Path) -> list[str]:
    DF = pd.read_csv(csvPath, dtype=str)

    if "SMILES" in DF.columns:
        series = DF["SMILES"]
    else:
        series = DF.iloc[:, 0]

    smilesList = (
        series.dropna()
        .astype(str)
        .str.strip()
    )
    smilesList = [s for s in smilesList if s != ""]

    # unique while preserving order
    seen = set()
    uniqueSmiles = []
    for s in smilesList:
        if s not in seen:
            seen.add(s)
            uniqueSmiles.append(s)
    return uniqueSmiles


def savePerJobInputScript(jobDir: Path, jobName: str, targetSmiles: str) -> None:
    """
    Save a runnable per-job script into the job folder.
    This captures the exact inputs (starter, helpers, target, constraints).
    """
    scriptText = f"""\
#!/usr/bin/env python
import os
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import csv

# Restrict threading per process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DORANET_PATH = Path(r"{str(DORANET_PATH)}")
sys.path.insert(0, str(DORANET_PATH))

import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as post_processing

start_time = time.time()

job_name = "{jobName}"

user_starters = {USER_STARTERS}

user_helpers = {USER_HELPERS}

user_target = {{"{targetSmiles}"}}

max_atoms = {MAX_ATOMS}
generations = {GENERATIONS}
ruleset = "{RULESET}"

print(f"Starter: {{list(user_starters)[0]}}")
print(f"Target:  {{list(user_target)[0]}}")
print(f"Job:     {{job_name}}")

forward_network = enzymatic.generate_network(
    job_name=job_name,
    starters=user_starters,
    gen=generations,
    max_atoms=max_atoms,
    direction="forward",
    targets=user_target,
    ruleset=ruleset
)

smiles_list = list(user_starters) + [mol.uid for mol in forward_network.mols if mol.uid not in user_starters]
print(f"Generated {{len(smiles_list) - len(user_starters)}} new molecules + {{len(user_starters)}} starters")

all_targets = set(smiles_list) - user_starters - user_helpers
print(f"Using {{len(all_targets)}} generated molecules as targets for pathway finding")

output_path = Path(f"{{job_name}}_molecules.csv")
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['SMILES', 'Is_Starter', 'MolFormula', 'MolWeight', 'NumHeavyAtoms'])
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mol_weight = round(rdMolDescriptors.CalcExactMolWt(mol), 4)
            num_heavy = mol.GetNumHeavyAtoms()
        else:
            formula, mol_weight, num_heavy = "N/A", 0, 0
        writer.writerow([smi, smi in user_starters, formula, mol_weight, num_heavy])

print(f"Saved molecules to {{output_path}}")

if all_targets:
    post_processing.one_step(
        networks={{forward_network}},
        total_generations=generations,
        starters=user_starters,
        helpers=user_helpers,
        target=all_targets,
        job_name=job_name,
    )
    print(f"Pathway files generated with prefix: {{job_name}}")
else:
    print("No targets found for pathway generation")

print(f"Time: {{time.time() - start_time:.2f}} s")
"""
    scriptPath = jobDir / "doranet_pathways.py"
    scriptPath.write_text(textwrap.dedent(scriptText))
    scriptPath.chmod(0o755)


def runOneTargetJob(jobIndex: int, targetSmiles: str, baseOutDir: Path) -> dict:
    startTime = time.time()

    jobName = f"{BASE_JOB_PREFIX}{jobIndex}"
    jobDir = baseOutDir / jobName
    jobDir.mkdir(parents=True, exist_ok=True)

    # Save the per-job "input script" into this folder
    savePerJobInputScript(jobDir, jobName, targetSmiles)

    # Imports inside worker for joblib processes
    sys.path.insert(0, str(DORANET_PATH))
    import doranet.modules.enzymatic as enzymatic
    import doranet.modules.post_processing as post_processing

    oldCwd = Path.cwd()
    os.chdir(jobDir)

    try:
        userTarget = {targetSmiles}
        print(f"[START] {jobName}  target={targetSmiles}", flush=True)

        forwardNetwork = enzymatic.generate_network(
            job_name=jobName,
            starters=USER_STARTERS,
            gen=GENERATIONS,
            max_atoms=MAX_ATOMS,
            direction="forward",
            targets=userTarget,
            ruleset=RULESET
        )

        smilesList = list(USER_STARTERS) + [
            mol.uid for mol in forwardNetwork.mols if mol.uid not in USER_STARTERS
        ]

        allTargets = set(smilesList) - USER_STARTERS - USER_HELPERS

        outputPath = Path(f"{jobName}_molecules.csv")
        with open(outputPath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES", "Is_Starter", "MolFormula", "MolWeight", "NumHeavyAtoms"])
            for smi in smilesList:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    molWeight = round(rdMolDescriptors.CalcExactMolWt(mol), 4)
                    numHeavy = mol.GetNumHeavyAtoms()
                else:
                    formula, molWeight, numHeavy = "N/A", 0, 0
                writer.writerow([smi, smi in USER_STARTERS, formula, molWeight, numHeavy])

        if allTargets:
            post_processing.one_step(
                networks={forwardNetwork},
                total_generations=GENERATIONS,
                starters=USER_STARTERS,
                helpers=USER_HELPERS,
                target=allTargets,
                job_name=jobName,
            )

        elapsed = time.time() - startTime
        print(f"[DONE]  {jobName}  genMols={len(smilesList)}  pathwayTargets={len(allTargets)}  time={elapsed:.2f}s", flush=True)

        return {
            "jobName": jobName,
            "jobDir": str(jobDir),
            "targetSmiles": targetSmiles,
            "numMolecules": len(smilesList),
            "numPathwayTargets": len(allTargets),
            "seconds": elapsed,
            "status": "ok",
        }

    except Exception as e:
        elapsed = time.time() - startTime
        print(f"[FAIL]  {jobName}  time={elapsed:.2f}s  error={e}", flush=True)
        return {
            "jobName": jobName,
            "jobDir": str(jobDir),
            "targetSmiles": targetSmiles,
            "seconds": elapsed,
            "status": "fail",
            "error": str(e),
        }

    finally:
        os.chdir(oldCwd)


def main():
    startTimeMain = time.time()

    baseOutDir = Path.cwd()
    targets = loadTargetSmilesList(TARGETS_CSV_PATH)

    # Choose subset if requested
    if TARGETS_TO_TEST is None:
        selected = list(enumerate(targets, start=1))  # (1-based index, smiles)
    else:
        wanted = set(TARGETS_TO_TEST)
        selected = [(i, s) for i, s in enumerate(targets, start=1) if i in wanted]

    print(f"Loaded {len(targets)} targets from {TARGETS_CSV_PATH}", flush=True)
    print(f"Selected {len(selected)} targets for processing", flush=True)
    print(f"Running with {NUM_PARALLEL_JOBS} parallel jobs", flush=True)

    results = Parallel(n_jobs=NUM_PARALLEL_JOBS, backend="loky", verbose=10)(
        delayed(runOneTargetJob)(jobIndex=i, targetSmiles=targetSmiles, baseOutDir=baseOutDir)
        for i, targetSmiles in selected
    )

    resultsDF = pd.DataFrame(results)
    resultsDF.to_csv(baseOutDir / "Ebola_High_pPotency_jobs_summary.csv", index=False)
    print("\nSaved summary to Ebola_High_pPotency_jobs_summary.csv", flush=True)
    print(resultsDF["status"].value_counts(dropna=False), flush=True)

    print(f"Time: {time.time() - startTimeMain:.2f} s", flush=True)


if __name__ == "__main__":
    main()
