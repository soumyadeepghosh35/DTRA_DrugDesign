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

doranetPath = Path(r"/users/sghosh6/DTRA_project/MACAW/doranet")
sys.path.insert(0, str(doranetPath))

import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as postProcessing

jobName = "high_pPotency_molecule_pathway3_wGen1"
starters = {'C1=NC2=C(C(=O)N1)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O'}
helpers = {'O', 'N', 'O=[N+]([O-])O', 'O=S(O)O', 'N#CO', 'O=NO', 'C=O', 'C#N', 'N#N', '[H][H]', 'O=S(=O)(O)O', 'Br', '[Br][Br]', 'O=O', '[C-]#[O+]', 'O=S=O', 'CO', 'S', 'C=C', 'O=C=O', 'NO'}
target = {"Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n3cnc4c(N)ncnc43)[C@H](O)[C@@H]1O)c1[nH]c(=O)[nH]c(=O)c1N2"}
maxAtoms = {'C': 15, 'N': 6, 'O': 8, 'S': 3}
generations = 1
ruleset = "JN3604IMT"
searchDepth = 1
maxNumRxns = 1
minRxnAtomEconomy = 0.0
filterExactNumRxns = True
exactNumRxns = 1
runRanking = True
runVisualization = False


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
    pathwaysPath = Path(f"{jobName}_pathways.txt")
    backupPath = Path(f"{jobName}_pathways_unfiltered.txt")
    exactPath = Path(f"{jobName}_pathways_exact{exactSteps}.txt")
    if not pathwaysPath.exists():
        return 0, 0
    blocks = splitPathwayBlocks(pathwaysPath)
    exactBlocks = [b for b in blocks if countPathwayBlockSteps(b) == exactSteps]
    if not backupPath.exists():
        shutil.copy2(pathwaysPath, backupPath)
    exactText = "

".join("
".join(block) for block in exactBlocks)
    if exactText:
        exactText += "
"
    exactPath.write_text(exactText, encoding="utf-8")
    pathwaysPath.write_text(exactText, encoding="utf-8")
    return len(blocks), len(exactBlocks)


t0 = time.time()
network = enzymatic.generate_network(
    job_name=jobName,
    starters=starters,
    gen=generations,
    max_atoms=maxAtoms,
    direction="forward",
    targets=target,
    ruleset=ruleset,
)

smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]
with Path(f"{jobName}_molecules.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES", "isStarter", "molFormula", "molWeight", "numHeavyAtoms"])
    for s in smilesList:
        formula, weight, heavyAtoms = getSmilesProps(s)
        writer.writerow([s, s in starters, formula, weight, heavyAtoms])

# Important: target=target, not all generated molecules.
postProcessing.pretreat_networks(
    networks={network},
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
    print(f"Exact-step filter Gen{generations}: {exact}/{original} pathways retained")
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

print(f"Done {jobName} in {time.time() - t0:.2f} s")
