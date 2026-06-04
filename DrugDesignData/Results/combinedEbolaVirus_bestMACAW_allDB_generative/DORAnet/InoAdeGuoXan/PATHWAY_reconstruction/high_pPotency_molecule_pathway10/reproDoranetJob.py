#!/usr/bin/env python3
import os, sys, csv, time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

doranetPath = Path(r"/users/sghosh6/DTRA_project/MACAW/doranet")
sys.path.insert(0, str(doranetPath))
import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as postProcessing

jobName = "high_pPotency_molecule_pathway10"
starters = {'C1=NC2=C(C(=O)N1)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O'}
helpers = {'O=NO', 'CO', '[C-]#[O+]', 'S', 'O=S(=O)(O)O', '[H][H]', 'O=S=O', 'O', 'C=O', 'C#N', 'O=[N+]([O-])O', 'N#CO', 'O=O', 'Br', 'NO', 'N', '[Br][Br]', 'O=S(O)O', 'C=C', 'N#N', 'O=C=O'}
target = {"O=C(O)CC(=O)OP(=O)(O)O"}
maxAtoms = {'C': 15, 'N': 6, 'O': 8, 'S': 3}
generations = 3
ruleset = "JN3604IMT"

t0 = time.time()
network = enzymatic.generate_network(
    job_name=jobName, starters=starters, gen=generations, max_atoms=maxAtoms,
    direction="forward", targets=target, ruleset=ruleset
)

smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]
allTargets = set(smilesList) - starters - helpers

outPath = Path(f"{jobName}_molecules.csv")
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
        networks={network}, total_generations=generations, starters=starters,
        helpers=helpers, target=allTargets, job_name=jobName
    )

print(f"Done {jobName} in {time.time()-t0:.2f} s")
