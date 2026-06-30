#!/usr/bin/env python3
import ast, csv, importlib, os, shutil, sys, time
from pathlib import Path
from typing import Callable

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

sys.path.insert(0, r"/users/sghosh6/DTRA_project/MACAW/doranet")
import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as postProcessing

jobName = "high_pPotency_molecule_pathway5_wGen3"
starters = {'C1=NC2=C(C(=O)N1)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O'}
helpers = {'O=S(O)O', 'CO', '[C-]#[O+]', 'C=C', '[H][H]', 'O=S=O', 'O=O', 'Br', 'O=[N+]([O-])O', 'C=O', 'NO', 'O=NO', 'O=S(=O)(O)O', 'C#N', 'S', '[Br][Br]', 'N#N', 'N', 'O=C=O', 'N#CO', 'O'}
target = {"CC(=O)OP(=O)(O)OC(C)=O"}
maxAtoms = {'C': 15, 'N': 6, 'O': 8, 'S': 3}
generations = 3
ruleset = "JN3604IMT"
searchDepth = 3
maxNumRxns = 3
minRxnAtomEconomy = 0.0
filterExactNumRxns = True
exactNumRxns = 3
runRanking = True
runVisualization = False

thermoBackend = "pathermo"
thermoCalculatorModule = ""
thermoCalculatorFunction = ""
thermoCalculatorPath = ""
thermoRequired = True
maxRxnThermoChange = 15.0
transformEnolsFlag = False


def getSmilesProps(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "N/A", 0.0, 0
    return rdMolDescriptors.CalcMolFormula(mol), round(rdMolDescriptors.CalcExactMolWt(mol), 4), mol.GetNumHeavyAtoms()


def countPathwayBlockSteps(blockLines):
    prefix = "reaction SMILES stoichiometry "
    for line in blockLines:
        if line.startswith(prefix):
            try:
                parsed = ast.literal_eval(line[len(prefix):].strip())
                return len(parsed) if isinstance(parsed, list) else None
            except Exception:
                return None
    return None


def splitPathwayBlocks(pathwaysTxtPath):
    if not pathwaysTxtPath.exists():
        return []
    blocks, current = [], []
    for line in pathwaysTxtPath.read_text(encoding="utf-8").splitlines():
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
    exactText = "\n\n".join("\n".join(block) for block in exactBlocks)
    exactText = exactText + "\n" if exactText else exactText
    exactPath.write_text(exactText, encoding="utf-8")
    pathwaysPath.write_text(exactText, encoding="utf-8")
    return len(blocks), len(exactBlocks)


def buildJobackCalculator():
    from thermo.group_contribution.joback import Joback

    knownHfKcalPerMol = {
        "O": -57.80, "N": -11.02, "S": -4.93, "[H][H]": 0.00, "N#N": 0.00,
        "C=O": -25.95, "[C-]#[O+]": -26.42, "O=[N+]([O-])O": -32.10,
        "O=S(=O)(O)O": None, "O=S(O)O": None,
    }

    def calculateHf(smiles):
        mol = Chem.MolFromSmiles(smiles)
        canon = Chem.MolToSmiles(mol) if mol else smiles
        if canon in knownHfKcalPerMol:
            return knownHfKcalPerMol[canon]
        try:
            j = Joback(smiles)
            if j.status != "OK":
                return None
            return j.Hf(j.counts) / 4184
        except Exception:
            return None

    return calculateHf


def buildPathermoCalculator():
    from pathermo.properties import Hf as pathermoHf
    from thermo.group_contribution.joback import Joback

    def calculateHf(smiles):
        hf = pathermoHf(smiles)
        if hf is not None:
            return hf
        try:
            j = Joback(smiles)
            return j.Hf(j.counts) / 4184 if j.status == "OK" else None
        except Exception:
            return None

    return calculateHf


def loadThermoCalculator() -> Callable[[str], float] | None:
    if thermoBackend == "none":
        return None
    try:
        if thermoBackend == "pathermo":
            return buildPathermoCalculator()
        if thermoBackend == "joback":
            return buildJobackCalculator()
        if thermoBackend == "pgthermo":
            from pgthermo.properties import Hf as pgthermoHf

            return lambda smiles: pgthermoHf(smiles) / 1000
        if thermoBackend == "custom":
            if thermoCalculatorPath:
                sys.path.insert(0, thermoCalculatorPath)
            module = importlib.import_module(thermoCalculatorModule)
            calculator = getattr(module, thermoCalculatorFunction)
            if not callable(calculator):
                raise TypeError(f"{thermoCalculatorModule}.{thermoCalculatorFunction} is not callable")
            return calculator
        raise ValueError(f"Unknown thermoCalculator backend: {thermoBackend!r}")
    except Exception as exc:
        if thermoRequired:
            raise RuntimeError(f"Could not load thermoCalculator {thermoBackend!r}: {exc}") from exc
        print(f"WARNING: thermoCalculator {thermoBackend!r} failed ({exc}); continuing with No_Thermo.")
        return None


thermoCalculator = loadThermoCalculator()

t0 = time.time()
network = enzymatic.generate_network(
    job_name=jobName,
    starters=starters,
    gen=generations,
    max_atoms=maxAtoms,
    direction="forward",
    targets=target,
    ruleset=ruleset,
    rxn_thermo_calculator=thermoCalculator,
    max_rxn_thermo_change=maxRxnThermoChange,
)

smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]
with Path(f"{jobName}_molecules.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES", "isStarter", "molFormula", "molWeight", "numHeavyAtoms"])
    for s in smilesList:
        formula, weight, heavyAtoms = getSmilesProps(s)
        writer.writerow([s, s in starters, formula, weight, heavyAtoms])

postProcessing.pretreat_networks(
    networks={network},
    total_generations=generations,
    starters=starters,
    helpers=helpers,
    job_name=jobName,
    transform_enols_flag=transformEnolsFlag,
    molecule_thermo_calculator=thermoCalculator,
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
    postProcessing.pathway_ranking(starters=starters, helpers=helpers, target=target, job_name=jobName, num_process=1)
if runVisualization:
    postProcessing.pathway_visualization(starters=starters, helpers=helpers, job_name=jobName, num_process=1)

print(f"Done {jobName} in {time.time() - t0:.2f} s")
