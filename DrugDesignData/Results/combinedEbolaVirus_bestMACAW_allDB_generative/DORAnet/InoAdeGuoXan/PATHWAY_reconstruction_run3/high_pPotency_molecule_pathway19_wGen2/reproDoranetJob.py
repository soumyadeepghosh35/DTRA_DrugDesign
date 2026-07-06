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

jobName = "high_pPotency_molecule_pathway19_wGen2"
starters = {'C1=NC2=C(C(=O)N1)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O'}
helpers = {'CO', 'O=S(=O)(O)O', 'O=[N+]([O-])O', 'Br', 'O=S(O)O', 'N#N', 'O=C=O', 'C#N', 'NO', '[H][H]', 'N#CO', 'O=NO', 'O=O', 'N', 'O', 'O=S=O', 'C=C', 'S', '[Br][Br]', 'C=O', '[C-]#[O+]'}
target = {"CC(=O)CC(=O)O"}
maxAtoms = {'C': 15, 'N': 6, 'O': 8, 'S': 3}
generations = 2
ruleset = "JN3604IMT"
searchDepth = 2
maxNumRxns = 2
minRxnAtomEconomy = 0.0
filterExactNumRxns = True
exactNumRxns = 2
runRanking = True
runVisualization = False

filterByThermodynamics = False
maxRxnThermoChange = float("0.0")
compoundsDbPath = "compounds.sqlite"
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


_componentContribution = None
_localCompoundCache = None


def getEquilibrator(compoundsDbPath):
    global _componentContribution, _localCompoundCache
    if _componentContribution is None:
        from equilibrator_api import ComponentContribution
        from equilibrator_assets.local_compound_cache import LocalCompoundCache

        _localCompoundCache = LocalCompoundCache()
        dbFile = Path(compoundsDbPath)
        if not dbFile.exists():
            _localCompoundCache.generate_local_cache_from_default_zenodo(str(dbFile))
        else:
            _localCompoundCache.ccache = _localCompoundCache.ccache.__class__(str(dbFile))
        _componentContribution = ComponentContribution(ccache=_localCompoundCache.ccache)
    return _componentContribution, _localCompoundCache


def buildRxnDg(compoundsDbPath):
    from equilibrator_api import Reaction

    def rxnDg(rxn):
        try:
            cc, lc = getEquilibrator(compoundsDbPath)
            allSmiles = list(rxn["reactants"]) + list(rxn["products"])
            compounds = lc.get_compounds(allSmiles)
            if any(c is None for c in compounds):
                return None
            smilesToCompound = dict(zip(allSmiles, compounds))
            stoich = {}
            for smi in rxn["reactants"]:
                stoich[smilesToCompound[smi]] = stoich.get(smilesToCompound[smi], 0) - 1
            for smi in rxn["products"]:
                stoich[smilesToCompound[smi]] = stoich.get(smilesToCompound[smi], 0) + 1
            reaction = Reaction(stoich)
            if not reaction.is_balanced():
                return None
            return cc.standard_dg_prime(reaction).value.m_as("kJ/mol") / 4.184
        except Exception:
            return None

    return rxnDg


rxnDgCalculator = buildRxnDg(compoundsDbPath) if filterByThermodynamics else None

t0 = time.time()
network = enzymatic.generate_network(
    job_name=jobName,
    starters=starters,
    gen=generations,
    max_atoms=maxAtoms,
    direction="forward",
    targets=target,
    ruleset=ruleset,
    rxn_thermo_calculator=rxnDgCalculator,
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
    molecule_thermo_calculator=None,
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
