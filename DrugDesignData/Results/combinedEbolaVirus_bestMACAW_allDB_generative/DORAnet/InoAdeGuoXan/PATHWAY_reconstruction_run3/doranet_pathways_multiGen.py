#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import os
import shutil
import sys
import time
import textwrap
from pathlib import Path
from typing import Any, Callable

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


def loadYaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def asList(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def loadTargetSmilesFromCsv(csvPath: Path, columnName: str | None = "SMILES") -> list[str]:
    df = pd.read_csv(csvPath, dtype=str)
    series = df[columnName] if (columnName and columnName in df.columns) else df.iloc[:, 0]
    smiles = [s.strip() for s in series.dropna().astype(str) if s.strip()]
    return list(dict.fromkeys(smiles))


def loadTargets(config: dict[str, Any]) -> list[str]:
    # Priority: targetSmiles > targetSmilesList > targetsCsvPath + targetsColumn.
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
    raw = config.get("generationsToRun", [config.get("generations", 3)])
    raw = [raw] if isinstance(raw, int) else raw
    generations = [int(g) for g in raw]
    if any(g <= 0 for g in generations):
        raise ValueError(f"Generation values must be positive integers, got {generations}")
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
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def countPathwayBlockSteps(blockLines: list[str]) -> int | None:
    # DORAnet writes "reaction SMILES stoichiometry [...]"; list length = number of steps.
    prefix = "reaction SMILES stoichiometry "
    for line in blockLines:
        if line.startswith(prefix):
            try:
                parsed = ast.literal_eval(line[len(prefix):].strip())
                return len(parsed) if isinstance(parsed, list) else None
            except Exception:
                return None
    return None


def splitPathwayBlocks(pathwaysTxtPath: Path) -> list[list[str]]:
    if not pathwaysTxtPath.exists():
        return []
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in pathwaysTxtPath.read_text(encoding="utf-8").splitlines():
        if line.startswith("pathway number ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def filterPathwaysTxtByExactSteps(jobName: str, exactSteps: int) -> dict[str, Any]:
    # Overwrites {jobName}_pathways.txt with exactly N-step blocks; backs up the original.
    pathwaysPath = Path(f"{jobName}_pathways.txt")
    backupPath = Path(f"{jobName}_pathways_unfiltered.txt")
    exactPath = Path(f"{jobName}_pathways_exact{exactSteps}.txt")

    if not pathwaysPath.exists():
        return {"pathwaysTxtFound": False, "numPathwayBlocksOriginal": 0, "numPathwayBlocksExact": 0, "exactPathwaysTxt": str(exactPath)}

    blocks = splitPathwayBlocks(pathwaysPath)
    exactBlocks = [b for b in blocks if countPathwayBlockSteps(b) == exactSteps]

    if not backupPath.exists():
        shutil.copy2(pathwaysPath, backupPath)

    exactText = "\n\n".join("\n".join(block) for block in exactBlocks)
    exactText = exactText + "\n" if exactText else exactText
    exactPath.write_text(exactText, encoding="utf-8")
    pathwaysPath.write_text(exactText, encoding="utf-8")

    return {
        "pathwaysTxtFound": True,
        "numPathwayBlocksOriginal": len(blocks),
        "numPathwayBlocksExact": len(exactBlocks),
        "exactPathwaysTxt": str(exactPath),
        "backupPathwaysTxt": str(backupPath),
    }


_localCompoundCache = None
_componentContribution = None


def _getEquilibrator(compoundsDbPath: str):
    # Lazy, per-process singleton: ComponentContribution() takes 10-20s to load
    # and must not be rebuilt per reaction. Each parallel worker (joblib/loky)
    # builds its own once. Not verified end-to-end (Zenodo network access
    # required) -- confirm with a real reaction before trusting a full run.
    global _localCompoundCache, _componentContribution
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


def buildRxnDg(compoundsDbPath: str) -> Callable[[dict], float | None]:
    """Reaction-level standard dG' (kcal/mol) via eQuilibrator's component-contribution
    method -- the approach DORAnet's own team uses for enzymatic reactions (their
    official example notebook), and the physically appropriate metric for biochemical
    reactions vs. enthalpy from group additivity. Already the dict -> float shape
    enzymatic.generate_network's rxn_thermo_calculator requires, so unlike Joback/pathermo
    this needs no per-molecule-to-reaction adapter. Returns None if any compound can't
    be resolved/registered via SMILES, or the reaction isn't atomically balanced.
    """
    from equilibrator_api import Reaction

    def rxnDg(rxn: dict) -> float | None:
        try:
            cc, lc = _getEquilibrator(compoundsDbPath)
            allSmiles = list(rxn["reactants"]) + list(rxn["products"])
            compounds = lc.get_compounds(allSmiles)
            if any(c is None for c in compounds):
                return None
            smilesToCompound = dict(zip(allSmiles, compounds))

            stoich: dict[Any, int] = {}
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


def maybeRunPostProcessing(
    network: Any,
    jobName: str,
    userTarget: set[str],
    starters: set[str],
    helpers: set[str],
    config: dict[str, Any],
    generationRun: int,
) -> dict[str, Any]:
    """Run DORAnet post-processing (pretreat -> find pathways -> rank/visualize) for userTarget."""
    import doranet.modules.post_processing as postProcessing

    searchDepth = generationRun if config.get("searchDepthEqualsGeneration", True) else int(config.get("searchDepth", generationRun))
    maxNumRxns = generationRun if config.get("maxNumRxnsEqualsGeneration", True) else int(config.get("maxNumRxns", generationRun))
    minRxnAtomEconomy = float(config.get("minRxnAtomEconomy", 0.0))
    runRanking = bool(config.get("runRanking", True))
    runVisualization = bool(config.get("runVisualization", False))
    numProcess = int(config.get("numProcess", 1))

    # dH/dG per reaction is already set during generate_network(); no per-molecule
    # calculator here since eQuilibrator's dG is reaction-level only. This only
    # ever fed the (off-by-default) enol-transform correction, a no-op either way
    # unless you've set transformEnolsFlag elsewhere.
    postProcessing.pretreat_networks(
        networks={network},
        total_generations=generationRun,
        starters=starters,
        helpers=helpers,
        job_name=jobName,
        remove_pure_helpers_rxns=bool(config.get("removePureHelpersRxns", False)),
        sanitize=bool(config.get("sanitize", True)),
        transform_enols_flag=bool(config.get("transformEnolsFlag", False)),
        molecule_thermo_calculator=None,
    )

    postProcessing.pathway_finder(
        starters=starters,
        helpers=helpers,
        target=userTarget,
        search_depth=searchDepth,
        max_num_rxns=maxNumRxns,
        min_rxn_atom_economy=minRxnAtomEconomy,
        job_name=jobName,
        consider_name_difference=bool(config.get("considerNameDifference", True)),
    )

    filterInfo: dict[str, Any] = {"pathwaysTxtFound": Path(f"{jobName}_pathways.txt").exists()}
    if bool(config.get("filterExactNumRxns", True)):
        exactSteps = generationRun if config.get("exactNumRxnsEqualsGeneration", True) else int(config.get("exactNumRxns", generationRun))
        filterInfo = filterPathwaysTxtByExactSteps(jobName, exactSteps)

    if runRanking:
        postProcessing.pathway_ranking(
            starters=starters,
            helpers=helpers,
            target=userTarget,
            weights=config.get("weights"),
            num_process=numProcess,
            reaxys_result_name=config.get("reaxysResultName"),
            job_name=jobName,
            cool_reactions=config.get("coolReactions"),
        )

    if runVisualization:
        postProcessing.pathway_visualization(
            starters=starters,
            helpers=helpers,
            num_process=numProcess,
            reaxys_result_name=config.get("reaxysResultName", "default"),
            job_name=jobName,
            exclude_smiles=config.get("excludeSmiles"),
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


def saveReproScript(jobDir: Path, config: dict[str, Any], jobName: str, targetSmiles: str, generationRun: int) -> None:
    """Save a standalone, thermodynamics-aware reproducer for one target/generation."""
    searchDepth = generationRun if config.get("searchDepthEqualsGeneration", True) else int(config.get("searchDepth", generationRun))
    maxNumRxns = generationRun if config.get("maxNumRxnsEqualsGeneration", True) else int(config.get("maxNumRxns", generationRun))
    exactNumRxns = generationRun if config.get("exactNumRxnsEqualsGeneration", True) else int(config.get("exactNumRxns", generationRun))

    filterByThermodynamics = bool(config.get("filterByThermodynamics", False))
    maxRxnThermoChange = float(config.get("maxRxnThermoChange", 0))
    compoundsDbPath = config.get("compoundsDbPath", "compounds.sqlite")
    transformEnolsFlag = bool(config.get("transformEnolsFlag", False))

    scriptText = f'''\
#!/usr/bin/env python3
import ast, csv, importlib, os, shutil, sys, time
from pathlib import Path
from typing import Callable

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

sys.path.insert(0, r"{config['doranetPath']}")
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

filterByThermodynamics = {filterByThermodynamics}
maxRxnThermoChange = float("{maxRxnThermoChange}")
compoundsDbPath = "{compoundsDbPath}"
transformEnolsFlag = {transformEnolsFlag}


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
    pathwaysPath = Path(f"{{jobName}}_pathways.txt")
    backupPath = Path(f"{{jobName}}_pathways_unfiltered.txt")
    exactPath = Path(f"{{jobName}}_pathways_exact{{exactSteps}}.txt")
    if not pathwaysPath.exists():
        return 0, 0
    blocks = splitPathwayBlocks(pathwaysPath)
    exactBlocks = [b for b in blocks if countPathwayBlockSteps(b) == exactSteps]
    if not backupPath.exists():
        shutil.copy2(pathwaysPath, backupPath)
    exactText = "\\n\\n".join("\\n".join(block) for block in exactBlocks)
    exactText = exactText + "\\n" if exactText else exactText
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
            stoich = {{}}
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
    direction="{config.get('direction', 'forward')}",
    targets=target,
    ruleset=ruleset,
    rxn_thermo_calculator=rxnDgCalculator,
    max_rxn_thermo_change=maxRxnThermoChange,
)

smilesList = list(starters) + [m.uid for m in network.mols if m.uid not in starters]
with Path(f"{{jobName}}_molecules.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES", "isStarter", "molFormula", "molWeight", "numHeavyAtoms"])
    for s in smilesList:
        formula, weight, heavyAtoms = getSmilesProps(s)
        writer.writerow([s, s in starters, formula, weight, heavyAtoms])

postProcessing.pretreat_networks(
    networks={{network}},
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
    print(f"Exact-step filter Gen{{generations}}: {{exact}}/{{original}} pathways retained")
if runRanking:
    postProcessing.pathway_ranking(starters=starters, helpers=helpers, target=target, job_name=jobName, num_process=1)
if runVisualization:
    postProcessing.pathway_visualization(starters=starters, helpers=helpers, job_name=jobName, num_process=1)

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
        sys.path.insert(0, str(Path(config["doranetPath"]).expanduser().resolve()))
        import doranet.modules.enzymatic as enzymatic

        starters = set(config["starters"])
        helpers = set(config["helpers"])
        userTarget = {targetSmiles}

        # Built per-worker process rather than passed in, since this runs under joblib/loky.
        # When filtering is off, skip the calculator entirely rather than just raising the
        # threshold: Rxn_dH_Filter rejects NaN dH regardless of max_dH (NaN < anything is
        # always False in Python), so any reaction touching a compound eQuilibrator can't
        # resolve/register gets killed no matter how permissive the threshold is. Only
        # rxn_thermo_calculator=None makes every reaction "No_Thermo" and always pass.
        filterByThermodynamics = bool(config.get("filterByThermodynamics", False))
        # 0, not 15: DORAnet's own enzymatic example keeps a reaction only if it's
        # exergonic as written (dG' < this value) -- a different quantity and cutoff
        # than an enthalpy-based threshold.
        maxRxnThermoChange = float(config.get("maxRxnThermoChange", 0))
        compoundsDbPath = config.get("compoundsDbPath", "compounds.sqlite")
        rxnDgCalculator = buildRxnDg(compoundsDbPath) if filterByThermodynamics else None

        network = enzymatic.generate_network(
            job_name=jobName,
            starters=starters,
            gen=generationRun,
            max_atoms=config["maxAtoms"],
            direction=config.get("direction", "forward"),
            targets=userTarget,
            ruleset=config["ruleset"],
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

        targetCanonical = canonicalSmiles(targetSmiles)
        targetInGeneratedMolecules = targetCanonical in {canonicalSmiles(s) for s in smilesList} if targetCanonical else False

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
            "filterByThermodynamics": filterByThermodynamics,
            "maxRxnThermoChange": maxRxnThermoChange,
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
    requiredKeys = ["doranetPath", "baseJobPrefix", "numParallelJobs", "starters", "helpers", "maxAtoms", "ruleset", "summaryCsv"]
    missing = [k for k in requiredKeys if k not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    if not resolveGenerationRuns(config):
        raise ValueError("No generations to run. Set generationsToRun, e.g. [1, 2, 3].")

    filterByThermodynamics = bool(config.get("filterByThermodynamics", False))
    # Fail before dispatching any parallel jobs if eQuilibrator can't initialize
    # (e.g. compoundsDbPath missing and Zenodo unreachable) -- cheaper to find out
    # now than after N jobs each hit the same error.
    if filterByThermodynamics:
        _getEquilibrator(config.get("compoundsDbPath", "compounds.sqlite"))


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

    targetsToTest = config.get("targetsToTest")
    selectedTargets = (
        list(enumerate(targets, start=1))
        if targetsToTest is None
        else [(i, s) for i, s in enumerate(targets, start=1) if i in {int(x) for x in targetsToTest}]
    )

    workItems = [
        (targetIndex, targetSmiles, generationRun)
        for targetIndex, targetSmiles in selectedTargets
        for generationRun in generations
    ]

    print(f"Config: {configPath} | Output dir: {outputDir}")
    print(f"Targets: {len(targets)} loaded, {len(selectedTargets)} selected | Generations: {generations}")
    print(f"Total jobs: {len(workItems)} with nJobs={config['numParallelJobs']}")
    filterByThermodynamics = bool(config.get("filterByThermodynamics", False))
    thermoStatus = f"ON (eQuilibrator dG', cutoff {config.get('maxRxnThermoChange', 0)} kcal/mol)" if filterByThermodynamics else "OFF (No_Thermo, unfiltered)"
    print(f"Thermo filtering: {thermoStatus}")

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
    if {"generationRun", "numPathwayBlocksExact"}.issubset(resultsDf.columns):
        print("\nExact pathway blocks by generation:")
        print(resultsDf.groupby("generationRun")["numPathwayBlocksExact"].sum().to_string())
    print(f"Total time: {time.time() - mainStart:.2f} s")


if __name__ == "__main__":
    main()