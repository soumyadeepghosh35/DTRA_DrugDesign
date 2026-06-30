#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import importlib
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


def buildJobackCalculator() -> Callable[[str], float | None]:
    """SMILES -> Hf (kcal/mol) via the Joback group-contribution method (pip install thermo),
    with a small literature-value lookup for inorganics Joback structurally can't fragment.

    Joback is organic-only: it has no group definitions for bare H2O, NH3, H2, N2, CO, etc.
    Those molecules show up as helpers in nearly every reaction, and inside DORAnet's own
    Chem_Rxn_dH_Calculator, ONE unresolved Hf in a reaction makes the whole reaction's dH
    NaN -- which Rxn_dH_Filter then rejects outright at generation time, not just in ranking.
    Without this table, thermodynamics-on network generation collapses to near-zero reactions.

    Values are gas-phase dfH (kcal/mol), sourced from NIST WebBook / CODATA unless noted.
    H2 and N2 are zero by definition. Confidence is high for H2O/NH3/CO/H2S; lower for HNO3;
    H2SO4/H2SO3 are left unresolved (None) since gas-phase data for those is too inconsistent
    across sources to assert here -- verify directly against NIST WebBook if your network
    leans on them and extend this table yourself.
    """
    from thermo.group_contribution.joback import Joback

    knownHfKcalPerMol = {
        "O": -57.80,             # H2O
        "N": -11.02,             # NH3
        "S": -4.93,              # H2S
        "[H][H]": 0.00,          # H2, reference element
        "N#N": 0.00,             # N2, reference element
        "C=O": -25.95,           # CH2O (formaldehyde)
        "[C-]#[O+]": -26.42,     # CO
        "O=[N+]([O-])O": -32.10,  # HNO3, lower confidence
        "O=S(=O)(O)O": None,     # H2SO4, gas-phase data too uncertain
        "O=S(O)O": None,         # H2SO3, same caveat
    }

    def calculateHf(smiles: str) -> float | None:
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


def buildPathermoCalculator() -> Callable[[str], float | None]:
    """SMILES -> Hf (kcal/mol) using pathermo (Benson group additivity; pip install
    git+https://github.com/dmdqy/pathermo.git or clone + `pip install -e .`), the
    calculator actually used in the published DORAnet paper (Zhang et al., Digital
    Discovery, 2025) -- including its 15 kcal/mol enthalpy threshold, which this
    same paper confirms is the value the authors used, not an arbitrary default.

    pathermo returns kcal/mol directly (no unit conversion needed) and ships with
    a bundled small-molecule table that resolves H2O, NH3, H2S, H2, N2, O2, CO2,
    HBr, Br2, etc. natively -- better helper-molecule coverage than Joback alone.

    Falls back to Joback only when pathermo returns None, which happens for
    structures its Benson group set has no value for -- notably fused polycyclic
    heteroaromatics such as the purine scaffold (verified: pathermo can't resolve
    a fused-ring carbon bonded to two ring nitrogens, a core feature of
    adenine/guanine/hypoxanthine/xanthine chemistry). Joback's SMARTS-based
    fragmentation handles those fine, so this fallback covers the gap.
    """
    from pathermo.properties import Hf as pathermoHf
    from thermo.group_contribution.joback import Joback

    def calculateHf(smiles: str) -> float | None:
        hf = pathermoHf(smiles)
        if hf is not None:
            return hf
        try:
            j = Joback(smiles)
            return j.Hf(j.counts) / 4184 if j.status == "OK" else None
        except Exception:
            return None

    return calculateHf


def loadThermoCalculator(config: dict[str, Any]) -> Callable[[str], float] | None:
    """Resolve the SMILES -> Hf (kcal/mol) callable named by thermoCalculator:
    pathermo/joback/pgthermo/custom/none."""
    backend = str(config.get("thermoCalculator", "pathermo")).lower()
    thermoRequired = bool(config.get("thermoRequired", True))

    if backend == "none":
        return None

    try:
        if backend == "pathermo":
            return buildPathermoCalculator()
        if backend == "joback":
            return buildJobackCalculator()
        if backend == "pgthermo":
            from pgthermo.properties import Hf as pgthermoHf  # not verified installable; see "pathermo" instead

            return lambda smiles: pgthermoHf(smiles) / 1000  # Hf returned in cal/mol
        if backend == "custom":
            if config.get("thermoCalculatorPath"):
                sys.path.insert(0, str(Path(config["thermoCalculatorPath"]).expanduser().resolve()))
            module = importlib.import_module(config["thermoCalculatorModule"])
            calculator = getattr(module, config["thermoCalculatorFunction"])
            if not callable(calculator):
                raise TypeError(f"{config['thermoCalculatorModule']}.{config['thermoCalculatorFunction']} is not callable")
            return calculator
        raise ValueError(f"Unknown thermoCalculator backend: {backend!r}")
    except Exception as exc:
        if thermoRequired:
            raise RuntimeError(f"Could not load thermoCalculator {backend!r}: {exc}") from exc
        print(f"WARNING: thermoCalculator {backend!r} failed ({exc}); continuing with No_Thermo.")
        return None


def maybeRunPostProcessing(
    network: Any,
    jobName: str,
    userTarget: set[str],
    starters: set[str],
    helpers: set[str],
    config: dict[str, Any],
    generationRun: int,
    thermoCalculator: Callable[[str], float] | None,
) -> dict[str, Any]:
    """Run DORAnet post-processing (pretreat -> find pathways -> rank/visualize) for userTarget."""
    import doranet.modules.post_processing as postProcessing

    searchDepth = generationRun if config.get("searchDepthEqualsGeneration", True) else int(config.get("searchDepth", generationRun))
    maxNumRxns = generationRun if config.get("maxNumRxnsEqualsGeneration", True) else int(config.get("maxNumRxns", generationRun))
    minRxnAtomEconomy = float(config.get("minRxnAtomEconomy", 0.0))
    runRanking = bool(config.get("runRanking", True))
    runVisualization = bool(config.get("runVisualization", False))
    numProcess = int(config.get("numProcess", 1))

    # dH per reaction is already set during generate_network(); the calculator here
    # only feeds the optional keto-enol correction when transformEnolsFlag is True.
    postProcessing.pretreat_networks(
        networks={network},
        total_generations=generationRun,
        starters=starters,
        helpers=helpers,
        job_name=jobName,
        remove_pure_helpers_rxns=bool(config.get("removePureHelpersRxns", False)),
        sanitize=bool(config.get("sanitize", True)),
        transform_enols_flag=bool(config.get("transformEnolsFlag", False)),
        molecule_thermo_calculator=thermoCalculator,
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

    thermoBackend = str(config.get("thermoCalculator", "joback")).lower()
    thermoCalculatorModule = config.get("thermoCalculatorModule", "")
    thermoCalculatorFunction = config.get("thermoCalculatorFunction", "")
    thermoCalculatorPath = str(Path(config["thermoCalculatorPath"]).expanduser().resolve()) if config.get("thermoCalculatorPath") else ""
    thermoRequired = bool(config.get("thermoRequired", True))
    maxRxnThermoChange = float(config.get("maxRxnThermoChange", 15))
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

thermoBackend = "{thermoBackend}"
thermoCalculatorModule = "{thermoCalculatorModule}"
thermoCalculatorFunction = "{thermoCalculatorFunction}"
thermoCalculatorPath = "{thermoCalculatorPath}"
thermoRequired = {thermoRequired}
maxRxnThermoChange = {maxRxnThermoChange}
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


def buildJobackCalculator():
    from thermo.group_contribution.joback import Joback

    knownHfKcalPerMol = {{
        "O": -57.80, "N": -11.02, "S": -4.93, "[H][H]": 0.00, "N#N": 0.00,
        "C=O": -25.95, "[C-]#[O+]": -26.42, "O=[N+]([O-])O": -32.10,
        "O=S(=O)(O)O": None, "O=S(O)O": None,
    }}

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
                raise TypeError(f"{{thermoCalculatorModule}}.{{thermoCalculatorFunction}} is not callable")
            return calculator
        raise ValueError(f"Unknown thermoCalculator backend: {{thermoBackend!r}}")
    except Exception as exc:
        if thermoRequired:
            raise RuntimeError(f"Could not load thermoCalculator {{thermoBackend!r}}: {{exc}}") from exc
        print(f"WARNING: thermoCalculator {{thermoBackend!r}} failed ({{exc}}); continuing with No_Thermo.")
        return None


thermoCalculator = loadThermoCalculator()

t0 = time.time()
network = enzymatic.generate_network(
    job_name=jobName,
    starters=starters,
    gen=generations,
    max_atoms=maxAtoms,
    direction="{config.get('direction', 'forward')}",
    targets=target,
    ruleset=ruleset,
    rxn_thermo_calculator=thermoCalculator,
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
        thermoCalculator = loadThermoCalculator(config)
        maxRxnThermoChange = float(config.get("maxRxnThermoChange", 15))

        network = enzymatic.generate_network(
            job_name=jobName,
            starters=starters,
            gen=generationRun,
            max_atoms=config["maxAtoms"],
            direction=config.get("direction", "forward"),
            targets=userTarget,
            ruleset=config["ruleset"],
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

        targetCanonical = canonicalSmiles(targetSmiles)
        targetInGeneratedMolecules = targetCanonical in {canonicalSmiles(s) for s in smilesList} if targetCanonical else False

        postInfo = maybeRunPostProcessing(network, jobName, userTarget, starters, helpers, config, generationRun, thermoCalculator)

        return {
            "jobName": jobName,
            "jobDir": str(jobDir),
            "targetIndex": targetIndex,
            "targetSmiles": targetSmiles,
            "targetCanonical": targetCanonical,
            "generationRun": generationRun,
            "targetInGeneratedMolecules": targetInGeneratedMolecules,
            "numMolecules": len(smilesList),
            "thermoCalculatorBackend": str(config.get("thermoCalculator", "joback")).lower(),
            "thermoCalculatorActive": thermoCalculator is not None,
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

    backend = str(config.get("thermoCalculator", "joback")).lower()
    if backend == "custom" and not (config.get("thermoCalculatorModule") and config.get("thermoCalculatorFunction")):
        raise ValueError("thermoCalculator is 'custom' but thermoCalculatorModule/thermoCalculatorFunction aren't both set.")

    # Fail before dispatching any parallel jobs if the calculator can't load.
    if bool(config.get("thermoRequired", True)) and backend != "none":
        loadThermoCalculator(config)


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
    print(f"Thermo backend: {str(config.get('thermoCalculator', 'joback')).lower()} | max dH cutoff: {config.get('maxRxnThermoChange', 15)} kcal/mol")

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