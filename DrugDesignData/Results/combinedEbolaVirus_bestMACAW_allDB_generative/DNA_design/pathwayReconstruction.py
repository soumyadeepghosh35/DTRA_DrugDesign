#!/usr/bin/env python3
"""
Folder-wise DORAnet pathway reconstruction.

This script is designed for DORAnet job folders like:

root/
  high_pPotency_molecule_pathway1/
    reproDoranetJob.py
    high_pPotency_molecule_pathway1_network_pretreated.json  # or another *.json

It reconstructs exact-depth reaction chains using product-to-reactant overlap.
By default it searches depths 1, 2, and 3 independently:

  depth 1: starter-containing reaction also produces target
  depth 2: starter-containing reaction -> target-producing reaction
  depth 3: starter-containing reaction -> linked reaction -> target-producing reaction

For any depth N:
  step 1: starter must be one reactant of reaction 1
  step k: at least one product from reaction k-1 must be one reactant of reaction k
  final: target must be one product of reaction N

Other reactants are treated as cofactors / side reactants and do not block the chain.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("Missing dependency: pandas. Install with: pip install pandas") from exc

try:
    from rdkit import Chem
except ImportError:
    Chem = None



SCRIPT_VERSION = "doranet_pathway_reconstruct_v6"

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -----------------------------------------------------------------------------
# SMILES handling
# -----------------------------------------------------------------------------


def canonicalize_smiles(
    smiles: Any,
    remove_stereochemistry: bool = False,
    strict: bool = False,
) -> Optional[str]:
    """Return canonical RDKit SMILES when possible; otherwise return stripped string."""
    if smiles is None:
        return None
    s = str(smiles).strip().strip('"').strip("'")
    if not s or s.lower() in {"nan", "none"}:
        return None

    if Chem is None:
        return s

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        if strict:
            return None
        return s
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_set(
    values: Iterable[Any],
    remove_stereochemistry: bool = False,
) -> Set[str]:
    out: Set[str] = set()
    for v in values or []:
        cv = canonicalize_smiles(v, remove_stereochemistry=remove_stereochemistry)
        if cv:
            out.add(cv)
    return out


# -----------------------------------------------------------------------------
# Parse reproDoranetJob.py safely with AST
# -----------------------------------------------------------------------------


def parse_repro_job_script(script_path: Path) -> Dict[str, Any]:
    """Parse literal top-level assignments from reproDoranetJob.py.

    Supports assignments like:
      jobName = "..."
      starters = {'SMILES'}
      helpers = {'O', 'N'}
      target = {"SMILES"}
      generations = 3
      maxAtoms = {'C': 15, ...}
      ruleset = "JN3604IMT"
    """
    text = script_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(script_path))
    allowed = {"jobName", "starters", "helpers", "target", "generations", "maxAtoms", "ruleset"}
    values: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in allowed:
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        # Non-literal assignment; ignore.
                        pass
    return values


def as_string_set(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set(str(k) for k in value.keys())
    if isinstance(value, (set, list, tuple)):
        return set(str(x) for x in value)
    return {str(value)}


# -----------------------------------------------------------------------------
# Reaction parsing
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactionRecord:
    rxn_id: int
    folder: str
    job_name: str
    raw_reaction: str
    reactants: Tuple[str, ...]
    products: Tuple[str, ...]
    name: str
    mid: str
    dH: str
    reaction_type: str
    reactant_string: str
    product_string: str
    clean_reaction_smiles: str


def flatten_json_reaction_payload(payload: Any) -> List[str]:
    """Extract reaction strings from several common JSON layouts."""
    if isinstance(payload, list):
        out: List[str] = []
        for item in payload:
            out.extend(flatten_json_reaction_payload(item))
        return out

    if isinstance(payload, str):
        return [payload]

    if isinstance(payload, dict):
        # Common names first.
        for key in ["reactions", "rxns", "data", "reactionStrings", "reaction_strings"]:
            if key in payload:
                return flatten_json_reaction_payload(payload[key])
        # Otherwise recurse over values.
        out = []
        for v in payload.values():
            out.extend(flatten_json_reaction_payload(v))
        return out

    return []


def split_dot_mols(mol_block: str, remove_stereochemistry: bool) -> Tuple[str, ...]:
    mols = []
    for m in str(mol_block).split("."):
        cm = canonicalize_smiles(m, remove_stereochemistry=remove_stereochemistry)
        if cm:
            mols.append(cm)
    return tuple(sorted(set(mols)))


def parse_doranet_reaction_string(
    rxn: str,
    rxn_id: int,
    folder: str,
    job_name: str,
    remove_stereochemistry: bool,
) -> Optional[ReactionRecord]:
    # DORAnet format: reactants>ruleName>dH$reactStoich$productStoich$reactionType>products
    parts = str(rxn).split(">")
    if len(parts) != 4:
        return None

    reactant_block, name, mid, product_block = parts
    reactants = split_dot_mols(reactant_block, remove_stereochemistry)
    products = split_dot_mols(product_block, remove_stereochemistry)
    if not reactants or not products:
        return None

    mid_parts = str(mid).split("$")
    dH = mid_parts[0] if len(mid_parts) > 0 else ""
    reaction_type = mid_parts[3] if len(mid_parts) > 3 else ""

    reactant_string = ".".join(reactants)
    product_string = ".".join(products)
    clean_rxn = f"{reactant_string}>>{product_string}"

    return ReactionRecord(
        rxn_id=rxn_id,
        folder=folder,
        job_name=job_name,
        raw_reaction=str(rxn),
        reactants=reactants,
        products=products,
        name=str(name),
        mid=str(mid),
        dH=str(dH),
        reaction_type=str(reaction_type),
        reactant_string=reactant_string,
        product_string=product_string,
        clean_reaction_smiles=clean_rxn,
    )


def load_reactions_from_jsons(
    folder_path: Path,
    job_name: str,
    cfg: Dict[str, Any],
    remove_stereochemistry: bool,
) -> Tuple[List[ReactionRecord], List[str]]:
    paths_cfg = cfg.get("paths", {})
    preferred = paths_cfg.get("preferred_network_json_name", "{jobName}_network_pretreated.json")
    preferred_name = preferred.format(jobName=job_name)
    glob_patterns = paths_cfg.get("reaction_json_globs", ["*_network_pretreated.json", "*.json"])

    candidate_paths: List[Path] = []
    preferred_path = folder_path / preferred_name
    if preferred_path.exists():
        candidate_paths.append(preferred_path)

    for pat in glob_patterns:
        for p in sorted(folder_path.glob(pat)):
            if p not in candidate_paths:
                candidate_paths.append(p)

    if not candidate_paths:
        return [], []

    # Prefer the first JSON that actually contains valid DORAnet reaction strings.
    errors = []
    for json_path in candidate_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{json_path.name}: JSON read error {exc}")
            continue

        raw_rxns = flatten_json_reaction_payload(payload)
        records: List[ReactionRecord] = []
        seen_keys: Set[Tuple[Tuple[str, ...], str, str, Tuple[str, ...]]] = set()
        for raw in raw_rxns:
            rec = parse_doranet_reaction_string(
                raw,
                rxn_id=len(records),
                folder=folder_path.name,
                job_name=job_name,
                remove_stereochemistry=remove_stereochemistry,
            )
            if rec is None:
                continue
            key = (rec.reactants, rec.name, rec.mid, rec.products)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # Reassign rxn_id after de-duplication.
            records.append(ReactionRecord(**{**asdict(rec), "rxn_id": len(records)}))

        if records:
            return records, [str(json_path)]
        errors.append(f"{json_path.name}: no valid DORAnet reaction strings found")

    return [], errors


# -----------------------------------------------------------------------------
# Chain search
# -----------------------------------------------------------------------------


@dataclass
class ChainPath:
    rxn_ids_forward: Tuple[int, ...]
    connection_mols_by_step: Tuple[Tuple[str, ...], ...]


def build_product_index(reactions: Sequence[ReactionRecord]) -> Dict[str, List[int]]:
    product_to_rxns: Dict[str, List[int]] = defaultdict(list)
    for r in reactions:
        for p in r.products:
            product_to_rxns[p].append(r.rxn_id)
    return product_to_rxns


def build_reactant_index(reactions: Sequence[ReactionRecord]) -> Dict[str, List[int]]:
    reactant_to_rxns: Dict[str, List[int]] = defaultdict(list)
    for r in reactions:
        for rea in r.reactants:
            reactant_to_rxns[rea].append(r.rxn_id)
    return reactant_to_rxns


def reconstruct_fixed_depth_chains_backward(
    reactions: Sequence[ReactionRecord],
    starters: Set[str],
    targets: Set[str],
    depth: int,
    max_routes: int,
    max_parent_candidates_per_reactant: int,
    allow_reuse_reaction: bool,
    allow_target_as_reactant: bool,
    exact_depth: bool,
) -> Tuple[List[ChainPath], Dict[str, Any]]:
    """Backward search for chains with product-to-reactant overlap.

    Start from reactions producing target. Prepend reactions that produce any reactant
    of the current first reaction. At depth N, require the first reaction to contain
    the starter.
    """
    if depth < 1:
        return [], {"error": "depth < 1"}

    rxn_by_id = {r.rxn_id: r for r in reactions}
    product_to_rxns = build_product_index(reactions)
    reactant_to_rxns = build_reactant_index(reactions)

    target_producers = sorted({rid for t in targets for rid in product_to_rxns.get(t, [])})
    step1_starter_reactions = sorted({rid for s in starters for rid in reactant_to_rxns.get(s, [])})

    diagnostics: Dict[str, Any] = {
        "num_target_producer_reactions": len(target_producers),
        "num_step1_starter_reactions": len(step1_starter_reactions),
        "target_producer_ids_sample": target_producers[:20],
        "step1_starter_reaction_ids_sample": step1_starter_reactions[:20],
        "partial_suffixes_by_length": {},
        "pruned_reused_reaction": 0,
        "pruned_target_as_reactant": 0,
        "pruned_no_starter_at_base": 0,
        "pruned_max_routes": False,
    }

    if not target_producers:
        return [], diagnostics

    # Each state is (suffix_rxn_ids_forward_order, connection_mols_by_forward_step)
    # Initially suffix has only final reaction; no connections yet.
    states: List[Tuple[Tuple[int, ...], Tuple[Tuple[str, ...], ...]]] = [
        ((rid,), tuple()) for rid in target_producers
    ]
    diagnostics["partial_suffixes_by_length"]["1"] = len(states)

    completed: List[ChainPath] = []

    # Build suffix backwards until length == depth. For exact_depth=False, also accept
    # shorter paths when first reaction already has starter.
    for current_len in range(1, depth + 1):
        next_states: List[Tuple[Tuple[int, ...], Tuple[Tuple[str, ...], ...]]] = []

        for rxn_ids, conns in states:
            first_rxn = rxn_by_id[rxn_ids[0]]

            if (not allow_target_as_reactant) and targets.intersection(first_rxn.reactants):
                diagnostics["pruned_target_as_reactant"] += 1
                continue

            first_has_starter = bool(starters.intersection(first_rxn.reactants))
            if first_has_starter:
                if (not exact_depth) or current_len == depth:
                    completed.append(ChainPath(rxn_ids_forward=rxn_ids, connection_mols_by_step=conns))
                    if len(completed) >= max_routes:
                        diagnostics["pruned_max_routes"] = True
                        return completed, diagnostics
                # If exact depth is required and we are not deep enough, keep expanding.

            # Stop when the requested route length has been reached. For exact-depth
            # mode, routes that do not start from a starter at this depth are rejected.
            if current_len == depth:
                if not first_has_starter:
                    diagnostics["pruned_no_starter_at_base"] += 1
                continue

            # Parent candidates: reactions producing any reactant of the current first rxn.
            parent_ids: List[int] = []
            for rea in first_rxn.reactants:
                ids = product_to_rxns.get(rea, [])[:max_parent_candidates_per_reactant]
                parent_ids.extend(ids)
            # Deduplicate while preserving order.
            seen: Set[int] = set()
            parent_ids_unique = []
            for pid in parent_ids:
                if pid not in seen:
                    seen.add(pid)
                    parent_ids_unique.append(pid)

            for pid in parent_ids_unique:
                if (not allow_reuse_reaction) and pid in rxn_ids:
                    diagnostics["pruned_reused_reaction"] += 1
                    continue
                parent_rxn = rxn_by_id[pid]
                connection = tuple(sorted(set(parent_rxn.products).intersection(first_rxn.reactants)))
                if not connection:
                    continue
                new_rxn_ids = (pid,) + rxn_ids
                new_conns = (connection,) + conns
                next_states.append((new_rxn_ids, new_conns))

                # Bound growth.
                if len(next_states) >= max_routes * 20:
                    break
            if len(next_states) >= max_routes * 20:
                break

        if current_len < depth:
            states = next_states
            diagnostics["partial_suffixes_by_length"][str(current_len + 1)] = len(states)
            if not states:
                break

    # Deduplicate completed chains.
    unique: List[ChainPath] = []
    seen_chains: Set[Tuple[int, ...]] = set()
    for p in completed:
        if p.rxn_ids_forward in seen_chains:
            continue
        seen_chains.add(p.rxn_ids_forward)
        unique.append(p)
        if len(unique) >= max_routes:
            diagnostics["pruned_max_routes"] = True
            break

    return unique, diagnostics


def annotate_chain_steps(
    chain: ChainPath,
    reactions: Sequence[ReactionRecord],
    starters: Set[str],
    helpers: Set[str],
    targets: Set[str],
    folder: str,
    job_name: str,
    route_id: str,
) -> List[Dict[str, Any]]:
    rxn_by_id = {r.rxn_id: r for r in reactions}
    records = []
    rxn_ids = list(chain.rxn_ids_forward)
    conns = list(chain.connection_mols_by_step)

    for step_idx, rid in enumerate(rxn_ids, start=1):
        r = rxn_by_id[rid]
        if step_idx == 1:
            connection = tuple(sorted(starters.intersection(r.reactants)))
            connection_type = "starter_in_reactants"
        else:
            connection = conns[step_idx - 2] if step_idx - 2 < len(conns) else tuple()
            connection_type = "previous_product_in_reactants"

        side_reactants = sorted(set(r.reactants) - set(connection))
        helper_side = sorted(set(side_reactants).intersection(helpers))
        nonhelper_side = sorted(set(side_reactants) - set(helpers))

        records.append({
            "dirName": folder,
            "jobName": job_name,
            "routeId": route_id,
            "step": step_idx,
            "numSteps": len(rxn_ids),
            "rxnId": rid,
            "reactionName": r.name,
            "reactionType": r.reaction_type,
            "dH": r.dH,
            "reactionSMILES": r.clean_reaction_smiles,
            "rawReactionString": r.raw_reaction,
            "reactants": ".".join(r.reactants),
            "products": ".".join(r.products),
            "connectionType": connection_type,
            "connectionMolecules": ".".join(connection),
            "sideReactantsAllowedAsCofactors": ".".join(side_reactants),
            "sideReactantsListedAsHelpers": ".".join(helper_side),
            "sideReactantsNotListedAsHelpers": ".".join(nonhelper_side),
            "starterInReactants": bool(starters.intersection(r.reactants)),
            "targetInProducts": bool(targets.intersection(r.products)),
        })
    return records


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------


def find_job_folders(root: Path, job_script_name: str, only_dir_names: Optional[List[str]]) -> List[Path]:
    if only_dir_names:
        return [root / d for d in only_dir_names if (root / d).is_dir()]
    folders = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / job_script_name).exists():
            folders.append(p)
    # Also allow root itself to be a single job folder.
    if (root / job_script_name).exists():
        folders.insert(0, root)
    return folders


def resolve_depths_to_search(generations: int, settings: Dict[str, Any]) -> List[int]:
    """Resolve exact route depths to search.

    Priority:
      1. settings.search_depths, e.g. [1, 2, 3]
      2. if explore_depths_up_to_generations=true, use range(1, generations + 1)
      3. otherwise use the single configured depth/generation

    If use_generations_as_max_depth=true, explicit depths larger than the folder's
    generations value are removed.
    """
    explicit = settings.get("search_depths", None)
    default_depth = int(settings.get("default_depth", 3))
    use_generations_as_max_depth = bool(settings.get("use_generations_as_max_depth", True))

    depths: List[int] = []
    if explicit is not None:
        if isinstance(explicit, int):
            raw_depths = [explicit]
        else:
            raw_depths = list(explicit)
        for d in raw_depths:
            try:
                di = int(d)
            except Exception:
                continue
            if di > 0:
                depths.append(di)
    elif bool(settings.get("explore_depths_up_to_generations", True)):
        depths = list(range(1, max(1, int(generations)) + 1))
    else:
        if bool(settings.get("use_generations_as_depth", True)):
            depths = [int(generations)]
        else:
            depths = [default_depth]

    if use_generations_as_max_depth:
        depths = [d for d in depths if d <= int(generations)]

    depths = sorted(set(depths))
    if not depths:
        depths = [min(default_depth, int(generations)) if use_generations_as_max_depth else default_depth]
    return depths


def process_folder(
    folder_path: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    paths_cfg = cfg.get("paths", {})
    settings = cfg.get("settings", {})
    canon_cfg = cfg.get("canonicalization", {})

    job_script_name = paths_cfg.get("job_script_name", "reproDoranetJob.py")
    remove_stereo = bool(canon_cfg.get("remove_stereochemistry_for_matching", True))

    script_path = folder_path / job_script_name
    job_info = parse_repro_job_script(script_path)
    job_name = str(job_info.get("jobName", folder_path.name))

    starters = canonicalize_set(as_string_set(job_info.get("starters")), remove_stereo)
    helpers = canonicalize_set(as_string_set(job_info.get("helpers")), remove_stereo)
    targets = canonicalize_set(as_string_set(job_info.get("target")), remove_stereo)

    try:
        generations = int(job_info.get("generations", settings.get("default_depth", 3)))
    except Exception:
        generations = int(settings.get("default_depth", 3))

    depths_to_search = resolve_depths_to_search(generations, settings)
    exact_depth = bool(settings.get("exact_depth", True))
    max_routes = int(settings.get("max_routes_per_folder", 5000))
    max_parent_candidates = int(settings.get("max_parent_candidates_per_reactant", 2000))
    allow_reuse_reaction = bool(settings.get("allow_reuse_reaction", False))
    allow_target_as_reactant = bool(settings.get("allow_target_as_reactant", False))

    reactions, json_sources_or_errors = load_reactions_from_jsons(folder_path, job_name, cfg, remove_stereo)

    base_summary: Dict[str, Any] = {
        "dirName": folder_path.name,
        "jobName": job_name,
        "scriptPath": str(script_path),
        "jsonSources": ";".join(json_sources_or_errors),
        "numStarters": len(starters),
        "numHelpers": len(helpers),
        "numTargets": len(targets),
        "starterSMILES": ";".join(sorted(starters)),
        "targetSMILES": ";".join(sorted(targets)),
        "generationsInJobScript": generations,
        "depthsSearched": ";".join(str(x) for x in depths_to_search),
        "exactDepth": exact_depth,
        "numReactionsLoaded": len(reactions),
    }

    unique_reaction_rows: List[Dict[str, Any]] = []
    for r in reactions:
        rd = asdict(r)
        rd["reactants"] = ".".join(r.reactants)
        rd["products"] = ".".join(r.products)
        unique_reaction_rows.append(rd)

    summary_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    if not starters or not targets or not reactions:
        for depth in depths_to_search:
            row = dict(base_summary)
            row.update({
                "status": "missing_starter_target_or_reactions",
                "searchDepthUsed": depth,
                "numRoutes": 0,
                "num_target_producer_reactions": 0,
                "num_step1_starter_reactions": 0,
                "partial_suffixes_by_length_json": "{}",
            })
            summary_rows.append(row)
        return summary_rows, route_rows, step_rows, unique_reaction_rows

    rxn_by_id = {r.rxn_id: r for r in reactions}

    for depth in depths_to_search:
        chains, diag = reconstruct_fixed_depth_chains_backward(
            reactions=reactions,
            starters=starters,
            targets=targets,
            depth=depth,
            max_routes=max_routes,
            max_parent_candidates_per_reactant=max_parent_candidates,
            allow_reuse_reaction=allow_reuse_reaction,
            allow_target_as_reactant=allow_target_as_reactant,
            exact_depth=exact_depth,
        )

        summary = dict(base_summary)
        summary.update({
            "status": "ok",
            "searchDepthUsed": depth,
            "numRoutes": len(chains),
            "num_target_producer_reactions": diag.get("num_target_producer_reactions", 0),
            "num_step1_starter_reactions": diag.get("num_step1_starter_reactions", 0),
            "partial_suffixes_by_length_json": json.dumps(diag.get("partial_suffixes_by_length", {})),
            "pruned_no_starter_at_base": diag.get("pruned_no_starter_at_base", 0),
            "pruned_reused_reaction": diag.get("pruned_reused_reaction", 0),
            "pruned_target_as_reactant": diag.get("pruned_target_as_reactant", 0),
            "pruned_max_routes": diag.get("pruned_max_routes", False),
        })
        summary_rows.append(summary)

        for i, chain in enumerate(chains, start=1):
            route_id = f"{folder_path.name}_d{depth}_route_{i:06d}"
            rxns = [rxn_by_id[rid] for rid in chain.rxn_ids_forward]
            route_rows.append({
                "dirName": folder_path.name,
                "jobName": job_name,
                "searchDepthUsed": depth,
                "routeId": route_id,
                "numSteps": len(rxns),
                "rxnIds": ";".join(str(r.rxn_id) for r in rxns),
                "reactionNames": " | ".join(r.name for r in rxns),
                "multiStepReactionSMILES": "  ||  ".join(r.clean_reaction_smiles for r in rxns),
                "starterSMILES": ";".join(sorted(starters)),
                "targetSMILES": ";".join(sorted(targets)),
                "connectionMoleculesByStep": " | ".join(".".join(c) for c in chain.connection_mols_by_step),
            })
            annotated_steps = annotate_chain_steps(
                chain, reactions, starters, helpers, targets,
                folder_path.name, job_name, route_id,
            )
            for rec in annotated_steps:
                rec["searchDepthUsed"] = depth
            step_rows.extend(annotated_steps)

    return summary_rows, route_rows, step_rows, unique_reaction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct DORAnet pathways folderwise across exact depths 1,2,3 using job folders directly.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    paths_cfg = cfg.get("paths", {})
    root = Path(paths_cfg.get("doranet_jobs_dir", ".")).expanduser().resolve()
    output_dir = Path(paths_cfg.get("output_dir", "pathway_reconstruction_results")).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    job_script_name = paths_cfg.get("job_script_name", "reproDoranetJob.py")
    only_dir_names = get_nested(cfg, ["settings", "only_dir_names"], None)
    max_folders = get_nested(cfg, ["settings", "max_folders"], None)

    folders = find_job_folders(root, job_script_name, only_dir_names)
    if max_folders:
        folders = folders[: int(max_folders)]

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Config: {config_path}")
    print(f"DORAnet jobs dir: {root}")
    print(f"Output dir: {output_dir}")
    print(f"Found job folders: {len(folders)}")

    all_summaries: List[Dict[str, Any]] = []
    all_routes: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    all_unique_rxns: List[Dict[str, Any]] = []

    for idx, folder in enumerate(folders, start=1):
        print(f"\n[{idx}/{len(folders)}] Folder: {folder.name}")
        summaries, routes, steps, unique_rxns = process_folder(folder, cfg)
        all_summaries.extend(summaries)
        all_routes.extend(routes)
        all_steps.extend(steps)
        all_unique_rxns.extend(unique_rxns)

        if summaries:
            first = summaries[0]
            print(
                f"  jobName={first.get('jobName')} starters={first.get('numStarters')} "
                f"helpers={first.get('numHelpers')} targets={first.get('numTargets')} "
                f"generations={first.get('generationsInJobScript')} depths={first.get('depthsSearched')}"
            )
            print(
                f"  reactions={first.get('numReactionsLoaded')} "
                f"target producers={first.get('num_target_producer_reactions', 'NA')} "
                f"starter reactions={first.get('num_step1_starter_reactions', 'NA')}"
            )
            for row in summaries:
                print(
                    f"    depth={row.get('searchDepthUsed')}: routes={row.get('numRoutes')} "
                    f"partial suffixes={row.get('partial_suffixes_by_length_json', '{}')}"
                )

    summary_df = pd.DataFrame(all_summaries)
    routes_df = pd.DataFrame(all_routes)
    steps_df = pd.DataFrame(all_steps)
    rxns_df = pd.DataFrame(all_unique_rxns)

    summary_path = output_dir / "doranet_chain_pathway_summary_by_depth.csv"
    routes_path = output_dir / "doranet_chain_reconstructed_pathways_by_depth.csv"
    steps_path = output_dir / "doranet_chain_reconstructed_pathway_steps_by_depth.csv"
    rxns_path = output_dir / "doranet_chain_unique_reactions.csv"
    config_out_path = output_dir / "resolved_config_v6.json"

    summary_df.to_csv(summary_path, index=False)
    routes_df.to_csv(routes_path, index=False)
    steps_df.to_csv(steps_path, index=False)
    rxns_df.to_csv(rxns_path, index=False)
    config_out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Routes : {routes_path}")
    print(f"Steps  : {steps_path}")
    print(f"Rxns   : {rxns_path}")

    if not summary_df.empty:
        print("\nTop summary rows:")
        cols = [
            "dirName", "jobName", "searchDepthUsed", "num_target_producer_reactions",
            "num_step1_starter_reactions", "numRoutes", "partial_suffixes_by_length_json"
        ]
        cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[cols].head(60).to_string(index=False))

        print("\nRoute counts by depth:")
        try:
            print(
                summary_df.pivot_table(
                    index="dirName",
                    columns="searchDepthUsed",
                    values="numRoutes",
                    aggfunc="sum",
                    fill_value=0,
                ).to_string()
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
