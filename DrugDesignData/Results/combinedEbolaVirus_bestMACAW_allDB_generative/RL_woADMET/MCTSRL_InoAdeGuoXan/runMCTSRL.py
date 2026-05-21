#!/usr/bin/env python3
"""
runMCTSRL.py

MCTS-RL workflow for reaction-constrained, potency-focused molecular
optimization with DORAnet, MACAW, and ART.

This script is intentionally designed as a companion to the existing runRL.py.
It reuses the chemistry, scoring, reward, tracing, plotting, and Pareto helpers
from runRL.py, but replaces the PPO training/rollout section with Monte Carlo
Tree Search:

    selection -> expansion -> rollout -> backpropagation

The reinforcement-learning part is the reward-driven action-value update inside
MCTS. The model of the environment is DORAnet, the reward oracle is ART potency,
and the tree policy uses a PUCT/UCB score to balance exploitation and exploration.

Usage
-----
python runMCTSRL.py config_mctsrl_gen1.yaml

Recommended first test
----------------------
Set:
    mode.run_smoke_tests: true
    mcts.num_simulations_per_seed: 4
    mcts.top_seed_count: 2

Then increase num_simulations_per_seed and top_seed_count.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# The existing runRL.py already contains the validated project-specific building blocks.
# Keep runMCTSRL.py in the same directory as runRL.py, or add that directory to PYTHONPATH.
from runRL import (
    addFrontierAndNoveltyColumns,
    addProjectPaths,
    CombinedMoleculeScorer,
    computeMultiObjectiveReward,
    canonicalizeSmiles,
    deriveMaxAtomsFromSeeds,
    DoranetAction,
    DoranetAdapter,
    ensureDir,
    finalizeRewardConfig,
    getDefaultPreferenceWeights,
    getDoranetGenList,
    loadConfig,
    loadSeedDF,
    MacawFeatureBuilder,
    ArtPotencyOracle,
    precomputeSeedDoranetActions,
    runPotencyBeamSearchDiagnostic,
    saveComparisonPlot,
    saveParetoOutputs,
    summarizeRoute,
)


# =============================================================================
# MCTS data structures
# =============================================================================

@dataclass
class MctsState:
    """A molecule state inside the search tree."""

    smiles: str
    seedSmiles: str
    seedPotency: float
    depth: int
    routeHistory: List[DoranetAction] = field(default_factory=list)


@dataclass
class MctsNode:
    """One node in the MCTS tree."""

    state: MctsState
    parent: Optional["MctsNode"] = None
    parentAction: Optional[DoranetAction] = None
    prior: float = 1.0
    visitCount: int = 0
    valueSum: float = 0.0
    children: Dict[str, "MctsNode"] = field(default_factory=dict)
    unexpandedActions: Optional[List[DoranetAction]] = None
    terminalScore: Optional[dict] = None
    terminalReward: Optional[dict] = None

    @property
    def meanValue(self) -> float:
        if self.visitCount <= 0:
            return 0.0
        return float(self.valueSum / self.visitCount)

    def isExpanded(self) -> bool:
        return self.unexpandedActions is not None

    def hasUnexpandedActions(self) -> bool:
        return self.unexpandedActions is not None and len(self.unexpandedActions) > 0


# =============================================================================
# MCTS planner
# =============================================================================

class MctsPlanner:
    """MCTS planner for DORAnet reaction-path search.

    DORAnet provides legal actions from a molecule. ART provides potency scores.
    MCTS learns state-action values by repeatedly simulating reaction paths and
    backpropagating potency-focused rewards through the tree.
    """

    def __init__(
        self,
        seedDF: pd.DataFrame,
        doranetAdapter: DoranetAdapter,
        combinedScorer: CombinedMoleculeScorer,
        rewardConfig: dict,
        config: dict,
        outputDir: Path,
    ):
        self.seedDF = seedDF.reset_index(drop=True).copy()
        self.doranetAdapter = doranetAdapter
        self.combinedScorer = combinedScorer
        self.rewardConfig = rewardConfig
        self.config = config
        self.outputDir = ensureDir(outputDir)
        self.mctsConfig = config.get("mcts", {}) or {}

        self.maxDepth = int(self.mctsConfig.get("max_depth", config.get("rl", {}).get("max_steps", 3)))
        self.numSimulationsPerSeed = int(self.mctsConfig.get("num_simulations_per_seed", 128))
        self.explorationConstant = float(self.mctsConfig.get("exploration_constant", 1.25))
        self.discountFactor = float(self.mctsConfig.get("discount_factor", 1.0))
        self.rolloutPolicy = str(self.mctsConfig.get("rollout_policy", "rank_softmax")).lower()
        self.rolloutTemperature = float(self.mctsConfig.get("rollout_temperature", 0.35))
        self.rolloutTopK = int(self.mctsConfig.get("rollout_top_k", 16))
        self.expandTopK = self.mctsConfig.get("expand_top_k", None)
        self.expandTopK = None if self.expandTopK is None else int(self.expandTopK)
        self.addIntermediateCandidates = bool(self.mctsConfig.get("add_intermediate_candidates", True))
        self.progressiveWideningConfig = self.mctsConfig.get("progressive_widening", {}) or {}
        self.progressiveWideningEnabled = bool(self.progressiveWideningConfig.get("enabled", True))
        self.progressiveWideningK = float(self.progressiveWideningConfig.get("k", 2.0))
        self.progressiveWideningAlpha = float(self.progressiveWideningConfig.get("alpha", 0.50))
        self.randomSeed = int(self.mctsConfig.get("random_seed", 42))
        self.rng = random.Random(self.randomSeed)
        self.npRng = np.random.default_rng(self.randomSeed)
        self.preferenceWeights = self._getPreferenceWeights()

        self.globalBestSeedPotency = float(
            rewardConfig.get(
                "globalBestSeedPotency",
                pd.to_numeric(self.seedDF["pPotency_prediction"], errors="coerce").max(),
            )
        )

        self.evaluatedRows: List[dict] = []
        self.edgeRows: List[dict] = []
        self.rootSummaryRows: List[dict] = []
        self.bestCandidateBySmiles: Dict[str, dict] = {}

    def _getPreferenceWeights(self) -> dict:
        """Use the first multi-objective preference set when available; otherwise reward.weights."""
        moConfig = self.config.get("multi_objective", {}) or {}
        preferenceSets = moConfig.get("preference_sets", {}) or {}
        if preferenceSets:
            firstName = list(preferenceSets.keys())[0]
            return preferenceSets[firstName]
        return getDefaultPreferenceWeights(self.rewardConfig)

    def getPriorForAction(self, actionIndex: int, numActions: int) -> float:
        """Rank-based prior; DoranetAdapter already returns actions after ART ranking."""
        if numActions <= 0:
            return 1.0
        temperature = float(self.mctsConfig.get("prior_temperature", 0.70))
        temperature = max(temperature, 1e-6)
        rank = float(actionIndex)
        weights = np.exp(-rank / temperature)
        denom = sum(np.exp(-idx / temperature) for idx in range(numActions))
        return float(weights / max(denom, 1e-12))

    def expandNodeActions(self, node: MctsNode) -> None:
        """Enumerate and cache DORAnet actions for a node."""
        if node.unexpandedActions is not None:
            return

        if node.state.depth >= self.maxDepth:
            node.unexpandedActions = []
            return

        actionList = self.doranetAdapter.enumerateActions(node.state.smiles)
        if self.expandTopK is not None:
            actionList = actionList[: self.expandTopK]
        node.unexpandedActions = list(actionList)

    def isTerminalNode(self, node: MctsNode) -> bool:
        if node.state.depth >= self.maxDepth:
            return True
        self.expandNodeActions(node)
        return len(node.unexpandedActions or []) == 0 and len(node.children) == 0

    def selectChild(self, node: MctsNode) -> MctsNode:
        """PUCT-style tree policy."""
        if not node.children:
            raise ValueError("Cannot select child from a node with no children.")

        parentVisits = max(node.visitCount, 1)
        bestScore = -float("inf")
        bestChildren: List[MctsNode] = []

        for child in node.children.values():
            qValue = child.meanValue
            exploration = (
                self.explorationConstant
                * child.prior
                * math.sqrt(parentVisits)
                / (1.0 + child.visitCount)
            )
            score = qValue + exploration
            if score > bestScore + 1e-12:
                bestScore = score
                bestChildren = [child]
            elif abs(score - bestScore) <= 1e-12:
                bestChildren.append(child)

        return self.rng.choice(bestChildren)

    def maxChildrenAllowed(self, node: MctsNode) -> int:
        """Progressive widening limit for large DORAnet action spaces."""
        if not self.progressiveWideningEnabled:
            return 10**9
        visits = max(node.visitCount, 1)
        return max(1, int(math.ceil(self.progressiveWideningK * (visits ** self.progressiveWideningAlpha))))

    def selectLeaf(self, root: MctsNode) -> MctsNode:
        """Walk down the tree until reaching a node to expand/evaluate."""
        node = root
        while True:
            if self.isTerminalNode(node):
                return node

            canWiden = len(node.children) < self.maxChildrenAllowed(node)
            if node.hasUnexpandedActions() and canWiden:
                return node

            if node.children:
                node = self.selectChild(node)
            else:
                return node

    def expandOneChild(self, node: MctsNode) -> MctsNode:
        """Expand one unvisited action from the selected leaf."""
        self.expandNodeActions(node)
        if not node.hasUnexpandedActions():
            return node

        expansionStrategy = str(self.mctsConfig.get("expansion_strategy", "prior_order")).lower()
        if expansionStrategy in {"random", "uniform"}:
            actionPos = self.rng.randrange(len(node.unexpandedActions))
        else:
            actionPos = 0

        actionObj = node.unexpandedActions.pop(actionPos)
        nextSmiles = canonicalizeSmiles(self.doranetAdapter.applyAction(node.state.smiles, actionObj))
        if nextSmiles is None:
            return node

        childState = MctsState(
            smiles=nextSmiles,
            seedSmiles=node.state.seedSmiles,
            seedPotency=node.state.seedPotency,
            depth=node.state.depth + 1,
            routeHistory=node.state.routeHistory + [actionObj],
        )
        childKey = f"{actionObj.actionIndex}:{nextSmiles}"
        prior = self.getPriorForAction(actionObj.actionIndex, max(actionObj.actionIndex + 1, len(node.children) + len(node.unexpandedActions) + 1))
        childNode = MctsNode(
            state=childState,
            parent=node,
            parentAction=actionObj,
            prior=prior,
        )
        node.children[childKey] = childNode

        if self.addIntermediateCandidates:
            self.evaluateState(childNode.state, source="expanded")

        return childNode

    def chooseRolloutAction(self, actionList: List[DoranetAction]) -> Optional[DoranetAction]:
        """Choose one action during the simulation/rollout phase."""
        if not actionList:
            return None

        if self.rolloutTopK > 0:
            actionList = actionList[: self.rolloutTopK]

        if self.rolloutPolicy in {"first", "greedy", "rank_greedy"}:
            return actionList[0]
        if self.rolloutPolicy in {"random", "uniform"}:
            return self.rng.choice(actionList)

        # rank_softmax: lower rank gets higher probability, but exploration remains.
        temperature = max(self.rolloutTemperature, 1e-6)
        ranks = np.arange(len(actionList), dtype=float)
        probs = np.exp(-ranks / temperature)
        probs = probs / probs.sum()
        idx = int(self.npRng.choice(len(actionList), p=probs))
        return actionList[idx]

    def rolloutFrom(self, state: MctsState) -> MctsState:
        """Simulate a reaction path from the expanded node to terminal depth."""
        currentState = copy.deepcopy(state)
        while currentState.depth < self.maxDepth:
            actionList = self.doranetAdapter.enumerateActions(currentState.smiles)
            if not actionList:
                break
            actionObj = self.chooseRolloutAction(actionList)
            if actionObj is None:
                break
            nextSmiles = canonicalizeSmiles(self.doranetAdapter.applyAction(currentState.smiles, actionObj))
            if nextSmiles is None:
                break
            currentState = MctsState(
                smiles=nextSmiles,
                seedSmiles=currentState.seedSmiles,
                seedPotency=currentState.seedPotency,
                depth=currentState.depth + 1,
                routeHistory=currentState.routeHistory + [actionObj],
            )

        return currentState

    def evaluateState(self, state: MctsState, source: str) -> Tuple[dict, dict, float]:
        """Score a molecule and compute the scalar MCTS reward."""
        scoreDict = self.combinedScorer.scoreOne(state.smiles)
        scoreDict = dict(scoreDict)
        scoreDict["seedSmiles"] = state.seedSmiles
        scoreDict["seedPotency"] = float(state.seedPotency)
        scoreDict["globalBestSeedPotency"] = float(self.globalBestSeedPotency)

        potencyValue = float(scoreDict.get("pPotency_prediction", np.nan))
        scoreDict["deltaPotency"] = potencyValue - float(state.seedPotency) if np.isfinite(potencyValue) else np.nan
        scoreDict["frontierDeltaPotency"] = potencyValue - float(self.globalBestSeedPotency) if np.isfinite(potencyValue) else np.nan

        rewardDict = computeMultiObjectiveReward(
            scoreDict=scoreDict,
            rewardConfig=self.rewardConfig,
            routeDepth=state.depth,
            preferenceWeights=self.preferenceWeights,
        )

        scalarReward = float(rewardDict["totalReward"])
        scalarReward *= float(self.discountFactor ** max(state.depth - 1, 0))

        row = dict(scoreDict)
        row["mctsSource"] = source
        row["mctsDepth"] = state.depth
        row["mctsReward"] = scalarReward
        row["routeLength"] = len(state.routeHistory)
        row.update(summarizeRoute(state.routeHistory))
        for key, value in rewardDict.items():
            if key == "rewardVector":
                for objName, objVal in value.items():
                    row[f"reward_{objName}"] = objVal
            else:
                row[f"reward_{key}"] = value

        self.evaluatedRows.append(row)

        smilesKey = canonicalizeSmiles(row.get("SMILES", state.smiles)) or state.smiles
        previous = self.bestCandidateBySmiles.get(smilesKey)
        if previous is None or float(row.get("mctsReward", -np.inf)) > float(previous.get("mctsReward", -np.inf)):
            self.bestCandidateBySmiles[smilesKey] = row

        return scoreDict, rewardDict, scalarReward

    def backpropagate(self, node: MctsNode, value: float) -> None:
        """Backpropagate rollout value to the root."""
        current = node
        while current is not None:
            current.visitCount += 1
            current.valueSum += float(value)
            current = current.parent

    def runSimulation(self, root: MctsNode) -> None:
        leaf = self.selectLeaf(root)
        expanded = self.expandOneChild(leaf)
        terminalState = self.rolloutFrom(expanded.state)
        _, _, rewardValue = self.evaluateState(terminalState, source="rollout_terminal")
        self.backpropagate(expanded, rewardValue)

    def saveTreeRowsForRoot(self, root: MctsNode, seedIndex: int) -> None:
        """Flatten tree edges for debugging and reproducibility."""
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                action = child.parentAction
                routeSummary = summarizeRoute(child.state.routeHistory)
                self.edgeRows.append(
                    {
                        "seedIndex": seedIndex,
                        "seedSmiles": root.state.seedSmiles,
                        "parentSmiles": node.state.smiles,
                        "childSmiles": child.state.smiles,
                        "depth": child.state.depth,
                        "actionIndex": action.actionIndex if action is not None else np.nan,
                        "actionProductSmiles": action.productSmiles if action is not None else "",
                        "actionSourceSmiles": action.sourceSmiles if action is not None else "",
                        "visitCount": child.visitCount,
                        "meanValue": child.meanValue,
                        "valueSum": child.valueSum,
                        "prior": child.prior,
                        **routeSummary,
                    }
                )
                stack.append(child)

    def runForSeed(self, seedRow: pd.Series, seedIndex: int) -> MctsNode:
        seedSmiles = str(seedRow["canonicalSmiles"])
        seedPotency = float(seedRow["pPotency_prediction"])
        rootState = MctsState(
            smiles=seedSmiles,
            seedSmiles=seedSmiles,
            seedPotency=seedPotency,
            depth=0,
            routeHistory=[],
        )
        root = MctsNode(state=rootState, parent=None, parentAction=None, prior=1.0)

        self.evaluateState(rootState, source="seed_root")

        for simIndex in range(self.numSimulationsPerSeed):
            try:
                self.runSimulation(root)
            except Exception as exc:
                print(f"WARNING: MCTS simulation failed for seedIndex={seedIndex}, sim={simIndex}: {exc}")

        bestChild = None
        if root.children:
            bestChild = max(root.children.values(), key=lambda child: (child.visitCount, child.meanValue))

        self.rootSummaryRows.append(
            {
                "seedIndex": seedIndex,
                "seedSmiles": seedSmiles,
                "seedPotency": seedPotency,
                "rootVisitCount": root.visitCount,
                "numRootChildren": len(root.children),
                "bestChildSmiles": bestChild.state.smiles if bestChild is not None else "",
                "bestChildVisitCount": bestChild.visitCount if bestChild is not None else 0,
                "bestChildMeanValue": bestChild.meanValue if bestChild is not None else np.nan,
            }
        )
        self.saveTreeRowsForRoot(root, seedIndex=seedIndex)
        return root

    def run(self) -> pd.DataFrame:
        """Run MCTS for the configured seed subset and save outputs."""
        startTime = time.time()
        topSeedCount = self.mctsConfig.get("top_seed_count", None)
        seedWorkDF = self.seedDF.copy()
        if topSeedCount is not None:
            seedWorkDF = seedWorkDF.sort_values("pPotency_prediction", ascending=False).head(int(topSeedCount)).reset_index(drop=True)

        print("\n================ MCTS-RL potency search ================")
        print(f"Seeds searched: {len(seedWorkDF)}")
        print(f"Simulations per seed: {self.numSimulationsPerSeed}")
        print(f"Max reaction-path depth: {self.maxDepth}")
        print(f"Exploration constant: {self.explorationConstant}")
        print(f"Rollout policy: {self.rolloutPolicy}")
        print(f"Progressive widening: {self.progressiveWideningEnabled}, k={self.progressiveWideningK}, alpha={self.progressiveWideningAlpha}")
        print(f"Preference weights: {self.preferenceWeights}")

        for localSeedIndex, seedRow in seedWorkDF.iterrows():
            print(
                f"\n[MCTS seed {localSeedIndex + 1}/{len(seedWorkDF)}] "
                f"pPotency={float(seedRow['pPotency_prediction']):.4f} "
                f"SMILES={seedRow['canonicalSmiles']}"
            )
            self.runForSeed(seedRow, seedIndex=int(localSeedIndex))

        allEvaluatedDF = pd.DataFrame(self.evaluatedRows)
        if allEvaluatedDF.empty:
            allEvaluatedDF = pd.DataFrame()

        uniqueRows = list(self.bestCandidateBySmiles.values())
        generatedDF = pd.DataFrame(uniqueRows)

        if not generatedDF.empty:
            generatedDF = addFrontierAndNoveltyColumns(
                generatedDF,
                seedDF=self.seedDF,
                globalBestSeedPotency=float(self.globalBestSeedPotency),
            )
            if bool(self.config.get("output", {}).get("exclude_seed_molecules_from_generated", True)):
                generatedDF = generatedDF.loc[generatedDF["isNovelVsSeedSet"]].copy()

            sortCols = [
                col for col in [
                    "frontierDeltaPotency",
                    "deltaPotency",
                    "pPotency_prediction",
                    "mctsReward",
                ]
                if col in generatedDF.columns
            ]
            if sortCols:
                generatedDF = generatedDF.sort_values(sortCols, ascending=[False for _ in sortCols])
            maxUnique = self.mctsConfig.get("max_unique_candidates", None)
            if maxUnique is not None:
                generatedDF = generatedDF.head(int(maxUnique)).copy()
            generatedDF = generatedDF.reset_index(drop=True)

        outputCsv = self.outputDir / self.mctsConfig.get("output_csv", "mctsrl_generated_candidates.csv")
        evaluatedCsv = self.outputDir / self.mctsConfig.get("evaluated_output_csv", "mctsrl_all_evaluated_candidates.csv")
        edgeCsv = self.outputDir / self.mctsConfig.get("edge_output_csv", "mctsrl_tree_edges.csv")
        rootCsv = self.outputDir / self.mctsConfig.get("root_summary_csv", "mctsrl_root_summary.csv")

        generatedDF.to_csv(outputCsv, index=False)
        allEvaluatedDF.to_csv(evaluatedCsv, index=False)
        pd.DataFrame(self.edgeRows).to_csv(edgeCsv, index=False)
        pd.DataFrame(self.rootSummaryRows).to_csv(rootCsv, index=False)

        print("\nSaved MCTS-RL generated candidates to:", outputCsv)
        print("Saved all evaluated MCTS states to:", evaluatedCsv)
        print("Saved MCTS tree edges to:", edgeCsv)
        print("Saved MCTS root summaries to:", rootCsv)
        print(f"MCTS-RL runtime: {time.time() - startTime:.2f} seconds")

        return generatedDF


# =============================================================================
# Smoke tests and workflow
# =============================================================================

def runMctsSmokeTests(
    seedDF: pd.DataFrame,
    doranetAdapter: DoranetAdapter,
    combinedScorer: CombinedMoleculeScorer,
    rewardConfig: dict,
    config: dict,
    outputDir: Path,
) -> None:
    """Small end-to-end test of DORAnet actions, ART scores, and MCTS."""
    print("\n================ MCTS-RL smoke tests ================")
    testDF = seedDF.sort_values("pPotency_prediction", ascending=False).head(int(config.get("mode", {}).get("smoke_test_n", 2))).reset_index(drop=True)
    actionRows = []

    for idx, row in testDF.iterrows():
        actions = doranetAdapter.enumerateActions(row["canonicalSmiles"])
        actionRows.append(
            {
                "seedIndex": idx,
                "seedSmiles": row["canonicalSmiles"],
                "seedPotency": row["pPotency_prediction"],
                "numActions": len(actions),
                "firstActionProduct": actions[0].productSmiles if actions else "",
            }
        )

    smokeActionDF = pd.DataFrame(actionRows)
    smokeActionDF.to_csv(outputDir / "mcts_smoke_seed_actions.csv", index=False)
    print("Saved smoke action summary to:", outputDir / "mcts_smoke_seed_actions.csv")

    smokeConfig = copy.deepcopy(config)
    smokeConfig.setdefault("mcts", {})["top_seed_count"] = len(testDF)
    smokeConfig.setdefault("mcts", {})["num_simulations_per_seed"] = int(config.get("mode", {}).get("smoke_mcts_simulations", 4))
    smokeConfig.setdefault("mcts", {})["max_unique_candidates"] = 100

    planner = MctsPlanner(
        seedDF=testDF,
        doranetAdapter=doranetAdapter,
        combinedScorer=combinedScorer,
        rewardConfig=rewardConfig,
        config=smokeConfig,
        outputDir=outputDir / "mcts_smoke_run",
    )
    smokeGeneratedDF = planner.run()
    print("MCTS smoke generated shape:", smokeGeneratedDF.shape)


def runSingleGenerationMctsWorkflow(
    config: dict,
    seedDF: pd.DataFrame,
    initialRewardConfig: dict,
    macawFeatureBuilder: MacawFeatureBuilder,
    artOracle: ArtPotencyOracle,
    outputDir: Path,
) -> pd.DataFrame:
    """Run one DORAnet generation depth with MCTS-RL."""
    outputDir = ensureDir(outputDir)
    gen = int(config.get("doranet", {}).get("gen", 1))

    print("\n" + "=" * 78)
    print(f"Starting MCTS-RL DORAnet generation run: gen={gen}")
    print("Output directory:", outputDir.resolve())
    print("DORAnet max_atoms:", config.get("doranet", {}).get("max_atoms"))
    print("DORAnet network directory:", config.get("doranet", {}).get("network_output_dir"))
    print("=" * 78)

    rewardConfig = finalizeRewardConfig(config, seedDF, initialRewardConfig)
    seedDF.to_csv(outputDir / "seedDF_used.csv", index=False)
    print("Seed DF shape:", seedDF.shape)
    print("Reward config:", rewardConfig)

    doranetAdapter = DoranetAdapter(config)
    combinedScorer = CombinedMoleculeScorer(
        artOracle=artOracle,
        toxicityOracle=None,
        computeQEDForOutput=False,
        scoreToxicityForOutput=False,
    )
    doranetAdapter.setSeedContext(seedDF)
    doranetAdapter.setActionScorer(combinedScorer)

    precomputeSeedDoranetActions(
        seedDF=seedDF,
        doranetAdapter=doranetAdapter,
        config=config,
        outputDir=outputDir,
    )

    runPotencyBeamSearchDiagnostic(
        seedDF=seedDF,
        doranetAdapter=doranetAdapter,
        combinedScorer=combinedScorer,
        config=config,
        outputDir=outputDir,
    )

    if bool(config.get("mode", {}).get("run_smoke_tests", False)):
        runMctsSmokeTests(
            seedDF=seedDF,
            doranetAdapter=doranetAdapter,
            combinedScorer=combinedScorer,
            rewardConfig=rewardConfig,
            config=config,
            outputDir=outputDir,
        )

    generatedDF = pd.DataFrame()
    if bool(config.get("mcts", {}).get("enabled", True)):
        planner = MctsPlanner(
            seedDF=seedDF,
            doranetAdapter=doranetAdapter,
            combinedScorer=combinedScorer,
            rewardConfig=rewardConfig,
            config=config,
            outputDir=outputDir,
        )
        generatedDF = planner.run()
    else:
        print("MCTS disabled by config.")

    generatedDF, paretoDF = saveParetoOutputs(generatedDF, config, outputDir)
    doranetAdapter.saveTraceTables(outputDir)

    if bool(config.get("plotting", {}).get("enabled", True)):
        saveComparisonPlot(seedDF, generatedDF, outputDir, config=config)

    return generatedDF


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DORAnet + ART potency-only MCTS-RL workflow.")
    parser.add_argument("config", help="Path to YAML config file.")
    args = parser.parse_args()

    startTime = time.time()
    config = loadConfig(args.config)
    addProjectPaths(config)

    genList = getDoranetGenList(config)
    if len(genList) != 1:
        raise ValueError(
            "This script is intentionally single-generation. Use one config per DORAnet gen job."
        )

    gen = int(genList[0])
    config.setdefault("doranet", {})["gen"] = gen
    outputDir = ensureDir(Path(config.get("output", {}).get("output_dir", f"./MCTSRL_gen{gen}")))

    print("Output directory:", outputDir.resolve())
    print("Single DORAnet generation:", gen)
    print(
        "Runtime thread environment:",
        {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    )

    seedDF, initialRewardConfig = loadSeedDF(config)
    maxAtoms, seedAtomCountDF = deriveMaxAtomsFromSeeds(seedDF, config)

    networkSubdir = config.get("doranet", {}).get("network_subdir", "doranet_networks")
    config.setdefault("doranet", {})["max_atoms"] = {str(k): int(v) for k, v in maxAtoms.items()}
    config.setdefault("doranet", {})["network_output_dir"] = str(outputDir / str(networkSubdir))

    seedAtomCountDF.to_csv(outputDir / "seed_atom_counts_used_for_max_atoms.csv", index=False)
    pd.DataFrame(
        [
            {
                "gen": int(gen),
                "multiplier": float((config.get("doranet", {}) or {}).get("max_atoms_multiplier", 1.5)),
                **{f"max_atoms_{k}": v for k, v in maxAtoms.items()},
            }
        ]
    ).to_csv(outputDir / "doranet_dynamic_max_atoms.csv", index=False)

    macawFeatureBuilder = MacawFeatureBuilder(config["macaw"]["transformer_path"])
    artOracle = ArtPotencyOracle(
        artModelPath=config["art"]["model_path"],
        macawFeatureBuilder=macawFeatureBuilder,
        artOutputDir=config["art"].get("output_dir"),
        inputFeaturePrefix=config["art"].get("input_feature_prefix", "MACAW_"),
    )

    generatedDF = runSingleGenerationMctsWorkflow(
        config=config,
        seedDF=seedDF,
        initialRewardConfig=initialRewardConfig,
        macawFeatureBuilder=macawFeatureBuilder,
        artOracle=artOracle,
        outputDir=outputDir,
    )

    nGenerated = 0 if generatedDF is None or generatedDF.empty else int(len(generatedDF))
    bestPotency = np.nan
    if generatedDF is not None and not generatedDF.empty and "pPotency_prediction" in generatedDF.columns:
        bestPotency = pd.to_numeric(generatedDF["pPotency_prediction"], errors="coerce").max()

    summaryDF = pd.DataFrame(
        [
            {
                "doranetGen": int(gen),
                "outputDir": str(outputDir),
                "nGeneratedUnique": nGenerated,
                "bestGeneratedPotency": bestPotency,
                "maxAtoms": "|".join(f"{k}:{v}" for k, v in maxAtoms.items()),
                "runtimeSeconds": time.time() - startTime,
            }
        ]
    )
    summaryPath = outputDir / "mcts_generation_run_summary.csv"
    summaryDF.to_csv(summaryPath, index=False)
    print("Saved MCTS generation run summary to:", summaryPath)
    print(f"Total runtime for DORAnet gen={gen}: {time.time() - startTime:.2f} seconds")


if __name__ == "__main__":
    main()
