"""Branch management, selection, and diversity measurement for the swarm."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import difflib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class BranchRecord:
    branch_id: str
    generation: int
    parent_a: Optional[str] = None
    parent_b: Optional[str] = None
    origin: str = "baseline"  # "baseline" | "mutation" | "crossover" | "diversity_injection" | "fallback_clone"
    val_bpb: float = float("inf")
    survived: bool = True
    code_hash: str = ""
    strategy_summary: str = ""

    def read_train_py(self) -> str:
        """Read the train.py file for this branch."""
        path = _branch_workdir(self.branch_id) / "train.py"
        if not path.exists():
            raise FileNotFoundError(f"train.py not found for branch {self.branch_id}: {path}")
        return path.read_text()

    def write_train_py(self, code: str) -> None:
        """Write new train.py code for this branch."""
        path = _branch_workdir(self.branch_id) / "train.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        self.code_hash = hashlib.sha256(code.encode()).hexdigest()

    def set_lineage(
        self,
        parent_a: Optional[BranchRecord],
        parent_b: Optional[BranchRecord],
        origin: str,
    ) -> None:
        self.parent_a = parent_a.branch_id if parent_a else None
        self.parent_b = parent_b.branch_id if parent_b else None
        self.origin = origin

    def update(self, result: TrainResult) -> None:
        self.val_bpb = result.val_bpb
        self.code_hash = hashlib.sha256(result.code.encode()).hexdigest()
        self.strategy_summary = result.strategy_summary

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    val_bpb: float
    code: str
    strategy_summary: str = ""
    crashed: bool = False


def _branch_workdir(branch_id: str) -> Path:
    """Return the working directory for a given branch."""
    return Path("branches") / branch_id


class Population:
    """Manages the population of branches for evolutionary search."""

    def __init__(self, branches: list[BranchRecord], config: dict):
        self.branches = branches
        self.config = config

    @classmethod
    def initialize(cls, population_size: int, config: dict) -> Population:
        """Create the initial population of branches."""
        branches = []
        for i in range(population_size):
            branch_id = f"branch-{i:02d}"
            branch = BranchRecord(branch_id=branch_id, generation=0, origin="baseline")
            # Create working directory
            workdir = _branch_workdir(branch_id)
            workdir.mkdir(parents=True, exist_ok=True)
            branches.append(branch)
        return cls(branches, config)

    def get_context_summary(self, generation: int) -> str:
        """Generate population context for the mutation prompt."""
        lines = []
        # Sort by val_bpb for readability
        sorted_branches = sorted(self.branches, key=lambda b: b.val_bpb)
        for b in sorted_branches:
            trend = "new" if b.generation == 0 else "active"
            line = (
                f"Branch {b.branch_id}: val_bpb={b.val_bpb:.4f}, "
                f"strategy={b.strategy_summary or 'baseline'}, trend={trend}"
            )
            lines.append(line)
        return "\n".join(lines)

    def get_best_bpb(self) -> float:
        """Return the best (lowest) val_bpb in the population."""
        return min(b.val_bpb for b in self.branches)

    def tournament_select(self, keep_ratio: float = 0.5) -> tuple[list[BranchRecord], list[BranchRecord]]:
        """Select survivors via truncation selection (lower bpb = better)."""
        sorted_branches = sorted(self.branches, key=lambda b: b.val_bpb)
        n_keep = max(1, int(len(sorted_branches) * keep_ratio))
        survivors = sorted_branches[:n_keep]
        killed = sorted_branches[n_keep:]

        for b in survivors:
            b.survived = True
        for b in killed:
            b.survived = False

        return survivors, killed

    @staticmethod
    def sample_parents(survivors: list[BranchRecord]) -> tuple[BranchRecord, BranchRecord]:
        """Sample two parents using tournament selection from survivors."""
        def pick() -> BranchRecord:
            a, b = random.sample(survivors, min(2, len(survivors)))
            if len(survivors) < 2:
                return a
            return a if a.val_bpb < b.val_bpb else b

        pa = pick()
        pb = pick()
        # Ensure different parents if possible
        attempts = 0
        while pb.branch_id == pa.branch_id and len(survivors) > 1 and attempts < 10:
            pb = pick()
            attempts += 1
        return pa, pb

    def compute_diversity(self) -> float:
        """Compute average pairwise diversity across the population."""
        codes = []
        for b in self.branches:
            try:
                codes.append(b.read_train_py())
            except FileNotFoundError:
                codes.append("")
        return population_diversity(codes)

    def inject_diversity(self) -> Optional[BranchRecord]:
        """Reset the most-central branch to encourage exploration.

        The most-central branch is the one with the highest average similarity
        to all other branches. We reset it and mark it for bold mutation.
        """
        codes = []
        for b in self.branches:
            try:
                codes.append(b.read_train_py())
            except FileNotFoundError:
                codes.append("")

        if len(codes) < 2:
            return None

        # Find most-central branch (highest avg similarity = lowest avg diversity)
        avg_diversities = []
        for i in range(len(codes)):
            divs = [
                pairwise_diversity(codes[i], codes[j])
                for j in range(len(codes))
                if i != j and codes[i] and codes[j]
            ]
            avg_diversities.append(sum(divs) / len(divs) if divs else 0.0)

        most_central_idx = min(range(len(avg_diversities)), key=lambda i: avg_diversities[i])
        target = self.branches[most_central_idx]
        target.origin = "diversity_injection"
        return target

    def log_generation(self, generation: int) -> None:
        """Append all branch records to the population log."""
        log_path = Path("results/population_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            for b in self.branches:
                record = b.to_dict()
                record["generation"] = generation
                f.write(json.dumps(record) + "\n")

    def save_phylogeny(self) -> None:
        """Rebuild and save the phylogeny tree from the population log."""
        from phylogeny import build_phylogeny
        build_phylogeny()


# --- Diversity functions ---

def code_fingerprint(code: str) -> dict:
    """Extract key architectural features from training code."""
    return {
        "has_rmsnorm": "RMSNorm" in code,
        "has_layernorm": "LayerNorm" in code and "RMSNorm" not in code,
        "has_groupnorm": "GroupNorm" in code,
        "optimizer": (
            "Muon" if "Muon" in code
            else "AdamW" if "AdamW" in code
            else "other"
        ),
        "has_rope": "rotary" in code.lower() or "RoPE" in code,
        "has_flash_attn": "flash" in code.lower(),
        "depth": int(m.group(1)) if (m := re.search(r"n_layer.*?=\s*(\d+)", code)) else -1,
        "width": int(m.group(1)) if (m := re.search(r"n_embd.*?=\s*(\d+)", code)) else -1,
        "lr": float(m.group(1)) if (m := re.search(r"learning_rate.*?=\s*([\d.e-]+)", code)) else -1,
    }


def pairwise_diversity(code_a: str, code_b: str) -> float:
    """Compute diversity between two code snippets (0=identical, 1=totally different)."""
    if not code_a or not code_b:
        return 1.0

    # Textual diff ratio
    diff_ratio = 1.0 - difflib.SequenceMatcher(
        None, code_a.splitlines(), code_b.splitlines()
    ).ratio()

    # Feature-based diversity
    fa, fb = code_fingerprint(code_a), code_fingerprint(code_b)
    feature_diversity = sum(fa[k] != fb[k] for k in fa) / len(fa)

    return 0.4 * diff_ratio + 0.6 * feature_diversity


def population_diversity(branch_codes: list[str]) -> float:
    """Compute average pairwise diversity across all branches."""
    valid = [c for c in branch_codes if c]
    if len(valid) < 2:
        return 1.0

    pairs = [
        (i, j)
        for i in range(len(valid))
        for j in range(i + 1, len(valid))
    ]
    if not pairs:
        return 1.0

    return sum(pairwise_diversity(valid[i], valid[j]) for i, j in pairs) / len(pairs)
