"""Lineage tracking and phylogeny tree construction."""

from __future__ import annotations

import json
from pathlib import Path


POPULATION_LOG = Path("results/population_log.jsonl")
PHYLOGENY_FILE = Path("results/phylogeny.json")


def load_population_log() -> list[dict]:
    """Load all records from the population log."""
    if not POPULATION_LOG.exists():
        return []
    records = []
    with open(POPULATION_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_phylogeny() -> dict:
    """Build a phylogeny tree from the population log and save to file.

    The phylogeny is a dict mapping branch_id to a list of generation records,
    plus a separate 'edges' list capturing parent-child relationships.

    Returns:
        The phylogeny dict.
    """
    records = load_population_log()

    # Group records by branch
    branches: dict[str, list[dict]] = {}
    for r in records:
        bid = r["branch_id"]
        if bid not in branches:
            branches[bid] = []
        branches[bid].append(r)

    # Build edges from lineage info
    edges = []
    seen_edges = set()
    for r in records:
        bid = r["branch_id"]
        gen = r["generation"]
        origin = r.get("origin", "baseline")

        if origin in ("crossover", "fallback_clone", "mutation", "diversity_injection"):
            pa = r.get("parent_a")
            pb = r.get("parent_b")

            if pa:
                edge_key = (pa, bid, gen)
                if edge_key not in seen_edges:
                    edges.append({
                        "from": pa,
                        "to": bid,
                        "generation": gen,
                        "type": origin,
                        "role": "parent_a",
                    })
                    seen_edges.add(edge_key)

            if pb:
                edge_key = (pb, bid, gen)
                if edge_key not in seen_edges:
                    edges.append({
                        "from": pb,
                        "to": bid,
                        "generation": gen,
                        "type": origin,
                        "role": "parent_b",
                    })
                    seen_edges.add(edge_key)

    phylogeny = {
        "branches": {
            bid: {
                "records": recs,
                "best_bpb": min(r["val_bpb"] for r in recs) if recs else float("inf"),
                "final_bpb": recs[-1]["val_bpb"] if recs else float("inf"),
                "origins": list({r.get("origin", "baseline") for r in recs}),
            }
            for bid, recs in branches.items()
        },
        "edges": edges,
        "total_generations": max((r["generation"] for r in records), default=0),
        "total_records": len(records),
    }

    PHYLOGENY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PHYLOGENY_FILE, "w") as f:
        json.dump(phylogeny, f, indent=2, default=str)

    return phylogeny


def get_lineage(branch_id: str) -> list[dict]:
    """Trace the full lineage of a branch back to its roots."""
    records = load_population_log()
    branch_records = {r["branch_id"]: r for r in records}

    lineage = []
    visited = set()
    queue = [branch_id]

    while queue:
        bid = queue.pop(0)
        if bid in visited:
            continue
        visited.add(bid)

        # Find all records for this branch
        branch_recs = [r for r in records if r["branch_id"] == bid]
        lineage.extend(branch_recs)

        # Trace parents
        for r in branch_recs:
            pa = r.get("parent_a")
            pb = r.get("parent_b")
            if pa and pa not in visited:
                queue.append(pa)
            if pb and pb not in visited:
                queue.append(pb)

    return sorted(lineage, key=lambda r: (r["generation"], r["branch_id"]))


if __name__ == "__main__":
    tree = build_phylogeny()
    print(json.dumps(tree, indent=2, default=str))
