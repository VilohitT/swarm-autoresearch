"""Compare swarm evolution results against vanilla autoresearch baseline."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


def load_swarm_log() -> list[dict]:
    path = Path("results/population_log.jsonl")
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_vanilla_log(path: str = "results/vanilla_log.jsonl") -> list[dict]:
    """Load vanilla autoresearch results.

    Expected format: one JSON object per line with at least
    {"experiment": int, "val_bpb": float}
    """
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compare():
    swarm_records = load_swarm_log()
    vanilla_records = load_vanilla_log()

    if not swarm_records:
        print("No swarm data found.")
        return

    # Compute swarm best-so-far by cumulative experiment count
    swarm_experiments = []
    best_so_far = float("inf")
    exp_count = 0
    gen_records: dict[int, list[float]] = {}
    for r in swarm_records:
        gen = r["generation"]
        gen_records.setdefault(gen, []).append(r["val_bpb"])

    for gen in sorted(gen_records.keys()):
        for bpb in gen_records[gen]:
            exp_count += 1
            if bpb < best_so_far:
                best_so_far = bpb
            swarm_experiments.append((exp_count, best_so_far))

    # Compute vanilla best-so-far
    vanilla_experiments = []
    if vanilla_records:
        best_so_far = float("inf")
        for i, r in enumerate(vanilla_records):
            bpb = r.get("val_bpb", float("inf"))
            if bpb < best_so_far:
                best_so_far = bpb
            vanilla_experiments.append((i + 1, best_so_far))

    fig, ax = plt.subplots(figsize=(10, 6))

    if swarm_experiments:
        sx, sy = zip(*swarm_experiments)
        ax.plot(sx, sy, "o-", color="#4CAF50", label="Swarm", linewidth=2, markersize=3)

    if vanilla_experiments:
        vx, vy = zip(*vanilla_experiments)
        ax.plot(vx, vy, "s-", color="#2196F3", label="Vanilla", linewidth=2, markersize=3)

    ax.set_xlabel("Cumulative Experiments")
    ax.set_ylabel("Best Validation BPB")
    ax.set_title("Swarm vs Vanilla AutoResearch")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("results/swarm_vs_vanilla.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")
    plt.close()

    # Print summary statistics
    if swarm_experiments:
        print(f"\nSwarm: {len(swarm_experiments)} experiments, "
              f"best bpb = {swarm_experiments[-1][1]:.4f}")
    if vanilla_experiments:
        print(f"Vanilla: {len(vanilla_experiments)} experiments, "
              f"best bpb = {vanilla_experiments[-1][1]:.4f}")

    # Save diversity log
    if swarm_records:
        save_diversity_log(swarm_records)


def save_diversity_log(records: list[dict]) -> None:
    """Compute and save per-generation diversity metrics."""
    from population import population_diversity, _branch_workdir

    gen_records: dict[int, list[str]] = {}
    for r in records:
        gen = r["generation"]
        bid = r["branch_id"]
        try:
            code = (_branch_workdir(bid) / "train.py").read_text()
            gen_records.setdefault(gen, []).append(code)
        except FileNotFoundError:
            pass

    out_path = Path("results/diversity_log.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "diversity", "n_branches"])
        for gen in sorted(gen_records.keys()):
            codes = gen_records[gen]
            div = population_diversity(codes)
            writer.writerow([gen, f"{div:.4f}", len(codes)])

    print(f"Saved diversity log to {out_path}")


if __name__ == "__main__":
    compare()
