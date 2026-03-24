"""Plot convergence curves for the swarm evolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


def load_log() -> list[dict]:
    path = Path("results/population_log.jsonl")
    if not path.exists():
        print("No population_log.jsonl found. Run the swarm first.")
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_convergence():
    records = load_log()
    if not records:
        print("No records to plot.")
        return

    # Group by generation
    gen_data: dict[int, list[float]] = {}
    for r in records:
        gen = r["generation"]
        bpb = r["val_bpb"]
        if bpb < float("inf"):
            gen_data.setdefault(gen, []).append(bpb)

    if not gen_data:
        print("No valid bpb data to plot.")
        return

    gens = sorted(gen_data.keys())
    best_per_gen = [min(gen_data[g]) for g in gens]
    mean_per_gen = [sum(gen_data[g]) / len(gen_data[g]) for g in gens]
    worst_per_gen = [max(gen_data[g]) for g in gens]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(gens, worst_per_gen, best_per_gen, alpha=0.15, color="#2196F3")
    ax.plot(gens, best_per_gen, "o-", color="#4CAF50", label="Best", linewidth=2)
    ax.plot(gens, mean_per_gen, "s--", color="#FF9800", label="Mean", linewidth=1.5)
    ax.plot(gens, worst_per_gen, "v:", color="#F44336", label="Worst", linewidth=1)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation BPB")
    ax.set_title("Swarm AutoResearch — Convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("results/convergence.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved convergence plot to {out_path}")
    plt.close()

    # Also plot per-branch convergence
    branch_data: dict[str, dict[int, float]] = {}
    for r in records:
        bid = r["branch_id"]
        gen = r["generation"]
        bpb = r["val_bpb"]
        if bpb < float("inf"):
            branch_data.setdefault(bid, {})[gen] = bpb

    fig, ax = plt.subplots(figsize=(10, 6))
    for bid in sorted(branch_data.keys()):
        data = branch_data[bid]
        bgens = sorted(data.keys())
        bpbs = [data[g] for g in bgens]
        ax.plot(bgens, bpbs, "o-", label=bid, linewidth=1.5, markersize=4)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation BPB")
    ax.set_title("Swarm AutoResearch — Per-Branch Convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("results/convergence_branches.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved per-branch convergence plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_convergence()
