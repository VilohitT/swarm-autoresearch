"""Visualize the phylogeny tree of the swarm evolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


def load_phylogeny() -> dict:
    path = Path("results/phylogeny.json")
    if not path.exists():
        print("No phylogeny.json found. Run the swarm first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def visualize():
    phylogeny = load_phylogeny()
    branches = phylogeny["branches"]
    edges = phylogeny["edges"]
    total_gens = phylogeny["total_generations"]

    if not branches:
        print("No data to visualize.")
        return

    # Assign y-positions to branches
    branch_ids = sorted(branches.keys())
    y_pos = {bid: i for i, bid in enumerate(branch_ids)}

    fig, ax = plt.subplots(figsize=(max(12, total_gens * 1.2), max(6, len(branch_ids) * 0.8)))

    # Color map for origins
    origin_colors = {
        "baseline": "#4CAF50",
        "mutation": "#2196F3",
        "crossover": "#FF9800",
        "diversity_injection": "#E91E63",
        "fallback_clone": "#9E9E9E",
    }

    # Plot branch timelines
    for bid, info in branches.items():
        records = info["records"]
        if not records:
            continue

        gens = [r["generation"] for r in records]
        bpbs = [r["val_bpb"] for r in records]

        # Plot points colored by origin
        for r in records:
            color = origin_colors.get(r.get("origin", "baseline"), "#666666")
            ax.scatter(
                r["generation"], y_pos[bid],
                c=color, s=80, zorder=5, edgecolors="black", linewidths=0.5,
            )

        # Connect with lines
        ax.plot(gens, [y_pos[bid]] * len(gens), color="#CCCCCC", linewidth=1, zorder=1)

    # Plot edges
    for edge in edges:
        from_bid = edge["from"]
        to_bid = edge["to"]
        gen = edge["generation"]
        etype = edge.get("type", "crossover")

        if from_bid in y_pos and to_bid in y_pos:
            color = origin_colors.get(etype, "#999999")
            style = "--" if edge.get("role") == "parent_b" else "-"
            ax.annotate(
                "",
                xy=(gen, y_pos[to_bid]),
                xytext=(gen - 1, y_pos[from_bid]),
                arrowprops=dict(
                    arrowstyle="->", color=color,
                    linestyle=style, linewidth=1.5, alpha=0.7,
                ),
            )

    # Labels and formatting
    ax.set_yticks(range(len(branch_ids)))
    ax.set_yticklabels(branch_ids)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Branch")
    ax.set_title("Swarm AutoResearch — Phylogeny Tree")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=origin)
        for origin, c in origin_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = Path("results/phylogeny.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved phylogeny visualization to {out_path}")
    plt.close()


if __name__ == "__main__":
    visualize()
