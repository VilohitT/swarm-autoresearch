"""Main generation loop for the swarm evolutionary algorithm."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

from population import Population, BranchRecord, TrainResult, _branch_workdir
from crossover import llm_crossover, generate_strategy_summary
from phylogeny import build_phylogeny

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/swarm.log"),
    ],
)
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def copy_baseline_train_py(branch: BranchRecord) -> None:
    """Copy the baseline train.py from autoresearch/ into a branch's workdir."""
    baseline = Path("autoresearch/train.py")
    if not baseline.exists():
        raise FileNotFoundError(
            f"Baseline train.py not found at {baseline}. "
            "Please place the autoresearch fork in autoresearch/"
        )
    workdir = _branch_workdir(branch.branch_id)
    workdir.mkdir(parents=True, exist_ok=True)
    code = baseline.read_text()
    branch.write_train_py(code)


def run_training(branch: BranchRecord, time_budget: int = 300) -> TrainResult:
    """Run train.py for a branch and return the result.

    Args:
        branch: The branch to train.
        time_budget: Maximum training time in seconds.

    Returns:
        TrainResult with val_bpb, code, and crash status.
    """
    workdir = _branch_workdir(branch.branch_id)
    train_py = workdir / "train.py"

    if not train_py.exists():
        return TrainResult(val_bpb=float("inf"), code="", crashed=True)

    code = train_py.read_text()

    log.info(f"Training {branch.branch_id} (budget={time_budget}s)...")

    try:
        result = subprocess.run(
            ["uv", "run", "train.py"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=time_budget,
        )

        if result.returncode != 0:
            log.warning(
                f"Training {branch.branch_id} failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr[-500:]}"
            )
            return TrainResult(val_bpb=float("inf"), code=code, crashed=True)

        # Parse val_bpb from stdout
        val_bpb = parse_val_bpb(result.stdout)
        if val_bpb is None:
            log.warning(f"Could not parse val_bpb from {branch.branch_id} output")
            return TrainResult(val_bpb=float("inf"), code=code, crashed=True)

        log.info(f"{branch.branch_id} => val_bpb={val_bpb:.4f}")
        return TrainResult(val_bpb=val_bpb, code=code, crashed=False)

    except subprocess.TimeoutExpired:
        log.warning(f"Training {branch.branch_id} timed out after {time_budget}s")
        return TrainResult(val_bpb=float("inf"), code=code, crashed=True)

    except Exception as e:
        log.error(f"Training {branch.branch_id} error: {e}")
        return TrainResult(val_bpb=float("inf"), code=code, crashed=True)


def parse_val_bpb(output: str) -> float | None:
    """Parse the final validation bpb from training output.

    Looks for patterns like 'val_bpb: 0.9743' or 'val bpb 0.9743' in the
    last few lines of output.
    """
    import re

    lines = output.strip().splitlines()
    # Search from the end for the last val_bpb value
    for line in reversed(lines[-50:]):
        # Match various formats: val_bpb: X, val bpb X, val_bpb=X
        match = re.search(r"val[_\s]bpb[\s:=]+([0-9]+\.?[0-9]*)", line, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def run_agent_experiment(
    branch: BranchRecord,
    population_context: str,
    generation: int,
    best_bpb: float,
    config: dict,
) -> TrainResult:
    """Run the autoresearch agent for one mutation experiment on a branch.

    This invokes the LLM to mutate train.py based on population context,
    then runs training.
    """
    import anthropic

    workdir = _branch_workdir(branch.branch_id)
    train_py = workdir / "train.py"
    current_code = train_py.read_text()

    # Build the mutation prompt from program.md template
    prompt_template = Path("program.md").read_text()
    mutation_prompt = prompt_template.format(
        generation=generation,
        population_context=population_context,
        branch_id=branch.branch_id,
        best_bpb=best_bpb,
    )

    # Ask LLM to mutate
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.get("llm_model", "claude-sonnet-4-20250514"),
        max_tokens=8000,
        temperature=config.get("llm_temperature_mutation", 0.5),
        messages=[
            {
                "role": "user",
                "content": (
                    f"{mutation_prompt}\n\n"
                    f"## Current train.py\n```python\n{current_code}\n```\n\n"
                    "Output the COMPLETE modified train.py. Output ONLY valid Python code."
                ),
            }
        ],
    )

    from crossover import extract_python_code
    new_code = extract_python_code(response.content[0].text)

    # Write mutated code
    branch.write_train_py(new_code)
    branch.set_lineage(branch, None, origin="mutation")

    # Run training
    result = run_training(branch, config.get("train_time_budget", 300))

    if result.crashed:
        # Fallback: restore original code
        log.warning(f"{branch.branch_id} mutation crashed, restoring original")
        branch.write_train_py(current_code)
        result = run_training(branch, config.get("train_time_budget", 300))

    # Generate strategy summary
    try:
        result.strategy_summary = generate_strategy_summary(
            result.code or current_code,
            model=config.get("llm_model", "claude-sonnet-4-20250514"),
        )
    except Exception as e:
        log.warning(f"Failed to generate strategy summary: {e}")
        result.strategy_summary = branch.strategy_summary or "unknown"

    return result


def run_swarm() -> None:
    """Main entry point: run the full swarm evolutionary loop."""
    config = load_config("config.yaml")
    Path("results").mkdir(exist_ok=True)

    log.info("=== Swarm AutoResearch ===")
    log.info(f"Config: {config}")

    # Initialize population
    pop = Population.initialize(config["population_size"], config)

    # Generation 0: baseline evaluation
    log.info("=== Generation 0: Baseline ===")
    for branch in pop.branches:
        copy_baseline_train_py(branch)
        result = run_training(branch, config["train_time_budget"])
        branch.update(result)
        log.info(f"  {branch.branch_id}: val_bpb={branch.val_bpb:.4f}")

    pop.log_generation(0)

    # Main evolution loop
    for gen in range(1, config["max_generations"] + 1):
        log.info(f"\n=== Generation {gen}/{config['max_generations']} ===")

        # Phase 1 — Mutate each branch
        log.info("Phase 1: Mutation")
        context = pop.get_context_summary(gen)
        best_bpb = pop.get_best_bpb()

        for branch in pop.branches:
            log.info(f"  Mutating {branch.branch_id}...")
            result = run_agent_experiment(branch, context, gen, best_bpb, config)
            branch.update(result)
            log.info(f"  {branch.branch_id}: val_bpb={branch.val_bpb:.4f}")

        # Phase 2 — Tournament selection
        log.info("Phase 2: Selection")
        survivors, killed = pop.tournament_select(
            keep_ratio=config.get("selection_ratio", 0.5)
        )
        log.info(
            f"  Survivors: {[b.branch_id for b in survivors]} "
            f"(best={survivors[0].val_bpb:.4f})"
        )
        log.info(f"  Killed: {[b.branch_id for b in killed]}")

        # Phase 3 — LLM crossover to fill killed slots
        log.info("Phase 3: Crossover")
        for dead in killed:
            pa, pb = pop.sample_parents(survivors)
            log.info(
                f"  Crossing {pa.branch_id} x {pb.branch_id} -> {dead.branch_id}"
            )

            try:
                new_code = llm_crossover(
                    pa.read_train_py(),
                    pb.read_train_py(),
                    pa.val_bpb,
                    pb.val_bpb,
                    model=config.get("llm_model", "claude-sonnet-4-20250514"),
                    temperature=config.get("llm_temperature_crossover", 0.7),
                )
                dead.write_train_py(new_code)
                dead.set_lineage(pa, pb, origin="crossover")
                result = run_training(dead, config["train_time_budget"])

                if result.crashed:
                    log.warning(
                        f"  Crossover child {dead.branch_id} crashed, "
                        f"falling back to clone of {pa.branch_id}"
                    )
                    dead.write_train_py(pa.read_train_py())
                    dead.set_lineage(pa, None, origin="fallback_clone")
                    result = run_training(dead, config["train_time_budget"])

            except Exception as e:
                log.error(f"  Crossover failed for {dead.branch_id}: {e}")
                dead.write_train_py(pa.read_train_py())
                dead.set_lineage(pa, None, origin="fallback_clone")
                result = run_training(dead, config["train_time_budget"])

            dead.update(result)
            log.info(f"  {dead.branch_id}: val_bpb={dead.val_bpb:.4f}")

        # Phase 4 — Diversity check
        diversity = pop.compute_diversity()
        log.info(f"Phase 4: Diversity = {diversity:.4f} (threshold={config['diversity_threshold']})")

        if diversity < config["diversity_threshold"]:
            log.info("  Diversity below threshold, injecting diversity...")
            target = pop.inject_diversity()
            if target:
                log.info(f"  Resetting {target.branch_id} for bold exploration")
                # The diversity injection branch will get a bold mutation next generation

        # Log and save
        pop.log_generation(gen)
        pop.save_phylogeny()

        # Print generation summary
        sorted_pop = sorted(pop.branches, key=lambda b: b.val_bpb)
        log.info(f"\n  Generation {gen} Summary:")
        for b in sorted_pop:
            log.info(
                f"    {b.branch_id}: val_bpb={b.val_bpb:.4f} "
                f"({b.origin}) {b.strategy_summary}"
            )
        log.info(f"  Best: {sorted_pop[0].branch_id} = {sorted_pop[0].val_bpb:.4f}")
        log.info(f"  Diversity: {diversity:.4f}")

    log.info("\n=== Swarm complete ===")
    best = min(pop.branches, key=lambda b: b.val_bpb)
    log.info(f"Best branch: {best.branch_id} with val_bpb={best.val_bpb:.4f}")
    log.info(f"Strategy: {best.strategy_summary}")


if __name__ == "__main__":
    run_swarm()
