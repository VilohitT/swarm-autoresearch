"""LLM-mediated crossover of training scripts via Anthropic API."""

from __future__ import annotations

import re

import anthropic

from population import pairwise_diversity


CROSSOVER_PROMPT = """You are an ML research engineer combining two training scripts.

## Parent A (val_bpb: {score_a:.4f})
```python
{code_a}
```

## Parent B (val_bpb: {score_b:.4f})
```python
{code_b}
```

Instructions:
1. Identify key decisions in each: normalization, optimizer, LR schedule, depth/width, init, tricks.
2. Create a new train.py taking at least one significant decision from each parent.
3. Add a comment block at the top:
   # CROSSOVER
   # From Parent A: [what]
   # From Parent B: [what]
   # Hypothesis: [why this combo might work]
4. Output ONLY valid Python runnable via: uv run train.py
"""

STRONGER_CROSSOVER_PROMPT = """You are an ML research engineer combining two training scripts.
IMPORTANT: You MUST genuinely combine ideas from BOTH parents. Do NOT just copy one parent.

## Parent A (val_bpb: {score_a:.4f})
```python
{code_a}
```

## Parent B (val_bpb: {score_b:.4f})
```python
{code_b}
```

Instructions:
1. Identify key decisions in each: normalization, optimizer, LR schedule, depth/width, init, tricks.
2. Create a NOVEL train.py that takes at least TWO significant decisions from EACH parent.
3. Make the combination non-trivial — the resulting architecture should be meaningfully different from both parents.
4. Add a comment block at the top:
   # CROSSOVER
   # From Parent A: [what]
   # From Parent B: [what]
   # Hypothesis: [why this combo might work]
5. Output ONLY valid Python runnable via: uv run train.py
"""


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response, handling markdown fences."""
    # Try to find fenced code block
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code fence
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Assume the whole response is code
    return text.strip()


def llm_crossover(
    code_a: str,
    code_b: str,
    score_a: float,
    score_b: float,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_retries: int = 2,
) -> str:
    """Generate a crossover child from two parent training scripts using an LLM.

    If the initial crossover is too similar to a parent (pairwise_diversity < 0.1),
    re-prompts with a stronger instruction.

    Args:
        code_a: Source code of parent A's train.py
        code_b: Source code of parent B's train.py
        score_a: Parent A's validation bpb
        score_b: Parent B's validation bpb
        model: Anthropic model to use
        temperature: Sampling temperature
        max_retries: Number of retries with stronger prompt if child is too similar

    Returns:
        The crossover child's train.py code
    """
    client = anthropic.Anthropic()

    prompt = CROSSOVER_PROMPT.format(
        code_a=code_a, code_b=code_b, score_a=score_a, score_b=score_b
    )

    for attempt in range(1 + max_retries):
        if attempt > 0:
            # Use stronger prompt on retry
            prompt = STRONGER_CROSSOVER_PROMPT.format(
                code_a=code_a, code_b=code_b, score_a=score_a, score_b=score_b
            )

        response = client.messages.create(
            model=model,
            max_tokens=8000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        child_code = extract_python_code(response.content[0].text)

        # Check diversity against both parents
        div_a = pairwise_diversity(child_code, code_a)
        div_b = pairwise_diversity(child_code, code_b)

        if div_a >= 0.1 and div_b >= 0.1:
            return child_code

    # Return whatever we got on the last attempt
    return child_code


def generate_strategy_summary(code: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Generate a one-line strategy summary for a training script."""
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=100,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this ML training script in ONE short line (max 60 chars). "
                "Focus on: normalization, optimizer, depth, key tricks.\n\n"
                f"```python\n{code[:3000]}\n```"
            ),
        }],
    )

    return response.content[0].text.strip()
