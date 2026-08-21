"""Shared helpers for OpenRouter-based branch discovery generation."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
PREANALYSIS_OUT = os.path.join(os.path.dirname(ROOT), "PreAnalysis", "outputs")
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(ROOT)), ".env")

# Same instruction as the pre-analysis, so the task framing is identical across
# every source we generate from.
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

BASE_URL = "https://openrouter.ai/api/v1"


def load_key(path=ENV_PATH):
    """Read OPENROUTER_API_KEY from the .env file (tolerates spaces around '=')."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "OPENROUTER_API_KEY":
                return v.strip().strip('"').strip("'")
    raise RuntimeError(f"OPENROUTER_API_KEY not found in {path}")


def load_problems(n=None, path=None):
    """The same 200 stratified MATH problems used by the pre-analysis."""
    path = path or os.path.join(PREANALYSIS_OUT, "problems.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows if n is None else rows[:n]


def user_content(problem):
    return problem + "\n\n" + INSTRUCTION
