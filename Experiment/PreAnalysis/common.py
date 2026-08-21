"""Shared paths, model ids, and small helpers for the Cost(beta) pre-analysis."""
import os

TEACHER = "Qwen/Qwen3-14B"
ANCHOR = "Qwen/Qwen3-1.7B-Base"

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

SEED = 20260819

# Instruction appended to every problem. Kept minimal: it only fixes the answer
# format the verifier needs, and is identical for both models.
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(tokenizer, problem):
    """Teacher chat template, thinking mode, applied identically to both models."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem + "\n\n" + INSTRUCTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def extract_boxed(text):
    """Last \\boxed{...} in text, brace-balanced. None if absent."""
    key = "\\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out) if depth == 0 else None
