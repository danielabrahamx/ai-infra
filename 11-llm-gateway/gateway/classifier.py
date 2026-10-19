"""Prompt-complexity classifiers. Score is in [0, 1]. Higher = harder.

A classifier is any callable `(prompt: str) -> float`. Swap it in create_app().
"""
import re
from typing import Callable

Classifier = Callable[[str], float]

CODE_RE = re.compile(r"```|\bdef \w+\(|\bclass \w+|#include|\bfunction\b|=>|\bSELECT\b.+\bFROM\b", re.I)
MATH_RE = re.compile(r"[∑∫√≤≥≠±]|\\frac|\\sum|\\int|\d\s*[\^*/]\s*\d|\b(integral|derivative|prove|theorem|equation|matrix|probability)\b", re.I)
REASON_RE = re.compile(r"\b(step[- ]by[- ]step|explain why|analy[sz]e|compare|trade-?offs?|design|architecture|optimi[sz]e|"
                       r"derive|prove|debug|refactor|reason|plan|evaluate|critique|why)\b", re.I)

WEIGHTS = {"length": 0.25, "code": 0.25, "math": 0.2, "reasoning": 0.25, "multi_question": 0.05}


def features(prompt: str) -> dict:
    """Each feature is in [0, 1]."""
    words = len(prompt.split())
    return {
        "length": min(words / 300.0, 1.0),                     # long prompts are harder
        "code": 1.0 if CODE_RE.search(prompt) else 0.0,
        "math": 1.0 if MATH_RE.search(prompt) else 0.0,
        "reasoning": min(len(REASON_RE.findall(prompt)) / 2.0, 1.0),
        "multi_question": min(prompt.count("?") / 3.0, 1.0),
    }


def heuristic(prompt: str) -> float:
    """Weighted sum of simple text features."""
    f = features(prompt)
    return round(sum(WEIGHTS[k] * v for k, v in f.items()), 4)


def constant(score: float) -> Classifier:
    """Constant classifier. Useful for tests and forced routing."""
    return lambda _p: score
