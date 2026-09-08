"""Registry for the canonical closure-recovery benchmark problems."""

from __future__ import annotations

import importlib


_PROBLEMS = {
    "linear": "linear_system",
    "lorenz63": "lorenz63_system",
    "lorenz96": "lorenz96_system",
}


def available_problems():
    return tuple(_PROBLEMS.keys())


def get_problem(name: str):
    key = name.lower().strip()
    if key not in _PROBLEMS:
        raise ValueError(
            f"Unknown system '{name}'. Available: {', '.join(_PROBLEMS)}"
        )
    return importlib.import_module(_PROBLEMS[key])
