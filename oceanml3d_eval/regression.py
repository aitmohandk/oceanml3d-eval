"""Pin reference scores and fail when they move.

The migration risk of this project is not a crash, it is a *silent* change of the science: a
refactor that shifts an RMSE by 3% looks like nothing in a log. ``save_baseline`` freezes the
scores of a product on a benchmark; ``check_baseline`` re-computes and compares within tolerance,
so the reference NOSC experiment can gate a pull request.

    oceanml3d-eval baseline -b surface_currents_15m -p .../product.yaml --save
    oceanml3d-eval baseline -b surface_currents_15m -p .../product.yaml --check --rtol 1e-3
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

BASELINE_DIR = Path("baselines")
KEY_COLUMNS = ("product", "benchmark", "metric", "region")


def _records(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        key = "|".join(str(row[c]) for c in KEY_COLUMNS)
        out[key] = {k: float(v) for k, v in row.items()
                    if k not in KEY_COLUMNS and isinstance(v, int | float) and not isinstance(v, bool)}
    return out


def save_baseline(df: pd.DataFrame, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "scores": _records(df)}, indent=2, sort_keys=True))
    return path


def compare(df: pd.DataFrame, path: str | Path, rtol: float = 1e-3, atol: float = 1e-8) -> list[str]:
    """Return the list of differences with the stored baseline (empty = reproduced)."""
    ref = json.loads(Path(path).read_text())["scores"]
    new = _records(df)
    problems: list[str] = []
    for key in sorted(set(ref) - set(new)):
        problems.append(f"{key}: missing from the new run")
    for key in sorted(set(new) - set(ref)):
        problems.append(f"{key}: not in the baseline (new metric/region — re-save if intended)")
    for key in sorted(set(ref) & set(new)):
        for score, old in ref[key].items():
            if score not in new[key]:
                problems.append(f"{key}/{score}: missing from the new run")
                continue
            cur = new[key][score]
            if math.isnan(old) and math.isnan(cur):
                continue
            if math.isnan(old) or math.isnan(cur) or abs(cur - old) > atol + rtol * abs(old):
                delta = "nan" if math.isnan(old) or old == 0 else f"{100 * (cur - old) / abs(old):+.2f}%"
                problems.append(f"{key}/{score}: {old:.6g} -> {cur:.6g} ({delta})")
    return problems


def check_baseline(df: pd.DataFrame, path: str | Path, rtol: float = 1e-3, atol: float = 1e-8) -> None:
    problems = compare(df, path, rtol, atol)
    if problems:
        raise SystemExit(f"scores differ from the baseline {path} (rtol={rtol}):\n  - " + "\n  - ".join(problems))
