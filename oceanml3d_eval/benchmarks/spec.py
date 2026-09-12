"""A benchmark = a task definition: period, references, regions, metrics, baselines."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from oceanml3d_eval.reference.base import Reference

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"


@dataclass
class MetricSpec:
    name: str
    reference: str                     # key in BenchmarkSpec.references
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSpec:
    name: str
    description: str
    first: str
    last: str
    references: dict[str, Reference]
    regions: list[str]
    metrics: list[MetricSpec]
    baselines: dict[str, str] = field(default_factory=dict)   # label -> product manifest path
    variables: list[str] = field(default_factory=lambda: ["u", "v"])
    leaderboard: list[str] = field(default_factory=list)      # score names shown first
    path: Path | None = None

    @classmethod
    def load(cls, name_or_path: str | Path, benchmarks_dir: Path = BENCHMARKS_DIR) -> BenchmarkSpec:
        path = Path(name_or_path)
        if not path.exists():
            path = benchmarks_dir / f"{name_or_path}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"benchmark '{name_or_path}' not found; available: {available_benchmarks(benchmarks_dir)}")
        doc = yaml.safe_load(os.path.expandvars(path.read_text()))
        refs = {k: Reference(name=k, **v) for k, v in doc["references"].items()}
        metrics = [MetricSpec(**m) for m in doc["metrics"]]
        base = {k: str((path.parent / v).resolve()) if not Path(v).is_absolute() else v for k, v in doc.get("baselines", {}).items()}
        return cls(doc["name"], doc.get("description", ""), str(doc["period"][0]), str(doc["period"][1]),
                   refs, doc["regions"], metrics, base, doc.get("variables", ["u", "v"]),
                   doc.get("leaderboard", []), path)


def available_benchmarks(benchmarks_dir: Path = BENCHMARKS_DIR) -> list[str]:
    return sorted(p.stem for p in benchmarks_dir.glob("*.yaml"))
