from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

import pandas as pd
import xarray as xr

from oceanml3d_eval.regions import Region

_METRICS: dict[str, type] = {}


def register_metric(name: str):
    def deco(cls):
        _METRICS[name] = cls
        cls.name = name
        return cls
    return deco


def get_metric(name: str) -> type:
    if name not in _METRICS:
        for ep in entry_points(group="oceanml3d_eval.metrics"):
            if ep.name == name:
                _METRICS[name] = ep.load()
    if name not in _METRICS:
        raise KeyError(f"unknown metric '{name}', available: {list_metrics()}")
    return _METRICS[name]


def list_metrics() -> list[str]:
    return sorted(_METRICS)


@dataclass
class MetricResult:
    scores: dict[str, float]                                  # flat scalars -> leaderboard
    diagnostics: dict[str, xr.Dataset | pd.DataFrame] = field(default_factory=dict)  # maps, tables -> report
    meta: dict[str, Any] = field(default_factory=dict)


class Metric:
    """Subclass and implement ``compute``. ``options`` come from the benchmark YAML."""
    name: str = "base"
    needs: str = "drifters_points"      # reference kind this metric consumes

    def __init__(self, **options: Any):
        self.options = options

    def compute(self, product: xr.Dataset, reference: Any, region: Region, first: str, last: str) -> MetricResult:
        raise NotImplementedError
