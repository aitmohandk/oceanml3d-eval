from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import xarray as xr

_READERS: dict[str, Callable[..., Any]] = {}


def register_reference(kind: str):
    def deco(fn):
        _READERS[kind] = fn
        return fn
    return deco


@dataclass
class Reference:
    """Description of a reference dataset as written in a benchmark YAML."""
    name: str
    kind: str                     # drifters_points | gridded
    path: str
    options: dict[str, Any] = field(default_factory=dict)
    depth_m: float | None = None

    def load(self, first: str | None = None, last: str | None = None) -> pd.DataFrame | xr.Dataset:
        if self.kind not in _READERS:
            raise KeyError(f"unknown reference kind '{self.kind}', available: {sorted(_READERS)}")
        return _READERS[self.kind](self.path, first=first, last=last, **self.options)


def get_reference(cfg: dict[str, Any]) -> Reference:
    return Reference(**cfg)
