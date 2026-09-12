"""Point observations of velocity: columns ``time, lat, lon, u, v`` (+ optional ``id``).

Readers accept NetCDF (AOML yearly files, CMEMS drifter files, or any table-like NetCDF),
CSV/Parquet, and the ``.pyo.gz`` pickles of velocity_metrics.
"""
from __future__ import annotations

import gzip
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanml3d_eval.reference.base import register_reference

DEFAULT_COLUMNS = {"time": "time", "lat": "lat", "lon": "lon", "u": "u", "v": "v"}


def _standardise(df: pd.DataFrame, columns: dict[str, str], scale: float, max_speed: float | None) -> pd.DataFrame:
    df = df.rename(columns={v: k for k, v in columns.items()})
    df = df[[c for c in ("time", "lat", "lon", "u", "v", "id") if c in df]].copy()
    df["time"] = pd.to_datetime(df["time"])
    df[["u", "v"]] = df[["u", "v"]].astype("float64") * scale
    df["lon"] = ((df["lon"] + 180) % 360) - 180
    if max_speed is not None:
        df = df[(df.u.abs() < max_speed) & (df.v.abs() < max_speed)]
    return df.dropna(subset=["lat", "lon", "u", "v"]).reset_index(drop=True)


@register_reference("drifters_points")
def read_drifters(path: str, first: str | None = None, last: str | None = None,
                  columns: dict[str, str] | None = None, scale: float = 1.0,
                  max_speed: float | None = 10.0, **_) -> pd.DataFrame:
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    files = sorted(glob(path)) if any(ch in path for ch in "*?[") else [path]
    frames = []
    for f in files:
        f = Path(f)
        if f.suffix in (".csv", ".txt"):
            frames.append(pd.read_csv(f))
        elif f.suffix == ".parquet":
            frames.append(pd.read_parquet(f))
        elif f.name.endswith(".pyo.gz") or f.suffix == ".pyo":
            frames.append(_read_pyo(f))
        else:
            ds = xr.open_dataset(f)
            frames.append(ds[[cols[k] for k in ("time", "lat", "lon", "u", "v") if cols[k] in ds]].to_dataframe().reset_index())
    df = _standardise(pd.concat(frames, ignore_index=True), cols, scale, max_speed)
    if first:
        df = df[df.time >= pd.Timestamp(first)]
    if last:
        df = df[df.time <= pd.Timestamp(last) + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def _read_pyo(path: Path) -> pd.DataFrame:
    """velocity_metrics drifter pickle: dict of arrays (time, lon, lat, u, v, ...)."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as f:
        d = pickle.load(f)
    df = pd.DataFrame({k: np.asarray(d[k]).ravel() for k in d if np.ndim(d[k]) >= 1})
    if "time" in df and not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce") if df["time"].dtype.kind in "fi" else pd.to_datetime(df["time"])
    return df
