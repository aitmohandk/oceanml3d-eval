"""Turn a yearly/multi-year NetCDF (or a Zarr store) into the daily-file product layout + manifest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from oceanml3d_eval.product import _DIM_ALIASES


def split_to_product(src: str | Path, out_dir: str | Path, name: str, variables: dict[str, str],
                     depth_m: float | None = None, depth_index: int | None = None, time_slice: tuple | None = None) -> Path:
    """``variables``: canonical name -> name in ``src`` (e.g. {u: ugos, v: vgos})."""
    ds = xr.open_zarr(src) if str(src).endswith(".zarr") else xr.open_dataset(src, chunks={"time": 30})
    ds = ds.rename({k: v for k, v in _DIM_ALIASES.items() if k in ds.dims})
    if "depth" in ds.dims:
        ds = ds.isel(depth=depth_index or 0, drop=True)
    ds = ds[list(variables.values())].rename({v: k for k, v in variables.items()})
    if time_slice:
        ds = ds.sel(time=slice(*time_slice))
    daily = Path(out_dir) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for t in ds.time.values:
        day = pd.Timestamp(t).strftime("%Y-%m-%d")
        f = daily / f"{name}_{day}.nc"
        if not f.exists():
            ds.sel(time=[t]).load().to_netcdf(f)
    manifest = {"name": name, "path": str(daily.resolve()), "pattern": f"{name}_(\\d{{4}})-(\\d{{2}})-(\\d{{2}})\\.nc",
                "variables": {k: {"name": k} for k in variables}, "coords": {"lat": "lat", "lon": "lon", "time": "time"},
                "time_coverage_hours": 24, "depth_m": depth_m,
                "first_date": pd.Timestamp(ds.time.values[0]).strftime("%Y-%m-%d"),
                "last_date": pd.Timestamp(ds.time.values[-1]).strftime("%Y-%m-%d"), "attrs": {"source": str(src)}}
    p = Path(out_dir) / "product.yaml"
    p.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return p
