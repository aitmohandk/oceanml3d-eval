"""Turn a yearly/multi-year NetCDF (or a Zarr store) into the daily-file product layout + manifest.

The one converter that makes a *reference* out of data that is not a product yet: the GLORYS truth
of an OSSE, DUACS, GlobCurrent... For a 3D task, `depth_indices` writes one variable per level,
`<canonical>_d<ii>`, which is the naming the product format uses and what `oceanml3d-core` exports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from oceanml3d_eval.product import _DIM_ALIASES
from oceanml3d_eval.product_contract import PRODUCT_FORMAT_VERSION, check_manifest


def _depth_metres(ds: xr.Dataset, index: int) -> float | None:
    if "depth" not in ds.coords:
        return None
    try:
        return round(float(np.asarray(ds["depth"].values)[index]), 3)
    except (IndexError, TypeError, ValueError):
        return None


def _flatten_depth(ds: xr.Dataset, variables: dict[str, str], depth_indices: list[int]) -> tuple[xr.Dataset, dict]:
    """``thetao(depth, ...)`` -> ``thetao_d00, thetao_d02, ...``; 2D variables are kept as they are.

    Selecting by *position on the file's depth axis* -- not by metres -- because that is what the
    task's ``depth_index`` means in ``oceanml3d-core`` (``config/data/osse3d_gs21.yaml``) and what
    the product format's ``_d<ii>`` suffix records. Reading them as metres would silently shift
    every level the day the truth is prepared at another vertical sampling.
    """
    out, described = xr.Dataset(coords={c: ds[c] for c in ("time", "lat", "lon") if c in ds.coords}), {}
    for canon, src in variables.items():
        da = ds[src]
        if "depth" not in da.dims:
            out[canon] = da
            described[canon] = {"name": canon}
            continue
        n = da.sizes["depth"]
        for i in depth_indices:
            if i >= n:
                raise IndexError(f"depth index {i} is out of range for '{src}': the file has {n} levels")
            name = f"{canon}_d{i:02d}"
            level = da.isel(depth=i, drop=True)
            metres = _depth_metres(ds, i)
            level.attrs.update({k: v for k, v in (("depth_index", i), ("depth_m", metres)) if v is not None})
            out[name] = level
            described[name] = {"name": name, "depth_index": i, **({"depth_m": metres} if metres is not None else {})}
    return out, described


def split_to_product(src: str | Path, out_dir: str | Path, name: str, variables: dict[str, str],
                     depth_m: float | None = None, depth_index: int | None = None,
                     depth_indices: list[int] | None = None, time_slice: tuple | None = None) -> Path:
    """``variables``: canonical name -> name in ``src`` (e.g. {u: ugos, v: vgos}).

    ``depth_indices``: keep these levels, one variable per level (3D reference). ``depth_index``:
    a single level, kept under the plain canonical name (surface reference, the 2D default).
    """
    ds = xr.open_zarr(src) if str(src).endswith(".zarr") else xr.open_dataset(src, chunks={"time": 30})
    ds = ds.rename({k: v for k, v in _DIM_ALIASES.items() if k in ds.dims})
    missing = [v for v in variables.values() if v not in ds.variables]
    if missing:
        raise KeyError(f"{src}: no variable {missing} (has: {sorted(ds.data_vars)})")

    if depth_indices:
        ds, described = _flatten_depth(ds, variables, list(depth_indices))
    else:
        if "depth" in ds.dims:
            ds = ds.isel(depth=depth_index or 0, drop=True)
        ds = ds[list(variables.values())].rename({v: k for k, v in variables.items()})
        described = {k: {"name": k} for k in variables}
    if time_slice:
        ds = ds.sel(time=slice(*time_slice))
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"{src}: no time step left for {time_slice}")

    daily = Path(out_dir) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for t in ds.time.values:
        day = pd.Timestamp(t).strftime("%Y-%m-%d")
        f = daily / f"{name}_{day}.nc"
        if not f.exists():
            ds.sel(time=[t]).load().to_netcdf(f)
    manifest = {"format_version": PRODUCT_FORMAT_VERSION,
                "name": name, "path": str(daily.resolve()), "pattern": f"{name}_(\\d{{4}})-(\\d{{2}})-(\\d{{2}})\\.nc",
                "variables": described, "coords": {"lat": "lat", "lon": "lon", "time": "time"},
                "time_coverage_hours": 24, "depth_m": depth_m,
                "first_date": pd.Timestamp(ds.time.values[0]).strftime("%Y-%m-%d"),
                "last_date": pd.Timestamp(ds.time.values[-1]).strftime("%Y-%m-%d"),
                "attrs": {"source": str(src)}}
    check_manifest(manifest, source=str(Path(out_dir) / "product.yaml"))
    p = Path(out_dir) / "product.yaml"
    p.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"[split] {len(list(daily.glob(f'{name}_*.nc')))} daily files, "
          f"{len(described)} variables -> {p}")
    return p


def parse_indices(text: str, ds_levels: int | None = None) -> list[int]:
    """``"0,2,4"`` or ``"0-4"`` or ``"all"`` -> a list of depth indices."""
    text = str(text).strip()
    if text in ("all", "*"):
        if ds_levels is None:
            raise ValueError("'all' needs the number of levels")
        return list(range(ds_levels))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out
