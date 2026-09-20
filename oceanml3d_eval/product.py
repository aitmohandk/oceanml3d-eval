"""A *product* is any gridded field we want to evaluate: a model output exported by
``oceanml3d-core`` or an external product (DUACS, GlobCurrent, NeurOST, GLORYS…).

Both are described by the same ``product.yaml`` manifest (see oceanml3d-core
``docs/product_format.md``). Legacy NOSC ``metric/dictionary/*.json`` files are also accepted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

from oceanml3d_eval.product_contract import check_manifest

_DIM_ALIASES = {"latitude": "lat", "longitude": "lon", "nav_lat": "lat", "nav_lon": "lon"}


@dataclass
class ProductSpec:
    name: str
    path: str
    pattern: str
    variables: dict[str, str]                 # canonical name -> name in files  (u -> ugos)
    coords: dict[str, str] = field(default_factory=lambda: {"lat": "lat", "lon": "lon", "time": "time"})
    depth_m: float | None = None
    depth_index: int | None = None
    ensemble_size: int | None = None
    time_coverage_hours: int = 24
    label: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)   # canonical -> multiplicative factor (units fix)
    lon_0_360: bool = False

    @classmethod
    def load(cls, path: str | Path) -> ProductSpec:
        path = Path(path)
        text = path.read_text()
        doc = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if "data_type" in doc:                       # legacy NOSC json descriptor
            return cls(name=doc["data_type"], path=doc["path"], pattern=doc["match"],
                       variables={"u": doc["varu"], "v": doc["varv"]},
                       coords={"lat": doc.get("nlat", "lat"), "lon": doc.get("nlon", "lon"), "time": "time"},
                       time_coverage_hours=doc.get("time_coverage_hours", 24), label=doc.get("label"))
        check_manifest(doc, source=str(path))
        variables = {k: (v["name"] if isinstance(v, dict) and "name" in v else k) for k, v in doc["variables"].items()}
        spec = cls(name=doc["name"], path=doc["path"], pattern=doc["pattern"], variables=variables,
                   coords=doc.get("coords", {"lat": "lat", "lon": "lon", "time": "time"}),
                   depth_m=doc.get("depth_m"), depth_index=doc.get("depth_index"),
                   ensemble_size=doc.get("ensemble_size"),
                   time_coverage_hours=doc.get("time_coverage_hours", 24), label=doc.get("label"),
                   attrs=doc.get("attrs", {}), scale=doc.get("scale", {}), lon_0_360=doc.get("lon_0_360", False))
        if not Path(spec.path).is_absolute():
            spec.path = str((path.parent / spec.path).resolve())
        return spec

    def files(self, first: str | None = None, last: str | None = None) -> list[Path]:
        rx = re.compile(self.pattern)
        out = []
        for f in sorted(Path(self.path).iterdir()):
            m = rx.fullmatch(f.name)
            if not m:
                continue
            if m.groups() and (first or last):
                day = pd.Timestamp("-".join(m.groups()[:3]))
                if first and day < pd.Timestamp(first) or last and day > pd.Timestamp(last):
                    continue
            out.append(f)
        return out


def open_product(spec: ProductSpec | str | Path, first: str | None = None, last: str | None = None,
                 variables: list[str] | None = None) -> xr.Dataset:
    """Open a product as an xarray Dataset with canonical names ``u, v, ssh…`` and dims ``time, lat, lon``."""
    if not isinstance(spec, ProductSpec):
        spec = ProductSpec.load(spec)
    files = spec.files(first, last)
    if not files:
        raise FileNotFoundError(f"no file matching {spec.pattern!r} in {spec.path} for [{first}, {last}]")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 30}) if len(files) > 1 else xr.open_dataset(files[0], chunks={"time": 30})
    rename = {v: k for k, v in spec.coords.items() if v in ds.dims or v in ds.coords}
    rename.update({k: v for k, v in _DIM_ALIASES.items() if k in ds.dims and v not in rename.values()})
    ds = ds.rename(rename)
    if "depth" in ds.dims:
        ds = ds.isel(depth=spec.depth_index or 0, drop=True)
    # `variables is None` means "everything the manifest declares"; an *empty list* means the caller
    # asked for nothing, which is a bug upstream -- `variables or list(...)` turned it into
    # "everything" and scored a product the benchmark had found no variable in (the uo_ vs u_ case).
    if variables is not None and not variables:
        raise ValueError(f"open_product({spec.name}): empty variable list. The caller asked for no "
                         f"variable; the product declares {sorted(spec.variables)[:8]}...")
    wanted = list(spec.variables) if variables is None else list(variables)
    unknown = [v for v in wanted if v not in spec.variables]
    if unknown:
        raise KeyError(f"{spec.name}: no variable {unknown} in the manifest "
                       f"(declared: {sorted(spec.variables)[:12]}{'...' if len(spec.variables) > 12 else ''})")
    out = xr.Dataset(coords={c: ds[c] for c in ("time", "lat", "lon") if c in ds.coords})
    for canon in wanted:
        src = spec.variables[canon]
        out[canon] = ds[src] * spec.scale.get(canon, 1.0)
    if spec.lon_0_360 or float(out.lon.max()) > 180:
        out = out.assign_coords(lon=((out.lon + 180) % 360) - 180).sortby("lon")
    if first or last:
        out = out.sel(time=slice(first, last))
    out.attrs.update({"product": spec.name, "depth_m": spec.depth_m,
                      "ensemble_size": spec.ensemble_size or (out.sizes.get("member"))})
    lead = ("member",) if "member" in out.dims else ()
    return out.transpose(*lead, "time", "lat", "lon", ...)
