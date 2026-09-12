"""Bridge to the vendored OceanDataLab ``velocity_metrics`` package (NOSC's original diagnostics).

It works with legacy artefacts (product JSON descriptor, region JSON, drifter ``.pyo.gz`` /
position ``.json`` files, ``drifters_parameters.ini``); this module generates the descriptor
and region files from the oceanml3d objects and calls the package as ``run_lagrangian_GL.py`` did.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np

from oceanml3d_eval.product import ProductSpec
from oceanml3d_eval.regions import Region

FMT = "%Y%m%dT%H%M%SZ"


def write_product_descriptor(spec: ProductSpec, out: Path, depth: int = 0) -> Path:
    doc = {"data_type": spec.name, "label": spec.label or spec.name, "path": spec.path, "pattern": spec.name,
           "match": spec.pattern, "varu": spec.variables["u"], "varv": spec.variables["v"],
           "nlon": spec.coords.get("lon", "lon"), "nlat": spec.coords.get("lat", "lat"),
           "time_coverage_hours": spec.time_coverage_hours, "fmt": "%Y-%m-%dT%H:%M:%S.%fZ", "depth": depth}
    out.write_text(json.dumps(doc, indent=2))
    return out


def write_region(region: Region, out: Path) -> Path:
    poly = [list(p) for p in region.polygon] if region.polygon else [[region.lon_min, region.lat_min], [region.lon_max, region.lat_min],
                                                                     [region.lon_max, region.lat_max], [region.lon_min, region.lat_max],
                                                                     [region.lon_min, region.lat_min]]
    out.write_text(json.dumps({"name": region.name, "lllon": region.lon_min, "urlon": region.lon_max,
                               "lllat": region.lat_min, "urlat": region.lat_max, "coords": poly}))
    return out


def lagrangian(spec: ProductSpec, region: Region, first: str, last: str, work_dir: Path,
               drifter_positions: str, drifter_file: str, parameter_file: str, days: int = 10, sdepth: int = 0) -> dict[str, float]:
    """Advect fictive drifters in the product and compare with real trajectories (SDE score).
    ``drifter_positions``: velocity_metrics positions JSON; ``drifter_file``: real drifters ``.pyo.gz``;
    ``parameter_file``: ``drifters_parameters.ini`` (box size, advection settings)."""
    import velocity_metrics.lagrangian.cumulative_distance as sde
    import velocity_metrics.lagrangian.drifters as drifters

    work_dir.mkdir(parents=True, exist_ok=True)
    prod = write_product_descriptor(spec, work_dir / f"{spec.name}.json", sdepth)
    reg = write_region(region, work_dir / f"region_{region.name.replace(' ', '')}.json")
    f0 = dt.datetime.strptime(first, "%Y-%m-%d").strftime(FMT)
    f1 = dt.datetime.strptime(last, "%Y-%m-%d").strftime(FMT)
    out_dir = work_dir / "lagrangian"
    drifters.run_all_load_once(parameter_file, str(prod), drifter_positions, days_of_advection=days, output_dir=str(out_dir),
                               region=str(reg), first_date=f0, last_date=f1, sdepth=sdepth)
    artificial = out_dir / f"{spec.name}_region_{region.name}_dep{sdepth}.pyo.gz"
    res = sde.run(str(artificial), [drifter_file], output_dir=str(out_dir / "plot"),
                  output_filename=f"SDE_region_{region.name}_{first}-{last}", isplot=False)
    return _sde_scores(res, out_dir, days)


def _sde_scores(res, out_dir: Path, days: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(v, int | float | np.floating):
                scores[f"vm_{k}"] = float(v)
            elif isinstance(v, list | np.ndarray) and len(v) >= days:
                arr = np.asarray(v, float)
                scores[f"vm_{k}_day{days}"] = float(np.nanmean(arr[days - 1])) if arr.ndim else float(arr)
    for p in out_dir.glob("plot/*.pyo*"):
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if isinstance(d, dict) and "score" in d:
                scores.setdefault("vm_sde_score", float(np.nanmean(d["score"])))
        except Exception:  # noqa: BLE001
            continue
    return scores or {"vm_sde_score": float("nan")}
