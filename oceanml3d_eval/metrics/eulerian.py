"""Eulerian comparison against point velocity observations (drifters).

Product is interpolated (bilinear in space, nearest/linear in time) at every observation,
then RMSE / bias / correlation / explained variance are computed per region, and binned
maps of RMSE are returned as diagnostics. Replaces ``velocity_metrics.eulerian`` +
``run_rmse_GL.py`` with a vectorised xarray implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from oceanml3d_eval.metrics.base import Metric, MetricResult, register_metric
from oceanml3d_eval.regions import Region


def colocate(product: xr.Dataset, obs: pd.DataFrame, variables=("u", "v"), time_method: str = "nearest") -> pd.DataFrame:
    """Return ``obs`` with extra columns ``<var>_prod`` sampled from ``product``."""
    if obs.empty:
        return obs.assign(**{f"{v}_prod": np.nan for v in variables})
    pts = {"time": xr.DataArray(obs.time.values, dims="obs"),
           "lat": xr.DataArray(obs.lat.values, dims="obs"),
           "lon": xr.DataArray(obs.lon.values, dims="obs")}
    sampled = product[list(variables)].interp(lat=pts["lat"], lon=pts["lon"], method="linear")
    sampled = sampled.interp(time=pts["time"], method=time_method) if time_method != "nearest" \
        else sampled.sel(time=pts["time"], method="nearest")
    out = obs.copy()
    for v in variables:
        out[f"{v}_prod"] = np.asarray(sampled[v].values)
    return out


def scores(df: pd.DataFrame, variables=("u", "v")) -> dict[str, float]:
    res: dict[str, float] = {}
    speed_err2 = 0.0
    speed_var = 0.0
    n = 0
    for v in variables:
        d = df[[v, f"{v}_prod"]].dropna()
        if d.empty:
            res.update({f"rmse_{v}": np.nan, f"bias_{v}": np.nan, f"corr_{v}": np.nan, f"var_explained_{v}": np.nan})
            continue
        err = d[f"{v}_prod"] - d[v]
        res[f"rmse_{v}"] = float(np.sqrt((err ** 2).mean()))
        res[f"bias_{v}"] = float(err.mean())
        res[f"corr_{v}"] = float(np.corrcoef(d[v], d[f"{v}_prod"])[0, 1]) if len(d) > 2 else np.nan
        res[f"var_explained_{v}"] = float(1 - err.var() / d[v].var()) if d[v].var() > 0 else np.nan
        speed_err2 += float((err ** 2).sum())
        speed_var += float(((d[v] - d[v].mean()) ** 2).sum())
        n = max(n, len(d))
    res["rmse_vec"] = float(np.sqrt(speed_err2 / max(n, 1)))
    res["var_explained_vec"] = float(1 - speed_err2 / speed_var) if speed_var > 0 else np.nan
    res["n_obs"] = int(n)
    return res


def binned_rmse(df: pd.DataFrame, region: Region, bin_deg: float, variables=("u", "v")) -> xr.Dataset:
    lat_edges = np.arange(region.lat_min, region.lat_max + bin_deg, bin_deg)
    lon_edges = np.arange(region.lon_min, region.lon_max + bin_deg, bin_deg)
    lat_c, lon_c = 0.5 * (lat_edges[1:] + lat_edges[:-1]), 0.5 * (lon_edges[1:] + lon_edges[:-1])
    out = xr.Dataset(coords={"lat": lat_c, "lon": lon_c})
    for v in variables:
        d = df[[v, f"{v}_prod", "lat", "lon"]].dropna()
        err2 = (d[f"{v}_prod"] - d[v]) ** 2
        s, _, _ = np.histogram2d(d.lat, d.lon, bins=[lat_edges, lon_edges], weights=err2)
        c, _, _ = np.histogram2d(d.lat, d.lon, bins=[lat_edges, lon_edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"rmse_{v}"] = (("lat", "lon"), np.sqrt(s / c))
        out["n_obs"] = (("lat", "lon"), c)
    return out


@register_metric("eulerian_drifters")
class EulerianDrifters(Metric):
    needs = "drifters_points"

    def compute(self, product: xr.Dataset, reference: pd.DataFrame, region: Region, first: str, last: str) -> MetricResult:
        variables = tuple(self.options.get("variables", ("u", "v")))
        bin_deg = float(self.options.get("bin_deg", 2.0))
        obs = reference[(reference.time >= pd.Timestamp(first)) & (reference.time <= pd.Timestamp(last) + pd.Timedelta(days=1))]
        obs = obs[region.mask(obs.lat.values, obs.lon.values)]
        prod = region.subset(product).sel(time=slice(first, last)).load()
        df = colocate(prod, obs, variables, self.options.get("time_method", "nearest"))
        return MetricResult(scores(df, variables), {"rmse_map": binned_rmse(df, region, bin_deg, variables)},
                            {"region": region.name, "first": first, "last": last})
