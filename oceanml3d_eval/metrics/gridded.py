"""Grid-to-grid scores against a gridded reference (OSSE truth or reference L4 product).

Per variable: RMSE and bias in physical units, RMSE normalised by the truth's *anomaly* standard
deviation, anomaly correlation, explained variance, and the SSH data-challenge pair ``mu``/``sigma``.

Two choices that decide whether the numbers mean anything:

**Normalise by the anomaly, not by the field.** ``rmse / rms(truth)`` is the SSH data-challenge
convention, and SSH is a near-zero-mean field. Temperature is not: with a mean of 18 degC, the same
0.5 degC error scores ``nrmse = 0.028`` instead of ``0.49``, i.e. ``mu = 0.97`` for a mediocre model,
and the skill-vs-depth curve of a 3D task is flattened level by level. ``nrmse`` here divides by the
standard deviation of the truth's anomaly (its time mean removed at each cell), which is the same
question for every variable: how much of the variability is reproduced. ``mu``/``sigma`` are kept,
unchanged and clearly named, for continuity with the published SSH challenges.

**Resolve the scores in time.** One number over a year hides a drift, a bad season or a few broken
days: ``scores_by_date`` is saved as a diagnostic table (one row per date and variable, RMSE and
bias), and ``rmse_<v>_DJF`` … per season are added for the variables of ``seasonal_variables``
(every variable when a benchmark scores eight or fewer; a 64-variable 3D task would otherwise add
256 columns to the leaderboard).

**Weight by cell area.** A regular lat/lon grid over-samples high latitudes; an unweighted mean over
a global region gives a cell at 60 degN twice the weight of one at the equator. Every mean below is
weighted by ``cos(lat)``. Pass ``area_weighted: false`` in the metric options for the plain mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from oceanml3d_eval.metrics.base import Metric, MetricResult, register_metric
from oceanml3d_eval.regions import Region


def check_same_grid(product: xr.Dataset, truth: xr.Dataset, tol: float = 0.5) -> None:
    """Refuse a truth that is not on the product's grid, instead of interpolating it quietly.

    ``interp`` always succeeds: score a 0.25 deg product against a 1/12 deg truth and the truth is
    smoothed onto the coarse grid, which flatters the model exactly where it is weakest -- the fine
    scales. The rule is the one ``oceanml3d-core`` applies when it stacks variables: every target
    cell must sit within half a grid step of a source cell. Pass ``regrid: true`` in the metric
    options to interpolate deliberately.
    """
    for dim in ("lat", "lon"):
        if dim not in product.coords or dim not in truth.coords:
            continue
        dst = np.asarray(product[dim].values, float)
        src = np.asarray(truth[dim].values, float)
        if src.size == 0 or dst.size == 0:
            raise ValueError(f"empty '{dim}' axis: the region selects nothing of the product or the truth")
        steps = [float(np.median(np.abs(np.diff(a)))) for a in (src, dst) if a.size > 1]
        if not steps:
            continue
        step = min(steps)                    # the finer of the two grids sets the tolerance
        worst = float(np.abs(dst[:, None] - src[None, :]).min(axis=1).max())
        if worst > tol * step:
            raise ValueError(
                f"the truth is not on the product's grid along '{dim}': the worst product cell is "
                f"{worst:.4g} deg from its nearest truth cell, more than {tol} x the finer step "
                f"({step:.4g} deg). Prepare both at the same resolution, or set `regrid: true` in the "
                f"metric options to interpolate the truth on purpose.")


def align(product: xr.Dataset, truth: xr.Dataset, variables: list[str],
          regrid: bool = False, time_tolerance_h: int = 12) -> tuple[xr.Dataset, xr.Dataset]:
    truth = truth[variables]
    if not regrid:
        check_same_grid(product, truth)
    truth = truth.interp(lat=product.lat, lon=product.lon)
    matched = truth.reindex(time=product.time, method="nearest",
                            tolerance=np.timedelta64(time_tolerance_h, "h"))
    if "time" in matched.dims:
        missing = _missing_times(matched, variables)
        if missing == matched.sizes["time"]:
            raise ValueError(
                f"no truth time step within {time_tolerance_h} h of the product's "
                f"[{str(product.time.values[0])[:10]}, {str(product.time.values[-1])[:10]}]: "
                f"the two do not cover the same period.")
        if missing:
            print(f"[gridded] {missing}/{matched.sizes['time']} dates have no truth within "
                  f"{time_tolerance_h} h and are scored as missing")
    return product[variables], matched


def _missing_times(matched: xr.Dataset, variables: list[str]) -> int:
    first = matched[variables[0]]
    dims = [d for d in first.dims if d != "time"]
    return int((~np.isfinite(first)).all(dims).sum()) if dims else 0


def select_variables(product: xr.Dataset, reference: xr.Dataset, options: dict) -> list[str]:
    """Variables to score: the ``variables`` option, else every field shared by product and truth.

    The intersection default lets depth-resolved benchmarks (``thetao_d00``…) work without
    restating their 60+ variable names in every metric block.
    """
    requested = options.get("variables")
    if requested is None:
        return [v for v in product.data_vars if v in reference.data_vars]
    return [v for v in requested if v in product and v in reference]


def area_weights(da: xr.DataArray) -> xr.DataArray:
    """``cos(lat)`` broadcast over the horizontal dims (1 everywhere if there is no latitude)."""
    if "lat" not in da.coords:
        return xr.ones_like(da.isel({d: 0 for d in da.dims if d != "lat"}, missing_dims="ignore"))
    return np.cos(np.deg2rad(da.lat)).clip(min=0.0)


def weighted_mean(da: xr.DataArray, weights: xr.DataArray, dims=("lat", "lon")) -> xr.DataArray:
    dims = [d for d in dims if d in da.dims]
    return da.weighted(weights.fillna(0.0)).mean(dims) if dims else da


def scores_for(pred: xr.DataArray, truth: xr.DataArray, weighted: bool = True) -> dict[str, float]:
    """Every score of one variable, over the whole (time, lat, lon) window."""
    w = area_weights(truth) if weighted else xr.ones_like(truth.lat)
    err = (pred - truth).load()
    truth = truth.load()
    both = np.isfinite(err) & np.isfinite(truth)
    err, truth = err.where(both), truth.where(both)

    mean_t = weighted_mean(truth, w, ("time", "lat", "lon"))
    anom = truth - truth.mean("time") if "time" in truth.dims else truth - mean_t
    anom_pred = (pred.where(both) - truth.mean("time")) if "time" in truth.dims else (pred.where(both) - mean_t)
    all_dims = ("time", "lat", "lon")

    rmse = float(np.sqrt(weighted_mean(err ** 2, w, all_dims)))
    std_anom = float(np.sqrt(weighted_mean(anom ** 2, w, all_dims)))
    bias = float(weighted_mean(err, w, all_dims))
    var_err = float(weighted_mean((err - bias) ** 2, w, all_dims))

    a, b = anom, anom_pred
    a_m, b_m = float(weighted_mean(a, w, all_dims)), float(weighted_mean(b, w, all_dims))
    cov = float(weighted_mean((a - a_m) * (b - b_m), w, all_dims))
    sa = float(np.sqrt(weighted_mean((a - a_m) ** 2, w, all_dims)))
    sb = float(np.sqrt(weighted_mean((b - b_m) ** 2, w, all_dims)))

    # SSH data-challenge pair, per time step then averaged: rmse(t) / rms_truth(t).
    rmse_t = np.sqrt(weighted_mean(err ** 2, w))
    rms_truth_t = np.sqrt(weighted_mean(truth ** 2, w))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (rmse_t / rms_truth_t).values

    return {
        "rmse": rmse,
        "bias": bias,
        "nrmse": rmse / std_anom if std_anom > 0 else float("nan"),
        "anom_corr": cov / (sa * sb) if sa > 0 and sb > 0 else float("nan"),
        "var_explained": 1 - var_err / std_anom ** 2 if std_anom > 0 else float("nan"),
        "mu": float(1 - np.nanmean(ratio)),           # data-challenge score, RMS-normalised
        "sigma": float(np.nanstd(ratio)),
    }


SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def scores_by_date(pred: xr.DataArray, truth: xr.DataArray, weighted: bool = True) -> pd.DataFrame:
    """One row per date: RMSE and bias. A single number over a year hides a drift or one bad month."""
    import pandas as pd

    w = area_weights(truth) if weighted else xr.ones_like(truth.lat)
    err = (pred - truth).load()
    rmse = np.sqrt(weighted_mean(err ** 2, w))
    bias = weighted_mean(err, w)
    return pd.DataFrame({"time": pd.DatetimeIndex(err.time.values),
                         "rmse": np.asarray(rmse.values, float), "bias": np.asarray(bias.values, float)})


def seasonal_scores(pred: xr.DataArray, truth: xr.DataArray, weighted: bool = True) -> dict[str, float]:
    """``{DJF: rmse, MAM: ...}``: the same RMSE, per season, pooled over its dates."""
    out = {}
    months = pd.DatetimeIndex(truth.time.values).month if "time" in truth.dims else []
    for season in ("DJF", "MAM", "JJA", "SON"):
        keep = np.array([SEASONS[m] == season for m in months])
        if keep.any():
            out[season] = scores_for(pred.isel(time=keep), truth.isel(time=keep), weighted)["rmse"]
    return out


@register_metric("gridded_rmse")
class GriddedRMSE(Metric):
    needs = "gridded"

    def compute(self, product: xr.Dataset, reference: xr.Dataset, region: Region, first: str, last: str) -> MetricResult:
        variables = select_variables(product, reference, self.options)
        weighted = bool(self.options.get("area_weighted", True))
        prod = region.subset(product).sel(time=slice(first, last))
        prod, truth = align(prod, reference.sel(time=slice(first, last)), variables,
                            regrid=bool(self.options.get("regrid", False)))
        res: dict[str, float] = {}
        maps = xr.Dataset()
        per_date = []
        seasonal_for = self.options.get("seasonal_variables", variables if len(variables) <= 8 else [])
        for v in variables:
            for score, value in scores_for(prod[v], truth[v], weighted).items():
                res[f"{score}_{v}"] = value
            for season, value in (seasonal_scores(prod[v], truth[v], weighted) if v in seasonal_for else {}).items():
                res[f"rmse_{v}_{season}"] = value
            maps[f"rmse_{v}"] = np.sqrt(((prod[v] - truth[v]) ** 2).mean("time"))
            day = scores_by_date(prod[v], truth[v], weighted)
            day.insert(1, "variable", v)
            per_date.append(day)
        diagnostics: dict = {"rmse_map": maps}
        if per_date:
            diagnostics["scores_by_date"] = pd.concat(per_date, ignore_index=True)
        return MetricResult(res, diagnostics, {"region": region.name, "area_weighted": weighted})
