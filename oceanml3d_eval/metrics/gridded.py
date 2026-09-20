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

**Weight by cell area.** A regular lat/lon grid over-samples high latitudes; an unweighted mean over
a global region gives a cell at 60 degN twice the weight of one at the equator. Every mean below is
weighted by ``cos(lat)``. Pass ``area_weighted: false`` in the metric options for the plain mean.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from oceanml3d_eval.metrics.base import Metric, MetricResult, register_metric
from oceanml3d_eval.regions import Region


def align(product: xr.Dataset, truth: xr.Dataset, variables: list[str]) -> tuple[xr.Dataset, xr.Dataset]:
    truth = truth[variables].interp(lat=product.lat, lon=product.lon)
    truth = truth.reindex(time=product.time, method="nearest", tolerance=np.timedelta64(12, "h"))
    return product[variables], truth


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


@register_metric("gridded_rmse")
class GriddedRMSE(Metric):
    needs = "gridded"

    def compute(self, product: xr.Dataset, reference: xr.Dataset, region: Region, first: str, last: str) -> MetricResult:
        variables = select_variables(product, reference, self.options)
        weighted = bool(self.options.get("area_weighted", True))
        prod = region.subset(product).sel(time=slice(first, last))
        prod, truth = align(prod, reference.sel(time=slice(first, last)), variables)
        res: dict[str, float] = {}
        maps = xr.Dataset()
        for v in variables:
            for score, value in scores_for(prod[v], truth[v], weighted).items():
                res[f"{score}_{v}"] = value
            maps[f"rmse_{v}"] = np.sqrt(((prod[v] - truth[v]) ** 2).mean("time"))
        return MetricResult(res, {"rmse_map": maps}, {"region": region.name, "area_weighted": weighted})
