"""Probabilistic scores for ensemble products (member dimension), against a gridded truth.

Ported from ``4dvarnet-fm-opencode`` ``evaluation/metrics.py`` (CRPS, energy score, spread) and
extended with the two diagnostics an ensemble reanalysis is judged on: the spread-skill ratio and
the rank histogram (flatness = calibration). A deterministic product is scored as a 1-member
ensemble, so the same benchmark line compares a U-Net with an EnKF.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from oceanml3d_eval.metrics.base import Metric, MetricResult, register_metric
from oceanml3d_eval.metrics.gridded import align, select_variables
from oceanml3d_eval.regions import Region


def crps_ensemble(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Fair CRPS by the energy form, averaged over all finite truth points.

    ``ensemble``: (n_members, ...), ``truth``: (...). For one member this reduces to the MAE.
    """
    m = ensemble.shape[0]
    ok = np.isfinite(truth) & np.isfinite(ensemble).all(axis=0)
    if not ok.any():
        return float("nan")
    e, t = ensemble[:, ok], truth[ok]
    term1 = np.abs(e - t[None]).mean(axis=0)
    if m == 1:
        return float(term1.mean())
    diff = np.abs(e[:, None] - e[None, :]).sum(axis=(0, 1)) / (2 * m * (m - 1))   # fair (unbiased) form
    return float((term1 - diff).mean())


def energy_score(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Multivariate generalisation of CRPS: the state at one time is treated as one vector."""
    m = ensemble.shape[0]
    e = ensemble.reshape(m, -1)
    t = truth.reshape(-1)
    ok = np.isfinite(t) & np.isfinite(e).all(axis=0)
    if not ok.any():
        return float("nan")
    e, t = e[:, ok], t[ok]
    term1 = np.linalg.norm(e - t[None], axis=1).mean()
    if m == 1:
        return float(term1)
    pair = np.linalg.norm(e[:, None] - e[None, :], axis=2).sum() / (2 * m * (m - 1))
    return float(term1 - pair)


def spread_skill(ensemble: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """(spread, RMSE of the ensemble mean, calibration ratio).

    For an m-member ensemble drawn from the same distribution as the truth, the error of the
    ensemble *mean* is larger than the spread by ``sqrt((m+1)/m)`` (the mean averages m draws, the
    truth is one more draw). The ratio below carries that factor, so a calibrated ensemble scores
    ~1 whatever m; < 1 means over-confident (spread too small), > 1 over-dispersive.
    """
    m = ensemble.shape[0]
    mean = np.nanmean(ensemble, axis=0)
    ok = np.isfinite(truth) & np.isfinite(mean)
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    var = np.nanvar(ensemble, axis=0, ddof=1) if m > 1 else np.zeros_like(mean)
    spread = float(np.sqrt(np.nanmean(var[ok])))
    rmse = float(np.sqrt(np.nanmean((mean[ok] - truth[ok]) ** 2)))
    ratio = spread * np.sqrt((m + 1) / m) / rmse if rmse > 0 else float("nan")
    return spread, rmse, float(ratio)


def rank_histogram(ensemble: np.ndarray, truth: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """Counts of the truth's rank among the members (ties broken at random)."""
    m = ensemble.shape[0]
    ok = np.isfinite(truth) & np.isfinite(ensemble).all(axis=0)
    e, t = ensemble[:, ok], truth[ok]
    below = (e < t[None]).sum(axis=0)
    ties = (e == t[None]).sum(axis=0)
    rng = np.random.default_rng(rng_seed)
    ranks = below + (rng.integers(0, ties + 1) if ties.any() else 0)
    return np.bincount(ranks, minlength=m + 1)[: m + 1]


@register_metric("ensemble_scores")
class EnsembleScores(Metric):
    needs = "gridded"

    def compute(self, product: xr.Dataset, reference: xr.Dataset, region: Region, first: str, last: str) -> MetricResult:
        variables = select_variables(product, reference, self.options)
        prod = region.subset(product).sel(time=slice(first, last))
        prod, truth = align(prod, reference.sel(time=slice(first, last)), variables)
        res: dict[str, float] = {}
        diag: dict[str, xr.Dataset] = {}
        for v in variables:
            p = prod[v].load()
            if "member" not in p.dims:
                p = p.expand_dims(member=[0])
            e = p.transpose("member", ...).values
            t = truth[v].load().values
            res[f"crps_{v}"] = crps_ensemble(e, t)
            res[f"energy_score_{v}"] = energy_score(e, t)
            spread, rmse, ratio = spread_skill(e, t)
            res[f"spread_{v}"], res[f"rmse_mean_{v}"], res[f"spread_skill_{v}"] = spread, rmse, ratio
            if e.shape[0] > 1:
                counts = rank_histogram(e, t)
                diag[f"rank_histogram_{v}"] = xr.Dataset({"count": ("rank", counts)},
                                                        coords={"rank": np.arange(len(counts))})
        res["n_members"] = int(prod.sizes.get("member", 1))
        return MetricResult(res, diag, {"region": region.name})
