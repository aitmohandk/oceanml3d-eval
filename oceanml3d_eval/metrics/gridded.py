"""Grid-to-grid scores against a gridded reference (OSSE truth or reference L4 product):
RMSE, normalised RMSE, temporal mean/std of the error (SSH data-challenge ``mu``, ``sigma``)."""
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


@register_metric("gridded_rmse")
class GriddedRMSE(Metric):
    needs = "gridded"

    def compute(self, product: xr.Dataset, reference: xr.Dataset, region: Region, first: str, last: str) -> MetricResult:
        variables = select_variables(product, reference, self.options)
        prod = region.subset(product).sel(time=slice(first, last))
        prod, truth = align(prod, reference.sel(time=slice(first, last)), variables)
        res: dict[str, float] = {}
        maps = xr.Dataset()
        for v in variables:
            err = (prod[v] - truth[v]).load()
            rmse_t = np.sqrt((err ** 2).mean(("lat", "lon")))
            rms_truth_t = np.sqrt((truth[v] ** 2).mean(("lat", "lon"))).load()
            res[f"rmse_{v}"] = float(np.sqrt((err ** 2).mean()))
            res[f"nrmse_{v}"] = float((rmse_t / rms_truth_t).mean())
            res[f"mu_{v}"] = float(1 - (rmse_t / rms_truth_t).mean())          # DC-style score
            res[f"sigma_{v}"] = float((rmse_t / rms_truth_t).std())
            maps[f"rmse_{v}"] = np.sqrt((err ** 2).mean("time"))
        return MetricResult(res, {"rmse_map": maps}, {"region": region.name})
