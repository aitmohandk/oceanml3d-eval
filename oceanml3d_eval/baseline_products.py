"""Reference baselines built from the truth itself: the lines every model must beat.

A score is a number until something else scores worse. For an OSSE there are two baselines that
need no extra data and no model, only the truth:

* **climatology** -- the truth's mean seasonal cycle, estimated over a *training* window and then
  read for the dates being scored. It is the "no skill" line: a model that does not beat it has
  learnt nothing beyond the season. With the anomaly-normalised scores it lands on
  ``var_explained = 0`` and ``nrmse = 1`` by construction.
* **persistence** -- the truth shifted by ``lag`` days. It is the line a reconstruction must beat to
  be worth more than yesterday's state, and on a fast-moving front like the Gulf Stream it is a
  demanding one at depth.

Both are written as ordinary products (daily files + manifest), so they go through exactly the same
metrics as the model: ``oceanml3d-eval run -b <bench> -p <baseline>/product.yaml``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from oceanml3d_eval.product import ProductSpec, open_product
from oceanml3d_eval.product_contract import PRODUCT_FORMAT_VERSION, check_manifest

KINDS = ("climatology", "persistence")


def climatology(truth: xr.Dataset, dates: pd.DatetimeIndex, train: tuple[str, str] | None = None,
                smooth_days: int = 31) -> xr.Dataset:
    """Day-of-year mean of ``truth`` over ``train``, smoothed, then read for ``dates``.

    The smoothing matters: with a few years of training a raw day-of-year mean is still noisy, and a
    noisy climatology is an artificially weak baseline. A centred rolling mean over ``smooth_days``
    days of the year (wrapped at the turn of the year) is the usual estimate.
    """
    src = truth.sel(time=slice(*train)) if train else truth
    if src.sizes.get("time", 0) == 0:
        raise ValueError(f"the truth has no date in the training window {train}")
    if train and pd.Timestamp(train[1]) >= pd.Timestamp(dates[0]):
        print(f"[baseline] WARNING: the climatology is trained up to {train[1]}, which is inside the "
              f"scored window starting {pd.Timestamp(dates[0]).date()}: the baseline sees the answers")
    doy = src.groupby("time.dayofyear").mean("time")
    if smooth_days > 1 and doy.sizes["dayofyear"] > smooth_days:
        wrapped = xr.concat([doy.isel(dayofyear=slice(-smooth_days, None)), doy,
                             doy.isel(dayofyear=slice(0, smooth_days))], dim="dayofyear")
        rolled = wrapped.rolling(dayofyear=smooth_days, center=True, min_periods=1).mean()
        n = doy.sizes["dayofyear"]
        doy = rolled.isel(dayofyear=slice(smooth_days, smooth_days + n))
        doy = doy.assign_coords(dayofyear=np.arange(1, n + 1))
    out = doy.sel(dayofyear=xr.DataArray(dates.dayofyear, dims="time", coords={"time": dates}))
    return out.drop_vars("dayofyear", errors="ignore")


def persistence(truth: xr.Dataset, dates: pd.DatetimeIndex, lag_days: int = 1) -> xr.Dataset:
    """The truth ``lag_days`` earlier, read for ``dates`` (missing dates stay NaN)."""
    wanted = dates - pd.Timedelta(days=lag_days)
    shifted = truth.reindex(time=wanted, method="nearest", tolerance=pd.Timedelta(hours=12))
    return shifted.assign_coords(time=dates)


def make_baseline(truth: str | Path | ProductSpec, out_dir: str | Path, kind: str, first: str, last: str,
                  name: str | None = None, train: tuple[str, str] | None = None, lag_days: int = 1,
                  smooth_days: int = 31) -> Path:
    """Write a ``climatology`` or ``persistence`` product covering ``[first, last]``."""
    if kind not in KINDS:
        raise ValueError(f"unknown baseline kind '{kind}', available: {list(KINDS)}")
    spec = truth if isinstance(truth, ProductSpec) else ProductSpec.load(truth)
    name = name or f"{kind}_{spec.name}"
    # A climatology needs the years before the scored window; persistence only needs a few days.
    span = (train[0] if train else None, last) if kind == "climatology" else \
        (str(pd.Timestamp(first) - pd.Timedelta(days=lag_days + 1))[:10], last)
    try:
        source = open_product(spec, span[0], span[1])
    except FileNotFoundError as exc:
        what = f"the training window {train}" if kind == "climatology" and train else f"[{span[0]}, {span[1]}]"
        raise ValueError(f"{spec.name} has no date in {what}: a {kind} baseline cannot be built from it "
                         f"({exc})") from exc
    dates = pd.DatetimeIndex(source.sel(time=slice(first, last)).time.values)
    if len(dates) == 0:
        raise ValueError(f"{spec.name} has no date in [{first}, {last}]")

    if kind == "climatology":
        field = climatology(source, dates, train, smooth_days)
    else:
        field = persistence(source, dates, lag_days)

    daily = Path(out_dir) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    field = field.load()
    # open_product records depth_m / ensemble_size in attrs and they may be None, which netCDF
    # refuses to serialise; the manifest carries them anyway.
    field.attrs = {k: v for k, v in field.attrs.items() if v is not None}
    for t in field.time.values:
        day = pd.Timestamp(t).strftime("%Y-%m-%d")
        field.sel(time=[t]).to_netcdf(daily / f"{name}_{day}.nc")

    manifest = {"format_version": PRODUCT_FORMAT_VERSION, "name": name, "path": str(daily.resolve()),
                "pattern": f"{name}_(\\d{{4}})-(\\d{{2}})-(\\d{{2}})\\.nc",
                "variables": {v: {"name": v} for v in field.data_vars},
                "coords": {"lat": "lat", "lon": "lon", "time": "time"},
                "label": f"{kind} of {spec.name}" + (f" (lag {lag_days} d)" if kind == "persistence" else ""),
                "time_coverage_hours": 24,
                "first_date": str(pd.Timestamp(dates[0]).date()), "last_date": str(pd.Timestamp(dates[-1]).date()),
                "attrs": {"baseline": kind, "truth": spec.name,
                          **({"train": f"{train[0]}..{train[1]}", "smooth_days": smooth_days} if kind == "climatology"
                             else {"lag_days": lag_days})}}
    check_manifest(manifest, source=str(Path(out_dir) / "product.yaml"))
    path = Path(out_dir) / "product.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"[baseline] {kind}: {len(dates)} days, {len(field.data_vars)} variables -> {path}")
    return path
