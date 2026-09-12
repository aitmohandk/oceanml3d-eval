"""Lagrangian metric: advect virtual particles in the product velocity field from real
drifter positions and compare with the real trajectories (separation distance after N days).

Backend 1 (default): built-in RK4 advection on the regular grid (numpy/xarray).
Backend 2 (optional): ``third_party/velocity_metrics`` (OceanDataLab, LGPL) used by NOSC,
selected with ``backend: velocity_metrics`` in the benchmark options.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from oceanml3d_eval.metrics.base import Metric, MetricResult, register_metric
from oceanml3d_eval.regions import Region

R_EARTH = 6371e3


def _velocity_at(prod: xr.Dataset, t, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    pts = dict(lat=xr.DataArray(lat, dims="p"), lon=xr.DataArray(lon, dims="p"))
    s = prod.sel(time=t, method="nearest").interp(**pts, method="linear")
    return np.nan_to_num(s.u.values), np.nan_to_num(s.v.values)


def advect(prod: xr.Dataset, t0, lat0: np.ndarray, lon0: np.ndarray, days: int, dt_hours: float = 3.0) -> pd.DataFrame:
    """RK4 advection; returns positions every day."""
    lat, lon = lat0.astype(float).copy(), lon0.astype(float).copy()
    dt = dt_hours * 3600
    steps = int(days * 24 / dt_hours)
    out = [pd.DataFrame({"day": 0, "p": np.arange(len(lat)), "lat": lat, "lon": lon})]
    t = pd.Timestamp(t0)

    def f(tt, la, lo):
        u, v = _velocity_at(prod, tt, la, lo)
        dlat = v / R_EARTH * 180 / np.pi
        dlon = u / (R_EARTH * np.cos(np.deg2rad(la))) * 180 / np.pi
        return dlat, dlon

    for s in range(1, steps + 1):
        k1 = f(t, lat, lon)
        k2 = f(t + pd.Timedelta(seconds=dt / 2), lat + dt / 2 * k1[0], lon + dt / 2 * k1[1])
        k3 = f(t + pd.Timedelta(seconds=dt / 2), lat + dt / 2 * k2[0], lon + dt / 2 * k2[1])
        k4 = f(t + pd.Timedelta(seconds=dt), lat + dt * k3[0], lon + dt * k3[1])
        lat = lat + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        lon = lon + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t = t + pd.Timedelta(seconds=dt)
        if s % int(24 / dt_hours) == 0:
            out.append(pd.DataFrame({"day": s * dt_hours / 24, "p": np.arange(len(lat)), "lat": lat, "lon": lon}))
    return pd.concat(out, ignore_index=True)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH / 1e3 * np.arcsin(np.sqrt(a))


@register_metric("lagrangian_drifters")
class LagrangianDrifters(Metric):
    needs = "drifters_points"

    def compute(self, product: xr.Dataset, reference: pd.DataFrame, region: Region, first: str, last: str) -> MetricResult:
        if self.options.get("backend") == "velocity_metrics":
            return self._velocity_metrics_backend(product, reference, region, first, last)
        days = int(self.options.get("days", 5))
        if "id" not in reference:
            raise ValueError("lagrangian_drifters needs an 'id' column to follow trajectories")
        obs = reference[region.mask(reference.lat.values, reference.lon.values)]
        obs = obs[(obs.time >= pd.Timestamp(first)) & (obs.time <= pd.Timestamp(last))]
        prod = region.subset(product).sel(time=slice(first, last)).load()
        starts = obs.sort_values("time").groupby("id").first().reset_index()
        starts = starts[starts.time <= pd.Timestamp(last) - pd.Timedelta(days=days)]
        if starts.empty:
            return MetricResult({f"sep_km_day{days}": np.nan, "n_traj": 0})
        seps = {}
        for t0, grp in starts.groupby(starts.time.dt.floor("D")):
            traj = advect(prod, t0, grp.lat.values, grp.lon.values, days)
            for d, sub in traj.groupby("day"):
                if d == 0:
                    continue
                real = obs[obs.id.isin(grp.id)].copy()
                real["dd"] = (real.time - t0).dt.total_seconds() / 86400
                real = real[np.abs(real.dd - d) < 0.25].groupby("id").first()
                ids = grp.id.values[sub.p.values]
                m = pd.Series(ids).isin(real.index).values
                if not m.any():
                    continue
                r = real.loc[ids[m]]
                seps.setdefault(d, []).append(haversine_km(sub.lat.values[m], sub.lon.values[m], r.lat.values, r.lon.values))
        res = {f"sep_km_day{int(d)}": float(np.concatenate(v).mean()) for d, v in seps.items()}
        res["n_traj"] = int(len(starts))
        return MetricResult(res, {}, {"region": region.name, "days": days})

    def _velocity_metrics_backend(self, product, reference, region, first, last) -> MetricResult:
        """Exact NOSC diagnostics. Options: ``product_spec`` (set by the runner), ``drifter_positions``,
        ``drifter_file``, ``parameter_file``, ``days``, ``sdepth``, ``work_dir``."""
        try:
            import velocity_metrics  # noqa: F401
        except ImportError as exc:
            raise ImportError("pip install -e third_party/velocity_metrics") from exc
        from pathlib import Path

        from oceanml3d_eval.velocity_metrics_backend import lagrangian

        o = self.options
        missing = [k for k in ("product_spec", "drifter_positions", "drifter_file", "parameter_file") if k not in o]
        if missing:
            raise ValueError(f"velocity_metrics backend needs options {missing}")
        scores = lagrangian(o["product_spec"], region, first, last, Path(o.get("work_dir", "results/velocity_metrics")),
                            o["drifter_positions"], o["drifter_file"], o["parameter_file"], int(o.get("days", 10)), int(o.get("sdepth", 0)))
        return MetricResult(scores, {}, {"region": region.name, "backend": "velocity_metrics"})
